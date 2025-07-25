"""
This module contains the core logic for Supervised Fine-Tuning (SFT)
using the Hugging Face TRL and PEFT libraries.
"""
import logging
from pathlib import Path
from typing import Dict

import torch
from datasets import load_dataset
from peft import LoraConfig
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    PreTrainedModel,
)
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from trl import SFTTrainer, SFTConfig

from .config import Backend, get_backend

logger = logging.getLogger(__name__)


def find_lora_target_modules(model) -> list[str]:
    """Find all linear layers in the model to apply LoRA."""
    target_modules = set()
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear) and "lm_head" not in name:
            # Split the name by dots and take the last part
            module_name = name.split('.')[-1]
            target_modules.add(module_name)
    
    # Prioritize common module names
    priority_order = ["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    
    sorted_modules = sorted(list(target_modules), key=lambda x: priority_order.index(x) if x in priority_order else len(priority_order))
    
    logger.info(f"Found LoRA target modules: {sorted_modules}")
    return sorted_modules


class SFTTrainerWrapper:
    """
    A wrapper class to manage the SFT process.
    """

    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        training_run_dir: Path,
        sft_data_path: Path,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.training_run_dir = training_run_dir
        self.sft_data_path = sft_data_path
        self.output_dir = self.training_run_dir

    def _format_dataset_entry(self, entry: Dict) -> str:
        """Format a dataset entry into a single string for training."""
        # The model should learn to generate the completion given the prompt
        return f"{entry['prompt']}\n\n{entry['completion']}"

    def train(self):
        """
        Executes the SFT process.
        """
        logger.info("--- Starting SFT ---")

        # 1. Load dataset
        logger.info(f"Loading dataset from {self.sft_data_path}")
        dataset = load_dataset("json", data_files=str(self.sft_data_path), split="train")

        # 2. Configure LoRA adapter
        logger.info("Configuring LoRA adapter...")
        lora_config = LoraConfig(
            r=16,
            lora_alpha=32,
            target_modules=find_lora_target_modules(self.model),
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
        )

        # 3. Set up SFTTrainer
        # Use a smaller batch size if on CPU
        backend = get_backend()
        per_device_train_batch_size = 1 if backend == Backend.CPU else 4
        gradient_accumulation_steps = 4 if backend == Backend.CPU else 1

        training_args = SFTConfig(
            output_dir=str(self.output_dir),
            num_train_epochs=3,
            per_device_train_batch_size=per_device_train_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            learning_rate=2e-4,
            logging_steps=10,
            save_strategy="epoch",
            optim="paged_adamw_8bit" if backend == Backend.CUDA else "adamw_torch",
            fp16=backend == Backend.CUDA,  # Enable fp16 only for CUDA
            packing=True,
        )

        logger.info("Initializing SFTTrainer...")
        trainer = SFTTrainer(
            model=self.model,
            args=training_args,
            peft_config=lora_config,
            train_dataset=dataset,
            formatting_func=self._format_dataset_entry,
            processing_class=self.tokenizer,
        )

        # 4. Run training
        logger.info("Starting training...")
        trainer.train()
        logger.info("Training finished.")

        # 5. Save adapter
        logger.info(f"Saving LoRA adapter to {self.output_dir}")
        trainer.save_model(str(self.output_dir))
        logger.info("SFT process completed successfully.")
