"""
This module contains the core logic for Supervised Fine-Tuning (SFT)
using the Hugging Face TRL and PEFT libraries.
"""
import logging
from pathlib import Path
from typing import Dict, cast

import torch
from datasets import load_dataset, Dataset
from peft import LoraConfig
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    PreTrainedModel,
)
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from trl import SFTTrainer, SFTConfig

from .config import Backend, get_backend, is_accelerated

logger = logging.getLogger(__name__)

# LoRA target modules to use (subset of commonly available modules)
# Focus on attention layers as specified in the project spec
LORA_TARGET_MODULES = ["q_proj", "v_proj", "k_proj", "o_proj"]


def find_lora_target_modules(model) -> list[str]:
    """Find available linear layers in the model and return a subset for LoRA."""
    available_modules = set()
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear) and "lm_head" not in name:
            # Split the name by dots and take the last part
            module_name = name.split('.')[-1]
            available_modules.add(module_name)
    
    # Select only the modules from our predefined subset that are available
    target_modules = [module for module in LORA_TARGET_MODULES if module in available_modules]
    
    logger.info(f"Available modules: {sorted(available_modules)}")
    logger.info(f"Selected LoRA target modules: {target_modules}")
    return target_modules


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

    def train(self):
        """
        Executes the SFT process.
        """
        logger.info("--- Starting SFT ---")

        # 1. Load dataset
        logger.info(f"Loading dataset from {self.sft_data_path}")
        dataset = cast(Dataset, load_dataset("json", data_files=str(self.sft_data_path), split="train"))

        # 2. Configure LoRA adapter
        logger.info("Configuring LoRA adapter...")
        lora_config = LoraConfig(
            r=8,
            lora_alpha=16,
            target_modules=find_lora_target_modules(self.model),
            lora_dropout=0.01,
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
            logging_steps=1,
            save_strategy="epoch",
            optim="paged_adamw_8bit" if is_accelerated(backend) else "adamw_torch",
            bf16=is_accelerated(backend),
            completion_only_loss=True,
            packing=False,  # Disable packing to avoid cross-contamination without flash attention
            logging_first_step=True,
            max_seq_length=2048,
            dataloader_num_workers=0,
        )

        logger.info("Initializing SFTTrainer...")
        trainer = SFTTrainer(
            model=self.model,
            args=training_args,
            peft_config=lora_config,
            train_dataset=dataset,
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
