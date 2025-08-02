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


def format_dataset(dataset: Dataset, tokenizer: PreTrainedTokenizerBase) -> Dataset:
    """
    Formats the dataset by applying the chat template to the 'messages' column.
    """
    def apply_template(example):
        return {"text": tokenizer.apply_chat_template(example["messages"], tokenize=False)}

    return dataset.map(apply_template, remove_columns=list(dataset.features))


def train_sft(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    training_run_dir: Path,
    dataset_generator,
):
    """
    Executes the SFT process using a dataset generator.
    """
    logger.info("--- Starting SFT ---")

    # 1. Create Hugging Face Dataset from the generator
    logger.info("Creating dataset from generator...")
    train_dataset = Dataset.from_generator(dataset_generator)
    # Ensure correct Dataset type for further processing
    train_dataset = cast(Dataset, train_dataset)
    logger.info("Dataset created from generator.")

    # 2. Format dataset
    formatted_dataset = format_dataset(train_dataset, tokenizer)
    logger.info("Dataset formatted for training.")

    # 3. Configure LoRA adapter
    logger.info("Configuring LoRA adapter...")
    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=find_lora_target_modules(model),
        lora_dropout=0.01,
        bias="none",
        task_type="CAUSAL_LM",
    )

    # 4. Set up SFTTrainer
    backend = get_backend()
    per_device_train_batch_size = 1 if backend == Backend.CPU else 4
    gradient_accumulation_steps = 4 if backend == Backend.CPU else 1

    training_args = SFTConfig(
        output_dir=str(training_run_dir),
        num_train_epochs=3,
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=2e-4,
        logging_steps=1,
        save_strategy="epoch",
        optim="paged_adamw_8bit" if is_accelerated(backend) else "adamw_torch",
        bf16=is_accelerated(backend),
        packing=True,
        logging_first_step=True,
        max_seq_length=2048,
        dataloader_num_workers=0,
        dataset_num_proc=4,
    )

    logger.info("Initializing SFTTrainer...")
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        peft_config=lora_config,
        train_dataset=formatted_dataset,
    )

    # 5. Run training
    logger.info("Starting training...")
    trainer.train()
    logger.info("Training finished.")

    # 6. Save adapter
    logger.info(f"Saving LoRA adapter to {training_run_dir}")
    trainer.save_model(str(training_run_dir))
    logger.info("SFT process completed successfully.")
