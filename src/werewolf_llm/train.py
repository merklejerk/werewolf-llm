import argparse
import logging
from pathlib import Path
import sys
from typing import Tuple, Optional

import torch
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.utils.quantization_config import BitsAndBytesConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define project structure paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
CHECKPOINTS_DIR = PROJECT_ROOT / "checkpoints"


class TrainingManifest(BaseModel):
    """Defines the structure of the manifest.json file in a checkpoints directory."""
    base_model: str


def load_or_create_model(
    training_name: str, base_model_name: Optional[str]
) -> Tuple[AutoModelForCausalLM, AutoTokenizer, Path]:
    """
    Load a model from a training checkpoint, or create a new one from a base model.
    """
    training_run_dir = CHECKPOINTS_DIR / training_name
    
    # Configure quantization for memory efficiency
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    if training_run_dir.exists():
        logger.info(f"Loading existing model from checkpoint: {training_run_dir}")
        model = AutoModelForCausalLM.from_pretrained(
            training_run_dir,
            quantization_config=bnb_config,
            device_map="auto",
        )
        tokenizer = AutoTokenizer.from_pretrained(training_run_dir)
    else:
        if not base_model_name:
            logger.error(
                f"Training run '{training_name}' not found. "
                "Please specify a --base model to create it."
            )
            sys.exit(1)
            
        logger.info(f"Creating new training run '{training_name}' from base model '{base_model_name}'")
        training_run_dir.mkdir(parents=True, exist_ok=True)

        model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            quantization_config=bnb_config,
            device_map="auto",
        )
        tokenizer = AutoTokenizer.from_pretrained(base_model_name)
        tokenizer.pad_token = tokenizer.eos_token

        logger.info(f"Saving initial checkpoint to {training_run_dir}")
        model.save_pretrained(training_run_dir)
        tokenizer.save_pretrained(training_run_dir)
        
        # Save a manifest for future reference
        manifest = TrainingManifest(base_model=base_model_name)
        with open(training_run_dir / "manifest.json", "w") as f:
            f.write(manifest.model_dump_json(indent=2))

    return model, tokenizer, training_run_dir


def run_sft(model: AutoModelForCausalLM, tokenizer: AutoTokenizer, training_run_dir: Path, sft_data_path: Path):
    """Stub for the Supervised Fine-Tuning process."""
    logger.info("--- Running Supervised Fine-Tuning (SFT) ---")
    logger.info(f"Loading SFT dataset from: {sft_data_path}")
    # TODO: Implement SFT
    # 1. Load dataset from sft_data_path
    # 2. Configure LoRA adapter
    # 3. Set up SFTTrainer
    # 4. Run training
    # 5. Save adapter to training_run_dir
    logger.info("SFT logic will be implemented here.")
    logger.info(f"Model and tokenizer are loaded. Checkpoint is at {training_run_dir}")


def run_rl(model: AutoModelForCausalLM, tokenizer: AutoTokenizer, training_run_dir: Path, rl_data_path: Path):
    """Stub for the Reinforcement Learning process."""
    logger.info("--- Running Reinforcement Learning (RL) ---")
    logger.info(f"Loading RL configuration from: {rl_data_path}")
    # TODO: Implement RL
    # 1. Load SFT adapter from path provided in args
    # 2. Set up game environment/simulator
    # 3. Configure PPO trainer
    # 4. Run self-play training loop
    # 5. Save final adapter to training_run_dir
    logger.info("RL logic will be implemented here.")
    logger.info(f"Model and tokenizer are loaded. Checkpoint is at {training_run_dir}")


def main():
    """Main function to orchestrate the training process."""
    parser = argparse.ArgumentParser(description="Run training for Werewolf LLM.")
    parser.add_argument("name", type=str,
                        help="A unique name for this training run (e.g., 'my-llama3-run').")
    parser.add_argument("-b", "--base", type=str, dest="base_model_name",
                        help="The Hugging Face model ID to use as a base (required for new runs).")
    
    # Mutually exclusive group for training mode based on data
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument("--sft", type=Path,
                            help="Path to the SFT dataset file. Triggers SFT mode.")
    mode_group.add_argument("--rl", type=Path,
                            help="Path to the RL configuration file. Triggers RL mode.")

    args = parser.parse_args()
    
    logger.info(f"Starting training script for run '{args.name}'...")
    
    try:
        model, tokenizer, training_run_dir = load_or_create_model(
            args.name, args.base_model_name
        )
    except Exception as e:
        logger.error(f"Failed to load or create model: {e}")
        logger.error("If using a gated model, ensure you are logged in via `huggingface-cli login`.")
        sys.exit(1)
        
    # Execute the selected training mode
    if args.sft:
        run_sft(model, tokenizer, training_run_dir, args.sft)
    elif args.rl:
        run_rl(model, tokenizer, training_run_dir, args.rl)

    logger.info("Script finished.")


if __name__ == "__main__":
    main()
