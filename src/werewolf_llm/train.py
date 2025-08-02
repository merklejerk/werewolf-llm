import argparse
import logging
from pathlib import Path
import sys
from typing import Tuple, Optional, Dict, Any
import asyncio
import json

from dataclasses import asdict

import torch
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM, PreTrainedModel
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from transformers.utils.quantization_config import BitsAndBytesConfig

from .sft_trainer import train_sft
# from .ppo_trainer import PPOTrainerWrapper
# from .vllm_agent import VLLMServer
from .config import Backend, get_backend, is_accelerated
from .generate_sft_data import generate_sft_dataset


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define project structure paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
CHECKPOINTS_DIR = PROJECT_ROOT / "checkpoints"
MAX_SEQ_LENGTH = 8192


class TrainingManifest(BaseModel):
    """Defines the structure of the manifest.json file in a checkpoints directory."""
    base_model: str


def load_or_create_model(
    training_name: str, base_model_name: Optional[str]
) -> Tuple[PreTrainedModel, PreTrainedTokenizerBase, Path]:
    """
    Load a model from a training checkpoint, or create a new one from a base model.
    """
    training_run_dir = CHECKPOINTS_DIR / training_name
    config_path = training_run_dir / "config.json"
    
    # Configure quantization for memory efficiency
    backend = get_backend()
    if is_accelerated(backend):
        # Use bfloat16 for GPU backends (CUDA/ROCm)
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        logger.info(f"Using 4-bit quantization with bfloat16 for {backend.value} backend")
    else:
        # Use float32 for CPU backend
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
        )
        logger.info(f"Using 4-bit quantization with float16 for {backend.value} backend")
        bnb_config = None # Stalls with valid bnb config on CPU backend?


    model_kwargs: Dict[str, Any] = {
        "device_map": "auto",
        "attn_implementation": "flash_attention_2" if is_accelerated(backend) else "eager",
    }
    if bnb_config is not None:
        model_kwargs["quantization_config"] = bnb_config
    
    if training_run_dir.exists() and config_path.exists():
        logger.info(f"Loading existing model from checkpoint: {training_run_dir}")
            
        model = AutoModelForCausalLM.from_pretrained(
            training_run_dir,
            **model_kwargs
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
            **model_kwargs
        )
        tokenizer = AutoTokenizer.from_pretrained(base_model_name, max_seq_length=MAX_SEQ_LENGTH)
        tokenizer.pad_token = tokenizer.eos_token

        logger.info(f"Saving initial checkpoint to {training_run_dir}")
        model.save_pretrained(training_run_dir)
        tokenizer.save_pretrained(training_run_dir)
        
        # Save a manifest for future reference
        manifest = TrainingManifest(base_model=base_model_name)
        with open(training_run_dir / "manifest.json", "w") as f:
            f.write(manifest.model_dump_json(indent=2))

    return model, tokenizer, training_run_dir


def run_sft(model: PreTrainedModel, tokenizer: PreTrainedTokenizerBase, training_run_dir: Path, dataset_count: int):
    """Runs the Supervised Fine-Tuning process by generating data on the fly."""
    def sft_dataset_generator():
        # Generate SFT data synchronously and yield one data point at a time
        for dp in generate_sft_dataset(dataset_count):
            yield asdict(dp)

    train_sft(
        model=model,
        tokenizer=tokenizer,
        training_run_dir=training_run_dir,
        dataset_generator=sft_dataset_generator,
    )


def run_rl(model: PreTrainedModel, tokenizer: PreTrainedTokenizerBase, training_run_dir: Path, rl_config_path: Path):
    raise NotImplementedError("Reinforcement Learning (RL) training is not yet implemented.")
    # """Runs the Reinforcement Learning (PPO) process."""
    # logger.info("--- Running Reinforcement Learning (RL) ---")
    # logger.info(f"Loading RL configuration from: {rl_config_path}")
    
    # # Load RL configuration
    # try:
    #     with open(rl_config_path) as f:
    #         rl_config = json.load(f)
    # except Exception as e:
    #     logger.error(f"Failed to load RL config: {e}")
    #     sys.exit(1)
    
    # # Extract configuration parameters
    # vllm_model_path = rl_config.get("vllm_model_path", str(training_run_dir))
    # games_per_batch = rl_config.get("games_per_batch", 8)
    # max_rounds = rl_config.get("max_rounds", 3)
    # num_rollouts = rl_config.get("num_rollouts", 1000)
    # sync_frequency = rl_config.get("sync_frequency", 100)
    # vllm_port = rl_config.get("vllm_port", 8000)
    
    # async def run_rl_training():
    #     # Start VLLM server
    #     vllm_server = VLLMServer(
    #         model_path=vllm_model_path,
    #         port=vllm_port,
    #     )
        
    #     try:
    #         logger.info("Starting VLLM server...")
    #         await vllm_server.start()
            
    #         # Initialize PPO trainer
    #         ppo_trainer = PPOTrainerWrapper(
    #             model=model,
    #             tokenizer=tokenizer,
    #             training_run_dir=training_run_dir,
    #             vllm_model_path=vllm_model_path,
    #             games_per_batch=games_per_batch,
    #             max_rounds=max_rounds,
    #             sync_frequency=sync_frequency,
    #         )
            
    #         # Run training
    #         await ppo_trainer.train(num_rollouts=num_rollouts)
            
    #     finally:
    #         # Clean up VLLM server
    #         vllm_server.stop()
    
    # # Run the async training loop
    # asyncio.run(run_rl_training())


def main():
    """Main function to orchestrate the training process."""
    parser = argparse.ArgumentParser(description="Run training for Werewolf LLM.")
    parser.add_argument("name", type=str,
                        help="A unique name for this training run (e.g., 'my-llama3-run').")
    parser.add_argument("-b", "--base", type=str, dest="base_model_name",
                        help="The Hugging Face model ID to use as a base (required for new runs).")
    
    # Mutually exclusive group for training mode
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument("--sft", type=int,
                            help="Size of dataset to generate. Triggers SFT mode.")
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
    if args.sft is not None:
        run_sft(model, tokenizer, training_run_dir, args.sft)
    elif args.rl:
        run_rl(model, tokenizer, training_run_dir, args.rl)

    logger.info("Script finished.")


if __name__ == "__main__":
    main()
