"""
Configuration settings for the Werewolf LLM project.
"""
import logging
import os
from enum import Enum

logger = logging.getLogger(__name__)


class Backend(Enum):
    """Enum for supported hardware backends."""

    CPU = "cpu"
    CUDA = "cuda"
    ROCM = "rocm"


def get_backend() -> Backend:
    """
    Resolves the hardware backend from the BACKEND environment variable.
    """
    backend_str = os.getenv("BACKEND", "cpu").lower()
    try:
        backend = Backend(backend_str)
        logger.info(f"Using backend: {backend.value}")
        return backend
    except ValueError:
        logger.warning(
            f"Invalid BACKEND value '{backend_str}'. Defaulting to 'cpu'. "
            f"Valid options are: {[b.value for b in Backend]}"
        )
        return Backend.CPU


def is_accelerated(backend: Backend) -> bool:
    """
    Returns True if the backend supports GPU acceleration (CUDA/ROCm features).
    """
    return backend in (Backend.CUDA, Backend.ROCM)
