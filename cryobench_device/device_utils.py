"""Device detection and management utilities for cross-platform GPU support.

This module provides utilities for detecting and configuring GPU devices across
different backends (CUDA, MPS/Metal, CPU) to enable CryoBench to run on various
hardware platforms including NVIDIA GPUs, Apple Silicon (M-Series), and CPU-only
systems.

This module mirrors the device_utils API from the MPS-enabled cryodrgn fork
for consistency.
"""

import logging
import torch
from typing import Optional, Tuple

logger = logging.getLogger(__name__)


def get_available_device(
    device: Optional[str] = None, verbose: bool = True
) -> Tuple[torch.device, str]:
    """
    Detect and return the best available compute device.

    Priority order (if device not specified):
    1. CUDA (NVIDIA GPUs)
    2. MPS (Apple Metal/M-Series)
    3. CPU (fallback)

    Args:
        device: Optional device string ('cuda', 'mps', 'cpu', or None for auto-detect)
        verbose: Whether to log device selection information

    Returns:
        Tuple of (torch.device, device_type_string)
    """
    # If device explicitly specified, validate and return it
    if device is not None:
        device_lower = device.lower()
        if device_lower.startswith("cuda"):
            if not torch.cuda.is_available():
                raise RuntimeError(
                    "CUDA device requested but CUDA is not available. "
                    "Please check your PyTorch installation and GPU drivers."
                )
            device_obj = torch.device(device_lower)
            if verbose:
                logger.info(
                    f"Using CUDA device: {torch.cuda.get_device_name(0)} "
                    f"(explicitly requested)"
                )
            return device_obj, "cuda"
        elif device_lower == "mps":
            if not hasattr(torch.backends, "mps") or not torch.backends.mps.is_available():
                raise RuntimeError(
                    "MPS device requested but MPS is not available. "
                    "Please ensure you are running on Apple Silicon with PyTorch >= 2.0.0."
                )
            device_obj = torch.device("mps")
            if verbose:
                logger.info("Using MPS (Metal) device (explicitly requested)")
            return device_obj, "mps"
        elif device_lower == "cpu":
            device_obj = torch.device("cpu")
            if verbose:
                logger.info("Using CPU device (explicitly requested)")
            return device_obj, "cpu"
        else:
            raise ValueError(
                f"Invalid device '{device}'. Must be 'cuda', 'mps', 'cpu', or None."
            )

    # Auto-detect best available device
    if torch.cuda.is_available():
        device_obj = torch.device("cuda")
        device_str = "cuda"
        if verbose:
            logger.info(
                f"Using CUDA device: {torch.cuda.get_device_name(0)} "
                f"({torch.cuda.device_count()} GPU(s) available)"
            )
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device_obj = torch.device("mps")
        device_str = "mps"
        if verbose:
            logger.info("Using MPS (Metal) device on Apple Silicon")
    else:
        device_obj = torch.device("cpu")
        device_str = "cpu"
        if verbose:
            logger.info("Using CPU device (no GPU acceleration available)")

    return device_obj, device_str


def get_device_string(device_index: Optional[int] = None) -> str:
    """
    Get appropriate device string for the current platform.

    Args:
        device_index: Optional CUDA device index (ignored for MPS/CPU)

    Returns:
        Device string suitable for torch.device()
    """
    if torch.cuda.is_available():
        if device_index is not None:
            return f"cuda:{device_index}"
        return "cuda:0"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"


def set_default_tensor_type(device_type: str):
    """
    Set the default tensor type based on device.

    Note: This function is provided for backward compatibility.
    Modern PyTorch code should use .to(device) instead.

    Args:
        device_type: Device type string ('cuda', 'mps', 'cpu')
    """
    if device_type == "cuda":
        torch.set_default_tensor_type(torch.cuda.FloatTensor)
    elif device_type == "mps":
        # MPS doesn't have a default tensor type setter
        # Operations should use explicit .to(device) calls
        logger.info("MPS device: using explicit .to(device) for tensor placement")
    else:
        # CPU is the default
        torch.set_default_tensor_type(torch.FloatTensor)


def log_device_info():
    """Log detailed information about available devices."""
    logger.info("=" * 60)
    logger.info("Device Information:")

    if torch.cuda.is_available():
        logger.info(f"  CUDA available: True")
        logger.info(f"  CUDA version: {torch.version.cuda}")
        logger.info(f"  Number of GPUs: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            logger.info(f"    GPU {i}: {torch.cuda.get_device_name(i)}")
    else:
        logger.info(f"  CUDA available: False")

    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        logger.info(f"  MPS (Metal) available: True")
        logger.info("  Running on Apple Silicon (M-Series)")
    else:
        logger.info(f"  MPS (Metal) available: False")

    logger.info(f"  PyTorch version: {torch.__version__}")
    logger.info("=" * 60)
