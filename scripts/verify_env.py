"""
quantlab/scripts/verify_env.py

Environment sanity check script to verify PyTorch CUDA installation,
NVIDIA driver detection, and quantization backend functionality.
"""

import sys
import torch
import pynvml
import bitsandbytes as bnb


def verify_environment() -> None:
    print("=" * 60)
    print("           QUANTLAB ENVIRONMENT VERIFICATION           ")
    print("=" * 60)

    # 1. Python & PyTorch
    print(f"[+] Python Version      : {sys.version.split()[0]}")
    print(f"[+] PyTorch Version     : {torch.__version__}")

    # 2. CUDA Hardware Verification
    cuda_available = torch.cuda.is_available()
    print(f"[+] CUDA Available      : {cuda_available}")

    if not cuda_available:
        print("[ERROR] CUDA is not available to PyTorch. Check GPU drivers.")
        sys.exit(1)

    device_count = torch.cuda.device_count()
    device_name = torch.cuda.get_device_name(0)
    cuda_version = torch.version.cuda

    print(f"[+] CUDA Compute Version: {cuda_version}")
    print(f"[+] Detected Devices    : {device_count}")
    print(f"[+] Target Device (0)   : {device_name}")

    # 3. NVML Driver Connection Check
    try:
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        total_vram_gb = mem_info.total / (1024**3)
        print(f"[+] NVML Total VRAM     : {total_vram_gb:.2f} GB")
        pynvml.nvmlShutdown()
    except Exception as e:
        print(f"[WARNING] NVML initialization failed: {e}")

    # 4. BitsAndBytes Quantization Backend Check
    try:
        # Create a sample 8-bit linear layer to verify CUDA kernel compilation
        sample_layer = bnb.nn.Linear8bitLt(32, 64, bias=False).cuda()
        dummy_in = torch.randn(4, 32, device="cuda")
        dummy_out = sample_layer(dummy_in)
        print(f"[+] bitsandbytes Backend: FUNCTIONAL (INT8 Kernel Executed)")
    except Exception as e:
        print(f"[ERROR] bitsandbytes test failed: {e}")
        sys.exit(1)

    print("=" * 60)
    print("[SUCCESS] Environment fully configured and ready for Milestone 1!")
    print("=" * 60)


if __name__ == "__main__":
    verify_environment()
