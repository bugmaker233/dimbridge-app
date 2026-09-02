from __future__ import annotations

import platform
import sys

import torch


def main() -> int:
    print(f"Python: {platform.python_version()}")
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")

    if not torch.cuda.is_available():
        print("ERROR: PyTorch cannot access CUDA on this server.")
        return 1

    print(f"CUDA runtime: {torch.version.cuda}")
    print(f"GPU count: {torch.cuda.device_count()}")
    for index in range(torch.cuda.device_count()):
        print(f"GPU {index}: {torch.cuda.get_device_name(index)}")

    left = torch.randn((512, 512), device="cuda")
    right = torch.randn((512, 512), device="cuda")
    result = left @ right
    torch.cuda.synchronize()
    print(f"CUDA smoke result: {result.mean().item():.6f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
