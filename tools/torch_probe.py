#!/usr/bin/env python3
"""A small PyTorch ROCm job for rocprofv3: a BLAS matmul, an MIOpen conv and elementwise kernels, ROUNDS rounds.
Prints the wall time of each round, so cold clocks and warm rounds show. usage: torch_probe.py [ROUNDS] [SIZE]"""
import sys
import time

import torch

rounds = int(sys.argv[1]) if len(sys.argv) > 1 else 40
size = int(sys.argv[2]) if len(sys.argv) > 2 else 2048
print("torch", torch.__version__, "gpu", torch.cuda.get_device_name(0), flush=True)
dev = "cuda"
a = torch.randn(size, size, device=dev)
conv = torch.nn.Conv2d(32, 32, 3, padding=1).to(dev)
img = torch.randn(1, 32, 512, 512, device=dev)
start = time.time()
for i in range(rounds):
    t = time.time()
    b = a @ a
    c = conv(img)
    d = (b * 0.5 + b.sin()).mean() + c.relu().sum()
    d.item()
    print(f"round {i} {(time.time() - t) * 1e3:.1f} ms", flush=True)
print(f"total {time.time() - start:.2f} s", flush=True)
