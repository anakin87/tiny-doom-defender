"""
int8 quantization so policy + code fit on a 1.44MB floppy.

Symmetric int8: per-output-channel scales for >=2D tensors, per-tensor for 1D.
Scales ride along in the same safetensors file under "<key>::scale", and config.json
records the scheme so evaluation.build_policy knows to dequantize on load.

Usage:
  quantize-int8 --ckpt output/cnn-ppo/policy_best --out output/cnn-ppo/policy_best_int8
"""

import argparse
import os

import torch
from safetensors.torch import load_file, save_file

from tiny_doom_defender.configuration_doom import DoomConvStemConfig
from tiny_doom_defender.evaluation import INT8_SCHEME, SCALE_SUFFIX


def quantize_tensor(v):
    if v.ndim >= 2:
        s = v.abs().amax(dim=tuple(range(1, v.ndim)), keepdim=True) / 127.0
    else:
        s = v.abs().amax().reshape(1) / 127.0
    s = torch.clamp(s, min=1e-12)
    q = torch.clamp((v / s).round(), -127, 127).to(torch.int8)
    return q, s


def quantize_checkpoint(ckpt, out):
    sd = load_file(os.path.join(ckpt, "model.safetensors"))
    packed, err_report, n_copied = {}, [], 0
    for k, v in sd.items():
        if v.dtype != torch.float32:
            packed[k] = v
            n_copied += 1
            continue
        q, s = quantize_tensor(v)
        packed[k] = q
        packed[k + SCALE_SUFFIX] = s
        rel = ((q.float() * s - v).norm() / v.norm().clamp(min=1e-12)).item()
        err_report.append((rel, k))
    os.makedirs(out, exist_ok=True)
    save_file(packed, os.path.join(out, "model_int8.safetensors"))
    cfg = DoomConvStemConfig.from_pretrained(ckpt)
    cfg.quantization = INT8_SCHEME
    cfg.save_pretrained(out)
    return err_report, n_copied


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", default="output/cnn-ppo/policy_best")
    ap.add_argument("--out", default="output/cnn-ppo/policy_best_int8")
    args = ap.parse_args()

    if os.path.abspath(args.ckpt) == os.path.abspath(args.out):
        ap.error("--out must differ from --ckpt")

    report, n_copied = quantize_checkpoint(args.ckpt, args.out)
    size = os.path.getsize(os.path.join(args.out, "model_int8.safetensors"))
    print(f"Wrote {args.out}/model_int8.safetensors ({size:,} bytes)")
    print(f"{len(report)} tensors quantized, {n_copied} copied as-is (not fp32)")
    print("Worst relative quantization error per tensor:")
    for rel, k in sorted(report, reverse=True)[:5]:
        print(f"  {rel:.4f}  {k}")


if __name__ == "__main__":
    main()
