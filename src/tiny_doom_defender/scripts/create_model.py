"""
Initialize a fresh (untrained) conv-stem DOOM model and write it to --output as a
standard HF model dir (config.json + model.safetensors). Pass the dir to
train-sft --base-model to train a non-default architecture.

Usage:
    create-model --num-layers 6 --output models/doom-cnn-6L
"""

import argparse

from tiny_doom_defender.configuration_doom import DoomConvStemConfig, default_encoder_config
from tiny_doom_defender.modeling_doom import DoomConvStemForActionClassification


def main():
    ap = argparse.ArgumentParser(description="Create the conv-stem ModernBERT model")
    ap.add_argument("--hidden-size", type=int, default=None)
    ap.add_argument("--num-layers", type=int, default=None, help="4 -> ~1.05M brain.")
    ap.add_argument("--num-heads", type=int, default=None)
    ap.add_argument("--intermediate-size", type=int, default=None)
    ap.add_argument("--max-position-embeddings", type=int, default=None)
    ap.add_argument("--local-attention", type=int, default=None)
    ap.add_argument(
        "--global-attn-every-n-layers",
        type=int,
        default=None,
        help="With 4 layers, the default 3 keeps 0 and 3 global so the last (pool-read) layer sees the full grid.",
    )
    ap.add_argument("--vocab-size", type=int, default=None)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    print("=" * 60)
    print("DOOM Conv-Stem — Model Creator")
    print("=" * 60)

    overrides = {
        "hidden_size": args.hidden_size,
        "num_hidden_layers": args.num_layers,
        "num_attention_heads": args.num_heads,
        "intermediate_size": args.intermediate_size,
        "max_position_embeddings": args.max_position_embeddings,
        "local_attention": args.local_attention,
        "global_attn_every_n_layers": args.global_attn_every_n_layers,
        "vocab_size": args.vocab_size,
    }
    enc_cfg = default_encoder_config(**{k: v for k, v in overrides.items() if v is not None})
    config = DoomConvStemConfig(encoder_config=enc_cfg)

    n_layers = enc_cfg.num_hidden_layers
    print(f"\n[1/2] Building model ({n_layers}L, hidden {enc_cfg.hidden_size})...")
    model = DoomConvStemForActionClassification(config)
    global_layers = [i for i, t in enumerate(enc_cfg.layer_types) if t == "full_attention"]
    print(f"  Global-attention layers: {global_layers}  (last layer global: {n_layers - 1 in global_layers})")

    n_total = sum(p.numel() for p in model.parameters())
    n_stem = sum(p.numel() for p in model.stem.parameters())
    n_enc = sum(p.numel() for p in model.encoder.parameters())
    n_heads = n_total - n_stem - n_enc
    print("\n" + "=" * 60)
    print(f"  Stem:        {n_stem:>10,} ({n_stem / 1e6:.4f}M)")
    print(f"  Encoder:     {n_enc:>10,} ({n_enc / 1e6:.4f}M)")
    print(f"  Pool+heads:  {n_heads:>10,} ({n_heads / 1e6:.4f}M)")
    print(f"  TOTAL:       {n_total:>10,} ({n_total / 1e6:.4f}M)")
    print("=" * 60)

    print(f"\n[2/2] Saving model to {args.output}...")
    model.save_pretrained(args.output)
    print(f"Model saved to: {args.output}")


if __name__ == "__main__":
    main()
