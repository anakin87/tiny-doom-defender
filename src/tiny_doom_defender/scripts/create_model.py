"""
Initialize a fresh (untrained) conv-stem DOOM model and write it to --output.

Emits two artifacts:
  config.json + model.safetensors   the ModernBERT encoder (HF format). train_sft.py
                                     loads this as --base-model and wraps it with a
                                     freshly-initialized conv stem + heads.
  stem_config.json                  the conv-stem geometry (resolution, frames, grid).
                                     DoomConvStemClassifier reads it to rebuild the stem,
                                     so the model dir is self-describing.

The encoder's vocab is tiny on purpose: the stem feeds `inputs_embeds`, so its
token-embedding table is never used.

Usage:
    create-model --output models/doom-cnn-4L-no-fwd
"""

import argparse
import json
import os

from transformers import ModernBertConfig, ModernBertModel

from tiny_doom_defender.config import (
    GRID_H,
    GRID_W,
    IN_CH,
    N_ACTION_STATES,
    N_FRAMES,
    N_PREV,
    N_TOKENS,
    RES_H,
    RES_W,
)
from tiny_doom_defender.model import DoomConvStemClassifier


def main():
    ap = argparse.ArgumentParser(description="Create the conv-stem ModernBERT encoder")
    ap.add_argument("--hidden-size", type=int, default=128)
    ap.add_argument("--num-layers", type=int, default=4, help="4 -> ~1.05M brain.")
    ap.add_argument("--num-heads", type=int, default=4)
    ap.add_argument("--intermediate-size", type=int, default=512)
    ap.add_argument(
        "--max-position-embeddings",
        type=int,
        default=1536,
        help="RoPE cap; must be >= N_TOKENS (1000). Not a param table.",
    )
    ap.add_argument("--local-attention", type=int, default=128)
    ap.add_argument(
        "--global-attn-every-n-layers",
        type=int,
        default=3,
        help="With 4 layers, keeps 0 and 3 global so the last (pool-read) layer sees the full grid.",
    )
    ap.add_argument(
        "--vocab-size",
        type=int,
        default=32,
        help="Token-embedding table is dead weight (inputs_embeds only); keep the vocab tiny so it costs ~nothing.",
    )
    ap.add_argument("--output", default="models/doom-cnn-4L-no-fwd")
    args = ap.parse_args()

    print("=" * 60)
    print("DOOM Conv-Stem — Encoder Creator")
    print("=" * 60)

    config = ModernBertConfig(
        vocab_size=args.vocab_size,
        hidden_size=args.hidden_size,
        num_hidden_layers=args.num_layers,
        num_attention_heads=args.num_heads,
        intermediate_size=args.intermediate_size,
        max_position_embeddings=args.max_position_embeddings,
        model_type="modernbert",
        attention_bias=False,
        attention_dropout=0.0,
        local_attention=args.local_attention,
        global_attn_every_n_layers=args.global_attn_every_n_layers,
        local_rope_theta=10000.0,
        global_rope_theta=160000.0,
        hidden_activation="gelu",
        embedding_dropout=0.0,
        mlp_dropout=0.0,
        layer_norm_eps=1e-5,
        norm_eps=1e-5,
        mlp_bias=False,
        norm_bias=False,
        initializer_range=0.02,
        initializer_cutoff_factor=2.0,
        tie_word_embeddings=False,
        use_cache=False,
        # All unused (we feed inputs_embeds, no tokenizer) — pin to 0 so they stay
        # inside the tiny vocab and don't trip config validation warnings.
        pad_token_id=0,
        bos_token_id=0,
        eos_token_id=0,
        cls_token_id=0,
        sep_token_id=0,
    )

    print(f"\n[1/3] Building encoder ({args.num_layers}L, hidden {args.hidden_size})...")
    encoder = ModernBertModel(config)
    global_layers = [i for i in range(args.num_layers) if i % args.global_attn_every_n_layers == 0]
    print(f"  Global-attention layers: {global_layers}  (last layer global: {args.num_layers - 1 in global_layers})")

    print(f"\n[2/3] Saving encoder to {args.output}...")
    os.makedirs(args.output, exist_ok=True)
    encoder.save_pretrained(args.output)
    stem_config = {
        "input_resolution": [RES_H, RES_W],
        "n_frames": N_FRAMES,
        "in_channels": IN_CH,
        "n_prev_actions": N_PREV,
        "n_action_states": N_ACTION_STATES,
        "token_grid": [GRID_H, GRID_W],
        "n_tokens": N_TOKENS,
        "stem": "Conv2d(9->32,s2) -> ReLU -> Conv2d(32->128,s2) -> ReLU",
    }
    with open(os.path.join(args.output, "stem_config.json"), "w") as f:
        json.dump(stem_config, f, indent=2)

    print("\n[3/3] Assembling full classifier to count params...")
    clf = DoomConvStemClassifier(args.output)
    n_total = sum(p.numel() for p in clf.parameters())
    n_stem = sum(p.numel() for p in clf.stem.parameters())
    n_enc = sum(p.numel() for p in clf.encoder.parameters())
    n_heads = n_total - n_stem - n_enc

    print("\n" + "=" * 60)
    print(f"  Stem:        {n_stem:>10,} ({n_stem / 1e6:.4f}M)")
    print(f"  Encoder:     {n_enc:>10,} ({n_enc / 1e6:.4f}M)")
    print(f"  Pool+heads:  {n_heads:>10,} ({n_heads / 1e6:.4f}M)")
    print(f"  TOTAL:       {n_total:>10,} ({n_total / 1e6:.4f}M)")
    print("=" * 60)
    print(f"\nEncoder saved to: {args.output}")


if __name__ == "__main__":
    main()
