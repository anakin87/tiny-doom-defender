"""HF config for the conv-stem DOOM model: stem geometry + a nested ModernBERT encoder
config, so the full architecture travels with every checkpoint.

constants.py (the pipeline geometry) supplies the defaults; utils.check_pipeline_config
compares a saved model's config back against it.
"""

from huggingface_hub.dataclasses import strict
from transformers import ModernBertConfig
from transformers.configuration_utils import PreTrainedConfig

from tiny_doom_defender import constants as pipeline


def default_encoder_config(
    hidden_size=128,
    num_hidden_layers=4,
    num_attention_heads=4,
    intermediate_size=512,
    max_position_embeddings=1536,  # RoPE cap; must be >= n_tokens (1000). Not a param table.
    local_attention=128,
    global_attn_every_n_layers=3,
    vocab_size=32,  # dead weight: the stem feeds inputs_embeds, tokens are never used
):
    """The tiny doom encoder (~1.05M params at the 4L defaults)."""
    return ModernBertConfig(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        num_hidden_layers=num_hidden_layers,
        num_attention_heads=num_attention_heads,
        intermediate_size=intermediate_size,
        max_position_embeddings=max_position_embeddings,
        attention_bias=False,
        attention_dropout=0.0,
        local_attention=local_attention,
        global_attn_every_n_layers=global_attn_every_n_layers,
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
        # All unused (inputs_embeds only, no tokenizer) — pinned inside the tiny vocab.
        pad_token_id=0,
        bos_token_id=0,
        eos_token_id=0,
        cls_token_id=0,
        sep_token_id=0,
    )


@strict
class DoomConvStemConfig(PreTrainedConfig):
    model_type = "doom_conv_stem"
    sub_configs = {"encoder_config": ModernBertConfig}

    encoder_config: dict | ModernBertConfig | None = None
    res_h: int = pipeline.RES_H
    res_w: int = pipeline.RES_W
    n_frames: int = pipeline.N_FRAMES
    n_prev_actions: int = pipeline.N_PREV
    n_action_states: int = pipeline.N_ACTION_STATES
    stem_channels: int = 32
    quantization: str | None = None  # set by quantize_int8; None means plain fp32 weights

    def __post_init__(self, **kwargs):
        if self.encoder_config is None:
            self.encoder_config = default_encoder_config()
        elif isinstance(self.encoder_config, dict):
            cleaned = {k: v for k, v in self.encoder_config.items() if k not in ("model_type", "transformers_version")}
            self.encoder_config = ModernBertConfig(**cleaned)
        super().__post_init__(**kwargs)

    # Derived geometry: two stride-2 convs mean the grid is always resolution / 4.
    @property
    def in_channels(self):
        return 3 * self.n_frames

    @property
    def grid_h(self):
        return self.res_h // 4

    @property
    def grid_w(self):
        return self.res_w // 4

    @property
    def n_tokens(self):
        return self.grid_h * self.grid_w

    @property
    def hidden_size(self):
        return self.encoder_config.hidden_size


__all__ = ["DoomConvStemConfig", "default_encoder_config"]
