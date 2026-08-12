import torch

from models.dinov3_encoder import DEFAULT_VIEWS, FrozenDinoV3Encoder
from models.valueformer import ValueFormer, ValueFormerConfig


def test_encoder_output_shape_and_determinism():
    enc = FrozenDinoV3Encoder(use_real_dinov3=False)
    views = {v: torch.rand(3, 3, 32, 32) for v in DEFAULT_VIEWS}
    out1 = enc(views)
    out2 = enc(views)
    assert out1.shape == (3, 6144)
    assert torch.allclose(out1, out2)


def test_valueformer_forward_shapes_and_param_count_matches_paper_table_1():
    cfg = ValueFormerConfig()
    model = ValueFormer(cfg)
    b, s = 5, 16
    out = model(
        torch.randn(b, s, cfg.vision_dim),
        torch.randn(b, s, cfg.state_dim),
        torch.randn(b, s, cfg.time_feat_dim),
    )
    assert out["v_mc"].shape == (b,)
    assert out["v_bin"].shape == (b,)
    assert torch.all((out["v_mc"] >= 0) & (out["v_mc"] <= 1))
    assert torch.all((out["v_bin"] >= 0) & (out["v_bin"] <= 1))
    # Table I: trainable total = 3,459,586 for the six-view dual-head config.
    assert model.num_trainable_params() == 3_459_586


def test_valueformer_handles_short_sequences_via_left_padding():
    cfg = ValueFormerConfig(seq_len=16)
    model = ValueFormer(cfg)
    b, s = 2, 3  # shorter than seq_len; caller is responsible for left-padding upstream
    out = model(
        torch.randn(b, s, cfg.vision_dim),
        torch.randn(b, s, cfg.state_dim),
        torch.randn(b, s, cfg.time_feat_dim),
    )
    assert out["v_mc"].shape == (b,)


def _all_position_hidden_states(model: ValueFormer, vision, state, time_feat):
    b, s, _ = vision.shape
    x = model.vision_proj(vision) + model.state_proj(state) + model.time_proj(time_feat)
    positions = torch.arange(s, device=x.device).clamp(max=model.config.seq_len - 1)
    x = x + model.pos_embed(positions).unsqueeze(0)
    mask = model._causal_mask(s, x.device)
    h = model.body(x, mask=mask, is_causal=True)
    return model.final_norm(h)  # (B, S, d_model)


def test_causal_mask_blocks_future_positions():
    cfg = ValueFormerConfig()
    model = ValueFormer(cfg)
    model.eval()
    b, s = 1, 16
    vision = torch.randn(b, s, cfg.vision_dim)
    state = torch.randn(b, s, cfg.state_dim)
    time_feat = torch.randn(b, s, cfg.time_feat_dim)

    with torch.no_grad():
        h_orig = _all_position_hidden_states(model, vision, state, time_feat)

        vision_perturbed = vision.clone()
        perturb_pos = 10
        vision_perturbed[:, perturb_pos, :] += 50.0
        h_perturbed = _all_position_hidden_states(model, vision_perturbed, state, time_feat)

    # Positions before the perturbation must be unaffected (causal: they
    # never attended to a future frame); positions at/after it must change.
    assert torch.allclose(h_orig[:, :perturb_pos], h_perturbed[:, :perturb_pos], atol=1e-5)
    assert not torch.allclose(h_orig[:, perturb_pos:], h_perturbed[:, perturb_pos:], atol=1e-5)
