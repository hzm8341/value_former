import torch

from models.aop import ActionOutcomePredictor, AOPConfig, RankingWeights
from models.contact_aop import ContactActionOutcomePredictor, ContactAOPConfig
from models.lite_lda import LiteLDA, LiteLDAConfig


def test_aop_forward_shapes_and_ranking_score():
    b = 6
    aop = ActionOutcomePredictor(AOPConfig())
    out = aop(torch.randn(b, 256), torch.randn(b, 8))
    assert out["delta_progress"].shape == (b,)
    assert out["delta_ee_pose"].shape == (b, 6)
    assert torch.all((out["p_jam"] >= 0) & (out["p_jam"] <= 1))
    score = ActionOutcomePredictor.ranking_score(out, RankingWeights())
    assert score.shape == (b,)


def test_ranking_score_prefers_higher_progress_lower_risk():
    weights = RankingWeights()
    good = {
        "delta_progress": torch.tensor([0.5]), "p_jam": torch.tensor([0.0]),
        "p_slip": torch.tensor([0.0]), "p_success_h": torch.tensor([0.9]),
    }
    bad = {
        "delta_progress": torch.tensor([0.0]), "p_jam": torch.tensor([0.8]),
        "p_slip": torch.tensor([0.5]), "p_success_h": torch.tensor([0.1]),
    }
    assert ActionOutcomePredictor.ranking_score(good, weights) > ActionOutcomePredictor.ranking_score(bad, weights)


def test_contact_aop_adds_contact_side_and_recoverable_heads():
    b = 3
    caop = ContactActionOutcomePredictor(ContactAOPConfig())
    out = caop(torch.randn(b, 256), torch.randn(b, 8), torch.randn(b, 50, 6 + 16))
    assert out["contact_side_logits"].shape == (b, 5)
    assert out["p_recoverable"].shape == (b,)
    assert "delta_progress" in out  # base AOP heads still present


def test_lite_lda_grounding_heads_share_weights_across_t_and_future():
    b = 2
    lda = LiteLDA(LiteLDAConfig())
    z_t = torch.randn(b, 256)
    out = lda(z_t, torch.randn(b, 8, 8))
    assert out["z_future"].shape == (b, 256)
    assert out["physical_state_t"].shape == (b, 5)
    assert out["physical_state_future"].shape == (b, 5)
    assert torch.allclose(lda.grounding_head(z_t), out["physical_state_t"])
