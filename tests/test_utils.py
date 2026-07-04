"""Unit tests for Utils/utils.py — loss functions, metrics, and helpers."""

import sys
import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import torch as t
import torch.nn as nn
import pytest
import numpy as np

from Utils.utils import (
    cal_bpr,
    _crr_neg,
    cal_crr,
    cal_reg,
    cal_neg_aug_v1,
    cal_neg_aug_v2,
    cal_l2_distance,
    innerProduct,
    pairPredict,
    cal_positive_pred_align,
    cal_positive_pred_align_v2,
    cal_positive_pred_align_v3,
    calcRegLoss,
    SimGCL_calcRegLoss,
    SimGCL_calcRegLoss_v2,
    SimGCL_calcRegLoss_v3,
    infoNCE,
    KLDiverge,
    pointKLDiverge,
    print_args,
    contrast,
    contrastLoss,
    _safe_ratio,
    _cal_membership_attack_metrics,
    cal_mi_metrics,
)


# ---- Fixtures ----

@pytest.fixture
def small_embeds():
    t.manual_seed(42)
    return t.randn(4, 8)


@pytest.fixture
def anc_pos_neg():
    t.manual_seed(42)
    anc = t.randn(5, 8)
    pos = t.randn(5, 8)
    neg = t.randn(5, 8)
    return anc, pos, neg


# ---- cal_bpr ----

class TestCalBpr:
    def test_output_is_scalar(self, anc_pos_neg):
        anc, pos, neg = anc_pos_neg
        loss = cal_bpr(anc, pos, neg)
        assert loss.dim() == 0

    def test_positive_loss(self, anc_pos_neg):
        anc, pos, neg = anc_pos_neg
        loss = cal_bpr(anc, pos, neg)
        assert loss.item() >= 0

    def test_perfect_separation(self):
        anc = t.ones(3, 4)
        pos = t.ones(3, 4) * 10
        neg = t.ones(3, 4) * -10
        loss = cal_bpr(anc, pos, neg)
        assert loss.item() < 0.01

    def test_gradient_flows(self, anc_pos_neg):
        anc, pos, neg = [x.requires_grad_(True) for x in anc_pos_neg]
        loss = cal_bpr(anc, pos, neg)
        loss.backward()
        assert anc.grad is not None


# ---- _crr_neg ----

class TestCrrNeg:
    def test_output_is_scalar(self, small_embeds):
        result = _crr_neg(small_embeds, small_embeds, temp=0.1)
        assert result.dim() == 0

    def test_positive_output(self, small_embeds):
        result = _crr_neg(small_embeds, small_embeds, temp=0.1)
        assert result.item() > 0


# ---- cal_crr ----

class TestCalCrr:
    def test_inbatch_vs_full(self):
        t.manual_seed(0)
        usr = t.randn(10, 8)
        itm = t.randn(10, 8)
        ancs = t.tensor([0, 1, 2])
        poss = t.tensor([0, 1, 2])
        full = cal_crr(usr, itm, ancs, poss, temp=0.1, inbatch=False)
        inb = cal_crr(usr, itm, ancs, poss, temp=0.1, inbatch=True)
        assert full.dim() == 0
        assert inb.dim() == 0


# ---- cal_reg ----

class TestCalReg:
    def test_returns_scalar(self):
        model = nn.Linear(4, 2)
        result = cal_reg(model)
        assert result.dim() == 0

    def test_nonneg(self):
        model = nn.Linear(4, 2)
        result = cal_reg(model)
        assert result.item() >= 0

    def test_zero_weights(self):
        model = nn.Linear(4, 2, bias=False)
        nn.init.zeros_(model.weight)
        result = cal_reg(model)
        assert result.item() == pytest.approx(0.0, abs=1e-7)


# ---- cal_neg_aug_v1 / v2 ----

class TestCalNegAug:
    def test_v1_scalar(self):
        u = t.randn(5, 8)
        i = t.randn(5, 8)
        loss = cal_neg_aug_v1(u, i)
        assert loss.dim() == 0

    def test_v2_scalar(self):
        u = t.randn(5, 8)
        i = t.randn(5, 8)
        loss = cal_neg_aug_v2(u, i)
        assert loss.dim() == 0

    def test_v2_equals_mean_inner_product(self):
        t.manual_seed(1)
        u = t.randn(5, 8)
        i = t.randn(5, 8)
        expected = (u * i).sum(-1).mean()
        actual = cal_neg_aug_v2(u, i)
        assert actual.item() == pytest.approx(expected.item(), abs=1e-5)


# ---- cal_l2_distance ----

class TestCalL2Distance:
    def test_zero_distance(self):
        e = t.randn(3, 4)
        assert cal_l2_distance(e, e).item() == pytest.approx(0.0, abs=1e-6)

    def test_positive_distance(self):
        assert cal_l2_distance(t.ones(3, 4), t.zeros(3, 4)).item() > 0


# ---- innerProduct / pairPredict ----

class TestInnerProduct:
    def test_dot_product(self):
        a = t.tensor([[1.0, 2.0]])
        b = t.tensor([[3.0, 4.0]])
        assert innerProduct(a, b).item() == pytest.approx(11.0)

    def test_batch(self):
        a = t.randn(5, 8)
        b = t.randn(5, 8)
        result = innerProduct(a, b)
        assert result.shape == (5,)


class TestPairPredict:
    def test_shape(self):
        anc = t.randn(5, 8)
        pos = t.randn(5, 8)
        neg = t.randn(5, 8)
        result = pairPredict(anc, pos, neg)
        assert result.shape == (5,)

    def test_values(self):
        anc = t.ones(1, 4)
        pos = t.ones(1, 4) * 2
        neg = t.ones(1, 4) * 0
        # innerProduct(anc, pos) = 8, innerProduct(anc, neg) = 0
        assert pairPredict(anc, pos, neg).item() == pytest.approx(8.0)


# ---- cal_positive_pred_align variants ----

class TestCalPositivePredAlign:
    def test_v1_scalar(self):
        t.manual_seed(0)
        s_u = t.randn(3, 8)
        t_u = t.randn(3, 8)
        s_i = t.randn(3, 8)
        t_i = t.randn(3, 8)
        result = cal_positive_pred_align(s_u, t_u, s_i, t_i, nn.MSELoss())
        assert result.dim() == 0

    def test_v2_scalar(self):
        t.manual_seed(0)
        s_u = t.randn(3, 8)
        t_u = t.randn(3, 8)
        s_i = t.randn(3, 8)
        t_i = t.randn(3, 8)
        result = cal_positive_pred_align_v2(s_u, t_u, s_i, t_i, nn.MSELoss())
        assert result.dim() == 0

    def test_v3_scalar(self):
        t.manual_seed(0)
        s_u = t.randn(3, 8)
        t_u = t.randn(3, 8)
        s_i = t.randn(3, 8)
        t_i = t.randn(3, 8)
        result = cal_positive_pred_align_v3(s_u, t_u, s_i, t_i, nn.MSELoss())
        assert result.dim() == 0

    def test_identical_inputs_v2(self):
        e = t.randn(3, 8)
        result = cal_positive_pred_align_v2(e, e, e, e, cal_l2_distance)
        assert result.item() == pytest.approx(0.0, abs=1e-4)


# ---- calcRegLoss ----

class TestCalcRegLoss:
    def test_with_params(self):
        params = [t.randn(3, 4), t.randn(2, 5)]
        result = calcRegLoss(params=params)
        assert result.item() > 0

    def test_with_model(self):
        model = nn.Linear(4, 2)
        result = calcRegLoss(model=model)
        assert result.item() > 0

    def test_zero_params(self):
        params = [t.zeros(3, 4)]
        result = calcRegLoss(params=params)
        assert result.item() == pytest.approx(0.0, abs=1e-7)

    def test_both_params_and_model(self):
        params = [t.randn(3, 4)]
        model = nn.Linear(4, 2)
        result = calcRegLoss(params=params, model=model)
        expected = calcRegLoss(params=params).item() + calcRegLoss(model=model).item()
        assert result.item() == pytest.approx(expected, rel=1e-5)


# ---- SimGCL_calcRegLoss variants ----

class TestSimGCLCalcRegLoss:
    def test_v1(self):
        u = t.randn(10, 8)
        i = t.randn(5, 8)
        result = SimGCL_calcRegLoss(u, i)
        assert result.dim() == 0
        assert result.item() > 0

    def test_v2(self):
        u = t.randn(10, 8)
        i = t.randn(5, 8)
        result = SimGCL_calcRegLoss_v2(u, i)
        assert result.dim() == 0
        assert result.item() > 0

    def test_v3(self):
        u = t.randn(10, 8)
        i = t.randn(5, 8)
        result = SimGCL_calcRegLoss_v3(u, i)
        assert result.dim() == 0
        assert result.item() > 0

    def test_v1_zero_embeds(self):
        u = t.zeros(10, 8)
        i = t.zeros(5, 8)
        assert SimGCL_calcRegLoss(u, i).item() == pytest.approx(0.0, abs=1e-7)


# ---- infoNCE ----

class TestInfoNCE:
    def test_scalar_output(self):
        t.manual_seed(0)
        e1 = t.randn(10, 8)
        e2 = t.randn(10, 8)
        nodes = t.tensor([0, 1, 2, 3])
        result = infoNCE(e1, e2, nodes, temp=0.1)
        assert result.dim() == 0

    def test_positive_loss(self):
        t.manual_seed(0)
        e1 = t.randn(10, 8)
        e2 = t.randn(10, 8)
        nodes = t.tensor([0, 1, 2])
        result = infoNCE(e1, e2, nodes, temp=0.1)
        assert result.item() > 0


# ---- KLDiverge ----

class TestKLDiverge:
    def test_scalar(self):
        tpreds = t.randn(5)
        spreds = t.randn(5)
        result = KLDiverge(tpreds, spreds)
        assert result.dim() == 0

    def test_identical_preds(self):
        preds = t.randn(10)
        result = KLDiverge(preds, preds)
        assert result.dim() == 0


# ---- pointKLDiverge ----

class TestPointKLDiverge:
    def test_scalar(self):
        tp = t.tensor([0.3, 0.7])
        sp = t.tensor([0.3, 0.7])
        result = pointKLDiverge(tp, sp)
        assert result.dim() == 0


# ---- contrast / contrastLoss ----

class TestContrast:
    def test_scalar(self):
        t.manual_seed(0)
        e1 = t.randn(10, 8)
        e2 = t.randn(10, 8)
        nodes = t.tensor([0, 1, 2])
        result = contrast(e1, e2, nodes, temp=10)
        assert result.dim() == 0

    def test_positive(self):
        t.manual_seed(0)
        e1 = t.randn(10, 8)
        e2 = t.randn(10, 8)
        nodes = t.tensor([0, 1, 2])
        result = contrast(e1, e2, nodes, temp=10)
        assert result.item() > 0


class TestContrastLoss:
    def test_scalar(self):
        t.manual_seed(0)
        e1 = t.randn(10, 8)
        e2 = t.randn(10, 8)
        nodes = t.tensor([0, 1, 2])
        result = contrastLoss(e1, e2, nodes, temp=10)
        assert result.dim() == 0

    def test_positive(self):
        t.manual_seed(0)
        e1 = t.randn(10, 8)
        e2 = t.randn(10, 8)
        nodes = t.tensor([0, 1, 2])
        result = contrastLoss(e1, e2, nodes, temp=10)
        assert result.item() > 0


# ---- _safe_ratio ----

class TestSafeRatio:
    def test_normal_ratio(self):
        result = _safe_ratio(t.tensor(6.0), t.tensor(3.0))
        assert result == pytest.approx(2.0)

    def test_zero_denominator(self):
        result = _safe_ratio(t.tensor(1.0), t.tensor(0.0))
        assert result > 0  # clamped, not inf

    def test_negative_denominator(self):
        result = _safe_ratio(t.tensor(1.0), t.tensor(-0.5))
        # eps clamp prevents division by negative
        assert np.isfinite(result)


# ---- _cal_membership_attack_metrics ----

class TestCalMembershipAttackMetrics:
    def test_perfect_separation(self):
        drp = t.tensor([10.0, 9.0, 8.0])
        neg = t.tensor([1.0, 2.0, 3.0])
        auc, acc = _cal_membership_attack_metrics(drp, neg)
        assert auc == pytest.approx(1.0, abs=0.01)
        assert acc >= 0.9

    def test_random_scores(self):
        t.manual_seed(42)
        drp = t.randn(50)
        neg = t.randn(50)
        auc, acc = _cal_membership_attack_metrics(drp, neg)
        assert 0 <= auc <= 1
        assert 0 <= acc <= 1

    def test_output_types(self):
        drp = t.tensor([1.0, 2.0])
        neg = t.tensor([3.0, 4.0])
        auc, acc = _cal_membership_attack_metrics(drp, neg)
        assert isinstance(auc, float)
        assert isinstance(acc, float)


# ---- cal_mi_metrics ----

class TestCalMiMetrics:
    def test_returns_dict_keys(self):
        t.manual_seed(0)
        drp = t.randn(10)
        neg = t.randn(10)
        result = cal_mi_metrics(drp, neg)
        expected_keys = {'mi_bf', 'mi_ng', 'mi_auc', 'mi_acc',
                         'avg_before_prob', 'avg_after_prob', 'avg_neg_prob'}
        assert set(result.keys()) == expected_keys

    def test_with_before_scores(self):
        drp = t.randn(10)
        neg = t.randn(10)
        before = t.randn(10)
        result = cal_mi_metrics(drp, neg, before_drp_scores=before)
        assert 'mi_bf' in result

    def test_values_are_finite(self):
        t.manual_seed(0)
        drp = t.randn(20)
        neg = t.randn(20)
        result = cal_mi_metrics(drp, neg)
        for k, v in result.items():
            assert np.isfinite(v), f"{k} is not finite: {v}"


# ---- print_args ----

class TestPrintArgs:
    def test_does_not_crash(self, capsys):
        class FakeArgs:
            def __init__(self):
                self.lr = 0.001
                self.batch = 512
        print_args(FakeArgs())
        captured = capsys.readouterr()
        assert "lr" in captured.out
        assert "batch" in captured.out
