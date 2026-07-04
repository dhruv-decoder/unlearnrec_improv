"""Unit tests for models/Model.py — model layers and components.

Tests run on CPU to avoid CUDA dependency. We mock args where needed.
"""

import sys
import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import torch as t
import torch.nn as nn
import torch_sparse as ts
import numpy as np
import pytest

from config.params import args


def _make_sparse_tensor(n, m, nnz):
    """Create a small SparseTensor on CPU for testing."""
    rows = t.randint(0, n, (nnz,))
    cols = t.randint(0, m, (nnz,))
    vals = t.rand(nnz)
    return ts.SparseTensor(row=rows, col=cols, value=vals, sparse_sizes=(n, m))


# ---- get_shape ----

class TestGetShape:
    def test_sparse_tensor(self):
        from models.Model import get_shape
        sp = _make_sparse_tensor(4, 6, 10)
        assert tuple(get_shape(sp)) == (4, 6)

    def test_dense_tensor(self):
        from models.Model import get_shape
        dense = t.randn(3, 5)
        assert get_shape(dense) == t.Size([3, 5])


# ---- GCNLayer ----

class TestGCNLayer:
    def test_forward(self):
        from models.Model import GCNLayer
        layer = GCNLayer()
        adj = _make_sparse_tensor(5, 5, 10)
        embeds = t.randn(5, 8)
        out = layer(adj, embeds)
        assert out.shape == (5, 8)

    def test_output_differs_from_input(self):
        from models.Model import GCNLayer
        layer = GCNLayer()
        adj = _make_sparse_tensor(5, 5, 10)
        embeds = t.randn(5, 8)
        out = layer(adj, embeds)
        assert not t.allclose(out, embeds)


# ---- SpAdjDropEdge ----

class TestSpAdjDropEdge:
    def test_keep_rate_1(self):
        from models.Model import SpAdjDropEdge
        dropper = SpAdjDropEdge()
        adj = _make_sparse_tensor(5, 5, 10)
        result = dropper(adj, keepRate=1.0)
        # should return the same adj
        assert result is adj

    def test_drop_reduces_edges(self):
        from models.Model import SpAdjDropEdge
        dropper = SpAdjDropEdge()
        t.manual_seed(0)
        adj = _make_sparse_tensor(20, 20, 100)
        result = dropper(adj, keepRate=0.5)
        row, col, val = result.coo()
        # With 50% keep rate, expect roughly 50 edges (with randomness)
        assert len(val) < 100
        assert len(val) > 0

    def test_preserves_shape(self):
        from models.Model import SpAdjDropEdge
        dropper = SpAdjDropEdge()
        adj = _make_sparse_tensor(10, 10, 30)
        result = dropper(adj, keepRate=0.7)
        assert tuple(result.sizes()) == (10, 10)


# ---- FeedForwardLayer ----

class TestFeedForwardLayer:
    @pytest.fixture(autouse=True)
    def setup_args(self):
        self.old_latdim = args.latdim
        self.old_leaky = args.leaky
        args.latdim = 8
        args.leaky = 0.99
        yield
        args.latdim = self.old_latdim
        args.leaky = self.old_leaky

    def test_identity_act(self):
        from models.Model import FeedForwardLayer
        layer = FeedForwardLayer(8, 8, act=None).cpu()
        # move params to cpu
        layer.W = nn.Parameter(layer.W.data.cpu())
        layer.bias = nn.Parameter(layer.bias.data.cpu())
        x = t.randn(3, 8)
        out = layer(x)
        assert out.shape == (3, 8)

    def test_leaky_act(self):
        from models.Model import FeedForwardLayer
        layer = FeedForwardLayer(8, 8, act='leaky').cpu()
        layer.W = nn.Parameter(layer.W.data.cpu())
        layer.bias = nn.Parameter(layer.bias.data.cpu())
        x = t.randn(3, 8)
        out = layer(x)
        assert out.shape == (3, 8)

    def test_relu_act(self):
        from models.Model import FeedForwardLayer
        layer = FeedForwardLayer(8, 8, act='relu').cpu()
        layer.W = nn.Parameter(layer.W.data.cpu())
        layer.bias = nn.Parameter(layer.bias.data.cpu())
        x = t.randn(3, 8)
        out = layer(x)
        assert out.shape == (3, 8)

    def test_relu6_act(self):
        from models.Model import FeedForwardLayer
        layer = FeedForwardLayer(8, 8, act='relu6').cpu()
        layer.W = nn.Parameter(layer.W.data.cpu())
        layer.bias = nn.Parameter(layer.bias.data.cpu())
        x = t.randn(3, 8)
        out = layer(x)
        assert out.shape == (3, 8)

    def test_invalid_act_raises(self):
        from models.Model import FeedForwardLayer
        with pytest.raises(Exception):
            FeedForwardLayer(8, 8, act='gelu')

    def test_residual_skip(self):
        from models.Model import FeedForwardLayer
        layer = FeedForwardLayer(8, 8, act='leaky').cpu()
        layer.W = nn.Parameter(layer.W.data.cpu())
        layer.bias = nn.Parameter(layer.bias.data.cpu())
        x = t.randn(3, 8)
        out = layer(x)
        # With residual skip: output = act(x @ W + bias) + x
        # Output should differ from just x
        assert not t.allclose(out, x)

    def test_gradient_flows(self):
        from models.Model import FeedForwardLayer
        layer = FeedForwardLayer(8, 8, act='leaky').cpu()
        layer.W = nn.Parameter(layer.W.data.cpu())
        layer.bias = nn.Parameter(layer.bias.data.cpu())
        x = t.randn(3, 8, requires_grad=True)
        out = layer(x)
        out.sum().backward()
        assert x.grad is not None
        assert layer.W.grad is not None


# ---- SimGclGCNLayer ----

class TestSimGclGCNLayer:
    def test_no_perturb(self):
        from models.Model import SimGclGCNLayer
        layer = SimGclGCNLayer(perturb=False)
        adj = _make_sparse_tensor(5, 5, 10)
        embeds = t.randn(5, 8)
        out = layer(adj, embeds)
        assert out.shape == (5, 8)
        # Without perturbation, should be deterministic
        out2 = layer(adj, embeds)
        assert t.allclose(out, out2)

    def test_with_perturb(self):
        from models.Model import SimGclGCNLayer
        old_eps = args.eps
        args.eps = 0.2
        layer = SimGclGCNLayer(perturb=True)
        adj = _make_sparse_tensor(5, 5, 10)
        embeds = t.randn(5, 8)
        t.manual_seed(0)
        out1 = layer(adj, embeds)
        t.manual_seed(1)
        out2 = layer(adj, embeds)
        assert out1.shape == (5, 8)
        # Two calls with different seeds should produce different results
        assert not t.allclose(out1, out2)
        args.eps = old_eps


# ---- HGNNLayer ----

class TestHGNNLayer:
    @pytest.fixture(autouse=True)
    def setup_args(self):
        self.old_latdim = args.latdim
        self.old_leaky = args.leaky
        args.latdim = 8
        args.leaky = 0.99
        yield
        args.latdim = self.old_latdim
        args.leaky = self.old_leaky

    def test_leaky_forward(self):
        from models.Model import HGNNLayer
        layer = HGNNLayer(8, 8, act='leaky').cpu()
        layer.W1 = nn.Parameter(layer.W1.data.cpu())
        layer.bias1 = nn.Parameter(layer.bias1.data.cpu())
        layer.W2 = nn.Parameter(layer.W2.data.cpu())
        layer.bias2 = nn.Parameter(layer.bias2.data.cpu())
        x = t.randn(3, 8)
        out = layer(x)
        assert out.shape == (3, 8)

    def test_relu_forward(self):
        from models.Model import HGNNLayer
        layer = HGNNLayer(8, 8, act='relu').cpu()
        layer.W1 = nn.Parameter(layer.W1.data.cpu())
        layer.bias1 = nn.Parameter(layer.bias1.data.cpu())
        layer.W2 = nn.Parameter(layer.W2.data.cpu())
        layer.bias2 = nn.Parameter(layer.bias2.data.cpu())
        x = t.randn(3, 8)
        out = layer(x)
        assert out.shape == (3, 8)


# ---- GAIEDecoder ----

class TestGAIEDecoder:
    def test_forward(self):
        from models.Model import GAIEDecoder
        decoder = GAIEDecoder()
        z = t.randn(10, 8)
        drp_rows = [0, 1, 2]
        drp_cols = [5, 6, 7]
        out = decoder(z, drp_rows, drp_cols)
        assert out.shape == (3,)

    def test_output_is_inner_product(self):
        from models.Model import GAIEDecoder
        decoder = GAIEDecoder()
        z = t.randn(10, 8)
        drp_rows = [0]
        drp_cols = [5]
        out = decoder(z, drp_rows, drp_cols)
        expected = (z[0] * z[5]).sum()
        assert out.item() == pytest.approx(expected.item(), abs=1e-5)


# ---- HyperNetwork ----

class TestHyperNetwork:
    def test_forward_shape(self):
        from models.Model import HyperNetwork
        hyper = HyperNetwork(latent_dim=8, num_nodes=20, embed_dim=8, hyper_rank=4)
        z = t.randn(8)
        W_u = hyper(z)
        assert W_u.shape == (20, 8)

    def test_gradient_flows(self):
        from models.Model import HyperNetwork
        hyper = HyperNetwork(latent_dim=8, num_nodes=20, embed_dim=8, hyper_rank=4)
        z = t.randn(8, requires_grad=True)
        W_u = hyper(z)
        W_u.sum().backward()
        assert z.grad is not None

    def test_different_inputs_different_outputs(self):
        from models.Model import HyperNetwork
        hyper = HyperNetwork(latent_dim=8, num_nodes=20, embed_dim=8, hyper_rank=4)
        z1 = t.randn(8)
        z2 = t.randn(8)
        W1 = hyper(z1)
        W2 = hyper(z2)
        assert not t.allclose(W1, W2)


# ---- GAIEEncoder ----

class TestGAIEEncoder:
    def test_forward_shape(self):
        from models.Model import GAIEEncoder
        encoder = GAIEEncoder(in_dim=8, hidden_dim=16, latent_dim=8, num_layers=3)
        adj = _make_sparse_tensor(10, 10, 20)
        node_feats = t.randn(10, 8)
        mu, logvar = encoder(adj, node_feats)
        assert mu.shape == (10, 8)
        assert logvar.shape == (10, 8)

    def test_gradient_flows(self):
        from models.Model import GAIEEncoder
        encoder = GAIEEncoder(in_dim=8, hidden_dim=16, latent_dim=8, num_layers=3)
        adj = _make_sparse_tensor(10, 10, 20)
        node_feats = t.randn(10, 8, requires_grad=True)
        mu, logvar = encoder(adj, node_feats)
        (mu.sum() + logvar.sum()).backward()
        assert node_feats.grad is not None


# ---- GATInfluenceLayer ----

class TestGATInfluenceLayer:
    @pytest.fixture(autouse=True)
    def setup_args(self):
        self.old_latdim = args.latdim
        args.latdim = 8
        yield
        args.latdim = self.old_latdim

    def test_forward_shape(self):
        from models.Model import GATInfluenceLayer
        layer = GATInfluenceLayer(in_dim=8, out_dim=8)
        adj = _make_sparse_tensor(10, 10, 20)
        h = t.randn(10, 8)
        out = layer(adj, h)
        assert out.shape == (10, 8)

    def test_gradient_flows(self):
        from models.Model import GATInfluenceLayer
        layer = GATInfluenceLayer(in_dim=8, out_dim=8)
        adj = _make_sparse_tensor(10, 10, 20)
        h = t.randn(10, 8, requires_grad=True)
        out = layer(adj, h)
        out.sum().backward()
        assert h.grad is not None


# ---- LightGCN (basic forward) ----

class TestLightGCN:
    @pytest.fixture(autouse=True)
    def setup_args(self):
        self.old_user = getattr(args, 'user', None)
        self.old_item = getattr(args, 'item', None)
        self.old_latdim = args.latdim
        self.old_gnn_layer = args.gnn_layer
        args.user = 5
        args.item = 5
        args.latdim = 8
        args.gnn_layer = 2
        yield
        if self.old_user is not None:
            args.user = self.old_user
        if self.old_item is not None:
            args.item = self.old_item
        args.latdim = self.old_latdim
        args.gnn_layer = self.old_gnn_layer

    def test_forward_shape(self):
        from models.Model import LightGCN

        class FakeHandler:
            ts_ori_adj = _make_sparse_tensor(10, 10, 20)
        handler = FakeHandler()
        model = LightGCN(handler)
        adj = _make_sparse_tensor(10, 10, 20)
        usr_emb, itm_emb = model(adj)
        assert usr_emb.shape == (5, 8)
        assert itm_emb.shape == (5, 8)

    def test_forward_all_layer(self):
        from models.Model import LightGCN

        class FakeHandler:
            ts_ori_adj = _make_sparse_tensor(10, 10, 20)
        handler = FakeHandler()
        model = LightGCN(handler)
        adj = _make_sparse_tensor(10, 10, 20)
        embeds_list, embeds = model(adj, all_layer=True)
        assert len(embeds_list) == args.gnn_layer + 1
        assert embeds.shape == (10, 8)
