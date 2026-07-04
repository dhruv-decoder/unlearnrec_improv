"""Unit tests for data/data_handler.py — TrnData, TstData, normalization."""

import sys
import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import numpy as np
import pytest
from scipy.sparse import coo_matrix, csr_matrix
import torch as t

from config.params import args
from data.data_handler import TrnData, TstData, DataHandler


# ---- TrnData ----

class TestTrnData:
    @pytest.fixture
    def sample_coomat(self):
        rows = np.array([0, 0, 1, 2, 3])
        cols = np.array([0, 2, 1, 3, 0])
        vals = np.ones(5, dtype=np.float32)
        return coo_matrix((vals, (rows, cols)), shape=(4, 5))

    def test_len(self, sample_coomat):
        ds = TrnData(sample_coomat)
        assert len(ds) == 5

    def test_getitem(self, sample_coomat):
        ds = TrnData(sample_coomat)
        row, col, neg = ds[0]
        assert row == 0
        assert col == 0
        assert neg == 0  # negs initialized to 0

    def test_neg_sampling(self, sample_coomat):
        # Set args.item so neg_sampling can draw negatives
        old_item = getattr(args, 'item', None)
        args.item = 5
        ds = TrnData(sample_coomat)
        ds.neg_sampling()
        # After neg_sampling, negs should be valid item indices
        for i in range(len(ds)):
            _, _, neg = ds[i]
            assert 0 <= neg < args.item
        if old_item is not None:
            args.item = old_item

    def test_neg_sampling_avoids_positives(self, sample_coomat):
        old_item = getattr(args, 'item', None)
        args.item = 5
        ds = TrnData(sample_coomat)
        ds.neg_sampling()
        dokmat = sample_coomat.todok()
        for i in range(len(ds)):
            r, c, neg = ds[i]
            # negative should not be a known positive
            assert (r, neg) not in dokmat
        if old_item is not None:
            args.item = old_item


# ---- TstData ----

class TestTstData:
    @pytest.fixture
    def sample_tst_trn(self):
        # 3 users, 4 items
        tst_rows = np.array([0, 0, 1, 2])
        tst_cols = np.array([1, 3, 2, 0])
        tst_vals = np.ones(4, dtype=np.float32)
        tst_mat = coo_matrix((tst_vals, (tst_rows, tst_cols)), shape=(3, 4))

        trn_rows = np.array([0, 1, 2])
        trn_cols = np.array([0, 1, 3])
        trn_vals = np.ones(3, dtype=np.float32)
        trn_mat = coo_matrix((trn_vals, (trn_rows, trn_cols)), shape=(3, 4))

        return tst_mat, trn_mat

    def test_len(self, sample_tst_trn):
        tst_mat, trn_mat = sample_tst_trn
        ds = TstData(tst_mat, trn_mat)
        assert len(ds) == 3  # 3 users with test data

    def test_getitem_returns_user_and_mask(self, sample_tst_trn):
        tst_mat, trn_mat = sample_tst_trn
        ds = TstData(tst_mat, trn_mat)
        usr, mask = ds[0]
        assert isinstance(usr, (int, np.integer))
        assert mask.shape == (4,)

    def test_tst_locs(self, sample_tst_trn):
        tst_mat, trn_mat = sample_tst_trn
        ds = TstData(tst_mat, trn_mat)
        # user 0 has test items at cols 1 and 3
        assert set(ds.tst_locs[0]) == {1, 3}
        assert set(ds.tst_locs[1]) == {2}
        assert set(ds.tst_locs[2]) == {0}

    def test_mask_marks_train_items(self, sample_tst_trn):
        tst_mat, trn_mat = sample_tst_trn
        ds = TstData(tst_mat, trn_mat)
        # user 0 has train item at col 0 -> mask[0] should be 1
        usr, mask = ds[0]
        actual_usr = ds.tst_usrs[0]
        if actual_usr == 0:
            assert mask[0] == 1.0


# ---- DataHandler._normalize_adj ----

class TestNormalizeAdj:
    @pytest.fixture
    def handler(self):
        old_data = args.data
        old_adv = args.adversarial_attack
        args.data = 'ml1m'
        args.adversarial_attack = False
        h = DataHandler.__new__(DataHandler)
        args.data = old_data
        args.adversarial_attack = old_adv
        return h

    def test_square_symmetric(self, handler):
        # Simple 3x3 adjacency
        mat = coo_matrix(np.array([
            [0, 1, 1],
            [1, 0, 0],
            [1, 0, 0],
        ], dtype=np.float32))
        normed = handler._normalize_adj(mat)
        assert normed.shape == (3, 3)
        # Normalized adjacency should have values in [0, 1]
        assert normed.max() <= 1.0 + 1e-6
        assert normed.min() >= -1e-6

    def test_rectangular(self, handler):
        mat = coo_matrix(np.array([
            [1, 0, 1],
            [0, 1, 0],
        ], dtype=np.float32))
        normed = handler._normalize_adj(mat)
        assert normed.shape == (2, 3)

    def test_isolated_node(self, handler):
        # Node 2 has no connections
        mat = coo_matrix(np.array([
            [0, 1, 0],
            [1, 0, 0],
            [0, 0, 0],
        ], dtype=np.float32))
        normed = handler._normalize_adj(mat)
        # Row 2 should be all zeros
        row2 = normed.toarray()[2]
        assert np.allclose(row2, 0)


# ---- DataHandler.random_drop_edges ----

class TestRandomDropEdges:
    @pytest.fixture
    def handler_with_args(self):
        old_data = args.data
        old_adv = args.adversarial_attack
        old_user = getattr(args, 'user', None)
        old_item = getattr(args, 'item', None)

        args.data = 'ml1m'
        args.adversarial_attack = False
        args.user = 5
        args.item = 5

        h = DataHandler.__new__(DataHandler)
        yield h

        args.data = old_data
        args.adversarial_attack = old_adv
        if old_user is not None:
            args.user = old_user
        if old_item is not None:
            args.item = old_item

    def test_drop_rate_splits(self, handler_with_args):
        rows = np.array([0, 0, 1, 2, 3, 4, 0, 1, 2, 3])
        cols = np.array([0, 1, 2, 3, 4, 0, 2, 3, 4, 1])
        vals = np.ones(10, dtype=np.float32)
        mat = coo_matrix((vals, (rows, cols)), shape=(5, 5))

        pk_mat, mask, drp_mat, dropped_edges, picked_edges = handler_with_args.random_drop_edges(mat, rate=0.3)

        # 30% drop rate on 10 edges = 3 dropped
        assert drp_mat.nnz == 3
        assert pk_mat.nnz == 7
        assert len(dropped_edges[0]) == 3
        assert len(picked_edges[0]) == 7

    def test_zero_drop_rate(self, handler_with_args):
        rows = np.array([0, 1, 2])
        cols = np.array([1, 2, 3])
        vals = np.ones(3, dtype=np.float32)
        mat = coo_matrix((vals, (rows, cols)), shape=(5, 5))

        pk_mat, mask, drp_mat, dropped_edges, picked_edges = handler_with_args.random_drop_edges(mat, rate=0.0)

        assert drp_mat.nnz == 0
        assert pk_mat.nnz == 3

    def test_mask_shape(self, handler_with_args):
        rows = np.array([0, 1, 2])
        cols = np.array([1, 2, 3])
        vals = np.ones(3, dtype=np.float32)
        mat = coo_matrix((vals, (rows, cols)), shape=(5, 5))

        _, mask, _, _, _ = handler_with_args.random_drop_edges(mat, rate=0.3)
        assert mask.shape[0] == args.user + args.item
