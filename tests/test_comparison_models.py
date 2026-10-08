"""Comparison architectures: shapes, finite outputs, independence from
trailing undetected frames, and exactness of the dilated temporal convolution."""

import numpy as np
import pytest
import torch
import torch.nn as nn

from comparison_models import COMPARISON_MODELS, DilatedTemporalConv, spatial_partition


def _batch(T=200):
    torch.manual_seed(0)
    x = torch.zeros(3, T, 33, 2)
    for b, n in enumerate((60, 90, 130)):
        x[b, :n] = torch.randn(n, 33, 2) * 0.5
    x[0, 20:25] = 0.0                      # an internal detection gap
    return x


@pytest.mark.parametrize("name", list(COMPARISON_MODELS))
def test_forward_finite_and_trailing_invariant(name):
    torch.manual_seed(0)
    m = COMPARISON_MODELS[name]().eval()
    x = _batch()
    with torch.no_grad():
        p, l = m(x)
        _, l2 = m(torch.cat([x, torch.zeros(3, 100, 33, 2)], 1))
    assert p.shape == (3,) and torch.isfinite(l).all()
    assert torch.allclose(l, l2, atol=1e-5)


@pytest.mark.parametrize("name", list(COMPARISON_MODELS))
def test_all_zero_sequence_is_finite(name):
    m = COMPARISON_MODELS[name]().eval()
    with torch.no_grad():
        _, l = m(torch.zeros(2, 120, 33, 2))
    assert torch.isfinite(l).all()


@pytest.mark.parametrize("dims,d,s,T", [(1, 2, 1, 37), (1, 4, 1, 60), (1, 8, 1, 181),
                                        (2, 2, 1, 50), (2, 2, 2, 61), (2, 1, 2, 40)])
def test_dilated_temporal_conv_equals_dilated_conv(dims, d, s, T):
    torch.manual_seed(0)
    k = 3 if dims == 1 else 5
    m = DilatedTemporalConv(4, 6, k, d, s, dims)
    ref = (nn.Conv1d(4, 6, k, s, (k - 1) * d // 2, d) if dims == 1
           else nn.Conv2d(4, 6, (k, 1), (s, 1), ((k - 1) * d // 2, 0), (d, 1)))
    ref.weight.data.copy_(m.conv.weight.data)
    ref.bias.data.copy_(m.conv.bias.data)
    x = torch.randn(2, 4, T) if dims == 1 else torch.randn(2, 4, T, 3)
    assert torch.allclose(m(x), ref(x), atol=1e-5)


def test_spatial_partition_columns_normalised():
    A = spatial_partition()
    assert A.shape == (3, 33, 33)
    np.testing.assert_allclose(A.sum(axis=(0, 1)), 1.0, atol=1e-6)
