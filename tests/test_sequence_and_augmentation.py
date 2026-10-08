"""Sequence preparation (plausibility check, onset alignment) and training
augmentation must respect the frame-validity mask."""

import numpy as np
import pytest

from dataset import augment_sequence
from sequence import (align_onset, fill_internal_gaps, mask_summary,
                      prepare_sequence, reject_implausible, valid_mask)


def _seq(lead=0, n=80, T=300, gap=None, seed=0):
    rng = np.random.default_rng(seed)
    s = np.zeros((T, 33, 2), np.float32)
    s[lead:lead + n] = rng.normal(0, 0.5, (n, 33, 2))
    if gap is not None:
        s[lead + gap[0]: lead + gap[1]] = 0
    return s


def test_align_onset_moves_first_detection_to_zero():
    s = _seq(lead=37, gap=(10, 14))
    a = align_onset(s)
    vm = valid_mask(a)
    assert vm[0] and vm.sum() == valid_mask(s).sum()
    np.testing.assert_array_equal(a[:80], s[37:117])


def test_reject_implausible_zeroes_only_outlier_frames():
    s = _seq()
    s[5, 3, 0] = 25.0
    r = reject_implausible(s, 10.0)
    vm = valid_mask(r)
    assert not vm[5] and vm.sum() == 79


def test_prepare_is_idempotent():
    s = _seq(lead=12, gap=(30, 40))
    p = prepare_sequence(s)
    np.testing.assert_array_equal(prepare_sequence(p), p)


def test_fill_internal_gaps_leaves_ends_untouched():
    s = _seq(lead=5, gap=(20, 25))
    f = fill_internal_gaps(s)
    vm = valid_mask(f)
    assert vm[5:85].all() and not vm[:5].any() and not vm[85:].any()


def test_mask_summary_counts():
    m = mask_summary(_seq(lead=10, n=50, gap=(20, 23)))
    assert m["first_valid"] == 10 and m["n_valid"] == 47 and m["gap_frames"] == 3
    assert m["gap_runs"] == 1 and m["longest_gap"] == 3


@pytest.mark.parametrize("lead,gap", [(0, None), (40, None), (0, (20, 30)), (60, (5, 9))])
def test_augmentation_keeps_onset_and_never_blanks(lead, gap):
    s = _seq(lead=lead, n=70, gap=gap)
    first = int(np.argmax(valid_mask(s)))
    for seed in range(300):
        np.random.seed(seed)
        a = augment_sequence(s.copy())
        vm = valid_mask(a)
        assert vm.sum() >= 5, "augmentation produced a near-empty sequence"
        assert int(np.argmax(vm)) == first, "augmentation moved the first valid frame"
        assert not vm[:first].any(), "undetected leading frames became valid"


def test_augmentation_does_not_fill_gaps_with_noise():
    s = _seq(lead=0, n=100, gap=(40, 50))
    for seed in range(200):
        np.random.seed(seed)
        a = augment_sequence(s.copy())
        vm = valid_mask(a)
        # a gap may shift slightly under time warping but never disappears
        assert (~vm[:valid_mask(a).nonzero()[0].max()]).any()


def test_augmentation_input_not_modified():
    s = _seq(lead=20)
    ref = s.copy()
    np.random.seed(0)
    augment_sequence(s)
    np.testing.assert_array_equal(s, ref)
