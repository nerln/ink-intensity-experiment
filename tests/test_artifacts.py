"""Numpy-only checks for the committed headline and calibration records."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "out"


def _topk_mask(values: np.ndarray, q: float) -> np.ndarray:
    flat = values.reshape(-1)
    k = min(flat.size, max(1, int(round(q * flat.size))))
    chosen = np.argpartition(flat, flat.size - k)[flat.size - k :]
    mask = np.zeros(flat.size, dtype=bool)
    mask[chosen] = True
    return mask


def _delta(a: np.ndarray, b: np.ndarray, q: float) -> float:
    ma, mb = _topk_mask(a, q), _topk_mask(b, q)
    return 1.0 - float(np.count_nonzero(ma & mb) / np.count_nonzero(ma | mb))


def _average_ranks(values: np.ndarray) -> np.ndarray:
    """Equivalent to scipy rankdata(method='average'), without a scipy dependency."""
    flat = values.reshape(-1)
    order = np.argsort(flat, kind="stable")
    ordered = flat[order]
    starts = np.flatnonzero(np.r_[True, ordered[1:] != ordered[:-1]])
    ends = np.r_[starts[1:], flat.size]
    ranked = np.empty(flat.size, dtype=np.float64)
    for start, end in zip(starts, ends):
        ranked[order[start:end]] = (start + end - 1) / 2.0
    return ranked


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    ra, rb = _average_ranks(a), _average_ranks(b)
    ra -= ra.mean()
    rb -= rb.mean()
    return float(np.dot(ra, rb) / np.sqrt(np.dot(ra, ra) * np.dot(rb, rb)))


def test_headline_recomputes_from_committed_arrays() -> None:
    a = np.load(OUT / "ts_hann_A.npy")
    b = np.load(OUT / "ts_hann_B.npy")
    held_out = np.load(OUT / "ts_hann_Bmap_0p6154_104p32.npy")

    observed = _delta(a, b, 0.05)
    remapped = _delta(a, held_out, 0.05)
    assert _delta(a, a, 0.05) == 0.0
    assert observed == pytest.approx(0.7573947667804323, abs=1e-15)
    assert remapped == pytest.approx(0.04519015659955261, abs=1e-15)
    assert (observed - remapped) / observed == pytest.approx(0.9403347387893252)
    assert _spearman(a, b) == pytest.approx(0.5100482744668753)
    assert _spearman(a, held_out) == pytest.approx(0.99914297554535)


def test_calibration_record_links_gate_and_experiment() -> None:
    """Audit committed gate metadata; raw published-map replay needs SOURCE_DATA.sha256."""
    calibration = json.loads((OUT / "calibration.json").read_text())
    experiment = json.loads((OUT / "experiment_timesformer.json").read_text())
    best = calibration["best"]

    assert calibration["chance_iou5"] == pytest.approx(0.05 / 1.95)
    assert best == max(calibration["sweep"], key=lambda row: row["iou_5"])
    assert (best["iou_5"] >= calibration["pass_mark_iou5"]) is calibration["passed"]
    assert (best["start"], best["count"], best["reverse"]) == (1, 26, False)
    assert best["iou_5"] == pytest.approx(0.44916800265354634)
    assert experiment["blends"]["hann"]["calibration_iou5"] == best["iou_5"]


def test_artifact_manifest() -> None:
    for line in (ROOT / "ARTIFACTS.sha256").read_text().splitlines():
        expected, relative = line.split(maxsplit=1)
        actual = hashlib.sha256((ROOT / relative).read_bytes()).hexdigest()
        assert actual == expected, relative
