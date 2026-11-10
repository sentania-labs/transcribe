import json
import math

import pytest

from app.voice import (VoiceConfigError, clean_regions, embedding_windows, mean_signature,
                       parse_voice_models, unit)

ENTRY = {"model": "m", "revision": "r1", "method": "clean-mean-v1", "path": "/p", "sha256": "abc"}


def test_clean_regions_remove_overlap():
    turns = [(0.0, 10.0, "A"), (8.0, 15.0, "B"), (14.0, 20.0, "A")]
    clean = clean_regions(turns)
    assert clean["A"] == [(0.0, 8.0), (15.0, 20.0)]
    assert clean["B"] == [(10.0, 14.0)]


def test_clean_regions_join_touching_and_ignore_empty():
    clean = clean_regions([(0.0, 5.0, "A"), (5.0, 9.0, "A"), (3.0, 3.0, "B")])
    assert clean == {"A": [(0.0, 9.0)]}


def test_embedding_windows_split_drop_and_budget():
    wins = embedding_windows([(0.0, 25.0), (30.0, 31.0)])
    assert all(2.0 <= b - a <= 10.0 for a, b in wins)
    assert abs(sum(b - a for a, b in wins) - 25.0) < 1e-9  # the 1 s piece is dropped
    many = [(i * 20.0, i * 20.0 + 10.0) for i in range(100)]  # 1000 s of 10 s pieces
    picked = embedding_windows(many, budget=100.0)
    assert sum(b - a for a, b in picked) <= 100.0
    assert picked[0][0] == 0.0 and picked[-1][0] == 1980.0  # first and last piece: spread over the whole span


def test_mean_signature_is_unit_and_skips_bad():
    v = mean_signature([[3.0, 4.0], [0.0, 0.0], [1.0, 0.0]], [1.0, 5.0, 1.0])
    assert abs(math.hypot(*v) - 1.0) < 1e-9
    assert unit([0.0, 0.0]) is None
    assert mean_signature([[0.0, 0.0]], [1.0]) is None


def test_parse_voice_models():
    assert parse_voice_models("") == []
    (e,) = parse_voice_models(json.dumps([ENTRY]))
    assert e["version"] == "r1+clean-mean-v1"
    two = parse_voice_models(json.dumps([ENTRY, {**ENTRY, "revision": "r2"}]))
    assert [x["version"] for x in two] == ["r1+clean-mean-v1", "r2+clean-mean-v1"]
    for bad in ("{", json.dumps({}), json.dumps([{**ENTRY, "sha256": ""}]),
                json.dumps([{**ENTRY, "method": "nope"}]), json.dumps([ENTRY, ENTRY])):
        with pytest.raises(VoiceConfigError):
            parse_voice_models(bad)
