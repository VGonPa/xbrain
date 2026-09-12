# tests/test_knowledge_fusion.py
"""Reciprocal Rank Fusion and the explanation it must keep (Plan 03 §4, TDD 15-17).

THESE TESTS PIN THE SHAPE, NEVER THE CONSTANTS. `RRF_K` and the channel weights are 03.7's to
move once the bake-off has measured them (Plan 03 §4.1, recorte atómico §2.2): a test holding
`60` as a literal would go red on the child that applies the measured winner, and the child that
MEASURES would be forced to rewrite the tests of the child that FUSES. So every expected score
here is computed from `fusion.RRF_K` and `fusion.CHANNEL_WEIGHTS` READ AT CALL TIME, and one
test moves both to prove the module reads them rather than closing over a copy.
"""

from __future__ import annotations

from dataclasses import fields

import pytest

from xbrain.knowledge import fusion
from xbrain.knowledge.fusion import FusedChunk, fuse


def _expected_score(**ranks: int) -> float:
    """The RRF sum over the channels a chunk appeared in, from the module's CURRENT constants."""
    return sum(
        fusion.CHANNEL_WEIGHTS[channel] / (fusion.RRF_K + rank) for channel, rank in ranks.items()
    )


def _by_id(fused: tuple[FusedChunk, ...]) -> dict[str, FusedChunk]:
    return {chunk.chunk_id: chunk for chunk in fused}


def test_a_chunk_both_channels_found_keeps_both_ranks_and_both_channels() -> None:
    """TDD 15: the fused match still says WHICH channels found it and WHERE each ranked it."""
    fused = _by_id(fuse({"lexical": ("a", "b"), "vector": ("c", "b")}))

    both = fused["b"]
    assert both.matched_by == ("lexical", "vector")
    assert both.lexical_rank == 2
    assert both.vector_rank == 2


def test_a_single_channel_chunk_carries_null_for_the_other_rank_not_zero() -> None:
    """TDD 16: absent from a channel is `None`. A `0` would read as a rank better than first."""
    fused = _by_id(fuse({"lexical": ("a",), "vector": ("c",)}))

    assert fused["a"].matched_by == ("lexical",)
    assert fused["a"].lexical_rank == 1
    assert fused["a"].vector_rank is None
    assert fused["c"].matched_by == ("vector",)
    assert fused["c"].vector_rank == 1
    assert fused["c"].lexical_rank is None


def test_the_fused_score_is_the_rrf_sum_of_the_constants_read_at_call_time(monkeypatch) -> None:
    """The formula is pinned; the numbers are not. Moving both constants moves the score.

    Seen red with a `k: int = RRF_K` default argument: the default is bound at import, so the
    monkeypatched value never reached the arithmetic and 03.7's hunk would have changed nothing.
    """
    rankings = {"lexical": ("a", "b"), "vector": ("b",)}
    before = _by_id(fuse(rankings))
    assert before["b"].score == pytest.approx(_expected_score(lexical=2, vector=1))
    assert before["a"].score == pytest.approx(_expected_score(lexical=1))

    monkeypatch.setattr(fusion, "RRF_K", fusion.RRF_K + 17)
    monkeypatch.setattr(
        fusion,
        "CHANNEL_WEIGHTS",
        {channel: weight * 3 for channel, weight in fusion.CHANNEL_WEIGHTS.items()},
    )
    after = _by_id(fuse(rankings))

    assert after["b"].score == pytest.approx(_expected_score(lexical=2, vector=1))
    assert after["b"].score != pytest.approx(before["b"].score)


def test_agreement_between_channels_outranks_either_channel_alone() -> None:
    """For ANY positive k and weights, being found twice at rank 1 beats being found once there."""
    fused = fuse({"lexical": ("solo", "both"), "vector": ("both",)})

    assert [chunk.chunk_id for chunk in fused][0] == "both"


def test_ties_are_broken_by_chunk_id_so_two_runs_agree(monkeypatch) -> None:
    """Spec §3.7.8. Equal weights make `z` and `a` tie exactly; the id settles it, not dict order."""
    monkeypatch.setattr(fusion, "CHANNEL_WEIGHTS", {"lexical": 1.0, "vector": 1.0})

    fused = fuse({"lexical": ("z",), "vector": ("a",)})

    assert fused[0].score == fused[1].score
    assert [chunk.chunk_id for chunk in fused] == ["a", "z"]
    assert fuse({"vector": ("a",), "lexical": ("z",)}) == fused


def test_the_fused_score_is_not_named_as_a_probability() -> None:
    """TDD 17: a fused rank has no calibrated scale, and the field name must not claim one."""
    names = {field.name for field in fields(FusedChunk)}

    assert "score" in names
    assert not [name for name in names if "prob" in name or "confidence" in name]


def test_a_chunk_repeated_inside_one_channel_is_refused() -> None:
    """A channel naming a chunk twice has no single rank for it; picking one would be a guess."""
    with pytest.raises(ValueError, match="dup"):
        fuse({"lexical": ("dup", "other", "dup")})


def test_an_unknown_channel_is_refused() -> None:
    """A channel with no declared weight is a typo, and weighting it zero would drop it silently."""
    with pytest.raises(ValueError, match="graphh"):
        fuse({"graphh": ("a",)})  # type: ignore[dict-item]
