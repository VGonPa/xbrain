"""Reciprocal Rank Fusion, and the explanation of a match that fusion must not lose.

Plan 03 §4. Two channels rank chunks on scales that share nothing — bm25 is negative-is-better
over one corpus's term statistics, cosine is a geometry fixed by one model — so a sum of scores
would need a per-corpus calibration redone with every model change. RRF reads only RANKS:

    RRF(chunk) = Σ  w_channel / (RRF_K + rank_channel(chunk))

**`RRF_K` and `CHANNEL_WEIGHTS` are starting points, not measurements.** Plan 03 §4.1 puts their
sweep inside the bake-off (03.7), which owns the hunk that applies the winner. Two consequences
live in this file: the constants are READ AT CALL TIME (a default argument would bind them at
import, and the measured winner would change nothing), and no test pins their values — only the
formula and the shape.

**The explanation is the product, not a by-product** (spec §5.3, criterion §13.7). A fused chunk
keeps `matched_by`, and one rank per channel that is `None` exactly when that channel did not
find it. `0` would be a lie with a number's shape: it reads as a rank better than first.

**`score` is a ranking signal.** It is uncalibrated and unnormalised, and nothing here calls it a
probability or a confidence — a fused rank has no scale to be one on.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import get_args

from xbrain.knowledge.contracts import Channel

# The standard RRF constant — see the module docstring: 03.7 moves it with numbers, not here.
RRF_K: int = 60

# One weight per channel the fusion accepts. `graph` is absent on purpose: it is Plan 04's, and
# a channel with no declared weight is refused rather than silently weighted zero.
CHANNEL_WEIGHTS: Mapping[Channel, float] = {"lexical": 1.0, "vector": 1.0}

# The order channels are NAMED in `matched_by`: the contract's own declaration order, so two
# fusions of the same rankings agree whatever order the caller built its mapping in.
_CHANNEL_ORDER: tuple[str, ...] = get_args(Channel)


@dataclass(frozen=True)
class FusedChunk:
    """One chunk after fusion: its id, who found it, where each channel ranked it, its signal."""

    chunk_id: str
    matched_by: tuple[Channel, ...]
    lexical_rank: int | None
    vector_rank: int | None
    score: float


def fuse(rankings: Mapping[Channel, Sequence[str]]) -> tuple[FusedChunk, ...]:
    """Fuse per-channel rankings of chunk ids, best first, ties broken by `chunk_id`.

    Each sequence is one channel's ranking, best first; a chunk's rank is its 1-based position.
    The tie-break is what makes two runs agree (spec §3.7.8): equal RRF sums are common — a chunk
    first on one channel and one first on the other tie exactly under equal weights.
    """
    ranks = _rank_by_channel(rankings)
    fused = [_explain(chunk_id, channel_ranks) for chunk_id, channel_ranks in ranks.items()]
    fused.sort(key=lambda chunk: (-chunk.score, chunk.chunk_id))
    return tuple(fused)


def _rank_by_channel(rankings: Mapping[Channel, Sequence[str]]) -> dict[str, dict[Channel, int]]:
    """`{chunk_id: {channel: rank}}`, refusing an unknown channel or a chunk ranked twice.

    A chunk repeated inside ONE channel has no single rank there, and keeping either position
    would be a guess about which the channel meant. An unknown channel is a typo, and weighting
    it zero would drop its whole ranking without a word.
    """
    weights = CHANNEL_WEIGHTS
    ranks: dict[str, dict[Channel, int]] = {}
    for channel, ranking in rankings.items():
        if channel not in weights:
            raise ValueError(
                f"Canal de fusión desconocido: {channel!r}. Los que tienen peso: "
                f"{', '.join(sorted(weights))}."
            )
        for position, chunk_id in enumerate(ranking, start=1):
            channel_ranks = ranks.setdefault(chunk_id, {})
            if channel in channel_ranks:
                raise ValueError(
                    f"El canal {channel!r} rankea el chunk {chunk_id} dos veces "
                    f"(posiciones {channel_ranks[channel]} y {position}): no tiene un rango único."
                )
            channel_ranks[channel] = position
    return ranks


def _rrf(channel_ranks: Mapping[Channel, int]) -> float:
    """The RRF sum over the channels that found the chunk, from the constants as they are NOW."""
    k, weights = RRF_K, CHANNEL_WEIGHTS
    return sum(weights[channel] / (k + rank) for channel, rank in channel_ranks.items())


def _explain(chunk_id: str, channel_ranks: Mapping[Channel, int]) -> FusedChunk:
    """The fused chunk, carrying WHY it matched: channels in contract order, `None` for absence."""
    matched_by = tuple(channel for channel in _CHANNEL_ORDER if channel in channel_ranks)
    return FusedChunk(
        chunk_id=chunk_id,
        matched_by=matched_by,  # type: ignore[arg-type]
        lexical_rank=channel_ranks.get("lexical"),
        vector_rank=channel_ranks.get("vector"),
        score=_rrf(channel_ranks),
    )
