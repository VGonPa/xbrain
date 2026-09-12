"""Shell out to an external embedding backend and read back one vector per text.

The thin, ML-free wrapper at the heart of the knowledge index's vector plane
(Plan 03 §1), and the third sibling of the same locked shape as
`xbrain.transcribe` and `xbrain.vision`: xbrain stays **mechanical**, the model
runs as an EXTERNAL subprocess located via config/PATH, and the CLI carries **no**
embedding/ML dependency. This module imports no model library — a test asserts it,
and a second test asserts that no test in the suite loads one either.

The embedder contract xbrain expects:

- It is invoked as ``<command> [--model M]`` where ``<command>`` comes from
  ``[embeddings].command`` and may itself be a multi-token command (a wrapper
  script), split with ``shlex`` and run WITHOUT a shell.
- The request travels as JSON on **stdin**, because argv cannot carry thousands of
  chunk texts::

      {"schema_version": "1", "model": "…|null", "texts": ["…", "…"]}

- The response travels as JSON on **stdout**::

      {"schema_version": "1", "model": "intfloat/multilingual-e5-base",
       "dimension": 768, "normalized": true, "vectors": [[0.01, …], …]}

- **There is no bundled default command.** ``[embeddings].command`` starts empty,
  exactly like ``[vision].command``: shipping a default would be choosing the
  embedding model without evaluating it, and the whole point of Plan 03 is that
  the model is chosen by the golden set. Unset, `search` keeps serving lexical
  results and this module raises `EmbedderNotFound` if asked for vectors.

Every row of the response is validated before it becomes a vector, because a
matrix that silently accepts a wrong-shaped, non-finite or wrong-model batch
answers queries with numbers nobody can trace back to a text:

============================================  ========================
Check                                         Failure
============================================  ========================
unknown ``schema_version``                    `EmbedderFailed`, naming the expected one
``len(vectors) != len(texts)``                `EmbedderFailed`, with both figures
dimension inconsistent between vectors        `EmbedderFailed`, with the culprit index
declared ``dimension`` ≠ the vectors sent     `EmbedderFailed`, with both figures
dimension ≠ the caller's (manifest) one       `EmbedderFailed` — models are never mixed
a non-finite value (``NaN``, ``inf``)         `EmbedderFailed`
a zero vector (no direction to normalize)     `EmbedderFailed`
stdout that is not UTF-8                      `EmbedderFailed`
============================================  ========================

**Normalization is a post-condition, not a claim.** The returned vectors are
always L2-normalized, so 03.3 can compute cosine similarity as a plain dot
product; `EmbeddingBatch.renormalized` records whether xbrain had to do the work.
The backend's ``normalized`` flag is VERIFIED rather than trusted — a batch of
norm 5 declared normalized would multiply every stored similarity by five,
silently, and checking costs the same single pass over the data that normalizing
does (CLAUDE.md rule 9: assert on the source, never on the reported conclusion).

**No failure message ever quotes an embedded text** (Plan 03 §10.5, §10.8). The
corpus is personal and an operator error ends up in a terminal, a log or a pasted
issue, so the messages carry counts, indices and dimensions only. **That includes
the backend's own stderr on a non-zero exit, which is therefore not relayed**: it
is text the model process chose to print, and the usual thing a crashing embedder
prints is a traceback whose frame quotes the `repr` of the text it failed on. A
missing stderr costs the operator one re-run by hand; a leaked one cannot be
un-pasted, and the batch that reaches this backend is the whole corpus.

Failures surface as clear operator errors (subclasses of `RuntimeError`, which the
CLI's `_handle_cli_errors` turns into a clean exit-1): a **missing / non-executable
/ unconfigured** command (`EmbedderNotFound`), or a **non-zero exit, timeout, or
unusable output** (`EmbedderFailed`). The `runner` (a `subprocess.run` stand-in) is
injectable so tests run offline against a fake — no real embedder, ever.
"""

from __future__ import annotations

import json
import math
import shlex
import subprocess  # nosec B404 - the embedder is an external subprocess by design (Plan 03 §1.1)
from collections.abc import Callable, Sequence
from dataclasses import dataclass

# The wire format version of the stdin/stdout contract above. It is sent on every
# request AND required on every response, so a backend upgraded out from under
# xbrain fails loudly instead of being reinterpreted. `scripts/xbrain-embed`
# spells this string independently (it runs under the system python and cannot
# import xbrain); a test pins the two equal.
SCHEMA_VERSION = "1"

# How many texts xbrain hands the backend per subprocess call. Owned HERE and
# imported by `config.py` so there is one definition (CLAUDE.md rule 5): a literal
# retyped into the config loader is a second definition that drifts the day this
# one moves, with nothing going red because each file stays internally consistent.
DEFAULT_BATCH_SIZE = 64

# A generous wall-clock cap per batch: embedding 64 chunks is seconds on CPU, but
# a wedged model process must not hang an index build forever.
DEFAULT_TIMEOUT_SECONDS = 600

# How far a vector's norm may sit from 1.0 before xbrain re-normalizes it. A real
# backend emits float32 printed as decimal, so an exactly-unit vector arrives a few
# ulps off; that is a round-trip artefact, not an unnormalized batch.
_UNIT_TOLERANCE = 1e-6

Runner = Callable[..., "subprocess.CompletedProcess[str]"]


class EmbeddingError(RuntimeError):
    """Base class for every embedder failure (a clean CLI exit-1)."""


class EmbedderNotFound(EmbeddingError):
    """The configured embedder command is missing / not executable / unconfigured."""


class EmbedderFailed(EmbeddingError):
    """The embedder ran but failed: non-zero exit, timeout, or unusable output."""


@dataclass(frozen=True)
class EmbeddingBatch:
    """One validated response: the vectors, and what produced them.

    `vectors` has exactly one row per input text, in the same order, and every row
    is L2-normalized (see the module docstring). `model`, `dimension` and
    `normalized` are what the index manifest records — changing any of them
    invalidates the vector plane. `renormalized` says whether xbrain had to do the
    normalizing itself, which the manifest also records.
    """

    model: str
    dimension: int
    normalized: bool
    vectors: tuple[tuple[float, ...], ...]
    renormalized: bool = False


def _build_argv(command: str, model: str | None) -> list[str]:
    """Assemble the subprocess argv: shlex-split command + the optional model.

    The texts are NOT here — they travel on stdin, because a corpus-sized batch
    would blow past the argv limit. `command` is `shlex`-split so a multi-token
    wrapper (`python -m my_embedder`) works, and the whole thing runs WITHOUT a
    shell.
    """
    argv = shlex.split(command)
    if model:
        argv += ["--model", model]
    return argv


def _request(texts: Sequence[str], model: str | None) -> str:
    """The JSON request body handed to the backend on stdin."""
    return json.dumps(
        {"schema_version": SCHEMA_VERSION, "model": model, "texts": list(texts)},
        ensure_ascii=False,
    )


def _run_embedder(argv: list[str], payload: str, runner: Runner, timeout_seconds: int) -> str:
    """Run the embedder; return its stdout, or raise a clear operator error."""
    try:
        completed = runner(
            argv,
            input=payload,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
        )
    except FileNotFoundError as exc:
        raise EmbedderNotFound(
            f"embedder {argv[0]!r} not found — install it or set a valid "
            f"[embeddings].command in config.toml ({exc})"
        ) from exc
    except subprocess.TimeoutExpired as exc:
        raise EmbedderFailed(f"embedder {argv[0]!r} timed out after {timeout_seconds}s") from exc
    except OSError as exc:  # not-executable, permission denied, etc.
        raise EmbedderNotFound(
            f"embedder {argv[0]!r} could not be executed — check "
            f"[embeddings].command in config.toml ({exc})"
        ) from exc
    except UnicodeDecodeError as exc:  # subprocess.run(text=True) on non-UTF-8 stdout
        raise EmbedderFailed(f"embedder {argv[0]!r} produced non-UTF-8 stdout: {exc}") from exc
    if completed.returncode != 0:
        # The backend's stderr is deliberately NOT relayed. It is whatever the model
        # process chose to print, and a Python traceback prints the `repr` of the
        # argument it choked on — here, the embedded text, i.e. the corpus (Plan 03
        # §10.5). The message carries what identifies the RUN and nothing that
        # identifies the DATA; the operator re-runs the command to read the rest.
        raise EmbedderFailed(
            f"embedder {argv[0]!r} exited {completed.returncode} — its stderr is not "
            "repeated here, because a backend traceback quotes the text it failed on; "
            "run the command by hand on a batch you own to see it"
        )
    return completed.stdout or ""


def _parse_response(raw: str) -> dict:
    """The backend's stdout as a JSON object of the expected schema version."""
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise EmbedderFailed(f"embedder output was not valid JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise EmbedderFailed(f"embedder output was not a JSON object: {type(data).__name__}")
    version = str(data.get("schema_version", ""))
    if version != SCHEMA_VERSION:
        raise EmbedderFailed(
            f"embedder speaks schema_version {version or '(absent)'!r}, xbrain speaks "
            f"{SCHEMA_VERSION!r} — upgrade one side; the two are not interchangeable"
        )
    if not str(data.get("model") or "").strip():
        raise EmbedderFailed(
            "embedder output carries no 'model' — the index manifest records which "
            "model produced its vectors, and a batch that cannot say is unusable"
        )
    return data


def _coerce_vector(raw: object, position: int) -> tuple[float, ...]:
    """One response row as a tuple of finite floats, or a typed failure.

    `json.loads` cheerfully accepts `NaN` and `Infinity`, and either one poisons
    every dot product it later touches — ranking unpredictably instead of failing.
    """
    if not isinstance(raw, list) or not raw:
        raise EmbedderFailed(
            f"embedder vector at index {position} is not a non-empty list of numbers"
        )
    values: list[float] = []
    for axis, value in enumerate(raw):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise EmbedderFailed(
                f"embedder vector at index {position} has a non-numeric value at position {axis}"
            )
        number = float(value)
        if not math.isfinite(number):
            raise EmbedderFailed(
                f"embedder vector at index {position} has a non-finite value at position {axis}"
            )
        values.append(number)
    return tuple(values)


def _normalize(vector: tuple[float, ...], position: int) -> tuple[tuple[float, ...], bool]:
    """`(unit_vector, was_renormalized)` — L2, with the zero vector refused.

    A zero vector has no direction: normalizing it divides by zero, and storing it
    unchanged would leave a row at cosine 0 from every query forever, retrievable
    by nothing and reported by no one.
    """
    norm = math.sqrt(math.fsum(value * value for value in vector))
    if norm == 0.0:
        raise EmbedderFailed(
            f"embedder vector at index {position} is the zero vector — it has no "
            "direction, so it can be neither normalized nor ever retrieved"
        )
    if abs(norm - 1.0) <= _UNIT_TOLERANCE:
        return vector, False
    return tuple(value / norm for value in vector), True


def _validated_vectors(data: dict, expected_count: int) -> tuple[tuple[float, ...], ...]:
    """Every row, coerced and shape-checked against the request.

    The count check is what keeps text *i* paired with vector *i*: a short batch
    would silently repoint every later chunk_id at a different text's geometry.
    """
    raw_vectors = data.get("vectors")
    if not isinstance(raw_vectors, list):
        raise EmbedderFailed("embedder output 'vectors' is not a list")
    if len(raw_vectors) != expected_count:
        raise EmbedderFailed(
            f"embedder returned {len(raw_vectors)} vectors for {expected_count} texts — "
            "the pairing between a text and its vector would be lost"
        )
    vectors = tuple(_coerce_vector(raw, index) for index, raw in enumerate(raw_vectors))
    width = len(vectors[0])
    for index, vector in enumerate(vectors):
        if len(vector) != width:
            raise EmbedderFailed(
                f"embedder vector at index {index} has dimension {len(vector)}, but "
                f"the batch started at dimension {width} — a ragged batch is not a matrix"
            )
    return vectors


def _checked_dimension(data: dict, width: int, expected_dimension: int | None) -> int:
    """The batch's dimension, agreed by the backend's claim and the caller's manifest.

    The declared `dimension` is a CONCLUSION the backend reports and the vectors are
    the SOURCE, so the two are compared rather than one being trusted.
    `expected_dimension` is what the caller read from the index manifest: a batch of
    a different width is refused here, because appending it would leave one matrix
    holding two models' geometry and ranking them against each other.
    """
    declared = data.get("dimension")
    if not isinstance(declared, int) or isinstance(declared, bool) or declared != width:
        raise EmbedderFailed(
            f"embedder declared dimension {declared!r} but sent vectors of dimension {width}"
        )
    if expected_dimension is not None and width != expected_dimension:
        raise EmbedderFailed(
            f"embedder returned dimension {width}, but this index holds "
            f"{expected_dimension}-dimensional vectors — models are never mixed in one "
            "matrix; rebuild the vector plane with `xbrain index build --embeddings`"
        )
    return width


def embed_texts(
    texts: Sequence[str],
    *,
    command: str,
    model: str | None,
    expected_dimension: int | None = None,
    timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS,
    runner: Runner | None = None,
) -> EmbeddingBatch:
    """Embed one batch of texts via the external embedder subprocess.

    Shells out to `command` (a multi-token wrapper is supported and run WITHOUT a
    shell), hands it the texts as JSON on stdin, and validates every row of the
    response before returning it. The result's vectors are L2-normalized and in the
    same order as `texts`.

    `model` is passed through as `--model M` (`None` → the backend's own default).
    `expected_dimension`, when given, is the dimension the index manifest already
    holds; a batch of any other width is refused. An empty / unconfigured `command`
    raises `EmbedderNotFound`, as does a missing or non-executable binary; a
    non-zero exit, a timeout or an unusable response raises `EmbedderFailed`.

    `texts` must be non-empty. This is a caller precondition rather than an
    operator error — there is nothing to embed, and no dimension the response could
    be validated against, so the subprocess is never invoked. `xbrain index build`
    checks for an empty corpus before it gets here.

    `runner` (a `subprocess.run` stand-in) is injectable for tests; it defaults to
    `subprocess.run`, resolved at call time so it stays monkeypatchable.
    """
    if not command.strip():
        raise EmbedderNotFound(
            "no [embeddings].command configured — set it in config.toml to build or "
            "query the vector plane (there is no bundled default embedder: the model "
            "is chosen by evaluation, not by a default)"
        )
    if not texts:
        raise ValueError("embed_texts requires at least one text")
    active_runner: Runner = runner if runner is not None else subprocess.run
    argv = _build_argv(command, model)
    stdout = _run_embedder(argv, _request(texts, model), active_runner, timeout_seconds)
    data = _parse_response(stdout)
    vectors = _validated_vectors(data, len(texts))
    dimension = _checked_dimension(data, len(vectors[0]), expected_dimension)
    normalized = [_normalize(vector, index) for index, vector in enumerate(vectors)]
    return EmbeddingBatch(
        model=str(data["model"]),
        dimension=dimension,
        normalized=True,
        vectors=tuple(vector for vector, _ in normalized),
        renormalized=any(changed for _, changed in normalized),
    )


def embed_passages(
    texts: Sequence[str],
    *,
    command: str,
    model: str | None,
    prefix: str = "",
    expected_dimension: int | None = None,
    timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS,
    runner: Runner | None = None,
) -> EmbeddingBatch:
    """Embed corpus fragments, applying the model's **passage** prefix.

    The prefix is a property of the MODEL, not of this code: the E5 and BGE
    families degrade notably when a passage is embedded without `"passage: "` or
    with the query prefix instead, and nothing downstream would ever report it —
    the vectors stay well-formed, unit-length and simply worse. It lives in
    `[embeddings].passage_prefix` so changing model does not mean editing Python,
    and it goes into the manifest, because changing it invalidates the stored
    vectors exactly as changing the model does.
    """
    return embed_texts(
        [f"{prefix}{text}" for text in texts],
        command=command,
        model=model,
        expected_dimension=expected_dimension,
        timeout_seconds=timeout_seconds,
        runner=runner,
    )


def embed_query(
    text: str,
    *,
    command: str,
    model: str | None,
    prefix: str = "",
    expected_dimension: int | None = None,
    timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS,
    runner: Runner | None = None,
) -> EmbeddingBatch:
    """Embed one search query, applying the model's **query** prefix.

    The mirror of `embed_passages`, and the reason the two are separate functions
    rather than one with a flag: the asymmetric-prefix models are the ones this
    index is built for, and a swap between the two sides is invisible in the data.
    Returns a one-row batch so the caller still sees the model and dimension it
    must check against the manifest before searching.

    There is deliberately NO `len(vectors) == 1` guard here. `embed_texts` already
    refuses any response whose vector count differs from the text count, and one
    text goes in — so such a guard could never fire. It would be a second copy of an
    invariant that module owns (rule 5), and a test appearing to exercise it would
    in fact be satisfied upstream: green, and testing nothing (rule 1).
    """
    return embed_texts(
        [f"{prefix}{text}"],
        command=command,
        model=model,
        expected_dimension=expected_dimension,
        timeout_seconds=timeout_seconds,
        runner=runner,
    )
