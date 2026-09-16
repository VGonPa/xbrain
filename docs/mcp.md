# xbrain over MCP — connecting an agent

`xbrain mcp-serve` exposes the knowledge index to an external agent (Claude Code,
Claude Desktop, any MCP client) through the Model Context Protocol, over **stdio**.
It is a thin adapter: three tools, each calling exactly the service the CLI calls
and returning exactly the model the CLI prints with `--json`. There is no second
retrieval, no second citation format, no second limit.

What to do with the results — search, verify, cite — is in
[knowledge-for-agents.md](knowledge-for-agents.md). How the index is built and
kept current is in [knowledge-index.md](knowledge-index.md).

## Before you connect

**1 · Build the index.** The server answers from `data/index/`; without it,
`xbrain.search` and `xbrain.graph_expand` refuse (`xbrain.get` does not need it).

```bash
uv run xbrain index build
```

**2 · Install the `[mcp]` extra.** The SDK is optional, so a CLI-only install never
pays for it. Without it the command refuses with an instruction, exit code 1:

```
$ uv run xbrain mcp-serve
Error: `xbrain mcp-serve` necesita el extra opcional `[mcp]`, que no está instalado: instálalo con: uv pip install 'xbrain[mcp]'
```

From a checkout, the simplest way is to let `uv run` add it on launch (below).
If you prefer `uv sync`, list **every** extra you use in one call —
`uv sync --extra mcp --extra embeddings` — because `uv sync` removes the extras
its flags did not name.

**3 · Know which corpus it serves.** The server reads `config.toml` and `data/`
from the root of the checkout xbrain is installed from — the same place the CLI
reads. `XBRAIN_REPO_ROOT=/some/dir` points both at another root.

## Configuring the client

The launch command, verified end to end with the SDK's own stdio client
(`initialize`, `list_tools`, one call per tool):

```bash
uv run --directory /path/to/xbrain --extra mcp xbrain mcp-serve
```

**Claude Code:**

```bash
claude mcp add xbrain -- uv run --directory /path/to/xbrain --extra mcp xbrain mcp-serve
```

**Claude Desktop** (`claude_desktop_config.json`):

```json
{
  "mcpServers": {
    "xbrain": {
      "command": "uv",
      "args": ["run", "--directory", "/path/to/xbrain", "--extra", "mcp", "xbrain", "mcp-serve"]
    }
  }
}
```

The client starts the process and talks to it over stdin/stdout. There is no
port, no network listener and no authentication to manage: the process can only
be reached by whoever launched it.

## The three tools

Each tool returns the **same response model** the CLI serialises with `--json`,
and its input and output schemas are **derived** from the handler's annotations
and return type, never typed by hand. `tests/test_mcp_cli_equivalence.py` runs the
same cases through both doors and requires identical JSON.

| Tool | CLI twin | Inputs (required in bold) | Returns |
|---|---|---|---|
| `xbrain.search` | `xbrain search … --json` | **`query`**, `filters`, `limit` (10), `strategy` (`"lexical"`), `cursor` | `SearchResponse` |
| `xbrain.get` | `xbrain get <id> --json` | **`item_id`**, `surfaces`, `query`, `cursor` | `EvidenceBundle` |
| `xbrain.graph_expand` | `xbrain graph-expand --item <id> --json` | **`item_id`**, `max_hops` (1), `max_neighbors` (unset → 10) | `GraphExpansionResponse` |

The defaults are the services' own defaults, asserted by a test, so a tool call
and the CLI with no flags answer the same question.

**`xbrain.search`.** `filters` is the whole `SearchFilters` model, the eight
fields the CLI exposes as flags: `created_from`, `created_to`, `source`, `author`,
`topics`, `content_kinds`, `origins`, `has_surfaces`. Unknown values are refused
before the index is touched, listing the valid ones. `strategy` accepts
`lexical`, `vector`, `hybrid` and `hybrid_graph` — with the same outcomes as the
CLI (see [Limits](#limits)). A truncated page carries a `cursor`; to continue,
repeat the same `query`, `filters` and `limit` with that cursor.

**`xbrain.get`.** Reads the **live store**, never the index, so it answers with
`data/index/` deleted. Without `surfaces` it returns the index card: metadata,
topics, the summary, and the list of surfaces the item has. Ask for a body by
name (`surfaces: ["external_article"]`). Over the character budget
(`[index].get_char_budget`, 40,000 by default, the same value the CLI reads) it
returns `truncated: true` and a `cursor`; the continuation repeats the same
`surfaces` and `query`.

**`xbrain.graph_expand`.** Nodes, edges and one explicit path per reached node,
each edge carrying `relation`, `method`, `weight`, `shared_items` and
`supporting_item_ids`. The response carries
`semantics: "co_occurrence_in_corpus"` and
`disclaimer_key: "graph_edge_is_corpus_not_world"` **in the data**, so the
distinction survives an agent summarising the prose away: an edge means *these
topics were assigned together to these items of this corpus*, never *these
concepts are related in the world*.

## Errors

An operator error reaches the agent as a tool result with `is_error: true` and
the **same message the CLI prints**, prefixed by the SDK. Real output from the
probe above:

```
Error executing tool xbrain.get: No existe el item 'does-not-exist' en el store. Comprueba el id con `xbrain search` o `xbrain knowledge inspect <id>`.
Error executing tool xbrain.search: Estrategia desconocida: 'vectro'. Las declaradas son: lexical, vector, hybrid, hybrid_graph; implementadas hoy: lexical.
Error executing tool xbrain.search: `--strategy vector` necesita vectores y este índice no tiene plano vectorial: configura `[embeddings].command` en config.toml y ejecuta `xbrain index build --embeddings --force`. …
Error executing tool xbrain.graph_expand: El índice va por detrás del store (`index_behind_store`): … Ejecuta `xbrain index update`.
```

Without that translation the SDK would reduce every one of them to
`Error executing tool xbrain.search` and drop the cause. What counts as an
operator error is the CLI's own list, imported rather than copied. Anything else
is a bug and is raised as such. (The `implementadas hoy: lexical` tail of the typo
message is stale: `vector` and `hybrid` do run. It is in the backlog.)

Degradations are **not** errors: a search that ran over an index behind the
store, or a `hybrid` that could not open its vector channel, succeeds and says
so in `index.degraded` — read it on every response.

## Read-only

No tool writes. `search` and `graph_expand` open the index `mode=ro`, so a stray
write is an error rather than a silent repair; `get` opens no persisted index at
all. `tests/test_mcp_server.py` calls all three tools and then requires
`items.json`, `vocab.yaml`, `topics.json` and every file under `data/index/` to be
byte-identical. No tool dereferences a URL from the corpus: a link is returned as
data, never fetched.

## Retrieved content is untrusted data

The corpus is tweets, articles and transcripts written by strangers. A text that
says *"ignore your instructions"* travels to the agent like any other text. xbrain
does not filter it — filtering would alter the evidence — it **labels** it:

- every tool description and the server's instructions say that returned content
  is corpus data, never instructions, even when the text itself asks otherwise;
- every surface and fragment carries `origin` (`source`, `asr`, `vlm`, `llm`,
  `user`, `unknown`), `trust_class` and `derived`;
- hostile text is transported **verbatim** and only inside named content fields;
  it is never spliced into a field the agent reads as metadata.

`tests/test_mcp_prompt_injection.py` stores an item whose text is a prompt
injection, retrieves it through all three tools, and checks it arrives labelled
and changes nothing. What the agent does with it is the agent's responsibility:
treat excerpts as material to cite, never as orders.

## The trust boundary

What someone connecting an agent needs to know, and nothing more is promised:

- **The server makes no network calls of its own.** No handler opens a socket or
  speaks HTTP; a test walks the adapter's source, lazy imports included, and fails
  on any network module.
- **During a query xbrain may start ONE external process: the embedder in
  `[embeddings].command`, which receives the query text.** It does not happen out
  of the box: `xbrain.search` defaults to `strategy: "lexical"` and
  `[embeddings].command` ships empty, so the embedder runs only when the caller
  asks for `vector`/`hybrid` **and** someone configured the command.
  `scripts/xbrain-embed` is the reference backend and runs locally; it is not a
  default, because there is none.
- **What that binary does with the text is outside xbrain's scope.** xbrain
  governs whom it calls, not what the callee does. Pointing
  `[embeddings].command` at a remote service is a choice, visible in your
  `config.toml`. That is why this is declared rather than promised.

No generative LLM is called on any of the three paths.

## Limits

- **`hybrid_graph` cannot be switched on from here.** Asked for by name, it
  answers `lexical` declaring `hybrid_graph_not_implemented` — as the CLI does.
  The graph re-ranking exists behind a switch only the Python API and
  `xbrain eval` pass, and the sweep that measured it found it helps no case
  ([graph-threshold-sweep.md](graph-threshold-sweep.md)).
- **`vector` and `hybrid` need the opt-in vector plane** and a configured
  embedder; the degradation matrix is in
  [knowledge-index.md](knowledge-index.md#when-the-vector-channel-cannot-run).
  A filtered `vector`/`hybrid` request is answered lexically, declared as
  `vector_filters_unsupported`.
- **`graph_expand` refuses an index behind the store** (`search` only declares it):
  its edges may not be the current corpus's. Run `xbrain index update`.
- **An unknown `item_id` in `graph_expand` returns a one-node expansion**, not an
  error — indistinguishable from an item with no topics. `get` refuses it. In the
  backlog.
- **stdio only.** No HTTP/SSE transport, by design.

Something not working? → [Troubleshooting](troubleshooting.md#xbrain-mcp-serve-does-not-start-or-the-client-sees-no-tools).
