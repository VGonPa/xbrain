# tests/test_knowledge_seams.py
"""The two seams of the knowledge layer, ENUMERATED BY REFERENCE (round 08).

Six rounds closed the FAIL-OPEN family one route at a time and four rounds closed the
ATTRIBUTION/LOCATOR family the same way. Round 06 gave each family ONE function — seam (a)
`index_build.describe_base` + `index_schema.require_database`, seam (b)
`chunking.fragment_locator` + `chunking.chunk_evidence` + `render._label` — and a sentinel
test per seam that every consumer must repeat. Round 07 ENUMERATED the consumers from the
source with `ast`, and both round-08 gates showed what that enumeration measured: the two
LITERAL forms (`open_index(...)`, `KnowledgeChunk(...)`) and no other — nine bypasses green
in one battery, fourteen of fourteen in another, four of them ordinary Pydantic idioms
(`model_validate`, `model_construct`, `model_copy(update=)`, a class alias), plus
`sqlite3.Connection`, `getattr`, a lambda, an `async def`, module-level code, a helper in
`cli.py` and a subpackage. A guard that says «all» and covers the literal form is the claim
rule 2 exists to cut down.

WHAT THIS FILE GUARANTEES NOW. Every module under `src/xbrain` — `cli.py` and every
subpackage included, not `knowledge/*.py` alone — is read with `ast`, and every REFERENCE to
a watched symbol is resolved through the module's imports (`import x as y`, `from x import y
as z`), through name aliases (`_KC = KnowledgeChunk`, `opener = _s.connect`, `fp =
_ids.chunk_fingerprint`), through attribute chains and through the package's re-exports
(`from xbrain.knowledge import KnowledgeChunk`), in EVERY scope: a function, an `async def`,
a lambda, a class body, the module level. A reference is a USE unless it sits in one of the
DECLARED harmless contexts — an annotation, an `isinstance`/`issubclass`/`cast` argument, an
introspection read (`.model_fields`), an element of the contract's model registry — and a use
is red unless it lies inside a declared consumer:

* seam (a): `sqlite3.connect`/`sqlite3.Connection` are used by `open_index` and
  `open_memory_index` and by nothing else; `sqlite3` is imported by three named modules and
  no other; `open_index` is used by the four declared DOORS and by nothing else; every door
  but the creator asks both halves of the question; a `describe_base` verdict is never
  discarded; `getattr` over `sqlite3` or over a watched module, and `__import__`/
  `import_module` of one, are red anywhere;
* seam (b): `KnowledgeChunk`/`SearchMatch` are used by the two declared FRAGMENT BUILDERS and
  by nothing else, never through `model_validate`/`model_construct`/`parse_*`/`type(x)(...)`
  /`.__class__(...)`, and each builder binds `locator=` to a value that IS a `fragment_locator`
  call (a dead call beside a hand-built locator is red); `model_copy` in the knowledge package
  and `cli.py` lives in `fragment_locator` only; `chunk_fingerprint` is used by the two
  declared EVIDENCE HASHERS and by nothing else, and every call passes a `chunk_evidence(...)`
  call as its evidence (a dead `chunk_evidence` beside a hand-built tuple is red); `hashlib`
  in the knowledge package and `cli.py` lives in `ids.py` and `index_build.py` only.

The battery at the end of this file is the positive control, kept IN the file: every form
both gates used to walk around the round-07 enumeration — and the two literal controls — is
injected into an in-memory copy of the package and must produce a violation. A form added to
the battery that the analyser does not see fails this file; a rule weakened until a form
passes fails this file. That is how «seen red» stays true after the round that wrote it.

WHAT IT DOES NOT GUARANTEE, said plainly. Enumeration by reference sees names; it does not
see values. A string handed to `getattr` on an object the analyser cannot resolve, an
`exec`/`eval`, a monkeypatch at runtime, a chunk passed into a module OUTSIDE the knowledge
package and `model_copy`-ed there, or a door that ASSIGNS the verdict of `describe_base` and
never reads it, are not seen here. What holds those is the behavioural layer: the sentinel
tests of `test_knowledge_index_invalidation.py` (every door repeats the seam's answer),
`test_knowledge_search_service.py` (a fragment served with a hand-built locator fails the
locator identity test; a forged row is excluded and counted) and `test_knowledge_render.py`
(the contract's string fields forged in totality). A new DOOR added to `DOORS` below is
guarded here structurally and must ALSO be added to the sentinel tests, which this file
cannot do for it.
"""

from __future__ import annotations

import ast
import functools
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from pathlib import Path

import pytest

import xbrain

PACKAGE_ROOT = Path(xbrain.__file__).parent  # src/xbrain, walked recursively
SRC_ROOT = PACKAGE_ROOT.parent

# ---------------------------------------------------------------------------
# the watched symbols, by CANONICAL dotted name, and the consumers declared for each
# ---------------------------------------------------------------------------

SQLITE_MODULE = "sqlite3"
SQLITE_OPENERS = frozenset({"sqlite3.connect", "sqlite3.Connection"})
SQLITE_OPENER_FUNCTIONS = frozenset(
    {
        "xbrain.knowledge.index_schema::open_index",
        "xbrain.knowledge.index_schema::open_memory_index",
    }
)
# The modules allowed to import `sqlite3` at all. `index_build` and `lexical` take a
# `sqlite3.Connection` and read `sqlite3.DatabaseError`; a fourth module that needs the
# exceptions is added here on purpose, never by accident.
SQLITE_IMPORTERS = frozenset(
    {"xbrain.knowledge.index_schema", "xbrain.knowledge.index_build", "xbrain.knowledge.lexical"}
)

DOOR_FUNCTION = "xbrain.knowledge.index_schema.open_index"
# Every function that opens `knowledge.db`. `build` is the ONE creator (`create=True`, G-2)
# and is exempt from the existence half: it unlinks and creates.
DOORS = frozenset(
    {
        "xbrain.knowledge.index_build::build",
        "xbrain.knowledge.index_build::update",
        "xbrain.knowledge.index_build::_index_contents",  # `status`
        "xbrain.knowledge.index_store::open_for_query",  # `search`
    }
)
CREATOR = "xbrain.knowledge.index_build::build"
EXISTENCE_HALF = "xbrain.knowledge.index_schema.require_database"
DESCRIBE_HALF = frozenset(
    {
        "xbrain.knowledge.index_build.describe_base",
        "xbrain.knowledge.index_build.require_consistent",
    }
)
DESCRIBE_BASE = "xbrain.knowledge.index_build.describe_base"
SEALER = "xbrain.knowledge.index_build.write_manifest"

FRAGMENT_MODELS = frozenset(
    {"xbrain.knowledge.models.KnowledgeChunk", "xbrain.knowledge.contracts.SearchMatch"}
)
# Every function that constructs a served fragment. `_chunk` builds the chunk `get` and the
# writer emit; `_match` builds the `SearchMatch` `search` returns.
FRAGMENT_BUILDERS = frozenset(
    {"xbrain.knowledge.chunking::_chunk", "xbrain.knowledge.search_service::_match"}
)
NARROWER = "xbrain.knowledge.chunking.fragment_locator"
# The Pydantic ways of making an instance without the class call. None is allowed on a
# watched class anywhere: a fragment built without validation, or hydrated from a dict, is
# a fragment whose locator nobody narrowed.
CONSTRUCTOR_METHODS = frozenset(
    {
        "model_validate",
        "model_validate_json",
        "model_validate_strings",
        "model_construct",
        "parse_obj",
        "parse_raw",
        "parse_file",
        "from_orm",
        "construct",
        "__new__",
        "__call__",
    }
)
MODEL_COPY_FUNCTIONS = frozenset({"xbrain.knowledge.chunking::fragment_locator"})

HASHER = "xbrain.knowledge.ids.chunk_fingerprint"
EVIDENCE = "xbrain.knowledge.chunking.chunk_evidence"
EVIDENCE_HASHERS = frozenset(
    {"xbrain.knowledge.chunking::_chunk", "xbrain.knowledge.index_store::verify_fingerprints"}
)
HASHLIB_MODULES = frozenset({"xbrain.knowledge.ids", "xbrain.knowledge.index_build"})

# Modules a `getattr(<module>, "…")` or an `import_module("…")` must never reach: the ones
# that define a watched symbol, plus `sqlite3`.
WATCHED_MODULES = frozenset(
    {
        SQLITE_MODULE,
        "xbrain.knowledge.models",
        "xbrain.knowledge.contracts",
        "xbrain.knowledge.ids",
        "xbrain.knowledge.chunking",
        "xbrain.knowledge.index_schema",
        "xbrain.knowledge.index_build",
    }
)
WATCHED_ATTRIBUTES = frozenset(
    {"connect", "Connection", "KnowledgeChunk", "SearchMatch", "chunk_fingerprint", "open_index"}
)

# The contexts in which a reference to a watched symbol is NOT a use, declared once.
HARMLESS_CALLS = frozenset({"isinstance", "issubclass", "cast", "typing.cast", "TypeVar"})
INTROSPECTION = frozenset(
    {"model_fields", "model_config", "model_json_schema", "__name__", "__qualname__", "__doc__"}
)
REGISTRIES = frozenset({("xbrain.knowledge.contracts", "CONTRACT_MODELS")})

# Where `model_copy` and `hashlib` are policed beside the knowledge package: the CLI, the
# realistic home of a helper that walks around a seam (gate round 08).
POLICED_PREFIXES = ("xbrain.knowledge", "xbrain.cli")


# ---------------------------------------------------------------------------
# reading the package
# ---------------------------------------------------------------------------


def _module_name(path: Path) -> str:
    parts = list(path.relative_to(SRC_ROOT).with_suffix("").parts)
    if parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts)


def read_package() -> dict[str, str]:
    """`{dotted module: source}` for every `.py` under `src/xbrain`, subpackages included."""
    return {
        _module_name(path): path.read_text(encoding="utf-8")
        for path in sorted(PACKAGE_ROOT.rglob("*.py"))
    }


def read_policed() -> dict[str, str]:
    """The knowledge package and the CLI alone — where every bypass of the battery is
    injected, so the battery pays 0.4 s per form instead of 0.9 s for the whole package.
    The assertion on the SHIPPED package walks all of it (`read_package`)."""
    return {name: source for name, source in read_package().items() if _policed(name)}


@dataclass
class _Module:
    name: str
    tree: ast.Module
    bindings: dict[str, str] = field(default_factory=dict)
    # Names DEFINED at the module's top level (`def`, `class`, an assignment): a bare
    # reference to one of them inside the same module is canonical to that module.
    defs: set[str] = field(default_factory=set)
    parents: dict[int, ast.AST] = field(default_factory=dict)
    owners: dict[int, str] = field(default_factory=dict)
    annotated: set[int] = field(default_factory=set)
    registry_nodes: set[int] = field(default_factory=set)


def _import_bindings(name: str, tree: ast.Module, *, is_package: bool) -> dict[str, str]:
    """`{local name: dotted target}` from every import statement in the module, any scope."""
    out: dict[str, str] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                out[alias.asname or alias.name.split(".")[0]] = (
                    alias.name if alias.asname else alias.name.split(".")[0]
                )
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                head = name.split(".") if is_package else name.split(".")[:-1]
                head = head[: len(head) - (node.level - 1)]
                base = ".".join([*head, node.module]) if node.module else ".".join(head)
            else:
                base = node.module or ""
            for alias in node.names:
                if alias.name != "*":
                    out[alias.asname or alias.name] = f"{base}.{alias.name}" if base else alias.name
    return out


def _dotted(node: ast.AST) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        head = _dotted(node.value)
        return None if head is None else f"{head}.{node.attr}"
    return None


class _Package:
    """The whole package, parsed once, with a resolver from a local name to a canonical one."""

    def __init__(self, sources: Mapping[str, str]) -> None:
        self.modules: dict[str, _Module] = {}
        for name, source in sources.items():
            tree = _parse(name, source)
            is_package = any(other.startswith(name + ".") for other in sources)
            module = _Module(
                name=name,
                tree=tree,
                bindings=_import_bindings(name, tree, is_package=is_package),
                defs=_definitions(tree),
            )
            self._index(module)
            self.modules[name] = module
        # Name aliases (`_KC = KnowledgeChunk`, `opener = _s.connect`) bind the alias to what
        # its value resolves to, in any scope, to a fixpoint: an alias of an alias resolves.
        for module in self.modules.values():
            for _ in range(8):
                if not self._bind_aliases(module):
                    break

    def _bind_aliases(self, module: _Module) -> bool:
        grew = False
        for node in ast.walk(module.tree):
            if isinstance(node, ast.Assign) and len(node.targets) == 1:
                target = node.targets[0]
                value = _dotted(node.value)
                if isinstance(target, ast.Name) and value is not None:
                    resolved = self.resolve(module, value)
                    if module.bindings.get(target.id) != resolved:
                        module.bindings[target.id] = resolved
                        grew = True
        return grew

    def _index(self, module: _Module) -> None:
        stack: list[str] = []

        def visit(node: ast.AST, in_annotation: bool, in_registry: bool) -> None:
            module.owners[id(node)] = f"{module.name}::" + (".".join(stack) or "<module>")
            if in_annotation:
                module.annotated.add(id(node))
            if in_registry:
                module.registry_nodes.add(id(node))
            annotation_children = _annotation_children(node)
            registry_children: set[int] = set()
            registry_target = _assigned_name(node)
            if registry_target is not None and (module.name, registry_target) in REGISTRIES:
                registry_children.add(id(getattr(node, "value", None)))
            pushed = False
            if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
                stack.append(node.name)
                pushed = True
            elif isinstance(node, ast.Lambda):
                stack.append("<lambda>")
                pushed = True
            for child in ast.iter_child_nodes(node):
                module.parents[id(child)] = node
                visit(
                    child,
                    in_annotation or id(child) in annotation_children,
                    in_registry or id(child) in registry_children,
                )
            if pushed:
                stack.pop()

        visit(module.tree, False, False)

    def resolve(self, module: _Module, dotted: str) -> str:
        """The canonical name a local dotted name denotes, through imports and re-exports."""
        head, *rest = dotted.split(".")
        if head in module.bindings:
            base = module.bindings[head]
        elif head in module.defs:
            base = f"{module.name}.{head}"
        else:
            base = head
        current = ".".join([base, *rest])
        for _ in range(8):
            parts = current.split(".")
            moved = False
            for cut in range(len(parts) - 1, 0, -1):
                prefix, tail = ".".join(parts[:cut]), parts[cut:]
                target = self.modules.get(prefix)
                if target is not None and tail[0] in target.bindings:
                    current = ".".join([target.bindings[tail[0]], *tail[1:]])
                    moved = True
                    break
            if not moved:
                break
        return current

    # -- the references -----------------------------------------------------

    def references(self) -> list[tuple[_Module, ast.AST, str]]:
        """Every OUTERMOST Name/Attribute node of the package with its canonical name."""
        if not hasattr(self, "_references"):
            self._references = list(self._walk_references())
        return self._references

    def _walk_references(self) -> Iterator[tuple[_Module, ast.AST, str]]:
        for module in self.modules.values():
            for node in ast.walk(module.tree):
                if not isinstance(node, ast.Name | ast.Attribute):
                    continue
                parent = module.parents.get(id(node))
                if isinstance(parent, ast.Attribute) and parent.value is node:
                    continue  # the chain above it carries the full name
                if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store | ast.Del):
                    continue
                dotted = _dotted(node)
                if dotted is not None:
                    yield module, node, self.resolve(module, dotted)

    def is_harmless(self, module: _Module, node: ast.AST) -> bool:
        """The declared contexts in which a reference is not a USE."""
        if id(node) in module.annotated or id(node) in module.registry_nodes:
            return True
        parent = module.parents.get(id(node))
        if isinstance(parent, ast.Call) and node in parent.args:
            callee = _dotted(parent.func)
            return callee is not None and self.resolve(module, callee) in HARMLESS_CALLS
        return False

    def uses(self, symbols: frozenset[str] | set[str]) -> dict[str, set[str]]:
        """`{canonical symbol: owners that USE it}` over the whole package.

        `Class.model_fields` and the other introspection reads resolve to a name BELOW the
        class and are therefore never a use of the class; `Class.model_validate` resolves
        the same way and is reported separately, as a construction.
        """
        out: dict[str, set[str]] = {}
        for module, node, canonical in self.references():
            if canonical in symbols and not self.is_harmless(module, node):
                out.setdefault(canonical, set()).add(module.owners[id(node)])
        return out

    def owner_uses(self, owner: str) -> set[str]:
        """Every canonical name an owner references (harmless contexts excluded)."""
        return {
            canonical
            for module, node, canonical in self.references()
            if module.owners[id(node)] == owner and not self.is_harmless(module, node)
        }

    def calls(self) -> Iterator[tuple[_Module, ast.Call, str | None]]:
        for module in self.modules.values():
            for node in ast.walk(module.tree):
                if isinstance(node, ast.Call):
                    callee = _dotted(node.func)
                    yield module, node, None if callee is None else self.resolve(module, callee)

    def importers_of(self, top: str) -> set[str]:
        found: set[str] = set()
        for module in self.modules.values():
            for node in ast.walk(module.tree):
                if isinstance(node, ast.Import) and any(
                    a.name == top or a.name.startswith(top + ".") for a in node.names
                ):
                    found.add(module.name)
                if isinstance(node, ast.ImportFrom) and (
                    node.module == top or (node.module or "").startswith(top + ".")
                ):
                    found.add(module.name)
        return found

    def function(self, declared: str) -> ast.AST | None:
        """The `FunctionDef` a `module::qualname` names, or None."""
        module_name, _, qualname = declared.partition("::")
        module = self.modules.get(module_name)
        if module is None:
            return None
        for node in ast.walk(module.tree):
            if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
                _, _, scope = module.owners[id(node)].partition("::")
                name = node.name if scope == "<module>" else f"{scope}.{node.name}"
                if name == qualname:
                    return node
        return None


@functools.lru_cache(maxsize=256)
def _parse(name: str, source: str) -> ast.Module:
    """Parsed once per (module, source): the battery re-analyses the package ~40 times and
    only ever changes one module of it."""
    return ast.parse(source, filename=name)


def _assigned_name(node: ast.AST) -> str | None:
    """The single `Name` an `Assign`/`AnnAssign` binds, or None."""
    if isinstance(node, ast.Assign) and len(node.targets) == 1:
        target = node.targets[0]
        return target.id if isinstance(target, ast.Name) else None
    if isinstance(node, ast.AnnAssign) and node.value is not None:
        return node.target.id if isinstance(node.target, ast.Name) else None
    return None


def _definitions(tree: ast.Module) -> set[str]:
    """Every name the module defines at its top level."""
    out: set[str] = set()
    for node in tree.body:
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
            out.add(node.name)
        elif (name := _assigned_name(node)) is not None:
            out.add(name)
    return out


def _annotation_children(node: ast.AST) -> set[int]:
    """The DIRECT children of `node` that are ANNOTATIONS: a type named there is not a use.

    An argument's annotation hangs off the `ast.arg` node (a grandchild of the function),
    so it is claimed there; the return annotation hangs off the function itself.
    """
    out: set[int] = set()
    if isinstance(node, ast.arg) and node.annotation is not None:
        out.add(id(node.annotation))
    if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef) and node.returns is not None:
        out.add(id(node.returns))
    if isinstance(node, ast.AnnAssign):
        out.add(id(node.annotation))
    return out


# ---------------------------------------------------------------------------
# the rules — ONE function, returning every violation it can see
# ---------------------------------------------------------------------------


def _policed(module_name: str) -> bool:
    return module_name.startswith(POLICED_PREFIXES)


def _literal(node: ast.AST | None) -> str | None:
    return node.value if isinstance(node, ast.Constant) and isinstance(node.value, str) else None


def _named(nodes: set[str]) -> str:
    return ", ".join(sorted(nodes)) or "—"


def _call_resolves_to(package: _Package, module: _Module, node: ast.AST, symbol: str) -> bool:
    return (
        isinstance(node, ast.Call)
        and (callee := _dotted(node.func)) is not None
        and package.resolve(module, callee) == symbol
    )


def seam_violations(
    sources: Mapping[str, str],
    *,
    doors: frozenset[str] = DOORS,
    fragment_builders: frozenset[str] = FRAGMENT_BUILDERS,
    evidence_hashers: frozenset[str] = EVIDENCE_HASHERS,
) -> list[str]:
    """Every way the package walks around seam (a) or seam (b), as sentences. Empty = closed."""
    package = _Package(sources)
    return [
        *_seam_a_violations(package, doors),
        *_seam_b_violations(package, fragment_builders, evidence_hashers),
    ]


def _seam_a_violations(package: _Package, doors: frozenset[str]) -> list[str]:
    out: list[str] = []
    opener_uses = package.uses(SQLITE_OPENERS)
    for symbol in sorted(SQLITE_OPENERS):
        strangers = opener_uses.get(symbol, set()) - SQLITE_OPENER_FUNCTIONS
        if strangers:
            out.append(f"{symbol} used outside the opener functions by {_named(strangers)}")
    if not opener_uses.get("sqlite3.connect", set()) >= SQLITE_OPENER_FUNCTIONS:
        out.append("an opener function stopped using sqlite3.connect")
    importers = package.importers_of(SQLITE_MODULE)
    if importers != SQLITE_IMPORTERS:
        out.append(f"sqlite3 importers moved: {_named(importers ^ SQLITE_IMPORTERS)}")

    door_uses = package.uses({DOOR_FUNCTION}).get(DOOR_FUNCTION, set())
    if door_uses != doors:
        out.append(f"open_index users are not the declared doors: {_named(door_uses ^ doors)}")
    for door in sorted(doors - {CREATOR}):
        used = package.owner_uses(door)
        if EXISTENCE_HALF not in used:
            out.append(f"{door} does not ask whether the base exists")
        if not used & DESCRIBE_HALF:
            out.append(f"{door} does not ask whether the manifest describes the base")
    if SEALER not in package.owner_uses(CREATOR):
        out.append("the creator no longer seals through write_manifest")

    for module, call, callee in package.calls():
        owner = module.owners[id(call)]
        if callee == DESCRIBE_BASE and isinstance(module.parents.get(id(call)), ast.Expr):
            out.append(f"{owner} discards the verdict of describe_base")
        if callee in {"getattr", "builtins.getattr"} and call.args:
            target = _dotted(call.args[0])
            resolved = package.resolve(module, target) if target else None
            attribute = _literal(call.args[1]) if len(call.args) > 1 else None
            if resolved in WATCHED_MODULES or attribute in WATCHED_ATTRIBUTES:
                out.append(f"{owner} reaches a watched symbol through getattr")
        if callee in {"__import__", "importlib.import_module", "import_module"}:
            name = _literal(call.args[0]) if call.args else None
            if name is None or name in WATCHED_MODULES or name.startswith("xbrain.knowledge"):
                out.append(f"{owner} imports a watched module dynamically")
    return out


def _seam_b_violations(
    package: _Package, fragment_builders: frozenset[str], evidence_hashers: frozenset[str]
) -> list[str]:
    out: list[str] = []
    fragment_uses = package.uses(FRAGMENT_MODELS)
    for symbol in sorted(FRAGMENT_MODELS):
        strangers = fragment_uses.get(symbol, set()) - fragment_builders
        if strangers:
            out.append(f"{symbol} used outside the fragment builders by {_named(strangers)}")
    builders_seen = set().union(*fragment_uses.values()) if fragment_uses else set()
    if not builders_seen >= fragment_builders:
        out.append("a fragment builder no longer constructs its fragment model")
    for module, node, canonical in package.references():
        head, _, attribute = canonical.rpartition(".")
        if head in FRAGMENT_MODELS and attribute in CONSTRUCTOR_METHODS:
            out.append(f"{module.owners[id(node)]} builds a fragment through .{attribute}")
    for module, call, _callee in package.calls():
        owner = module.owners[id(call)]
        func = call.func
        if isinstance(func, ast.Attribute) and func.attr == "__class__":
            out.append(f"{owner} constructs through .__class__")
        if isinstance(func, ast.Call) and _dotted(func.func) == "type":
            out.append(f"{owner} constructs through type(x)(...)")
        if (
            _policed(module.name)
            and isinstance(func, ast.Attribute)
            and func.attr == "model_copy"
            and owner not in MODEL_COPY_FUNCTIONS
        ):
            out.append(f"{owner} rebuilds a model through model_copy")
    for builder in sorted(fragment_builders):
        out += _builder_violations(package, builder)

    hasher_uses = package.uses({HASHER}).get(HASHER, set())
    if hasher_uses != evidence_hashers:
        out.append(
            "chunk_fingerprint users are not the declared hashers: "
            f"{_named(hasher_uses ^ evidence_hashers)}"
        )
    for module, call, callee in package.calls():
        if callee != HASHER:
            continue
        evidence = call.args[0] if call.args else None
        if not _call_resolves_to(package, module, evidence, EVIDENCE):
            out.append(f"{module.owners[id(call)]} hashes something other than chunk_evidence(…)")
    hashlib_importers = {m for m in package.importers_of("hashlib") if _policed(m)}
    if hashlib_importers - HASHLIB_MODULES:
        out.append(f"hashlib imported by {_named(hashlib_importers - HASHLIB_MODULES)}")
    return out


def _builder_violations(package: _Package, builder: str) -> list[str]:
    """Inside one fragment builder: every constructor binds `locator=` to `fragment_locator`."""
    function = package.function(builder)
    if function is None:
        return [f"{builder} does not exist"]
    module = package.modules[builder.partition("::")[0]]
    narrowed = {
        node.targets[0].id
        for node in ast.walk(function)
        if isinstance(node, ast.Assign)
        and len(node.targets) == 1
        and isinstance(node.targets[0], ast.Name)
        and _call_resolves_to(package, module, node.value, NARROWER)
    }
    constructors = [
        node
        for node in ast.walk(function)
        if isinstance(node, ast.Call)
        and (callee := _dotted(node.func)) is not None
        and package.resolve(module, callee) in FRAGMENT_MODELS
    ]
    if not constructors:
        return [f"{builder} constructs no fragment"]
    out = []
    for constructor in constructors:
        locator = next((kw.value for kw in constructor.keywords if kw.arg == "locator"), None)
        if locator is None:
            out.append(f"{builder} constructs a fragment with no locator= keyword")
        elif _call_resolves_to(package, module, locator, NARROWER):
            continue
        elif isinstance(locator, ast.Name) and locator.id in narrowed:
            continue
        else:
            out.append(f"{builder} binds locator= to something fragment_locator did not build")
    return out


# ---------------------------------------------------------------------------
# the assertion on the real package
# ---------------------------------------------------------------------------


def test_every_consumer_of_the_two_seams_is_declared_and_no_reference_walks_around_them() -> None:
    """The package as shipped produces no violation. Read the module docstring for what a
    violation is and for what this cannot see."""
    assert seam_violations(read_package()) == []


def test_the_declared_consumers_exist_in_the_shipped_package() -> None:
    """The sets are asserted EQUAL in both directions by `seam_violations`; this pins that the
    names declared above resolve to functions of the shipped package, so a renamed door is
    red by name rather than silently absent from the walk."""
    package = _Package(read_package())
    for declared in sorted(DOORS | FRAGMENT_BUILDERS | EVIDENCE_HASHERS | MODEL_COPY_FUNCTIONS):
        assert package.function(declared) is not None, declared


# ---------------------------------------------------------------------------
# the positive controls — every bypass both gates found, injected in memory, must be RED
# ---------------------------------------------------------------------------

KN = "xbrain.knowledge."


def _append(sources: dict[str, str], module: str, snippet: str) -> dict[str, str]:
    return {**sources, module: sources.get(module, "") + "\n\n" + snippet + "\n"}


def _replace(sources: dict[str, str], module: str, old: str, new: str) -> dict[str, str]:
    assert sources[module].count(old) == 1, (module, old)
    return {**sources, module: sources[module].replace(old, new)}


BYPASSES: dict[str, tuple[str, str]] = {
    # ---- the two literal controls of round 07 (were red then, must stay red)
    "control-open_index": (
        KN + "index_build",
        "def _peek_plain(d):\n    with open_index(d) as c:\n        return c\n",
    ),
    "control-SearchMatch(**d)": (
        KN + "search_service",
        "def _match_plain(d):\n    return SearchMatch(**d)\n",
    ),
    # ---- seam (a): the Fable gate's nine, and the DIFF/TESTS batteries
    "a1-sqlite3.Connection": (
        KN + "index_build",
        "def _peek_ctor(p):\n    return sqlite3.Connection(str(p))\n",
    ),
    "a2-getattr-connect": (
        KN + "index_build",
        'def _peek_getattr(p):\n    return getattr(sqlite3, "connect")(p)\n',
    ),
    "a3-local-alias": (
        KN + "index_build",
        "def _peek_alias(p):\n    import sqlite3 as _s\n    opener = _s.connect\n    return opener(p)\n",
    ),
    "a4-module-lambda": (KN + "index_build", "_PEEK = lambda d: open_index(d)  # noqa: E731\n"),
    "a5-from-import-as": (
        KN + "index_store",
        "def peek_alias(p):\n    from sqlite3 import connect as _c\n    return _c(p)\n",
    ),
    "a6-async-def": (
        KN + "index_store",
        "async def peek_async(p):\n    return sqlite3.connect(p)\n",
    ),
    "a7-cli-helper": (
        "xbrain.cli",
        "def _peek_index(cfg):\n    import sqlite3\n"
        "    return sqlite3.connect(cfg.index_path / 'knowledge.db')\n",
    ),
    "a8-subpackage": (
        KN + "adapters.mcp",
        "import sqlite3\n\n\ndef peek(path):\n    return sqlite3.connect(path)\n",
    ),
    "a9-partial": (
        KN + "index_build",
        "def _peek_partial(p):\n    import functools\n"
        "    return functools.partial(sqlite3.connect, p)()\n",
    ),
    "a10-door-alias": (
        KN + "index_build",
        "def _peek_door(d):\n    opener = open_index\n    return opener(db_path(d), read_only=True)\n",
    ),
    "a11-module-level": (KN + "index_build", "if False:\n    sqlite3.connect('never')\n"),
    "a12-contextlib-closing": (
        KN + "index_store",
        "def peek_closing(p):\n    import contextlib\n    import sqlite3\n"
        "    with contextlib.closing(sqlite3.connect(p)) as c:\n        return c\n",
    ),
    "a13-import_module": (
        KN + "index_store",
        "def peek_dyn(p):\n    import importlib\n"
        "    return importlib.import_module('sqlite3').connect(p)\n",
    ),
    "a14-reexported-door": (
        KN + "get_service",
        "from xbrain.knowledge.index_schema import open_index as _door\n\n\n"
        "def _peek(d):\n    return _door(d, read_only=True)\n",
    ),
    # ---- seam (b)
    "b1-model_validate": (
        KN + "get_service",
        "def _hydrate(row):\n    return KnowledgeChunk.model_validate(row)\n",
    ),
    "b2-model_copy": (
        KN + "search_service",
        "def _relabel(match, loc):\n    return match.model_copy(update={'locator': loc})\n",
    ),
    "b4-class-alias": (
        KN + "chunking",
        "_KC = KnowledgeChunk\n\n\ndef _emit_alias(**kw):\n    return _KC(**kw)\n",
    ),
    "b5-model_construct": (
        KN + "get_service",
        "def _construct(**kw):\n    return KnowledgeChunk.model_construct(**kw)\n",
    ),
    "b6-import-as-alias": (
        KN + "get_service",
        "from xbrain.knowledge.models import KnowledgeChunk as _KC2\n\n\n"
        "def _build(**kw):\n    return _KC2(**kw)\n",
    ),
    "b7-reexport": (
        KN + "get_service",
        "from xbrain.knowledge.contracts import KnowledgeChunk as _KC3\n\n\n"
        "def _build3(**kw):\n    return _KC3(**kw)\n",
    ),
    "b8-cli-match": (
        "xbrain.cli",
        "def _match_cli(position, hit):\n"
        "    from xbrain.knowledge.contracts import SearchMatch\n"
        "    from xbrain.knowledge.models import Locator\n"
        "    return SearchMatch(chunk_id=hit.chunk_id, locator=Locator(kind='item_text'))\n",
    ),
    "b9-partial-class": (
        KN + "get_service",
        "def _mk():\n    import functools\n    return functools.partial(KnowledgeChunk, chunk_id='x')\n",
    ),
    "b10-type-of": (
        KN + "get_service",
        "def _clone(chunk, **kw):\n    return type(chunk)(**{**dict(chunk), **kw})\n",
    ),
    "b11-getattr-class": (
        KN + "get_service",
        "def _dyn(row):\n    from xbrain.knowledge import models as _m\n"
        "    return getattr(_m, 'KnowledgeChunk')(**row)\n",
    ),
    "b12-dunder-class": (
        KN + "get_service",
        "def _clone2(chunk):\n    return chunk.__class__(**dict(chunk))\n",
    ),
    # ---- the hashers
    "c1-hasher-alias": (
        KN + "index_store",
        "def _rehash(v):\n    from xbrain.knowledge import ids as _ids\n"
        "    fp = _ids.chunk_fingerprint\n"
        "    return fp(('only', 'three', 'fields'), chunker_version=v)\n",
    ),
    "c2-hashlib-verifier": (
        KN + "index_store",
        "import hashlib\n\n\ndef verify_by_hand(hits):\n"
        "    return [h for h in hits if hashlib.sha256(h.text.encode()).hexdigest() == h.fingerprint]\n",
    ),
    "c3-from-import-hasher": (
        KN + "index_store",
        "from xbrain.knowledge.ids import chunk_fingerprint as _fp\n\n\n"
        "def _verify_by_hand(hits):\n"
        "    return tuple(h for h in hits if _fp((h.surface_id, str(h.chunk_index), h.text))"
        " == h.fingerprint)\n",
    ),
}

# The forms that EDIT an existing consumer instead of adding one: the dead calls.
DEAD_CALLS: dict[str, tuple[str, str, str]] = {
    "d1-dead-fragment_locator-in-_chunk": (
        KN + "chunking",
        "    locator = fragment_locator(surface.locator, start, end)\n",
        "    fragment_locator(surface.locator, start, end)\n    locator = surface.locator\n",
    ),
    "d2-dead-chunk_evidence-in-emitter": (
        KN + "chunking",
        "        fingerprint=chunk_fingerprint(\n            chunk_evidence(\n",
        "        fingerprint=chunk_fingerprint(\n"
        "            (surface.surface_id, str(index), text) if True else chunk_evidence(\n",
    ),
    "d3-dead-chunk_evidence-in-verifier": (
        KN + "index_store",
        "        expected = chunk_fingerprint(\n            chunk_evidence(\n",
        "        expected = chunk_fingerprint(\n"
        "            (hit.surface_id, str(hit.chunk_index), hit.text) if True else chunk_evidence(\n",
    ),
}


@pytest.mark.parametrize("name", sorted(BYPASSES))
def test_every_bypass_both_gates_found_is_red(name: str) -> None:
    """The positive control, kept in the file: each form injected into an in-memory copy of
    the shipped package must produce a violation. Seen GREEN on `36f694b` for every form
    but the two controls (gate Fable round 08 §4.3: 9 of 11; DIFF battery 13 of 17; TESTS
    battery 14 of 14)."""
    module, snippet = BYPASSES[name]
    assert seam_violations(_append(read_policed(), module, snippet)), name


@pytest.mark.parametrize("name", sorted(DEAD_CALLS))
def test_a_dead_call_beside_a_hand_built_value_is_red(name: str) -> None:
    """A consumer that keeps calling the seam and serves something else — the form that kept
    `…one_evidence_projection` green in round 07 (TESTS battery d2/d3) and the `_chunk` with a
    dead `fragment_locator` (DIFF b6) that only the behavioural sentinel caught."""
    module, old, new = DEAD_CALLS[name]
    assert seam_violations(_replace(read_policed(), module, old, new)), name


def test_a_fifth_door_that_discards_the_verdict_is_red_even_when_declared() -> None:
    """DIFF a11: a door ADDED to `DOORS` that calls both halves and drops the verdict of
    `describe_base` on the floor. Declaring it does not admit it: the verdict of the seam
    must be consumed — raised, returned or published — never a bare statement."""
    snippet = (
        "def open_ignoring_verdict(index_dir, manifest):\n"
        "    from xbrain.knowledge.index_build import describe_base\n"
        "    require_database(index_dir)\n"
        "    connection = open_index(db_path(index_dir), read_only=True)\n"
        "    describe_base(connection, manifest, db_path(index_dir), whole_file=False)\n"
        "    return connection\n"
    )
    sources = _append(read_policed(), KN + "index_store", snippet)
    doors = DOORS | {KN + "index_store::open_ignoring_verdict"}
    violations = seam_violations(sources, doors=doors)
    assert any("discards the verdict" in v for v in violations), violations


def test_the_analyser_sees_through_the_declared_harmless_contexts_and_nothing_more() -> None:
    """Rule 1 on the analyser itself: an annotation, an `isinstance`, an introspection read
    and a registry element are NOT uses (the package is full of them and is green), while
    the same name as a call argument IS. Pinned on two synthetic modules so a widening of the
    harmless contexts is red here."""
    harmless = (
        "from xbrain.knowledge.models import KnowledgeChunk\n"
        "def f(c: KnowledgeChunk) -> KnowledgeChunk:\n"
        "    assert isinstance(c, KnowledgeChunk)\n"
        "    return c\n"
        "FIELDS = KnowledgeChunk.model_fields\n"
    )
    assert seam_violations(_append(read_policed(), KN + "probe_harmless", harmless)) == []
    harmful = (
        "from xbrain.knowledge.models import KnowledgeChunk\n"
        "def g(rows):\n    return list(map(KnowledgeChunk, rows))\n"
    )
    assert seam_violations(_append(read_policed(), KN + "probe_harmful", harmful))
