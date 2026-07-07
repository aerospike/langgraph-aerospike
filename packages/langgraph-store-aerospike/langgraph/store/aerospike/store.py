import asyncio
import contextlib
import warnings
from collections.abc import Iterable
from datetime import datetime, timezone
from typing import Any

from aerospike_helpers import expressions as exp
from aerospike_helpers.operations import map_operations, operations

# `Result`, `SearchItem`, and `TTLConfig` are part of the documented public
# surface of `langgraph.store.base` (they appear in `BaseStore` method
# signatures and the public `Op` union), but upstream forgot to list them in
# `langgraph.store.base.__all__`. The `noinspection PyProtectedMember` comment
# keeps PyCharm quiet without pulling in genuinely private helpers
# (`_ensure_ttl`, `_ensure_refresh`, `_validate_namespace`).
# noinspection PyProtectedMember
from langgraph.store.base import (  # noqa: PLC2701
    BaseStore,
    GetOp,
    Item,
    ListNamespacesOp,
    NamespacePath,
    Op,
    PutOp,
    Result,
    SearchItem,
    SearchOp,
    TTLConfig,
)

import aerospike
import aerospike.exception  # noqa: F401  # expose `aerospike.exception` submodule for type checkers

SEP = "|"


def _now_utc() -> datetime:
    return datetime.now(tz=timezone.utc)


class AerospikeStore(BaseStore):
    """Aerospike-backed implementation of LangGraph's ``BaseStore``.

    ``BaseStore`` already provides concrete implementations of every
    high-level convenience method (``put``, ``get``, ``delete``, ``search``,
    ``list_namespaces`` and their ``a*`` async twins). Each of those methods
    validates inputs, resolves TTL/refresh defaults via the store's
    ``ttl_config``, and then funnels the work through ``self.batch(...)`` /
    ``self.abatch(...)``.

    Per the LangGraph integration contract, an adapter only needs to
    implement ``batch`` and ``abatch``. Everything else comes for free, so
    this class deliberately avoids overriding the public surface to:

    * keep the adapter small and focused on the Aerospike-specific bits,
    * inherit any future improvements to validation/TTL handling, and
    * avoid importing private helpers (``_ensure_ttl``, ``_ensure_refresh``,
      ``_validate_namespace``) from ``langgraph.store.base``.
    """

    supports_ttl: bool = True

    def __init__(
        self,
        client: aerospike.Client,
        namespace: str = "langgraph",
        set: str = "store",
        ttl_config: TTLConfig | None = None,
    ) -> None:
        self.client = client
        self.ns = namespace
        self.set = set
        self.ttl_config = ttl_config

        # refresh_on_read needs a positive default_ttl to do anything useful.
        if ttl_config is not None and ttl_config.get("refresh_on_read"):
            default_ttl = ttl_config.get("default_ttl")
            if not (default_ttl and default_ttl > 0):
                warnings.warn(
                    "refresh_on_read=True has no effect without a positive "
                    "default_ttl; TTL sliding on read is disabled.",
                    stacklevel=2,
                )

        self._ensure_indexes()

    def _ensure_indexes(self) -> None:
        """Create secondary indexes on ``ns_prefixes`` and ``ns_suffixes``.

        Each item stores denormalized prefix/suffix list bins, so anchored
        namespace lookups can use CDT list indexes instead of scanning the set.
        Swallows ``IndexFoundError`` so construction is idempotent.
        """
        for bin_name, index_name in (
            ("ns_prefixes", f"{self.set}_ns_prefixes_idx"),
            ("ns_suffixes", f"{self.set}_ns_suffixes_idx"),
        ):
            with contextlib.suppress(aerospike.exception.IndexFoundError):
                self.client.index_list_create(
                    self.ns,
                    self.set,
                    bin_name,
                    aerospike.INDEX_STRING,
                    index_name,
                )

    # --------------- Aerospike helper functions ------------------

    def _key(self, namespace: tuple[str, ...], key: str) -> tuple[str, str, str]:
        return (self.ns, self.set, SEP.join([*namespace, key]))

    @staticmethod
    def _ns_prefixes(namespace: tuple[str, ...]) -> list[str]:
        """Joined contiguous prefixes, e.g. ``("a","b","c")`` -> ``["a","a|b","a|b|c"]``.

        Joining with ``SEP`` keeps token boundaries explicit for index values.
        """
        return [SEP.join(namespace[: i + 1]) for i in range(len(namespace))]

    @staticmethod
    def _ns_suffixes(namespace: tuple[str, ...]) -> list[str]:
        """Joined contiguous suffixes, e.g. ``("a","b","c")`` -> ``["c","b|c","a|b|c"]``."""
        n = len(namespace)
        return [SEP.join(namespace[n - i - 1 :]) for i in range(n)]

    @staticmethod
    def _leading_anchor(path: NamespacePath) -> str | None:
        """Leading literal tokens before the first ``*``, joined with ``SEP``."""
        tokens: list[str] = []
        for token in path:
            if token == "*":
                break
            tokens.append(token)
        return SEP.join(tokens) if tokens else None

    @staticmethod
    def _trailing_anchor(path: NamespacePath) -> str | None:
        """Trailing literal tokens after the last ``*``, joined with ``SEP``."""
        tokens: list[str] = []
        for token in reversed(path):
            if token == "*":
                break
            tokens.append(token)
        tokens.reverse()
        return SEP.join(tokens) if tokens else None

    def _index_predicate(
        self, prefix: NamespacePath | None, suffix: NamespacePath | None
    ) -> Any | None:
        """Return an index predicate, or ``None`` if the path has no anchor.

        Aerospike queries accept one ``where`` clause. Prefer the prefix index;
        expression filters still enforce the full prefix/suffix pattern.
        """
        from aerospike import predicates  # local import to avoid global aerospike side-effects

        if prefix:
            anchor = self._leading_anchor(prefix)
            if anchor is not None:
                return predicates.contains("ns_prefixes", aerospike.INDEX_TYPE_LIST, anchor)
        if suffix:
            anchor = self._trailing_anchor(suffix)
            if anchor is not None:
                return predicates.contains("ns_suffixes", aerospike.INDEX_TYPE_LIST, anchor)
        return None

    def _run_query(self, predicate: Any | None, exprs: list) -> list:
        """Query the set, optionally narrowed by ``predicate``, filtered by ``exprs``."""
        query = self.client.query(self.ns, self.set)
        if predicate is not None:
            query.where(predicate)
        policy: dict[str, Any] = {}
        if len(exprs) == 1:
            policy["expressions"] = exprs[0].compile()
        elif exprs:
            policy["expressions"] = exp.And(*exprs).compile()
        return query.results(policy=policy)

    def _build_read_policy_for_refresh(self, refresh_ttl: bool | None) -> dict[str, Any]:
        policy: dict[str, Any] = {}
        if self.ttl_config is not None and self.ttl_config.get("refresh_on_read"):
            policy["read_touch_ttl_percent"] = 100
        if refresh_ttl:
            policy["read_touch_ttl_percent"] = 100
        return policy

    def _get_type_result(self, value: Any):
        if isinstance(value, bool):
            return exp.ResultType.BOOLEAN
        elif isinstance(value, int):
            return exp.ResultType.INTEGER
        elif isinstance(value, float):
            return exp.ResultType.FLOAT
        elif isinstance(value, str):
            return exp.ResultType.STRING
        elif isinstance(value, bytes):
            return exp.ResultType.BLOB
        elif isinstance(value, (dict, list)):
            return exp.ResultType.MAP if isinstance(value, dict) else exp.ResultType.LIST
        return exp.ResultType.STRING

    def _get_op_expression(self, bin_expr, value_expr, operator: str):
        ops = {
            "$eq": exp.Eq,
            "$ne": exp.NE,
            "$gt": exp.GT,
            "$gte": exp.GE,
            "$lt": exp.LT,
            "$lte": exp.LE,
        }

        if operator not in ops:
            raise ValueError(f"Unsupported operator: {operator}")

        return ops[operator](bin_expr, value_expr)

    def _build_path_filter(
        self, path: NamespacePath, bin_name: str, is_suffix: bool = False
    ) -> list:
        """Build a list of expressions to handle wildcards in a NamespacePath."""
        conditions = []
        path_len = len(path)
        size_check = exp.GE(exp.ListSize(None, exp.ListBin(bin_name)), exp.Val(path_len))
        conditions.append(size_check)
        for i, token in enumerate(path):
            if token == "*":
                continue
            algo_index = i - path_len if is_suffix else i
            result_type = self._get_type_result(token)
            match_condition = exp.Eq(
                exp.ListGetByIndex(
                    None,
                    aerospike.LIST_RETURN_VALUE,
                    result_type,
                    exp.Val(algo_index),
                    exp.ListBin(bin_name),
                ),
                exp.Val(token),
            )
            conditions.append(match_condition)

        return conditions

    def _build_filter_exprs_from_dict(self, filter_dict: dict[str, Any]) -> list:
        filter_exprs = []

        for key, condition in filter_dict.items():
            map_key_expr = exp.Val(key)
            if isinstance(condition, dict) and any(k.startswith("$") for k in condition):
                for op, val in condition.items():
                    result_type = self._get_type_result(val)
                    target_expr = exp.MapGetByKey(
                        None,
                        aerospike.MAP_RETURN_VALUE,
                        result_type,
                        map_key_expr,
                        exp.MapBin("value"),
                    )

                    op_expr = self._get_op_expression(target_expr, exp.Val(val), op)
                    filter_exprs.append(op_expr)

            else:
                result_type = self._get_type_result(condition)
                target_expr = exp.MapGetByKey(
                    None, aerospike.MAP_RETURN_VALUE, result_type, map_key_expr, exp.MapBin("value")
                )
                filter_exprs.append(exp.Eq(target_expr, exp.Val(condition)))

        return filter_exprs

    # --------------- Per-op handlers (called from batch) --------------------
    #
    # Each handler implements one Op variant against Aerospike. They are
    # grouped together so `batch` stays a thin dispatch table.

    def _handle_put(self, op: PutOp) -> None:
        p_key = self._key(op.namespace, op.key)

        if op.value is None:
            try:
                self.client.remove(p_key)
            except aerospike.exception.RecordNotFound:
                return
            except aerospike.exception.AerospikeError as e:
                raise RuntimeError(f"Aerospike remove failed for {op.key}: {e}") from e
            return

        # `op.ttl` has already been resolved by `BaseStore.put` via
        # `_ensure_ttl(ttl_config, ttl)`, so it is either `None` (no TTL
        # configured / caller asked for "no expiration") or a positive float
        # in minutes. We map `None` to Aerospike's "never expire" sentinel
        # (-1) so behavior is deterministic regardless of the namespace's
        # default-ttl.
        if op.ttl is None:
            time_to_live: int = -1
        else:
            time_to_live = -1 if op.ttl < 0 else int(op.ttl * 60)

        # `created_at` / `updated_at` live in a `meta` Map bin so we can
        # use `MAP_WRITE_FLAGS_CREATE_ONLY | MAP_WRITE_FLAGS_NO_FAIL`
        # for `created_at`: the value is set on first write and silently
        # left alone on every subsequent upsert.
        now = _now_utc().isoformat()
        ops = [
            operations.write("namespace", list(op.namespace)),
            operations.write("key", op.key),
            operations.write("value", op.value),
            operations.write("ns_prefixes", self._ns_prefixes(op.namespace)),
            operations.write("ns_suffixes", self._ns_suffixes(op.namespace)),
            map_operations.map_put(
                "meta",
                "created_at",
                now,
                map_policy={
                    "map_write_flags": (
                        aerospike.MAP_WRITE_FLAGS_CREATE_ONLY | aerospike.MAP_WRITE_FLAGS_NO_FAIL
                    ),
                },
            ),
            map_operations.map_put("meta", "updated_at", now),
        ]
        try:
            self.client.operate(p_key, ops, policy={"ttl": time_to_live})
        except aerospike.exception.AerospikeError as e:
            raise RuntimeError(f"Aerospike put failed for {op.key}: {e}") from e

    def _handle_get(self, op: GetOp) -> Item | None:
        p_key = self._key(op.namespace, op.key)
        read_policy = self._build_read_policy_for_refresh(op.refresh_ttl)
        try:
            if read_policy:
                _, _, bins = self.client.get(p_key, policy=read_policy)
            else:
                _, _, bins = self.client.get(p_key)
        except aerospike.exception.RecordNotFound:
            return None
        except aerospike.exception.AerospikeError as e:
            raise RuntimeError(f"Aerospike get failed for {op.key}: {e}") from e

        value = bins.get("value")
        if value is None:
            return None

        ns = tuple(bins.get("namespace", op.namespace))
        k = bins.get("key", op.key)
        # Timestamps live in the `meta` Map bin (see `_handle_put`).
        meta = bins.get("meta") or {}
        now = _now_utc().isoformat()
        created_at = meta.get("created_at", now)
        updated_at = meta.get("updated_at", now)

        return Item(value=value, key=k, namespace=ns, created_at=created_at, updated_at=updated_at)

    def _handle_search(self, op: SearchOp) -> list[SearchItem]:
        if op.query:
            raise NotImplementedError(
                "AerospikeStore does not support semantic/vector search. "
                "Use search without the `query` argument."
            )

        exprs: list = []
        if op.namespace_prefix:
            exprs.extend(self._build_path_filter(op.namespace_prefix, "namespace", is_suffix=False))
        if op.filter:
            exprs.extend(self._build_filter_exprs_from_dict(op.filter))

        # The index narrows the candidate set; expressions enforce the exact
        # namespace prefix and value filters.
        predicate = self._index_predicate(op.namespace_prefix, None)
        try:
            records = self._run_query(predicate, exprs)
        except aerospike.exception.AerospikeError as e:
            raise RuntimeError(f"Aerospike search failed: {e}") from e

        # Re-fetch each match to apply read-touch TTL (queries don't support it).
        read_policy = self._build_read_policy_for_refresh(op.refresh_ttl)
        out: list[SearchItem] = []

        for pkey, _, bins in records:
            if read_policy:
                try:
                    _, _, bins = self.client.get(pkey, policy=read_policy)
                except aerospike.exception.RecordNotFound:
                    continue
                except aerospike.exception.AerospikeError as e:
                    raise RuntimeError(f"Aerospike search refresh failed: {e}") from e
            ns = tuple(bins.get("namespace", ()))
            key = bins.get("key")
            value = bins.get("value")
            meta = bins.get("meta") or {}
            now = _now_utc().isoformat()
            created_at = meta.get("created_at", now)
            updated_at = meta.get("updated_at", now)

            out.append(
                SearchItem(
                    namespace=ns,
                    key=key,
                    value=value,
                    created_at=created_at,
                    updated_at=updated_at,
                    score=None,
                )
            )

        if op.offset:
            out = out[op.offset :]
        if op.limit is not None:
            out = out[: op.limit]

        return out

    def _handle_list_namespaces(self, op: ListNamespacesOp) -> list[tuple[str, ...]]:
        prefix: NamespacePath | None = None
        suffix: NamespacePath | None = None
        if op.match_conditions:
            for condition in op.match_conditions:
                if condition.match_type == "prefix":
                    prefix = condition.path
                elif condition.match_type == "suffix":
                    suffix = condition.path
                else:
                    raise ValueError(f"Match type {condition.match_type} must be prefix or suffix.")

        exprs: list = []
        if prefix:
            exprs.extend(self._build_path_filter(prefix, "namespace", is_suffix=False))
        if suffix:
            exprs.extend(self._build_path_filter(suffix, "namespace", is_suffix=True))

        # Use one namespace index when possible; expression filters handle
        # wildcards and the other match condition.
        predicate = self._index_predicate(prefix, suffix)
        try:
            records = self._run_query(predicate, exprs)
        except aerospike.exception.AerospikeError as e:
            raise RuntimeError(f"Aerospike list_namespaces failed: {e}") from e

        all_namespaces: set[tuple[str, ...]] = set()
        for _, _, bins in records:
            ns = tuple(bins.get("namespace", ()))
            if op.max_depth is not None:
                ns = ns[: op.max_depth]
            all_namespaces.add(ns)

        # Sort for stable pagination (query order is digest-hash, not lexical).
        result = sorted(all_namespaces)
        if op.offset:
            result = result[op.offset :]
        if op.limit:
            result = result[: op.limit]
        return result

    # --------------- BaseStore implementation ------------------

    def batch(self, ops: Iterable[Op]) -> list[Result]:
        result: list[Result] = []
        for op in ops:
            if isinstance(op, GetOp):
                result.append(self._handle_get(op))
            elif isinstance(op, PutOp):
                self._handle_put(op)
                result.append(None)
            elif isinstance(op, SearchOp):
                result.append(self._handle_search(op))
            elif isinstance(op, ListNamespacesOp):
                result.append(self._handle_list_namespaces(op))
            else:
                raise TypeError(f"Unsupported operation type: {type(op)}")

        return result

    async def abatch(self, ops: Iterable[Op]) -> list[Result]:
        return await asyncio.to_thread(self.batch, ops)
