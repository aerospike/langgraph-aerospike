import asyncio
import contextlib
import warnings
from collections.abc import Iterable
from datetime import datetime, timezone
from typing import Any

from aerospike_sdk import (
    AerospikeError,
    CollectionIndexType,
    DataSet,
    Exp,
    ExpType,
    IndexAlreadyExistsError,
    ListReturnType,
    MapReturnType,
    RecordNotFoundError,
    SyncSession,
)

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
        session: SyncSession,
        namespace: str = "langgraph",
        set: str = "store",
        ttl_config: TTLConfig | None = None,
        vector_session: Any | None = None,
    ) -> None:
        self.session = session
        self.ns = namespace
        self.set = set
        self.ttl_config = ttl_config
        # ``vector_session`` is retained for backward compatibility but is
        # no longer required: the Aerospike SDK session passed as
        # ``session`` can read and write native VECTOR bins directly.
        self.vector_session = vector_session

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
        Swallows ``IndexAlreadyExistsError`` so construction is idempotent.
        """
        ds = DataSet.of(self.ns, self.set)
        for bin_name, index_name in (
            ("ns_prefixes", f"{self.set}_ns_prefixes_idx"),
            ("ns_suffixes", f"{self.set}_ns_suffixes_idx"),
        ):
            with contextlib.suppress(IndexAlreadyExistsError):
                self.session.index(dataset=ds).on_bin(bin_name).named(
                    index_name
                ).collection(CollectionIndexType.LIST).string().create()

    # --------------- Aerospike helper functions ------------------

    def _dataset(self) -> DataSet:
        return DataSet.of(self.ns, self.set)

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
        """Return an index-friendly expression, or ``None`` if the path has no anchor.

        Aerospike queries accept one ``where`` clause. Prefer the prefix index;
        expression filters still enforce the full prefix/suffix pattern.
        """
        if prefix:
            anchor = self._leading_anchor(prefix)
            if anchor is not None:
                return Exp.in_list(Exp.val(anchor), Exp.list_bin("ns_prefixes"))
        if suffix:
            anchor = self._trailing_anchor(suffix)
            if anchor is not None:
                return Exp.in_list(Exp.val(anchor), Exp.list_bin("ns_suffixes"))
        return None

    @staticmethod
    def _combined_where_expr(predicate: Any | None, exprs: list) -> Any | None:
        """AND ``predicate`` (an index-friendly expression, or ``None``) with ``exprs``.

        Returns ``None`` when there is nothing to filter on, so callers can
        skip ``.where(...)`` entirely (a bare query/scan).
        """
        conditions: list = []
        if predicate is not None:
            conditions.append(predicate)
        conditions.extend(exprs)
        if not conditions:
            return None
        return conditions[0] if len(conditions) == 1 else Exp.and_(conditions)

    def _run_query(self, predicate: Any | None, exprs: list) -> list:
        """Query the set, optionally narrowed by ``predicate``, filtered by ``exprs``."""
        query = self.session.query(self._dataset())
        where_expr = self._combined_where_expr(predicate, exprs)
        if where_expr is not None:
            query = query.where(where_expr)
        stream = query.execute()
        return [
            result.record
            for result in stream
            if result.is_ok and result.record is not None
        ]

    def _build_read_policy_for_refresh(self, refresh_ttl: bool | None) -> int | None:
        """Return the configured TTL in seconds for a read refresh, or ``None``."""
        if refresh_ttl:
            return self._ttl_seconds()
        if self.ttl_config is not None and self.ttl_config.get("refresh_on_read"):
            return self._ttl_seconds()
        return None

    def _ttl_seconds(self) -> int | None:
        """Configured default TTL in seconds, or ``None`` when unset/non-positive."""
        if self.ttl_config is None:
            return None
        minutes = self.ttl_config.get("default_ttl")
        if minutes is None:
            return None
        seconds = int(minutes) * 60
        return seconds if seconds > 0 else None

    @staticmethod
    def _get_type_result(value: Any):
        if isinstance(value, bool):
            return ExpType.BOOL
        elif isinstance(value, int):
            return ExpType.INT
        elif isinstance(value, float):
            return ExpType.FLOAT
        elif isinstance(value, str):
            return ExpType.STRING
        elif isinstance(value, bytes):
            return ExpType.BLOB
        elif isinstance(value, (dict, list)):
            return ExpType.MAP if isinstance(value, dict) else ExpType.LIST
        return ExpType.STRING

    def _get_op_expression(self, bin_expr, value_expr, operator: str):
        ops = {
            "$eq": Exp.eq,
            "$ne": Exp.ne,
            "$gt": Exp.gt,
            "$gte": Exp.ge,
            "$lt": Exp.lt,
            "$lte": Exp.le,
        }

        if operator not in ops:
            raise ValueError(f"Unsupported operator: {operator}")

        return ops[operator](bin_expr, value_expr)

    # Metric name -> (Exp distance-expression builder, sort order for "best
    # first"). Cosine similarity and dot product rank higher-is-closer;
    # squared euclidean distance ranks lower-is-closer.
    _VECTOR_METRICS: dict[str, tuple[str, str]] = {
        "cosine": ("cosine_similarity", "desc"),
        "euclidean": ("euclidean_squared_distance", "asc"),
        "dotproduct": ("dot_product", "desc"),
    }

    def _build_vector_distance_expr(self, query_vector: list, metric: str = "cosine") -> Any:
        """Build a server-side vector distance expression for ``metric``.

        Requires the ``aie/vector`` branch of ``aerospike-sdk`` and Aerospike
        Server 8.1.3+. The returned expression is meant to be projected via
        ``.with_op_projection(ExpOperation.read(...))`` and ranked with
        ``.order_by(...).top_k(...)`` -- see
        ``_handle_vector_search``.

        Args:
            query_vector: List of floats representing the query embedding
            metric: One of "cosine", "euclidean", "dotproduct"

        Returns:
            An Aerospike expression for computing distance
        """
        from aerospike_sdk import Vector

        if metric not in self._VECTOR_METRICS:
            raise ValueError(
                f"Unsupported distance metric: {metric}. "
                f"Choose from: {list(self._VECTOR_METRICS.keys())}"
            )

        exp_fn_name, _ = self._VECTOR_METRICS[metric]
        distance_fn = getattr(Exp, exp_fn_name)
        return distance_fn(Vector(query_vector), Exp.vector_bin("embedding"))

    def _build_path_filter(
        self, path: NamespacePath, bin_name: str, is_suffix: bool = False
    ) -> list:
        """Build a list of expressions to handle wildcards in a NamespacePath."""
        conditions = []
        path_len = len(path)
        conditions.append(
            Exp.ge(Exp.list_size(Exp.list_bin(bin_name), ()), Exp.val(path_len))
        )
        for i, token in enumerate(path):
            if token == "*":
                continue
            # Prefixes are matched from the front; suffixes from the back.
            # This preserves ``("users", "123", "*")`` as a true prefix and
            # ``("*", "settings")`` as a true suffix.
            algo_index = i - path_len if is_suffix else i
            result_type = self._get_type_result(token)
            match_condition = Exp.eq(
                Exp.list_get_by_index(
                    ListReturnType.VALUE,
                    result_type,
                    Exp.val(algo_index),
                    Exp.list_bin(bin_name),
                    (),
                ),
                Exp.val(token),
            )
            conditions.append(match_condition)

        return conditions

    def _build_filter_exprs_from_dict(self, filter_dict: dict[str, Any]) -> list:
        filter_exprs = []

        for key, condition in filter_dict.items():
            map_key_expr = Exp.val(key)
            if isinstance(condition, dict) and any(k.startswith("$") for k in condition):
                for op, val in condition.items():
                    result_type = self._get_type_result(val)
                    target_expr = Exp.map_get_by_key(
                        MapReturnType.VALUE,
                        result_type,
                        map_key_expr,
                        Exp.map_bin("value"),
                        (),
                    )

                    op_expr = self._get_op_expression(target_expr, Exp.val(val), op)
                    filter_exprs.append(op_expr)

            else:
                result_type = self._get_type_result(condition)
                target_expr = Exp.map_get_by_key(
                    MapReturnType.VALUE,
                    result_type,
                    map_key_expr,
                    Exp.map_bin("value"),
                    (),
                )
                filter_exprs.append(Exp.eq(target_expr, Exp.val(condition)))

        return filter_exprs

    # --------------- Per-op handlers (called from batch) --------------------
    #
    # Each handler implements one Op variant against Aerospike. They are
    # grouped together so `batch` stays a thin dispatch table.

    def _handle_put(self, op: PutOp) -> None:
        p_key = self._key(op.namespace, op.key)
        key = self._dataset().id(p_key[2])

        if op.value is None:
            try:
                self.session.delete(key).execute()
            except RecordNotFoundError:
                return
            except AerospikeError as e:
                raise RuntimeError(f"Aerospike remove failed for {op.key}: {e}") from e
            return

        # Extract embedding if present (Pre-release: requires Aerospike Server 8.1.3+).
        # ``PutOp`` is immutable, so strip ``__embedding__`` into a local copy
        # rather than mutating ``op.value``.
        value = op.value
        embedding = None
        if isinstance(value, dict) and "__embedding__" in value:
            embedding = value.get("__embedding__")
            # Remove __embedding__ from stored value to avoid duplication
            value = {k: v for k, v in value.items() if k != "__embedding__"}

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
        # use `map_insert_items(..., no_fail=True)` for `created_at`: the
        # value is set on first write and silently left alone on every
        # subsequent upsert.
        now = _now_utc().isoformat()
        builder = self.session.upsert(key)
        builder.bin("namespace").set_to(list(op.namespace))
        builder.bin("key").set_to(op.key)
        builder.bin("value").set_to(value)
        builder.bin("ns_prefixes").set_to(self._ns_prefixes(op.namespace))
        builder.bin("ns_suffixes").set_to(self._ns_suffixes(op.namespace))
        builder.bin("meta").map_insert_items({"created_at": now}, no_fail=True)
        builder.bin("meta").map_upsert_items({"updated_at": now})

        if embedding is not None:
            from aerospike_sdk import Vector

            builder.bin("embedding").set_to_vector(Vector(list(embedding)))

        if time_to_live < 0:
            builder = builder.never_expire()
        else:
            builder = builder.expire_record_after_seconds(time_to_live)

        try:
            builder.execute()
        except AerospikeError as e:
            raise RuntimeError(f"Aerospike put failed for {op.key}: {e}") from e

    def _handle_get(self, op: GetOp) -> Item | None:
        p_key = self._key(op.namespace, op.key)
        key = self._dataset().id(p_key[2])
        try:
            record = self.session.get(key)
        except RecordNotFoundError:
            return None
        except AerospikeError as e:
            raise RuntimeError(f"Aerospike get failed for {op.key}: {e}") from e

        bins = record.bins
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

        refresh_seconds = self._build_read_policy_for_refresh(op.refresh_ttl)
        if refresh_seconds is not None:
            try:
                self.session.touch(key).expire_record_after_seconds(refresh_seconds).execute()
            except RecordNotFoundError:
                pass
            except AerospikeError as e:
                raise RuntimeError(f"Aerospike refresh failed for {op.key}: {e}") from e

        return Item(value=value, key=k, namespace=ns, created_at=created_at, updated_at=updated_at)

    def _handle_search(self, op: SearchOp) -> list[SearchItem]:
        """Search items by namespace, scalar filter, or vector similarity.

        For vector search (requires the ``aie/vector`` branch of
        ``aerospike-sdk`` and Aerospike Server 8.1.3+):
        - Pass embedding as op.query (list of floats)
        - OR pass {"embedding": [...], "metric": "cosine|euclidean|dotproduct"}
        - Ranking is pushed down server-side (see ``_handle_vector_search``);
          results carry the distance in ``score``.
        """
        # Detect vector query and extract parameters.
        # ``SearchOp.query`` is typed as ``str | None`` upstream; this integration
        # also accepts a list of floats or a dict of ``{embedding, metric}``.
        raw_query: Any = op.query
        vector_query = None
        distance_metric = "cosine"

        if raw_query:
            if isinstance(raw_query, dict):
                vector_query = raw_query.get("embedding")
                if vector_query is None:
                    raise ValueError("dict query must contain an 'embedding' key")
                distance_metric = raw_query.get("metric", "cosine")
            elif isinstance(raw_query, list):
                vector_query = raw_query
            else:
                raise ValueError(
                    f"Unsupported query type {type(raw_query)}: pass a list of floats "
                    "or {'embedding': [...], 'metric': ...} for vector search."
                )

        # Build expressions for namespace prefix and value filters
        exprs: list = []
        if op.namespace_prefix:
            exprs.extend(self._build_path_filter(op.namespace_prefix, "namespace", is_suffix=False))
        if op.filter:
            exprs.extend(self._build_filter_exprs_from_dict(op.filter))

        # Branch on vector vs scalar search
        if vector_query:
            # Pre-release: requires Aerospike Server 8.1.3+
            return self._handle_vector_search(op, vector_query, distance_metric, exprs)
        else:
            return self._handle_scalar_search(op, exprs)

    def _handle_scalar_search(self, op: SearchOp, exprs: list) -> list[SearchItem]:
        """Handle scalar (non-vector) search queries."""
        # The index narrows the candidate set; expressions enforce the exact
        # namespace prefix and value filters.
        predicate = self._index_predicate(op.namespace_prefix, None)
        try:
            records = self._run_query(predicate, exprs)
        except AerospikeError as e:
            raise RuntimeError(f"Aerospike search failed: {e}") from e

        out: list[SearchItem] = []
        for record in records:
            bins = record.bins
            ns = tuple(bins.get("namespace", ()))
            key = bins.get("key")
            value = bins.get("value")
            meta = bins.get("meta") or {}
            now = _now_utc().isoformat()
            created_at = meta.get("created_at", now)
            updated_at = meta.get("updated_at", now)

            refresh_seconds = self._build_read_policy_for_refresh(op.refresh_ttl)
            if refresh_seconds is not None:
                try:
                    self.session.touch(record.key).expire_record_after_seconds(refresh_seconds).execute()
                except RecordNotFoundError:
                    continue
                except AerospikeError as e:
                    raise RuntimeError(f"Aerospike search refresh failed: {e}") from e

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

    # Aerospike's Top-K pushdown caps how many ranked records a single
    # query can return.
    _MAX_TOP_K = 1000

    def _handle_vector_search(
        self,
        op: SearchOp,
        vector_query: list,
        distance_metric: str,
        scalar_exprs: list,
    ) -> list[SearchItem]:
        """Handle vector similarity search via server-side Top-K pushdown.

        Requires the ``aie/vector`` branch of ``aerospike-sdk`` (vector
        support is not yet on a published release) and Aerospike Server
        8.1.3+. Ranking is computed on the server: the distance expression
        from ``_build_vector_distance_expr`` is projected into a named
        result bin via ``.with_op_projection(ExpOperation.read(...))``, and
        ``.order_by(...).top_k(...)`` performs the ``ORDER BY <distance>
        LIMIT k`` pushdown (merged client-side only when a queried node
        doesn't support it) -- see the SDK's "Vector bins and Top-K
        queries" README section and ``examples/vector_topk_query.py``.
        This is still a brute-force scan-and-rank over the queried set
        (there is no ANN/HNSW index), just computed where the data lives
        instead of after shipping every candidate record to the client.

        Args:
            op: SearchOp with namespace_prefix, filter, limit, offset, refresh_ttl
            vector_query: Query embedding vector (list of floats)
            distance_metric: One of "cosine", "euclidean", "dotproduct"
            scalar_exprs: List of expressions for namespace/filter filtering

        Returns:
            List of SearchItem ranked by distance (with score field populated)
        """
        try:
            from aerospike_async import ExpOperation, ExpReadFlags, Operation
            from aerospike_sdk import Order, OrderByType
        except ImportError as exc:
            raise ImportError(
                "Vector search requires the `aie/vector` branch of "
                "aerospike-sdk (not yet on a published release). Install "
                "with:\n"
                "  pip install 'aerospike-sdk @ "
                "git+https://github.com/aerospike/"
                "aerospike-client-python-sdk.git@aie/vector'"
            ) from exc

        if distance_metric not in self._VECTOR_METRICS:
            raise ValueError(f"Unsupported metric: {distance_metric}")

        distance_expr = self._build_vector_distance_expr(vector_query, distance_metric)
        _, sort_direction = self._VECTOR_METRICS[distance_metric]
        order = Order.DESC if sort_direction == "desc" else Order.ASC

        # Top-K ranks and truncates server-side, so ask for exactly as many
        # ranked candidates as offset+limit needs (capped at the server's
        # per-query maximum) instead of fetching every match.
        offset = op.offset or 0
        limit = op.limit if op.limit is not None else self._MAX_TOP_K
        top_k = max(1, min(self._MAX_TOP_K, offset + limit))

        predicate = self._index_predicate(op.namespace_prefix, None)
        where_expr = self._combined_where_expr(predicate, scalar_exprs)

        query = self.session.query(self._dataset())
        if where_expr is not None:
            query = query.where(where_expr)
        query = (
            query.with_op_projection(
                Operation.get_bin("namespace"),
                Operation.get_bin("key"),
                Operation.get_bin("value"),
                Operation.get_bin("meta"),
                # EVAL_NO_FAIL drops the "distance" bin instead of failing
                # the whole query when a record's vector is missing,
                # dimension-mismatched, or not a VECTOR particle.
                ExpOperation.read("distance", distance_expr, ExpReadFlags.EVAL_NO_FAIL),
            )
            .order_by("distance", OrderByType.DOUBLE, order)
            .top_k(top_k)
        )

        try:
            stream = query.execute()
        except AerospikeError as e:
            raise RuntimeError(f"Aerospike vector search failed: {e}") from e

        # top_k() results already arrive ranked best-first; only offset
        # needs to be applied client-side.
        ranked: list[Any] = [
            result.record
            for result in stream
            if result.is_ok
            and result.record is not None
            and "distance" in result.record.bins
        ][offset : offset + limit]

        out: list[SearchItem] = []
        for record in ranked:
            bins = record.bins
            ns = tuple(bins.get("namespace", ()))
            key = bins.get("key")
            value = bins.get("value")
            meta = bins.get("meta") or {}
            now = _now_utc().isoformat()
            created_at = meta.get("created_at", now)
            updated_at = meta.get("updated_at", now)

            refresh_seconds = self._build_read_policy_for_refresh(op.refresh_ttl)
            if refresh_seconds is not None:
                try:
                    self.session.touch(record.key).expire_record_after_seconds(refresh_seconds).execute()
                except RecordNotFoundError:
                    continue
                except AerospikeError as e:
                    raise RuntimeError(f"Aerospike search refresh failed: {e}") from e

            out.append(
                SearchItem(
                    namespace=ns,
                    key=key,
                    value=value,
                    created_at=created_at,
                    updated_at=updated_at,
                    score=bins["distance"],
                )
            )

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
        except AerospikeError as e:
            raise RuntimeError(f"Aerospike list_namespaces failed: {e}") from e

        all_namespaces: set[tuple[str, ...]] = set()
        for record in records:
            ns = tuple(record.bins.get("namespace", ()))
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
