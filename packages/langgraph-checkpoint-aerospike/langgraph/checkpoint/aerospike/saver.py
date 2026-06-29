from __future__ import annotations

import asyncio
import builtins
import contextlib
from collections.abc import AsyncIterator, Iterator, Mapping, Sequence
from datetime import datetime, timezone
from typing import Any, cast

from aerospike_helpers.operations import map_operations, operations
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import (
    WRITES_IDX_MAP,
    BaseCheckpointSaver,
    ChannelVersions,
    Checkpoint,
    CheckpointMetadata,
    CheckpointTuple,
    SerializerProtocol,
)

import aerospike
import aerospike.exception  # noqa: F401  # expose `aerospike.exception` submodule for type checkers

SEP = "|"


def _now_ns() -> datetime:
    return datetime.now(tz=timezone.utc)


class AerospikeSaver(BaseCheckpointSaver):
    def __init__(
        self,
        client: aerospike.Client,
        namespace: str = "test",
        set_cp: str = "lg_cp",
        set_writes: str = "lg_cp_w",
        set_meta: str = "lg_cp_meta",
        ttl: dict[str, Any] | None = None,
        *,
        serde: SerializerProtocol | None = None,
    ) -> None:
        # `BaseCheckpointSaver.__init__` registers `self.serde`, wrapping it
        # in `maybe_add_typed_methods` for backwards compatibility. Skipping
        # this call leaves `self.serde` pointing at the class-level default
        # and bypasses any future bookkeeping upstream adds to the base
        # constructor, so always forward.
        super().__init__(serde=serde)

        self.client = client
        self.ns = namespace
        self.set_cp = set_cp
        self.set_writes = set_writes
        self.set_meta = set_meta
        self.ttl = ttl or {}
        self.timeline_max: int | None = None
        self._ttl_minutes: int | None = self.ttl.get("default_ttl")
        self._refresh_on_read: bool = bool(self.ttl.get("refresh_on_read", False))

        self._ensure_indexes()

    def _ensure_indexes(self) -> None:
        """Create the secondary indexes ``delete_thread`` relies on.

        Idempotent: if an index with the same name already exists,
        ``IndexFoundError`` is raised by the client and silently swallowed.
        Any other failure (auth, namespace missing, ...) is propagated so
        misconfiguration surfaces at construction time rather than later
        when ``delete_thread`` is called.
        """
        for set_name in (self.set_cp, self.set_writes, self.set_meta):
            index_name = f"{set_name}_thread_id_idx"
            with contextlib.suppress(aerospike.exception.IndexFoundError):
                self.client.index_single_value_create(
                    self.ns,
                    set_name,
                    "thread_id",
                    aerospike.INDEX_STRING,
                    index_name,
                )

    # ---------- config parsing ----------
    @staticmethod
    def _ids_from_config(
        config: Mapping[str, Any] | None,
    ) -> tuple[str, str, str | None]:
        """Returns ``(thread_id, checkpoint_ns, checkpoint_id)`` from a RunnableConfig."""
        cfg = config or {}
        c = cfg.get("configurable", {}) or {}
        md = cfg.get("metadata", {}) or {}

        thread_id = c.get("thread_id") or md.get("thread_id")
        if not thread_id:
            raise ValueError("configurable.thread_id is required in RunnableConfig")

        checkpoint_ns = c.get("checkpoint_ns") or md.get("checkpoint_ns") or ""

        checkpoint_id = c.get("checkpoint_id")

        return thread_id, checkpoint_ns, checkpoint_id

    # ---------- keys ----------
    def _key_cp(self, thread_id: str, checkpoint_ns: str, checkpoint_id: str):
        return (self.ns, self.set_cp, f"{thread_id}{SEP}{checkpoint_ns}{SEP}{checkpoint_id}")

    def _key_writes(self, thread_id: str, checkpoint_ns: str, checkpoint_id: str):
        return (self.ns, self.set_writes, f"{thread_id}{SEP}{checkpoint_ns}{SEP}{checkpoint_id}")

    def _key_latest(self, thread_id: str, checkpoint_ns: str):
        return (self.ns, self.set_meta, f"{thread_id}{SEP}{checkpoint_ns}{SEP}__latest__")

    def _key_timeline(self, thread_id: str, checkpoint_ns: str):
        return (self.ns, self.set_meta, f"{thread_id}{SEP}{checkpoint_ns}{SEP}__timeline__")

    # ---------- aerospike io ----------
    def _ttl_policy(self) -> dict[str, Any] | None:
        """Return ``{"ttl": seconds}`` for the configured TTL, or ``None``.

        Passed as ``policy=`` to both ``client.put`` and ``client.operate``.
        """
        minutes = self._ttl_minutes
        if minutes is None:
            return None
        seconds = int(minutes) * 60
        return {"ttl": seconds} if seconds > 0 else None

    def _put(self, key, bins: dict[str, Any]) -> None:
        policy = self._ttl_policy()
        try:
            if policy is not None:
                self.client.put(key, bins, policy=policy)
            else:
                self.client.put(key, bins)
        except aerospike.exception.AerospikeError as e:
            raise RuntimeError(f"Aerospike put failed for {key}: {e}") from e

    def _get(self, key) -> tuple | None:
        # `read_touch_ttl_percent=100` refreshes the TTL on every
        # successful read, server-side, in the same round-trip.
        policy: dict[str, Any] | None = None
        if self._refresh_on_read and self._ttl_minutes is not None and self._ttl_minutes > 0:
            policy = {"read_touch_ttl_percent": 100}

        try:
            if policy is not None:
                rec = self.client.get(key, policy=policy)
            else:
                rec = self.client.get(key)
        except aerospike.exception.RecordNotFound:
            return None
        except aerospike.exception.AerospikeError as e:
            raise RuntimeError(f"Aerospike get failed for {key}: {e}") from e

        return rec

    def _read_timeline_items(self, timeline_key) -> builtins.list[tuple[str, str]]:
        """Return timeline entries as ``(iso_timestamp, checkpoint_id)`` pairs.

        On-disk shape is a Map bin (``timeline``) keyed by
        ``checkpoint_id`` with ISO-timestamp values. We sort by ``ts``
        descending here so callers see reverse-chronological order.
        """
        rec = self._get(timeline_key)
        if rec is None:
            return []
        bins = rec[2]
        timeline = bins.get("timeline") or {}
        if not isinstance(timeline, dict):
            return []
        pairs: list[tuple[str, str]] = [
            (ts, cid)
            for cid, ts in timeline.items()
            if isinstance(ts, str) and isinstance(cid, str)
        ]
        pairs.sort(key=lambda p: p[0], reverse=True)
        return pairs

    def _delete(self, key) -> None:
        try:
            self.client.remove(key)
        except aerospike.exception.RecordNotFound:
            pass
        except aerospike.exception.AerospikeError as e:
            raise RuntimeError(f"Aerospike delete failed for {key}: {e}") from e

    # ---------- public API (RunnableConfig-based) ----------
    def put(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: ChannelVersions,
    ) -> RunnableConfig:
        thread_id, checkpoint_ns, parent_checkpoint_id = self._ids_from_config(config)
        checkpoint_id = checkpoint.get("id")
        if not checkpoint_id:
            raise ValueError("checkpoint_id is required for put()")

        # `Checkpoint.ts` is a required TypedDict field, but be defensive in case
        # an older serialized format ever omits it.
        ts: str = checkpoint.get("ts") or _now_ns().isoformat()
        checkpoint["ts"] = ts

        cp_type, cp_bytes = self.serde.dumps_typed(checkpoint)
        metadata = metadata.copy()
        extra_metadata = cast(CheckpointMetadata, config.get("metadata") or {})
        metadata.update(extra_metadata)

        meta_type, meta_bytes = self.serde.dumps_typed(metadata)

        key = self._key_cp(thread_id, checkpoint_ns, checkpoint_id)
        rec: dict[str, Any] = {
            "thread_id": thread_id,
            "p_checkpoint_id": parent_checkpoint_id,
            "cp_type": cp_type,
            "checkpoint": cp_bytes,
            "meta_type": meta_type,
            "metadata": meta_bytes,
            "ts": ts,
        }
        self._put(key, rec)

        latest_key = self._key_latest(thread_id, checkpoint_ns)
        self._put(
            latest_key,
            {
                "thread_id": thread_id,
                "checkpoint_id": checkpoint_id,
                "ts": ts,
            },
        )

        timeline_key = self._key_timeline(thread_id, checkpoint_ns)
        # `map_put` upserts atomically: re-`put()`ing the same
        # `checkpoint_id` overwrites in place, and concurrent `put()`s
        # against the same thread/ns can't clobber each other's entries.
        timeline_ops: list[dict[str, Any]] = [
            operations.write("thread_id", thread_id),
            map_operations.map_put("timeline", checkpoint_id, ts),
        ]
        timeline_policy = self._ttl_policy()
        try:
            if timeline_policy is not None:
                self.client.operate(timeline_key, timeline_ops, policy=timeline_policy)
            else:
                self.client.operate(timeline_key, timeline_ops)
        except aerospike.exception.AerospikeError as e:
            raise RuntimeError(f"Aerospike operate failed for {timeline_key}: {e}") from e

        cfg_conf: dict[str, Any] = {**(config.get("configurable") or {})}
        cfg_conf.update(
            {
                "thread_id": thread_id,
                "checkpoint_ns": checkpoint_ns,
                "checkpoint_id": checkpoint_id,
            }
        )
        new_config: RunnableConfig = {**config, "configurable": cfg_conf}
        return new_config

    def put_writes(
        self,
        config: RunnableConfig,
        writes: Sequence[tuple[str, Any]],
        task_id: str,
        task_path: str = "",
    ) -> None:
        """Persist pending writes for a checkpoint.

        Each write is stored inside a Map bin (``writes``) keyed by
        ``f"{task_id}|{idx}"``, written via a single ``client.operate``
        call. ``map_put`` is server-atomic, giving us upsert-on-retry
        and tolerating concurrent callers against the same checkpoint.

        The ``thread_id`` bin is rewritten on every call so that
        ``delete_thread``'s secondary-index query keeps finding the
        record; the value is the same every time for a given key.
        """
        if not writes:
            return

        thread_id, checkpoint_ns, checkpoint_id = self._ids_from_config(config)
        if not checkpoint_id:
            return

        key = self._key_writes(thread_id, checkpoint_ns, checkpoint_id)
        now_ts = _now_ns().isoformat()

        ops: list[dict[str, Any]] = [operations.write("thread_id", thread_id)]
        for idx, (channel, value) in enumerate(writes):
            idx_val = WRITES_IDX_MAP.get(channel, idx)
            type_, serialized = self.serde.dumps_typed(value)
            new_item = {
                "task_id": task_id,
                "task_path": task_path,
                "channel": channel,
                "idx": idx_val,
                "type": type_,
                "value": serialized,
                "ts": now_ts,
            }
            map_key = f"{task_id}{SEP}{idx_val}"
            ops.append(map_operations.map_put("writes", map_key, new_item))

        policy = self._ttl_policy()
        try:
            if policy is not None:
                self.client.operate(key, ops, policy=policy)
            else:
                self.client.operate(key, ops)
        except aerospike.exception.AerospikeError as e:
            raise RuntimeError(f"Aerospike operate failed for {key}: {e}") from e

    def get_tuple(
        self,
        config: RunnableConfig,
    ) -> CheckpointTuple | None:

        thread_id, checkpoint_ns, checkpoint_id = self._ids_from_config(config)

        if checkpoint_id is None:
            latest = self._get(self._key_latest(thread_id, checkpoint_ns))
            if latest is None or "checkpoint_id" not in latest[2]:
                return None
            checkpoint_id = latest[2]["checkpoint_id"]

        key = self._key_cp(thread_id, checkpoint_ns, checkpoint_id)
        got = self._get(key)
        if got is None:
            return None

        # Refresh timeline record TTL if refresh_on_read is configured
        if self._refresh_on_read and self._ttl_minutes is not None and self._ttl_minutes > 0:
            self._get(self._key_timeline(thread_id, checkpoint_ns))

        _, _, bins = got

        cp_type = bins.get("cp_type")
        raw_cp = bins.get("checkpoint")
        raw_meta = bins.get("metadata")
        meta_type = bins.get("meta_type")
        if cp_type is None or raw_cp is None:
            return None
        try:
            checkpoint = self.serde.loads_typed((cp_type, raw_cp))
        except Exception:
            return None

        if meta_type is None or raw_meta is None:
            return None
        try:
            metadata = self.serde.loads_typed((meta_type, raw_meta))
        except Exception:
            return None

        pending_writes: list[tuple[str, str, Any]] = []
        wrec = self._get(self._key_writes(thread_id, checkpoint_ns, checkpoint_id))
        if wrec is not None:
            _, _, wbins = wrec
            # `writes` is a Map bin (see `put_writes`); each value
            # carries its own `task_id`, `channel`, and `idx`, so we
            # don't depend on map iteration order.
            writes_map = wbins.get("writes") or {}
            for item in writes_map.values():
                try:
                    task_id = item.get("task_id", "")
                    channel = item["channel"]
                    type_ = item["type"]
                    serialized = item["value"]
                    value = self.serde.loads_typed((type_, serialized))
                    pending_writes.append((task_id, channel, value))
                except KeyError:
                    continue

        cp_config: RunnableConfig = {
            "configurable": {
                "thread_id": thread_id,
                "checkpoint_ns": checkpoint_ns,
                "checkpoint_id": checkpoint_id,
            }
        }

        parent_config: RunnableConfig | None = None
        if bins.get("p_checkpoint_id"):
            parent_config = {
                "configurable": {
                    "thread_id": thread_id,
                    "checkpoint_ns": checkpoint_ns,
                    "checkpoint_id": bins.get("p_checkpoint_id"),
                }
            }

        return CheckpointTuple(
            config=cp_config,
            checkpoint=checkpoint,
            metadata=metadata,
            parent_config=parent_config,
            pending_writes=pending_writes,
        )

    def delete_thread(self, thread_id: str) -> None:
        """Delete every checkpoint, pending-write, and meta record for ``thread_id``.

        Spans every ``checkpoint_ns`` belonging to the thread. Implemented
        with a per-set secondary-index query on the ``thread_id`` bin
        (created in ``_ensure_indexes``), so cost is O(records-for-thread)
        rather than O(set-size).
        """
        from aerospike import predicates  # local import to avoid global aerospike side-effects

        for set_name in (self.set_cp, self.set_writes, self.set_meta):
            digests: builtins.list[bytes] = []

            def _collect(record: tuple, _digests: builtins.list[bytes] = digests) -> None:
                (_, _, _, digest), _meta, _bins = record
                _digests.append(digest)

            query = self.client.query(self.ns, set_name)
            query.where(predicates.equals("thread_id", thread_id))
            query.foreach(_collect)

            for digest in digests:
                with contextlib.suppress(aerospike.exception.RecordNotFound):
                    self.client.remove((self.ns, set_name, None, digest))

    def list(
        self,
        config: RunnableConfig | None,
        *,
        filter: dict[str, Any] | None = None,
        before: RunnableConfig | None = None,
        limit: int | None = None,
    ) -> Iterator[CheckpointTuple]:

        thread_id, checkpoint_ns, _ = self._ids_from_config(config or {})

        timeline_key = self._key_timeline(thread_id, checkpoint_ns)
        items = self._read_timeline_items(timeline_key)

        before_id: str | None = None
        if before is not None:
            _, _, before_id = self._ids_from_config(before or {})

        if before_id:
            seen = False
            new_items: list[tuple[str, str]] = []
            for ts, cid in items:
                if not seen:
                    if cid == before_id:
                        seen = True
                    continue
                new_items.append((ts, cid))
            items = new_items

        yielded = 0
        for _, cid in items:
            if limit is not None and yielded >= limit:
                break

            cp_config: RunnableConfig = {
                "configurable": {
                    "thread_id": thread_id,
                    "checkpoint_ns": checkpoint_ns,
                    "checkpoint_id": cid,
                }
            }

            tpl = self.get_tuple(cp_config)
            if tpl is None:
                continue

            if filter:
                ok = True
                for k, v in filter.items():
                    if tpl.metadata.get(k) != v:
                        ok = False
                        break
                if not ok:
                    continue

            yielded += 1
            yield tpl

    async def aget(self, config: RunnableConfig) -> Checkpoint | None:
        if value := await self.aget_tuple(config):
            return value.checkpoint
        return None

    async def aget_tuple(self, config: RunnableConfig) -> CheckpointTuple | None:

        return await asyncio.to_thread(self.get_tuple, config)

    async def alist(
        self,
        config: RunnableConfig | None,
        *,
        filter: dict[str, Any] | None = None,
        before: RunnableConfig | None = None,
        limit: int | None = None,
    ) -> AsyncIterator[CheckpointTuple]:

        def _collect() -> list[CheckpointTuple]:
            return list(self.list(config, filter=filter, before=before, limit=limit))

        items = await asyncio.to_thread(_collect)
        for tpl in items:
            yield tpl

    async def aput(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: ChannelVersions,
    ) -> RunnableConfig:

        return await asyncio.to_thread(
            self.put,
            config,
            checkpoint,
            metadata,
            new_versions,
        )

    async def aput_writes(
        self,
        config: RunnableConfig,
        writes: Sequence[tuple[str, Any]],
        task_id: str,
        task_path: str = "",
    ) -> None:

        await asyncio.to_thread(
            self.put_writes,
            config,
            writes,
            task_id,
            task_path,
        )

    async def adelete_thread(self, thread_id: str) -> None:
        """Asynchronously delete all checkpoint history for ``thread_id``."""
        await asyncio.to_thread(self.delete_thread, thread_id)
