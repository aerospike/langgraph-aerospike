from __future__ import annotations

import asyncio
import builtins
import contextlib
import warnings
from collections.abc import AsyncIterator, Iterator, Mapping, Sequence
from datetime import datetime, timezone
from typing import Any, cast

from aerospike_sdk import (
    AerospikeError,
    DataSet,
    Exp,
    IndexAlreadyExistsError,
    Key,
    RecordNotFoundError,
    SyncSession,
)
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

SEP = "|"


def _now_ns() -> datetime:
    return datetime.now(tz=timezone.utc)


class AerospikeSaver(BaseCheckpointSaver):
    def __init__(
        self,
        session: SyncSession,
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

        self.session = session
        self.ns = namespace
        self.set_cp = set_cp
        self.set_writes = set_writes
        self.set_meta = set_meta
        ttl = ttl or {}
        self._ttl_minutes: int | None = ttl.get("default_ttl")
        self._refresh_on_read: bool = bool(ttl.get("refresh_on_read", False))

        # refresh_on_read needs a positive default_ttl to do anything useful.
        if self._refresh_on_read and not (self._ttl_minutes and self._ttl_minutes > 0):
            warnings.warn(
                "refresh_on_read=True has no effect without a positive "
                "default_ttl; TTL sliding on read is disabled.",
                stacklevel=2,
            )

        self._ensure_indexes()

    def _ensure_indexes(self) -> None:
        """Create a ``thread_id`` secondary index on each checkpoint set.

        Swallows ``IndexAlreadyExistsError`` so construction is idempotent.
        """
        for set_name in (self.set_cp, self.set_writes, self.set_meta):
            index_name = f"{set_name}_thread_id_idx"
            with contextlib.suppress(IndexAlreadyExistsError):
                self.session.index(dataset=DataSet.of(self.ns, set_name)).on_bin(
                    "thread_id"
                ).named(index_name).string().create()

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
    def _key_cp(self, thread_id: str, checkpoint_ns: str, checkpoint_id: str) -> Key:
        return DataSet.of(
            self.ns, self.set_cp
        ).id(f"{thread_id}{SEP}{checkpoint_ns}{SEP}{checkpoint_id}")

    def _key_writes(self, thread_id: str, checkpoint_ns: str, checkpoint_id: str) -> Key:
        return DataSet.of(
            self.ns, self.set_writes
        ).id(f"{thread_id}{SEP}{checkpoint_ns}{SEP}{checkpoint_id}")

    def _key_latest(self, thread_id: str, checkpoint_ns: str) -> Key:
        return DataSet.of(
            self.ns, self.set_meta
        ).id(f"{thread_id}{SEP}{checkpoint_ns}{SEP}__latest__")

    # ---------- aerospike io ----------
    def _ttl_seconds(self) -> int | None:
        """Configured default TTL in seconds, or ``None`` when unset/non-positive."""
        minutes = self._ttl_minutes
        if minutes is None:
            return None
        seconds = int(minutes) * 60
        return seconds if seconds > 0 else None

    def _should_refresh(self) -> bool:
        """True when reads should slide the TTL forward (sliding-TTL mode)."""
        return self._refresh_on_read and self._ttl_seconds() is not None

    def _ttl_builder(self, builder) -> Any:
        """Apply TTL policy to a write builder when configured."""
        seconds = self._ttl_seconds()
        if seconds is None:
            return builder
        return builder.expire_record_after_seconds(seconds)

    def _put(self, key: Key, bins: dict[str, Any]) -> None:
        builder = self._ttl_builder(self.session.upsert(key).put(bins))
        try:
            builder.execute()
        except AerospikeError as e:
            raise RuntimeError(f"Aerospike put failed for {key.value}: {e}") from e

    def _get(self, key: Key) -> Any | None:
        try:
            record = self.session.get(key)
        except RecordNotFoundError:
            return None
        except AerospikeError as e:
            raise RuntimeError(f"Aerospike get failed for {key.value}: {e}") from e

        # The SDK sync fast-path ``session.get`` does not expose read policy
        # overrides, so sliding TTL is implemented as a follow-up ``touch``.
        if self._should_refresh():
            seconds = self._ttl_seconds()
            if seconds is not None:
                try:
                    self.session.touch(key).expire_record_after_seconds(seconds).execute()
                except RecordNotFoundError:
                    pass
                except AerospikeError as e:
                    raise RuntimeError(
                        f"Aerospike touch failed for {key.value}: {e}"
                    ) from e

        return record

    def _touch_latest(self, thread_id: str, checkpoint_ns: str) -> None:
        """Refresh ``__latest__`` TTL on read-by-id.

        Read-by-id refreshes the checkpoint and writes records via ``_get``,
        but it never reads the ``__latest__`` pointer.
        """
        seconds = self._ttl_seconds()
        if seconds is None:
            return
        key = self._key_latest(thread_id, checkpoint_ns)
        try:
            self.session.touch(key).expire_record_after_seconds(seconds).execute()
        except RecordNotFoundError:
            pass
        except AerospikeError as e:
            raise RuntimeError(f"Aerospike touch failed for latest record: {e}") from e

    def _list_checkpoint_ids(
        self, thread_id: str, checkpoint_ns: str
    ) -> builtins.list[tuple[str, str]]:
        """Return ``(iso_timestamp, checkpoint_id)`` for a thread/ns, newest first.

        History is enumerated from checkpoint records through the ``thread_id``
        index, avoiding a single growing timeline record.
        """
        ds = DataSet.of(self.ns, self.set_cp)
        stream = self.session.query(ds).where(
            Exp.eq(Exp.string_bin("thread_id"), Exp.string_val(thread_id))
        ).bins(["checkpoint_ns", "checkpoint_id", "ts"]).execute()

        pairs: list[tuple[str, str]] = []
        for result in stream:
            if not result.is_ok or result.record is None:
                continue
            bins = result.record.bins
            if bins.get("checkpoint_ns") != checkpoint_ns:
                continue
            cid = bins.get("checkpoint_id")
            ts = bins.get("ts")
            if isinstance(cid, str) and isinstance(ts, str):
                pairs.append((ts, cid))
        pairs.sort(key=lambda p: (p[0], p[1]), reverse=True)
        return pairs

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
        # Bins duplicate key parts so list() can project them from index query results.
        rec: dict[str, Any] = {
            "thread_id": thread_id,
            "checkpoint_ns": checkpoint_ns,
            "checkpoint_id": checkpoint_id,
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
        ``f"{task_id}|{idx}"``, written via a single server-side
        operation chain. ``map_upsert_items`` is server-atomic, giving us
        upsert-on-retry and tolerating concurrent callers against the same
        checkpoint.

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

        builder = self.session.upsert(key).bin("thread_id").set_to(thread_id)
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
            builder = builder.bin("writes").map_upsert_items({map_key: new_item})

        builder = self._ttl_builder(builder)
        try:
            builder.execute()
        except AerospikeError as e:
            raise RuntimeError(f"Aerospike operate failed for {key.value}: {e}") from e

    def get_tuple(
        self,
        config: RunnableConfig,
    ) -> CheckpointTuple | None:

        thread_id, checkpoint_ns, checkpoint_id = self._ids_from_config(config)

        resolved_via_latest = checkpoint_id is None
        if checkpoint_id is None:
            latest = self._get(self._key_latest(thread_id, checkpoint_ns))
            if latest is None or "checkpoint_id" not in latest.bins:
                return None
            checkpoint_id = latest.bins["checkpoint_id"]

        key = self._key_cp(thread_id, checkpoint_ns, checkpoint_id)
        got = self._get(key)
        if got is None:
            return None

        # If we resolved through __latest__, _get already refreshed it.
        # Read-by-id does not touch __latest__, so refresh it explicitly.
        if not resolved_via_latest and self._should_refresh():
            self._touch_latest(thread_id, checkpoint_ns)

        bins = got.bins

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
            # `writes` is a Map bin (see `put_writes`); each value
            # carries its own `task_id`, `channel`, and `idx`, so we
            # don't depend on map iteration order.
            writes_map = wrec.bins.get("writes") or {}
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
        """Delete every checkpoint, pending-write, and meta record for ``thread_id``."""
        for set_name in (self.set_cp, self.set_writes, self.set_meta):
            ds = DataSet.of(self.ns, set_name)
            digests: builtins.list[bytes] = []
            stream = self.session.query(ds).where(
                Exp.eq(Exp.string_bin("thread_id"), Exp.string_val(thread_id))
            ).with_no_bins().execute()
            for result in stream:
                if result.is_ok and result.record is not None:
                    digests.append(result.record.key.digest)

            for digest in digests:
                with contextlib.suppress(RecordNotFoundError):
                    self.session.delete(ds.id_from_digest(digest)).execute()

    def list(
        self,
        config: RunnableConfig | None,
        *,
        filter: dict[str, Any] | None = None,
        before: RunnableConfig | None = None,
        limit: int | None = None,
    ) -> Iterator[CheckpointTuple]:

        thread_id, checkpoint_ns, _ = self._ids_from_config(config or {})

        items = self._list_checkpoint_ids(thread_id, checkpoint_ns)

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
