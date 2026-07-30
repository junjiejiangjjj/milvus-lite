# WAL Concurrent Write Hardening Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Serialize all mutations within a Collection and reject structurally invalid WAL batches during recovery without changing the WAL format or default fsync policy.

**Architecture:** Add a per-Collection reentrant write lock around every mutation and WAL/MemTable freeze path while retaining the existing maintenance lock for Manifest and background-maintenance state. Refactor WAL recovery so Arrow stream-read failures preserve the validated prefix, while a decoded batch that fails full validation raises `WALCorruptedError` immediately.

**Tech Stack:** Python 3.10+, `threading.RLock`, PyArrow IPC Streaming, pytest, concurrent futures/threading.

---

## File Structure

- Modify `milvus_lite/storage/wal.py`: classify WAL read failures, validate decoded batches, log partial recovery, and raise contextual corruption errors.
- Modify `milvus_lite/engine/collection.py`: create the per-Collection write lock and apply it to mutation, flush, and shutdown entry points.
- Modify `tests/storage/test_wal.py`: add a deterministic decoded-but-invalid Arrow batch fixture and recovery classification tests.
- Create `tests/engine/test_concurrent_writers.py`: verify same-Collection writer serialization, repeated concurrent flushes, reopen consistency, and cross-Collection independence.

### Task 1: Reject Decoded but Invalid WAL Batches

**Files:**
- Modify: `tests/storage/test_wal.py`
- Modify: `milvus_lite/storage/wal.py:14-56`

- [ ] **Step 1: Add a deterministic corrupt-offset test helper**

Add these imports to `tests/storage/test_wal.py`:

```python
import struct

from milvus_lite.exceptions import WALCorruptedError
```

Add this helper after `_make_delta_batch`:

```python
def _corrupt_partition_offsets(path: str) -> None:
    """Make _partition structurally invalid without breaking IPC decoding."""
    payload = bytearray(open(path, "rb").read())
    offsets = struct.pack("<iii", 0, 8, 16)
    position = payload.find(offsets)
    assert position >= 0, "expected _partition offset buffer in WAL fixture"
    payload[position + 4:position + 8] = struct.pack("<i", 100)
    with open(path, "wb") as sink:
        sink.write(payload)
```

The existing `_make_data_batch` uses two `_default` values, each eight bytes,
so its `_partition` offsets are deterministically `[0, 8, 16]`.

- [ ] **Step 2: Write the failing decoded-corruption test**

Add to the truncation section of `tests/storage/test_wal.py`:

```python
def test_decoded_invalid_batch_raises_wal_corrupted(
    wal_dir, wal_data_schema, wal_delta_schema
):
    wal = WAL(wal_dir, wal_data_schema, wal_delta_schema, wal_number=1)
    wal.write_insert(_make_data_batch(wal_data_schema))
    wal._data_writer.close()
    wal._data_sink.close()
    wal._closed = True

    path = os.path.join(wal_dir, "wal_data_000001.arrow")
    _corrupt_partition_offsets(path)

    with pa.OSFile(path, "rb") as source:
        batch = pa.ipc.open_stream(source).read_next_batch()
    batch.validate(full=False)
    with pytest.raises(pa.ArrowInvalid, match="Offset invariant failure"):
        batch.validate(full=True)

    with pytest.raises(WALCorruptedError) as exc_info:
        WAL.recover(wal_dir, 1)

    message = str(exc_info.value)
    assert path in message
    assert "batch 0" in message
    assert "Offset invariant failure" in message
```

- [ ] **Step 3: Run the test to verify current recovery accepts the batch**

Run:

```bash
python3 -m pytest tests/storage/test_wal.py::test_decoded_invalid_batch_raises_wal_corrupted -v
```

Expected: FAIL because `WAL.recover` returns the invalid batch instead of
raising `WALCorruptedError`.

- [ ] **Step 4: Implement separate read and validation exception boundaries**

Update imports and `_read_wal_file` in `milvus_lite/storage/wal.py`:

```python
import logging

from milvus_lite.exceptions import WALCorruptedError

logger = logging.getLogger(__name__)


def _read_wal_file(path: str) -> List[pa.RecordBatch]:
    """Return the validated, recoverable RecordBatch prefix from one WAL."""
    if not os.path.exists(path):
        return []

    batches: list[pa.RecordBatch] = []
    try:
        with pa.OSFile(path, "rb") as source:
            try:
                reader = pa.ipc.open_stream(source)
            except pa.ArrowInvalid as exc:
                logger.warning("WAL schema/stream is unreadable: %s: %s", path, exc)
                return []

            batch_index = 0
            while True:
                try:
                    batch = reader.read_next_batch()
                except StopIteration:
                    break
                except pa.ArrowInvalid as exc:
                    logger.warning(
                        "WAL tail is truncated; retained %d validated batches: %s: %s",
                        len(batches),
                        path,
                        exc,
                    )
                    break

                try:
                    batch.validate(full=True)
                except pa.ArrowInvalid as exc:
                    raise WALCorruptedError(
                        f"WAL {path!r} batch {batch_index} failed full validation: {exc}"
                    ) from exc

                batches.append(batch)
                batch_index += 1
    except (OSError, IOError) as exc:
        logger.warning(
            "WAL I/O error; retained %d validated batches: %s: %s",
            len(batches),
            path,
            exc,
        )

    return batches
```

Do not catch `WALCorruptedError` in this function. Do not add a broad generic
exception handler.

- [ ] **Step 5: Verify decoded corruption fails early and truncation still recovers**

Run:

```bash
python3 -m pytest \
  tests/storage/test_wal.py::test_decoded_invalid_batch_raises_wal_corrupted \
  tests/storage/test_wal.py::test_truncated_file_recovers_partial \
  tests/storage/test_wal.py::test_corrupted_file_returns_empty \
  tests/storage/test_wal.py::test_read_wal_file_does_not_swallow_real_bugs \
  -v
```

Expected: 4 passed.

- [ ] **Step 6: Commit the recovery validation change**

```bash
git add milvus_lite/storage/wal.py tests/storage/test_wal.py
git commit -m "fix: validate recovered WAL batches"
```

### Task 2: Serialize Collection Mutation and WAL Rotation

**Files:**
- Modify: `milvus_lite/engine/collection.py:20-40, 200-270, 339-562, 1275-1283, 1771-1835, 2132-2151`
- Create: `tests/engine/test_concurrent_writers.py`

- [ ] **Step 1: Write a failing same-Collection serialization test**

Create `tests/engine/test_concurrent_writers.py`:

```python
from __future__ import annotations

import threading
import time

import pytest

from milvus_lite.engine.collection import Collection
from milvus_lite.schema.types import CollectionSchema, DataType, FieldSchema


@pytest.fixture
def schema():
    return CollectionSchema(fields=[
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True),
        FieldSchema(name="vec", dtype=DataType.FLOAT_VECTOR, dim=2),
    ])


def _record(pk: int) -> dict:
    return {"id": pk, "vec": [float(pk), float(pk + 1)]}


def test_same_collection_mutations_are_serialized(tmp_path, schema, monkeypatch):
    col = Collection("c", str(tmp_path / "c"), schema)
    active = 0
    max_active = 0
    state_lock = threading.Lock()
    original_apply = col._apply

    def tracked_apply(op):
        nonlocal active, max_active
        with state_lock:
            active += 1
            max_active = max(max_active, active)
        try:
            time.sleep(0.01)
            return original_apply(op)
        finally:
            with state_lock:
                active -= 1

    monkeypatch.setattr(col, "_apply", tracked_apply)
    barrier = threading.Barrier(3)
    errors = []

    def writer(pk):
        try:
            barrier.wait()
            col.insert([_record(pk)])
        except BaseException as exc:
            errors.append(exc)

    threads = [threading.Thread(target=writer, args=(pk,)) for pk in (1, 2)]
    for thread in threads:
        thread.start()
    barrier.wait()
    for thread in threads:
        thread.join(timeout=5)

    assert errors == []
    assert max_active == 1
    col.close()
```

- [ ] **Step 2: Run the serialization test to demonstrate overlapping writers**

Run:

```bash
python3 -m pytest tests/engine/test_concurrent_writers.py::test_same_collection_mutations_are_serialized -v
```

Expected: FAIL with `assert 2 == 1` because both threads can enter `_apply`.

- [ ] **Step 3: Add the reusable write-lock decorator and lock instance**

Add `functools` to the imports in `milvus_lite/engine/collection.py`, then add
this module-level helper before `class Collection`:

```python
def _write_locked(method):
    """Serialize a complete Collection mutation transaction."""
    @functools.wraps(method)
    def wrapper(self, *args, **kwargs):
        with self._write_lock:
            return method(self, *args, **kwargs)
    return wrapper
```

Initialize the per-Collection lock before constructing background workers:

```python
self._write_lock: threading.RLock = threading.RLock()
self._maintenance_lock: threading.RLock = threading.RLock()
```

The write lock must be distinct from `_maintenance_lock`.

- [ ] **Step 4: Apply the lock to every scoped mutation and lifecycle path**

Add the exact decorator line `@_write_locked` immediately above each of these
existing declarations, without otherwise changing their signatures or
bodies:

```text
def insert(
def upsert(
def delete(
def flush(self) -> None:
def _trigger_flush(self) -> None:
def close(self) -> None:
```

Do not add `_write_lock` acquisition inside `WAL.write_insert` or
`WAL.write_delete`: WAL remains a storage primitive and Collection owns the
mutation transaction. Do not change `_maintenance_lock` acquisition sites.

The decorator is intentionally applied to both public methods and
`_trigger_flush`. `RLock` makes nested paths such as `upsert -> insert` and
`insert -> _trigger_flush` safe.

- [ ] **Step 5: Verify same-Collection writes are serialized**

Run:

```bash
python3 -m pytest tests/engine/test_concurrent_writers.py::test_same_collection_mutations_are_serialized -v
```

Expected: PASS.

- [ ] **Step 6: Run focused Collection and flush regression tests**

Run:

```bash
python3 -m pytest -q \
  tests/engine/test_collection.py \
  tests/engine/test_flush.py \
  tests/engine/test_background_flush.py \
  tests/engine/test_concurrency_stability.py
```

Expected: all tests pass.

- [ ] **Step 7: Commit Collection write serialization**

```bash
git add milvus_lite/engine/collection.py tests/engine/test_concurrent_writers.py
git commit -m "fix: serialize collection mutations"
```

### Task 3: Prove Concurrent Flush Safety and Per-Collection Independence

**Files:**
- Modify: `tests/engine/test_concurrent_writers.py`

- [ ] **Step 1: Add a repeated concurrent flush-and-reopen test**

Append:

```python
def test_concurrent_writers_flush_and_reopen(tmp_path, schema, monkeypatch):
    monkeypatch.setattr("milvus_lite.engine.collection.MEMTABLE_SIZE_LIMIT", 3)
    data_dir = str(tmp_path / "c")
    col = Collection("c", data_dir, schema)
    barrier = threading.Barrier(5)
    errors = []

    def writer(worker_id):
        try:
            barrier.wait()
            for offset in range(20):
                pk = worker_id * 1000 + offset
                col.insert([_record(pk)])
                if offset % 5 == 0:
                    col.flush()
        except BaseException as exc:
            errors.append(exc)

    threads = [threading.Thread(target=writer, args=(worker_id,)) for worker_id in range(4)]
    for thread in threads:
        thread.start()
    barrier.wait()
    for thread in threads:
        thread.join(timeout=20)

    assert not any(thread.is_alive() for thread in threads)
    assert errors == []
    col.close()

    reopened = Collection("c", data_dir, schema)
    try:
        reopened.load()
        rows = reopened.query("id >= 0", output_fields=["id"], limit=1000)
        assert {row["id"] for row in rows} == {
            worker_id * 1000 + offset
            for worker_id in range(4)
            for offset in range(20)
        }
    finally:
        reopened.close()
```

- [ ] **Step 2: Add a cross-Collection independence test**

Append:

```python
def test_different_collections_do_not_share_write_lock(tmp_path, schema, monkeypatch):
    first = Collection("first", str(tmp_path / "first"), schema)
    second = Collection("second", str(tmp_path / "second"), schema)
    entered = threading.Barrier(3)
    release = threading.Event()
    errors = []

    def block_apply(collection, original_apply):
        def tracked(op):
            entered.wait(timeout=5)
            assert release.wait(timeout=5)
            return original_apply(op)
        monkeypatch.setattr(collection, "_apply", tracked)

    block_apply(first, first._apply)
    block_apply(second, second._apply)

    def writer(collection, pk):
        try:
            collection.insert([_record(pk)])
        except BaseException as exc:
            errors.append(exc)

    threads = [
        threading.Thread(target=writer, args=(first, 1)),
        threading.Thread(target=writer, args=(second, 2)),
    ]
    for thread in threads:
        thread.start()

    entered.wait(timeout=5)
    release.set()
    for thread in threads:
        thread.join(timeout=5)

    assert errors == []
    first.close()
    second.close()
```

Both writers must reach `_apply` simultaneously. If the implementation uses a
global lock, the barrier times out and the test fails.

- [ ] **Step 3: Run the complete concurrent-writer test module**

Run:

```bash
python3 -m pytest tests/engine/test_concurrent_writers.py -v
```

Expected: 3 passed.

- [ ] **Step 4: Run the gRPC concurrency compatibility test**

Run:

```bash
python3 -m pytest \
  tests/compatibility/test_milvus_advanced.py::TestConcurrency::test_concurrent_insert_and_search \
  -v
```

Expected: PASS.

- [ ] **Step 5: Commit concurrency coverage**

```bash
git add tests/engine/test_concurrent_writers.py
git commit -m "test: cover concurrent collection writers"
```

### Task 4: Full Verification and Documentation Alignment

**Files:**
- Modify: `docs/modules.md:217-222`
- Modify: `milvus_lite/adapter/grpc/server.py:11-16`

- [ ] **Step 1: Update the architecture concurrency contract**

Replace the single-writer invariant in `docs/modules.md` with:

```markdown
7. **Serialized writers per Collection**. Mutation and flush entry points acquire a per-Collection reentrant write lock. Concurrent writes to the same Collection are serialized as complete WAL + MemTable transactions; writes to different Collections remain independent.
```

Update the concurrency paragraph in `milvus_lite/adapter/grpc/server.py` to:

```text
Concurrency model: gRPC's threadpool dispatches requests across worker
threads. The engine serializes mutation and flush operations per Collection;
concurrent reads remain independent, and different Collections can be written
in parallel.
```

- [ ] **Step 2: Run formatting and static repository checks**

Run:

```bash
git diff --check
python3 -m compileall -q milvus_lite
```

Expected: both commands exit 0 with no output.

- [ ] **Step 3: Run focused WAL, recovery, crash, flush, and concurrency suites**

Run:

```bash
python3 -m pytest -q \
  tests/storage/test_wal.py \
  tests/engine/test_recovery.py \
  tests/engine/test_crash_recovery.py \
  tests/engine/test_flush.py \
  tests/engine/test_collection.py \
  tests/engine/test_concurrent_writers.py \
  tests/engine/test_concurrency_stability.py \
  tests/engine/test_background_flush.py
```

Expected: all selected tests pass.

- [ ] **Step 4: Run the full test suite**

Run:

```bash
python3 -m pytest -q
```

Expected: all tests pass. If an environment-specific optional dependency test
is skipped, record the skip count; do not treat expected skips as failures.

- [ ] **Step 5: Inspect the final diff for scope and lock ordering**

Run:

```bash
git diff --stat HEAD~3
git diff HEAD~3 -- milvus_lite/storage/wal.py milvus_lite/engine/collection.py
```

Confirm:

- `_write_lock` is per Collection.
- Every path acquiring both locks takes `_write_lock` before
  `_maintenance_lock`.
- No background path acquires `_write_lock` while already holding
  `_maintenance_lock`.
- WAL format and default `sync_mode` are unchanged.
- `WALCorruptedError` is not swallowed.

- [ ] **Step 6: Commit documentation and final verification metadata**

```bash
git add docs/modules.md milvus_lite/adapter/grpc/server.py
git commit -m "docs: describe serialized collection writers"
```
