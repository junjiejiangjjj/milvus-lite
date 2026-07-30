# WAL Concurrent Write Hardening Design

## Summary

Milvus Lite currently relies on a single-writer-per-Collection contract, but
the gRPC server dispatches requests through a multi-worker thread pool. Two
concurrent mutation requests can therefore allocate sequence numbers, append
to the same Arrow IPC stream, or freeze and replace the active WAL and
MemTable at the same time. These operations are not protected by a Collection
write lock.

Recovery has a separate weakness: Arrow IPC decoding can return a
`RecordBatch` whose internal arrays are invalid. `_read_wal_file` currently
accepts such a batch without calling `RecordBatch.validate(full=True)`, so the
error can surface later during recovery replay, filtering, querying, or
search.

This change serializes mutations within each Collection and validates every
recovered WAL batch before it can enter the MemTable. It deliberately does not
change the WAL fsync policy.

## Goals

- Make concurrent insert, upsert, delete, explicit flush, and close operations
  safe for the same Collection.
- Prevent concurrent writers from sharing or replacing an Arrow IPC writer.
- Keep writes to different Collections independent.
- Reject structurally invalid recovered batches before they enter engine
  state.
- Preserve partial recovery of complete batches before a truncated WAL tail.
- Report validated WAL corruption with enough context to diagnose the file
  and batch involved.

## Non-goals

- Providing host-crash durability through per-batch `fsync`.
- Adding WAL checksums or changing the Arrow IPC file format.
- Recovering data from a structurally invalid batch.
- Making all Collection reads mutually exclusive with writes.
- Redesigning background compaction or the maintenance locking model.

## Collection Write Serialization

Each `Collection` owns a `threading.RLock` named `_write_lock`. The lock is
per-Collection rather than global, so separate Collections remain writable in
parallel.

The complete mutation transaction must execute while holding `_write_lock`:

1. Validate mutation-specific state that may change concurrently.
2. Allocate `_seq` values.
3. Build the WAL `RecordBatch`.
4. Append the batch to WAL.
5. Apply the same batch to MemTable.
6. If necessary, freeze the active WAL and MemTable and complete synchronous
   flush persistence.

The following public lifecycle and mutation entry points acquire the lock:

- `insert`
- `upsert`
- `delete`
- explicit `flush`
- `close`

Internal helpers that can also be called from already locked paths, including
`_trigger_flush`, use the same `RLock`. Reentrancy prevents deadlock when a
public mutation method triggers a flush.

The existing `_maintenance_lock` retains its current responsibility for
Manifest, segment cache, DeltaIndex, and background maintenance coordination.
The write lock does not replace it. When both locks are required, the order is
always:

1. `_write_lock`
2. `_maintenance_lock`

Background maintenance continues to acquire only `_maintenance_lock`, so it
cannot introduce the reverse order.

## WAL Recovery Validation

`_read_wal_file` validates each decoded batch before appending it to the
returned list:

```python
for batch_index, batch in enumerate(reader):
    batch.validate(full=True)
    batches.append(batch)
```

Fast validation is insufficient because it does not validate all offset and
value invariants. Full validation is required on recovery. Recovery is much
less frequent than normal writes, so this cost does not affect steady-state
insert throughput.

Validation is not added to the normal write path. WAL batches are constructed
internally against a concrete Arrow schema, and write-side validation cannot
detect corruption introduced after serialization or persistence.

## Error Classification

Recovery needs to distinguish an incomplete tail from a batch that was
successfully decoded but failed validation.

### Truncated read

If the Arrow stream reader raises `pa.ArrowInvalid` while requesting the next
batch, `_read_wal_file` preserves all batches that were already decoded and
validated. It logs a warning containing:

- WAL path
- number of batches retained
- the Arrow error

This preserves the existing crash-recovery behavior for a process terminated
during the last WAL append.

### Decoded but invalid batch

If `batch.validate(full=True)` raises `pa.ArrowInvalid`, `_read_wal_file`
raises `WALCorruptedError` instead of treating the condition as an ordinary
truncated tail. The error includes:

- WAL path
- zero-based batch index
- validation error text

The invalid batch and all later batches are not replayed. Collection opening
fails explicitly, preventing corrupted data from entering MemTable or being
silently discarded.

To keep classification unambiguous, the reader iteration and batch validation
must use separate exception boundaries. A single broad `except
pa.ArrowInvalid` around both operations would incorrectly classify validation
failure as truncation.

File-level `OSError` and `IOError` retain existing partial-read behavior but
must also emit a warning. Unexpected programming exceptions continue to
propagate.

## Shutdown Behavior

`Collection.close` acquires `_write_lock`, preventing shutdown from racing an
in-flight mutation or flush. Once close begins under the lock, no later writer
can append to the WAL being flushed or deleted.

Existing idempotence expectations remain unchanged. This design does not
change `MilvusLite.close` exception handling; any broader shutdown error
reporting change should be handled separately.

## Compatibility

- WAL file format remains Arrow IPC Streaming and is unchanged.
- Existing valid WAL files recover normally.
- Existing WALs with a truncated final message still recover their complete
  prefix.
- WALs containing a decoded but structurally invalid batch now fail opening
  explicitly instead of failing later or being silently accepted.
- Mutation throughput for one Collection becomes serialized, matching the
  engine's existing single-writer architecture. Different Collections retain
  concurrency.
- Default `sync_mode="close"` remains unchanged.

## Testing

### Recovery tests

- Construct a valid IPC stream, corrupt a variable-length array offset so the
  stream reader returns the batch, and assert that recovery raises
  `WALCorruptedError` with path and batch index.
- Verify `validate(full=False)` would not catch the fixture, ensuring the test
  exercises the required full-validation behavior.
- Truncate the final WAL message and verify all earlier complete batches are
  recovered.
- Verify a completely unreadable schema/file retains the defined empty-prefix
  recovery behavior and logs a warning.
- Verify unexpected exceptions are not swallowed.

### Concurrency tests

- Run multiple writer threads performing inserts and deletes on one
  Collection, then close and reopen it and verify data and sequence ordering.
- Force a very small MemTable limit so multiple writers repeatedly cross the
  flush threshold.
- Instrument WAL creation and `close_and_delete` to assert that no writer
  appends to a frozen or closed WAL.
- Run writes against two Collections and use a synchronization barrier to
  confirm their per-Collection locks do not serialize each other.
- Exercise concurrent gRPC mutation requests because the gRPC worker pool is
  the production path that exposes this race.

### Regression tests

- Run storage WAL tests, engine recovery tests, crash recovery tests, flush
  tests, Collection tests, and gRPC compatibility mutation tests.

## Acceptance Criteria

- Concurrent mutation requests against one Collection cannot call the same
  `RecordBatchStreamWriter` concurrently.
- WAL/MemTable freeze and replacement cannot overlap another mutation for the
  same Collection.
- Recovered batches pass `validate(full=True)` before replay.
- A decoded invalid batch produces `WALCorruptedError` during Collection open.
- A truncated final IPC message still preserves the validated prefix.
- Writes to different Collections can proceed concurrently.
- Existing test suites pass without changing the default fsync policy or WAL
  format.
