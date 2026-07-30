import threading

import pytest

from milvus_lite.engine.collection import Collection
from milvus_lite.schema.types import CollectionSchema, DataType, FieldSchema


@pytest.fixture
def schema():
    return CollectionSchema(fields=[
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True),
        FieldSchema(name="vec", dtype=DataType.FLOAT_VECTOR, dim=2),
    ])


def _record(pk):
    return {"id": pk, "vec": [float(pk), float(pk)]}


def test_same_collection_mutations_are_serialized(tmp_path, schema, monkeypatch):
    collection = Collection("test", str(tmp_path / "data"), schema)
    original_apply = collection._apply
    state_lock = threading.Lock()
    start = threading.Barrier(3)
    apply_overlap = threading.Barrier(2)
    active = 0
    max_active = 0
    errors = []

    def tracked_apply(op):
        nonlocal active, max_active
        with state_lock:
            active += 1
            max_active = max(max_active, active)
        try:
            try:
                apply_overlap.wait(timeout=1)
            except threading.BrokenBarrierError:
                pass
            return original_apply(op)
        finally:
            with state_lock:
                active -= 1

    monkeypatch.setattr(collection, "_apply", tracked_apply)

    def insert(pk):
        try:
            start.wait(timeout=5)
            collection.insert([_record(pk)])
        except BaseException as exc:
            errors.append(exc)

    threads = [
        threading.Thread(target=insert, args=(pk,), daemon=True)
        for pk in (1, 2)
    ]
    try:
        for thread in threads:
            thread.start()
        start.wait(timeout=5)
        for thread in threads:
            thread.join(timeout=5)
    finally:
        for thread in threads:
            thread.join(timeout=5)

    assert all(not thread.is_alive() for thread in threads)
    try:
        assert errors == []
        assert max_active == 1
    finally:
        collection.close()


def test_concurrent_writers_flush_and_reopen(tmp_path, schema, monkeypatch):
    monkeypatch.setattr("milvus_lite.engine.collection.MEMTABLE_SIZE_LIMIT", 3)
    data_dir = str(tmp_path / "data")
    collection = Collection("test", data_dir, schema)
    start = threading.Barrier(5)
    errors = []

    def insert_many(writer_id):
        try:
            start.wait(timeout=5)
            for offset in range(20):
                collection.insert([_record(writer_id * 1000 + offset)])
                if (offset + 1) % 5 == 0:
                    collection.flush()
        except BaseException as exc:
            errors.append(exc)

    threads = [
        threading.Thread(target=insert_many, args=(writer_id,), daemon=True)
        for writer_id in range(4)
    ]
    try:
        for thread in threads:
            thread.start()
        start.wait(timeout=5)
        for thread in threads:
            thread.join(timeout=5)
    finally:
        for thread in threads:
            thread.join(timeout=5)

    assert all(not thread.is_alive() for thread in threads)
    try:
        assert errors == []
    finally:
        collection.close()

    reopened = Collection("test", data_dir, schema)
    try:
        reopened.load()
        records = reopened.query("id >= 0", output_fields=["id"])
        expected_ids = {
            writer_id * 1000 + offset
            for writer_id in range(4)
            for offset in range(20)
        }
        assert len(records) == 80
        assert {record["id"] for record in records} == expected_ids
    finally:
        reopened.close()


def test_different_collections_do_not_share_write_lock(
    tmp_path, schema, monkeypatch
):
    collections = [
        Collection("first", str(tmp_path / "first"), schema),
        Collection("second", str(tmp_path / "second"), schema),
    ]
    both_applying = threading.Barrier(3)
    release = threading.Event()
    errors = []

    for collection in collections:
        original_apply = collection._apply

        def coordinated_apply(op, original_apply=original_apply):
            both_applying.wait(timeout=5)
            if not release.wait(timeout=5):
                raise TimeoutError("writers were not released")
            return original_apply(op)

        monkeypatch.setattr(collection, "_apply", coordinated_apply)

    def insert(collection, pk):
        try:
            collection.insert([_record(pk)])
        except BaseException as exc:
            errors.append(exc)

    threads = [
        threading.Thread(target=insert, args=(collection, pk), daemon=True)
        for collection, pk in zip(collections, (1, 2))
    ]
    try:
        for thread in threads:
            thread.start()
        both_applying.wait(timeout=5)
        release.set()
        for thread in threads:
            thread.join(timeout=5)
    finally:
        release.set()
        for thread in threads:
            thread.join(timeout=5)

    assert all(not thread.is_alive() for thread in threads)
    try:
        assert errors == []
    finally:
        try:
            collections[0].close()
        finally:
            collections[1].close()
