"""TIMESTAMPTZ collection-level behavior."""

from datetime import datetime, timezone

from milvus_lite.db import MilvusLite
from milvus_lite.engine.collection import Collection
from milvus_lite.schema.timestamptz import micros_to_utc_datetime
from milvus_lite.schema.types import CollectionSchema, DataType, FieldSchema


def _schema() -> CollectionSchema:
    return CollectionSchema(fields=[
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True),
        FieldSchema(name="vec", dtype=DataType.FLOAT_VECTOR, dim=2),
        FieldSchema(name="tsz", dtype=DataType.TIMESTAMPTZ, nullable=True),
    ])


def test_timestamptz_query_and_restart(tmp_path):
    data_dir = str(tmp_path / "data")
    col = Collection("events", data_dir, _schema())
    col.insert([
        {"id": 1, "vec": [0.1, 0.2], "tsz": "2025-01-01T00:00:00+08:00"},
        {"id": 2, "vec": [0.2, 0.3], "tsz": "2025-01-03T00:00:00Z"},
        {"id": 3, "vec": [0.3, 0.4], "tsz": None},
    ])

    rows = col.query(
        "tsz > ISO '2025-01-02T00:00:00Z'",
        output_fields=["id", "tsz"],
    )
    assert [r["id"] for r in rows] == [2]
    assert rows[0]["tsz"] == datetime(2025, 1, 3, tzinfo=timezone.utc)

    rows = col.query(
        "tsz + INTERVAL 'P1D' <= ISO '2025-01-02T16:00:00Z'",
        output_fields=["id"],
    )
    assert [r["id"] for r in rows] == [1]
    col.close()

    reopened = Collection("events", data_dir, _schema())
    got = reopened.get([1], output_fields=["id", "tsz"])
    assert got[0]["tsz"] == micros_to_utc_datetime("2025-01-01T00:00:00+08:00")
    reopened.close()


def test_timestamptz_collection_timezone_persists_and_parses_naive_values(tmp_path):
    db = MilvusLite(str(tmp_path / "db"))
    col = db.create_collection(
        "events",
        _schema(),
        properties={"timezone": "Asia/Shanghai"},
    )
    col.insert([
        {"id": 1, "vec": [0.1, 0.2], "tsz": "2025-01-01T00:00:00"},
        {"id": 2, "vec": [0.2, 0.3], "tsz": "2025-01-02T00:00:00"},
    ])
    db.close()

    reopened_db = MilvusLite(str(tmp_path / "db"))
    reopened = reopened_db.get_collection("events")
    assert reopened.schema.properties["timezone"] == "Asia/Shanghai"

    rows = reopened.query(
        "tsz == ISO '2025-01-01T00:00:00'",
        output_fields=["id", "tsz"],
    )

    assert [r["id"] for r in rows] == [1]
    assert rows[0]["tsz"] == datetime(2024, 12, 31, 16, tzinfo=timezone.utc)
    reopened_db.close()


def test_timestamptz_request_timezone_overrides_collection_timezone(tmp_path):
    col = Collection(
        "events",
        str(tmp_path / "data"),
        CollectionSchema(
            fields=_schema().fields,
            properties={"timezone": "UTC"},
        ),
    )
    col.insert([
        {"id": 1, "vec": [1.0, 0.0], "tsz": "2025-01-01T00:00:00Z"},
        {"id": 2, "vec": [0.0, 1.0], "tsz": "2024-12-31T16:00:00Z"},
    ])

    rows = col.query(
        "tsz == ISO '2025-01-01T00:00:00'",
        output_fields=["id"],
    )
    assert [r["id"] for r in rows] == [1]

    rows = col.query(
        "tsz == ISO '2025-01-01T00:00:00'",
        output_fields=["id"],
        timezone="Asia/Shanghai",
    )
    assert [r["id"] for r in rows] == [2]
    col.close()


def test_timestamptz_search_uses_request_timezone(tmp_path):
    col = Collection(
        "events",
        str(tmp_path / "search"),
        CollectionSchema(fields=_schema().fields),
    )
    col.insert([
        {"id": 1, "vec": [1.0, 0.0], "tsz": "2025-01-01T00:00:00Z"},
        {"id": 2, "vec": [0.0, 1.0], "tsz": "2024-12-31T16:00:00Z"},
    ])

    results = col.search(
        [[1.0, 0.0]],
        top_k=10,
        expr="tsz == ISO '2025-01-01T00:00:00'",
        output_fields=["id"],
        timezone="Asia/Shanghai",
    )

    assert [hit["id"] for hit in results[0]] == [2]
    col.close()


def test_timestamptz_alter_collection_timezone_persists(tmp_path):
    data_dir = str(tmp_path / "db_alter")
    db = MilvusLite(data_dir)
    col = db.create_collection(
        "events",
        _schema(),
        properties={"timezone": "UTC"},
    )
    col.insert([
        {"id": 1, "vec": [1.0, 0.0], "tsz": "2025-01-01T00:00:00Z"},
        {"id": 2, "vec": [0.0, 1.0], "tsz": "2024-12-31T16:00:00Z"},
    ])

    rows = col.query(
        "tsz == ISO '2025-01-01T00:00:00'",
        output_fields=["id"],
    )
    assert [r["id"] for r in rows] == [1]

    db.alter_collection_properties(
        "events",
        properties={"timezone": "Asia/Shanghai"},
    )
    rows = col.query(
        "tsz == ISO '2025-01-01T00:00:00'",
        output_fields=["id"],
    )
    assert [r["id"] for r in rows] == [2]
    db.close()

    reopened_db = MilvusLite(data_dir)
    reopened = reopened_db.get_collection("events")
    assert reopened.schema.properties["timezone"] == "Asia/Shanghai"
    rows = reopened.query(
        "tsz == ISO '2025-01-01T00:00:00'",
        output_fields=["id"],
    )
    assert [r["id"] for r in rows] == [2]
    reopened_db.close()
