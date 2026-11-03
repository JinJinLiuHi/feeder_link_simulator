"""Versioned SQLite device library; model metadata and exact NPY array blobs."""

from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
import hashlib
import io
import os
from pathlib import Path
import sqlite3
import numpy as np
from .device_models import DeviceModel, KINDS, MAX_BYTES, json_text


BEIJING_TIMEZONE = timezone(timedelta(hours=8))


def format_beijing_time(value):
    """Render both historical UTC timestamps and new +08:00 timestamps."""
    timestamp = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if timestamp.tzinfo is None:
        timestamp = timestamp.replace(tzinfo=timezone.utc)
    return timestamp.astimezone(BEIJING_TIMEZONE).strftime("%Y-%m-%d %H:%M:%S")


def default_database_path():
    return Path(os.environ.get("FEEDER_MODEL_DB", Path(__file__).resolve().parent/"data"/"device_models.sqlite3")).resolve()


class ModelStore:
    def __init__(self, path=None):
        self.path = Path(path) if path is not None else default_database_path()
        self.path = self.path.resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.connection() as db:
            db.execute("BEGIN IMMEDIATE")
            version = db.execute("PRAGMA user_version").fetchone()[0]
            if version not in (0, 1):
                raise ValueError("模型数据库版本较新，当前程序无法读取")
            if version == 0:
                existing = db.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
                if existing:
                    raise ValueError("所选 SQLite 文件不是器件模型库，请选择其他文件")
                schema = """
                    CREATE TABLE models (
                        id INTEGER PRIMARY KEY, kind TEXT NOT NULL, name TEXT NOT NULL,
                        revision INTEGER NOT NULL, description TEXT NOT NULL,
                        parameters_json TEXT NOT NULL, created_at TEXT NOT NULL,
                        checksum TEXT NOT NULL, UNIQUE(kind, name, revision));
                    CREATE TABLE model_arrays (
                        model_id INTEGER NOT NULL REFERENCES models(id) ON DELETE CASCADE,
                        name TEXT NOT NULL, npy_data BLOB NOT NULL, PRIMARY KEY(model_id, name));
                    CREATE INDEX models_kind_name ON models(kind, name);
                    PRAGMA application_id=1179403341;
                    PRAGMA user_version=1;
                """
                for statement in schema.split(";"):
                    if statement.strip():
                        db.execute(statement)
            else:
                if db.execute("PRAGMA application_id").fetchone()[0] != 1179403341:
                    raise ValueError("该文件不是本程序的器件模型数据库")

    @contextmanager
    def connection(self):
        db = sqlite3.connect(self.path, timeout=10)
        db.row_factory = sqlite3.Row
        db.execute("PRAGMA foreign_keys=ON")
        try:
            with db:
                yield db
        finally:
            db.close()

    @staticmethod
    def checksum(kind, parameters, blobs):
        digest = hashlib.sha256(kind.encode()+parameters.encode())
        for name, blob in sorted(blobs.items()):
            digest.update(name.encode()+b"\0"+blob)
        return digest.hexdigest()

    def save(self, model):
        model = model.validated()
        if not model.name:
            raise ValueError("请填写模型名称")
        parameters = json_text(model.parameters)
        blobs = {}
        for name, array in model.arrays.items():
            stream = io.BytesIO()
            np.save(stream, array, allow_pickle=False)
            blobs[name] = stream.getvalue()
        checksum = self.checksum(model.kind, parameters, blobs)
        with self.connection() as db:
            db.execute("BEGIN IMMEDIATE")
            revision = db.execute("SELECT COALESCE(MAX(revision),0)+1 FROM models WHERE kind=? AND name=?",
                                  (model.kind, model.name)).fetchone()[0]
            cursor = db.execute("""INSERT INTO models(kind,name,revision,description,parameters_json,created_at,checksum)
                    VALUES(?,?,?,?,?,?,?)""", (model.kind, model.name, revision, model.description, parameters,
                      datetime.now(BEIJING_TIMEZONE).isoformat(timespec="microseconds"), checksum))
            model_id = cursor.lastrowid
            db.executemany("INSERT INTO model_arrays(model_id,name,npy_data) VALUES(?,?,?)",
                           [(model_id, name, blob) for name, blob in blobs.items()])
        return model_id

    def list(self, kind=None, search=""):
        clauses, args = [], []
        if kind is not None:
            if kind not in KINDS:
                raise ValueError("未知器件类型")
            clauses.append("kind=?")
            args.append(kind)
        if search:
            literal = search.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
            clauses.append("name LIKE ? ESCAPE '\\'")
            args.append("%"+literal+"%")
        query = "SELECT id,kind,name,revision,description,created_at FROM models"
        if clauses:
            query += " WHERE "+" AND ".join(clauses)
        with self.connection() as db:
            return [dict(row) for row in db.execute(query+" ORDER BY id DESC", args)]

    def get(self, model_id):
        import json
        with self.connection() as db:
            row = db.execute("SELECT * FROM models WHERE id=?", (model_id,)).fetchone()
            if row is None:
                raise ValueError("模型记录不存在，请刷新模型库")
            blobs = {r["name"]: bytes(r["npy_data"]) for r in db.execute(
                "SELECT name,npy_data FROM model_arrays WHERE model_id=?", (model_id,))}
        if sum(map(len, blobs.values())) > MAX_BYTES*2:
            raise ValueError("模型系数过大")
        if self.checksum(row["kind"], row["parameters_json"], blobs) != row["checksum"]:
            raise ValueError("模型记录校验失败，系数或参数可能已损坏")
        arrays = {name: np.load(io.BytesIO(blob), allow_pickle=False) for name, blob in blobs.items()}
        return DeviceModel(row["kind"], json.loads(row["parameters_json"]), arrays,
                           row["name"], row["description"]).validated()
