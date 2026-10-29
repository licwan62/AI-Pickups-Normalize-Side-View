"""Local single-page workbench for reviewing dimensions and exporting SVGs.

Run with ``python -m pickup_measure.workbench``. No frontend build is needed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sqlite3
import threading
import uuid
import webbrowser
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlsplit

import yaml

from .src.identifier import QwenVehicleIdentifier
from .src.loader import SUPPORTED_EXTENSIONS, load_image
from .src.master_table import display_name, load_master_table, query_market
from .src.settings import load_settings
from .src.side_view import build_vehicle_record


ROOT = Path(__file__).resolve().parents[1]
BUSY = {"MATCH_QUEUED", "MATCHING", "EXPORT_QUEUED", "PROCESSING"}
MAX_UPLOAD = 40 * 1024 * 1024
LOGGER = logging.getLogger("pickup_measure.workbench")


class WorkbenchError(ValueError):
    def __init__(self, message: str, status: int = 400):
        super().__init__(message)
        self.status = status


def timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def file_hash(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def row_key(row: dict) -> str:
    values = [row[key] for key in (
        "dimension_id", "length_mm", "width_mm", "height_mm", "size"
    )]
    return hashlib.sha256(json.dumps(values, ensure_ascii=False).encode()).hexdigest()


def candidate_payload(candidate) -> dict:
    row = candidate.as_dict()
    row["row_key"] = row_key(row)
    return row


class Workbench:
    def __init__(self, config: Path, *, state_dir: Path | None = None,
                 table=None, identifier=None, processor=None):
        self.config = config.resolve()
        self.settings = load_settings(self.config)
        payload = yaml.safe_load(self.config.read_text(encoding="utf-8")) or {}
        self.images_dir = (self.config.parent / payload.get("images_dir", "img/input")).resolve()
        self.state_dir = state_dir or self.config.parent / "data/workbench"
        self.upload_dir = self.images_dir / "_workbench"
        self.output_dir = self.settings.output_dir / "workbench"
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.lock = threading.RLock()
        self.table_lock = threading.RLock()
        self.db = sqlite3.connect(self.state_dir / "tasks.sqlite3", check_same_thread=False)
        self.db.execute("CREATE TABLE IF NOT EXISTS tasks (id TEXT PRIMARY KEY, payload TEXT NOT NULL)")
        self.tasks = {key: json.loads(value) for key, value in self.db.execute("SELECT id, payload FROM tasks")}
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="workbench")
        self.table = table
        self.master = {"status": "READY" if table is not None else "LOADING", "error": ""}
        if table is not None:
            self.master.update(table.stats(), mode="provided")
        self.identifier = identifier
        self.processor = processor
        for task in self.tasks.values():
            if task["status"] in BUSY:
                self._update(task, status="FAILED", stage="INTERRUPTED",
                             error="上次运行中断，请重新匹配或重试生成。")

    def close(self):
        self.executor.shutdown(wait=True)
        self.db.close()

    def _update(self, task: dict, **values):
        with self.lock:
            task.update(values, updated_at=timestamp())
            self.db.execute("INSERT OR REPLACE INTO tasks VALUES (?, ?)",
                            (task["id"], json.dumps(task, ensure_ascii=False)))
            self.db.commit()

    def _task(self, task_id: str) -> dict:
        try:
            return self.tasks[task_id]
        except KeyError:
            raise WorkbenchError("任务不存在", 404) from None

    def _idle(self, task: dict):
        if task["status"] in BUSY:
            raise WorkbenchError("该任务正在排队或处理中", 409)

    def _get_table(self, refresh=False, *, check_source=False):
        with self.table_lock:
            if self.table is not None and not refresh and (
                not check_source or self.master.get("mode") == "provided"
            ):
                return self.table
            self.master = {"status": "LOADING", "error": ""}
            settings = self.settings.master_table
            mode = "configured"
            source = settings.source
            try:
                try:
                    table = load_master_table(settings, force_refresh=refresh)
                    if not source.exists():
                        mode = "cache"
                except (FileNotFoundError, OSError):
                    if settings.cache_path.is_file():
                        table = load_master_table(replace(settings, refresh="never"))
                        mode = "cache"
                    else:
                        local = self.config.parent / "data/dimensions/全量表_汇总.csv"
                        if not local.is_file():
                            raise
                        # Keep the local snapshot separate from the authoritative NAS cache.
                        settings = replace(settings, source=local, refresh="auto",
                                           cache_path=self.state_dir / "local_master.csv")
                        table = load_master_table(settings, force_refresh=refresh)
                        mode, source = "local", local
                self.table = table
                self.master = {"status": "READY", "error": "", "mode": mode,
                               "source": str(source), "loaded_at": timestamp(), **table.stats()}
                return table
            except Exception as exc:
                self.table = None
                self.master = {"status": "FAILED", "error": str(exc)}
                raise

    def refresh_master(self, *, force=True):
        def refresh():
            try:
                self._get_table(refresh=force)
            except Exception:
                LOGGER.exception("Master table refresh failed")
        self.executor.submit(refresh)

    def snapshot(self) -> dict:
        with self.lock:
            tasks = json.loads(json.dumps(list(self.tasks.values())))
        for task in tasks:
            task.pop("image_path", None)
            task.pop("svg_path", None)
            task.pop("crop_path", None)
        return {"tasks": sorted(tasks, key=lambda t: t["created_at"], reverse=True),
                "master": dict(self.master), "images_dir": str(self.images_dir),
                "output_dir": str(self.output_dir),
                "api_key_configured": bool(self.settings.detection.api_key)}

    def register(self, path: Path, *, auto_match: bool = False) -> dict:
        # Decoding validates the actual file; the extension alone is insufficient.
        image = load_image(path)
        digest = file_hash(path)
        with self.lock:
            for task in self.tasks.values():
                if task["sha256"] == digest:
                    if auto_match and task["status"] == "NEW":
                        self.queue_match(task["id"])
                    return dict(task)
            task_id = uuid.uuid4().hex
            task = {"id": task_id, "filename": path.name, "image_path": str(path.resolve()),
                    "sha256": digest, "created_at": timestamp(), "status": "NEW",
                    "stage": "IMPORTED", "error": "", "warning": "", "candidates": [],
                    "selected": None, "confirmed": False, "identity": None,
                    "query": path.stem, "market": query_market(path.stem),
                    "width_px": image.width, "height_px": image.height}
            thumbnail_dir = self.state_dir / "thumbnails"
            thumbnail_dir.mkdir(parents=True, exist_ok=True)
            image.thumbnail((1000, 600))
            image.save(thumbnail_dir / f"{task_id}.jpg", quality=88)
            self.tasks[task_id] = task
            self._update(task)
            if auto_match:
                self.queue_match(task_id)
            return dict(task)

    def upload(self, filename: str, raw: bytes) -> dict:
        filename = Path(filename.replace("\\", "/")).name
        if Path(filename).suffix.lower() not in SUPPORTED_EXTENSIONS:
            raise WorkbenchError("支持 JPG、PNG、WebP、AVIF 图片")
        if not raw or len(raw) > MAX_UPLOAD:
            raise WorkbenchError("图片不能为空且不得超过 40 MB", 413)
        folder = self.upload_dir / uuid.uuid4().hex
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / filename
        try:
            path.write_bytes(raw)
            task = self.register(path, auto_match=True)
            if Path(task["image_path"]) != path:
                path.unlink()
                folder.rmdir()
            return {"id": task["id"]}
        except Exception:
            path.unlink(missing_ok=True)
            folder.rmdir()
            raise

    def scan(self):
        with self.lock:
            known_paths = {t["image_path"] for t in self.tasks.values()}
        added, errors = 0, []
        for path in sorted(self.images_dir.rglob("*")):
            if (not path.is_file() or path.suffix.lower() not in SUPPORTED_EXTENSIONS
                    or self.upload_dir in path.parents or str(path.resolve()) in known_paths):
                continue
            try:
                before = len(self.tasks)
                self.register(path)
                added += int(len(self.tasks) > before)
            except Exception as exc:
                errors.append({"filename": path.name, "error": str(exc)})
        return {"added": added, "errors": errors}

    def queue_match(self, task_id: str, *, query: str = "", market: str = "", use_ai=False):
        with self.lock:
            task = self._task(task_id)
            self._idle(task)
            self._update(task, status="MATCH_QUEUED", stage="WAITING", error="",
                         confirmed=False, selected=None, svg_path="", crop_path="",
                         candidates=[], identity=None, warning="",
                         market=market.strip().upper() or query_market(task["filename"]))
            self.executor.submit(self._match, task_id, query.strip(), use_ai)

    def _match(self, task_id: str, query: str, use_ai: bool):
        task = self._task(task_id)
        try:
            self._update(task, status="MATCHING", stage="MASTER_TABLE")
            table = self._get_table()
            path = Path(task["image_path"])
            # External source images may have been replaced since import.
            digest = file_hash(path)
            if digest != task["sha256"]:
                image = load_image(path)
                self._update(task, sha256=digest, width_px=image.width, height_px=image.height)
                image.thumbnail((1000, 600))
                image.save(self.state_dir / "thumbnails" / f"{task_id}.jpg", quality=88)
            text = query or path.stem
            result = table.find(display_name(text) if task["market"] else text,
                                market=task["market"] or None,
                                min_score=self.settings.master_table.min_score)
            method = "manual" if query else "filename"
            # A filename exact match avoids a paid vision call. Fuzzy guesses
            # never suppress image identification on an automatic import.
            if use_ai or (not query and not (result.best and result.best.exact)):
                self._update(task, stage="IDENTIFY")
                identifier = self.identifier or QwenVehicleIdentifier.from_settings(self.settings)
                try:
                    identity = identifier.identify(load_image(path))
                    text = identity.query
                    market = task["market"] or identity.market
                    result = table.find(display_name(text) if market else text, market=market or None,
                                        min_score=self.settings.master_table.min_score)
                    self._update(task, identity=identity.as_dict(), market=market)
                    method = "qwen"
                except Exception as exc:
                    # Keep filename candidates available for manual review.
                    self._update(task, warning=f"图像识别未完成：{exc}。可用车型名称搜索全量表。")
            rows = [candidate_payload(c) for c in result.candidates]
            signatures = {(r["length_mm"], r["width_mm"], r["height_mm"]) for r in rows
                          if rows[0]["score"] - r["score"] <= 0.03 + 1e-9}
            self._update(task, status="NEEDS_REVIEW", stage="REVIEW", query=text,
                         candidates=rows, match_method=method, reason=result.reason,
                         ambiguous=len(signatures) > 1, master_snapshot=dict(self.master),
                         error="" if rows else "未找到候选，请输入品牌、车型、年份重新搜索。")
        except Exception as exc:
            LOGGER.exception("Matching failed for %s", task_id)
            self._update(task, status="FAILED", stage="MATCH_FAILED", error=str(exc))

    def _vehicle(self, selected: dict):
        result = self._get_table().find("", dimension_id=selected["dimension_id"])
        for candidate in result.candidates:
            if candidate.vehicle.dimension_id == selected["dimension_id"]:
                row = candidate_payload(candidate)
                if row["row_key"] == selected["row_key"]:
                    return candidate.vehicle
        raise WorkbenchError("全量表中的尺寸已变化或记录不存在，请重新匹配并核对。", 409)

    def confirm(self, task_id: str, key: str):
        self._get_table(check_source=True)
        with self.lock:
            task = self._task(task_id)
            self._idle(task)
            selected = next((r for r in task["candidates"] if r["row_key"] == key), None)
            if selected is None:
                raise WorkbenchError("请选择当前全量表候选记录", 409)
            vehicle = self._vehicle(selected)
            selected = {**selected, **vehicle.as_dict()}
            if file_hash(Path(task["image_path"])) != task["sha256"]:
                raise WorkbenchError("原图已变化，请重新匹配。", 409)
            self._update(task, selected=selected, confirmed=True, confirmed_at=timestamp(),
                         status="READY", stage="CONFIRMED", error="", svg_path="", crop_path="",
                         master_snapshot=dict(self.master))

    def queue_export(self, task_ids: list[str]):
        with self.lock:
            # Validate the entire request before enqueuing anything.
            tasks = [self._task(task_id) for task_id in dict.fromkeys(task_ids)]
            if not tasks:
                raise WorkbenchError("请先选择已核对尺寸的任务")
            for task in tasks:
                self._idle(task)
                if not task.get("confirmed") or not task.get("selected"):
                    raise WorkbenchError("生成前请先核对并确认长宽高", 409)
            for task in tasks:
                self._update(task, status="EXPORT_QUEUED", stage="WAITING", error="", svg_path="", crop_path="")
                self.executor.submit(self._export, task["id"])

    def _export(self, task_id: str):
        task = self._task(task_id)
        try:
            self._update(task, status="PROCESSING", stage="VALIDATE")
            if file_hash(Path(task["image_path"])) != task["sha256"]:
                raise WorkbenchError("原图已变化，请重新匹配并核对尺寸。", 409)
            self._get_table(check_source=True)
            vehicle = self._vehicle(task["selected"])
            record = build_vehicle_record(vehicle, Path(task["image_path"]))
            output = self.output_dir / task_id
            output.mkdir(parents=True, exist_ok=True)
            audit = {key: task.get(key) for key in (
                "id", "filename", "sha256", "identity", "selected", "confirmed_at", "master_snapshot"
            )}
            audit["master_at_export"] = dict(self.master)
            (output / "review.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
            if self.processor is None:
                from .main import process_vehicle
                processor = process_vehicle
            else:
                processor = self.processor
            owner = self

            class Progress(dict):
                def __setitem__(self, key, value):
                    super().__setitem__(key, value)
                    if key == "stage":
                        owner._update(task, stage=value)

            status = processor(record, output_root=output,
                               annotation_style=self.settings.annotation,
                               detection_settings=replace(self.settings.detection, manual_fallback=False),
                               progress_state=Progress())
            svg = output / record.size / f"{record.id}.svg"
            if status != "EXPORTED" or not svg.is_file():
                raise RuntimeError(f"SVG 未生成：{status}")
            self._update(task, status="EXPORTED", stage="DONE", error="", svg_path=str(svg),
                         crop_path=str(output / record.size / record.id / "crop_source.png"),
                         exported_at=timestamp())
        except Exception as exc:
            LOGGER.exception("Export failed for %s", task_id)
            self._update(task, status="FAILED", stage="EXPORT_FAILED", error=str(exc),
                         confirmed=False if isinstance(exc, WorkbenchError) else task["confirmed"],
                         svg_path="", crop_path="")

    def asset(self, task_id: str, kind: str) -> tuple[Path, str]:
        with self.lock:
            task = self._task(task_id)
            if kind == "thumbnail":
                return self.state_dir / "thumbnails" / f"{task_id}.jpg", "image/jpeg"
            if kind == "source":
                return Path(task["image_path"]), "application/octet-stream"
            if kind == "review" and task["confirmed"]:
                return self.output_dir / task_id / "review.json", "application/json"
            if task["status"] != "EXPORTED":
                raise WorkbenchError("当前任务尚未成功生成", 404)
            if kind in {"svg", "crop"}:
                return Path(task["svg_path" if kind == "svg" else "crop_path"]), (
                    "image/svg+xml" if kind == "svg" else "image/png")
            raise WorkbenchError("文件不存在", 404)


def make_handler(workbench: Workbench):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format, *args):
            LOGGER.debug(format, *args)

        def _check_local(self):
            # Prevent foreign websites from issuing actions against loopback.
            host = self.headers.get("Host", "")
            if host not in {f"127.0.0.1:{self.server.server_port}", f"localhost:{self.server.server_port}"}:
                raise WorkbenchError("无效的本地访问地址", 403)
            origin = self.headers.get("Origin")
            if origin and origin not in {f"http://127.0.0.1:{self.server.server_port}", f"http://localhost:{self.server.server_port}"}:
                raise WorkbenchError("仅允许本地工作台发起请求", 403)

        def _send(self, status, data, content_type="application/json; charset=utf-8", *, attachment=False):
            raw = json.dumps(data, ensure_ascii=False).encode("utf-8") if not isinstance(data, bytes) else data
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(raw)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            if attachment:
                self.send_header("Content-Disposition", "attachment")
            self.end_headers()
            self.wfile.write(raw)

        def do_GET(self):
            try:
                self._check_local()
                parts = urlsplit(self.path)
                if parts.path == "/":
                    self._send(200, (Path(__file__).parent / "web/index.html").read_bytes(), "text/html; charset=utf-8")
                elif parts.path == "/api/state":
                    self._send(200, workbench.snapshot())
                elif parts.path.startswith("/api/assets/"):
                    _, _, _, task_id, kind = parts.path.split("/")
                    path, mime = workbench.asset(task_id, kind)
                    if not path.is_file():
                        raise WorkbenchError("文件不存在", 404)
                    self._send(200, path.read_bytes(), mime, attachment="download" in parse_qs(parts.query))
                else:
                    raise WorkbenchError("页面不存在", 404)
            except (WorkbenchError, FileNotFoundError) as exc:
                self._send(getattr(exc, "status", 404), {"error": str(exc)})
            except Exception:
                LOGGER.exception("GET failed")
                self._send(500, {"error": "读取失败，请查看本地日志"})

        def do_POST(self):
            try:
                self._check_local()
                length = int(self.headers.get("Content-Length", "0"))
                limit = MAX_UPLOAD if self.path == "/api/upload" else 64 * 1024
                if not 0 < length <= limit:
                    raise WorkbenchError("请求为空或文件过大", 413)
                raw = self.rfile.read(length)
                if self.path == "/api/upload":
                    result = workbench.upload(unquote(self.headers.get("X-Filename", "image.png")), raw)
                else:
                    body = json.loads(raw)
                    if not isinstance(body, dict):
                        raise WorkbenchError("请求必须为 JSON 对象")
                    if self.path == "/api/scan":
                        result = workbench.scan()
                    elif self.path == "/api/match":
                        workbench.queue_match(body["id"], query=body.get("query", ""),
                                              market=body.get("market", ""), use_ai=bool(body.get("use_ai")))
                        result = {"ok": True}
                    elif self.path == "/api/confirm":
                        workbench.confirm(body["id"], body["row_key"])
                        result = {"ok": True}
                    elif self.path == "/api/export":
                        if not isinstance(body.get("ids"), list) or not all(isinstance(i, str) for i in body["ids"]):
                            raise WorkbenchError("ids 必须为任务 ID 数组")
                        workbench.queue_export(body["ids"])
                        result = {"ok": True}
                    elif self.path == "/api/master/refresh":
                        workbench.refresh_master()
                        result = {"ok": True}
                    else:
                        raise WorkbenchError("接口不存在", 404)
                self._send(200, result)
            except (WorkbenchError, ValueError, KeyError, TypeError, OSError) as exc:
                self._send(getattr(exc, "status", 400), {"error": str(exc)})
            except Exception:
                LOGGER.exception("POST failed")
                self._send(500, {"error": "处理失败，请查看本地日志"})

    return Handler


def main():
    parser = argparse.ArgumentParser(description="车辆侧视图本地工作台")
    parser.add_argument("--config", type=Path, default=ROOT / "config.yaml")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args()
    workbench = Workbench(args.config)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        handlers=[logging.FileHandler(workbench.state_dir / "workbench.log", encoding="utf-8")])
    server = ThreadingHTTPServer(("127.0.0.1", args.port), make_handler(workbench))
    workbench.refresh_master(force=False)
    url = f"http://127.0.0.1:{server.server_port}"
    print(f"车辆侧视图工作台: {url}", flush=True)
    if not args.no_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        workbench.close()


if __name__ == "__main__":
    main()
