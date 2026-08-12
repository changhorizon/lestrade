import asyncio
import json
import logging
import os

from .. import config
from ..rag.engine import engine

logger = logging.getLogger(__name__)

_STATE_FILE = config.FAISS_INDEX_PATH + ".watcher.json"


def _load_state() -> dict:
    if os.path.exists(_STATE_FILE):
        try:
            with open(_STATE_FILE, "r") as f:
                return json.load(f)
        except Exception:
            pass
    return {}


def _save_state(state: dict):
    os.makedirs(os.path.dirname(_STATE_FILE), exist_ok=True)
    with open(_STATE_FILE, "w") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)


def _list_dirs() -> list[str]:
    raw = config.KB_DIRS
    if not raw:
        return []
    return [d.strip() for d in raw.split(",") if d.strip()]


def _walk_files(dirs: list[str]) -> list[tuple[str, str]]:
    exts = config.WATCH_EXT
    files = []
    for d in dirs:
        if not os.path.isdir(d):
            continue
        for root, _, names in os.walk(d):
            for name in names:
                if any(name.endswith(e) for e in exts):
                    rel = os.path.relpath(os.path.join(root, name), os.path.dirname(d))
                    files.append((rel, os.path.join(root, name)))
    return files


def _read_file(path: str) -> str:
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        return f.read()


async def watch_loop():
    dirs = _list_dirs()
    if not dirs:
        logger.info("Watcher: no KB_DIRS configured, skipping")
        return

    state = _load_state()

    if not state and engine.entries:
        for _name, path in _walk_files(dirs):
            state[path] = os.path.getmtime(path)
        _save_state(state)
        logger.info("Watcher: seeded %d existing files from meta", len(state))

    logger.info("Watcher: monitoring %s every %ds", dirs, config.WATCH_INTERVAL)

    while True:
        try:
            current_paths = set()
            for name, path in _walk_files(dirs):
                current_paths.add(path)
                new_mtime = os.path.getmtime(path)
                old_mtime = state.get(path)

                try:
                    if old_mtime is None:
                        text = _read_file(path)
                        if text.strip():
                            count = await engine.add_text(text, source=name)
                            logger.info("Added %s: %d chunks", name, count)
                    elif old_mtime != new_mtime:
                        engine.remove_by_source(name)
                        text = _read_file(path)
                        if text.strip():
                            count = await engine.add_text(text, source=name)
                            logger.info("Updated %s: %d chunks", name, count)
                        else:
                            logger.info("Removed %s: content deleted", name)
                except Exception as e:
                    logger.warning("Skip %s: %s", name, e)

                state[path] = new_mtime

            stale = [p for p in state if p not in current_paths]
            for path in stale:
                name = os.path.basename(path)
                for d in dirs:
                    if path.startswith(d):
                        rel = os.path.relpath(path, os.path.dirname(d))
                        name = rel
                        break
                engine.remove_by_source(name)
                logger.info("Removed stale %s (file deleted/renamed)", name)
                del state[path]

            _save_state(state)
        except Exception as e:
            logger.warning("Watcher scan error: %s", e)

        await asyncio.sleep(config.WATCH_INTERVAL)
