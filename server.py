#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Whisper — офлайн-транскрибация аудио и видео.

Локальное веб-приложение: запускается двойным кликом по файлу
«Запустить Whisper.command», открывает окно в браузере и работает
полностью офлайн (модель распознавания уже скачана в кеш).

Рассчитано на слабые машины (8 ГБ RAM): аудио обрабатывается кусками,
куски удаляются сразу, память не растёт, зависшая задача не блокирует очередь.

Распознавание работает на одном из двух бэкендов: mlx-whisper (Apple Silicon)
или faster-whisper (процессор, любая система). Нужен ещё ffmpeg.
Используется только стандартная библиотека Python.
"""
from __future__ import annotations

import contextlib
import io
import json
import math
import os
import re
import shutil
import socket
import subprocess
import sys
import tempfile
import threading
import time
import uuid
import wave
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from queue import Queue
from urllib.parse import parse_qs, unquote, urlparse

# Служебные полосы прогресса tqdm глушим через подмену stderr
# (переменная TQDM_DISABLE на практике не помогает).
os.environ.setdefault("TQDM_DISABLE", "1")

# Жёсткий офлайн: не обращаемся к интернету вообще.
# Чтобы разрешить докачку модели, запустите с WHISPER_ALLOW_NET=1.
if os.environ.get("WHISPER_ALLOW_NET") != "1":
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

# --------------------------------------------------------------------------
# Бэкенд распознавания
#   mlx     — MLX, только Apple Silicon (быстро, видеокарта)
#   faster  — faster-whisper (CTranslate2), работает на CPU везде:
#             Windows, Linux, Intel-маки
# Выбор: WHISPER_BACKEND=mlx|faster, иначе автоматически.
# --------------------------------------------------------------------------
try:
    import mlx_whisper as _mlx_module
except Exception:  # pragma: no cover — на не-Apple платформах
    _mlx_module = None

try:
    import faster_whisper as _faster_module
except Exception:  # pragma: no cover — если не установлен
    _faster_module = None

mlx_whisper = _mlx_module
faster_whisper = _faster_module


def _pick_backend() -> str:
    want = (os.environ.get("WHISPER_BACKEND") or "").strip().lower()
    if want == "mlx":
        return "mlx" if mlx_whisper else ""
    if want == "faster":
        return "faster" if faster_whisper else ""
    if mlx_whisper is not None:
        return "mlx"          # на Apple Silicon он быстрее
    if faster_whisper is not None:
        return "faster"
    return ""


BACKEND = _pick_backend()
BACKEND_LABEL = {
    "mlx": "MLX (Apple Silicon)",
    "faster": "faster-whisper (CPU)",
}.get(BACKEND, "не найден")
IMPORT_ERROR = None if BACKEND else (
    "не установлен ни mlx-whisper, ни faster-whisper — "
    "запустите setup.sh (macOS) или setup.ps1 (Windows)")

# --------------------------------------------------------------------------
# Настройки и пути
# --------------------------------------------------------------------------
APP_NAME = "Whisper — офлайн-транскрибация"
VERSION = "2.3"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
UPLOAD_DIR = os.path.join(DATA_DIR, "uploads")
WORK_DIR = os.path.join(DATA_DIR, "work")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
JOBS_FILE = os.path.join(DATA_DIR, "jobs.json")
PORT_FILE = os.path.join(DATA_DIR, "port.txt")

SETTINGS_FILE = os.path.join(DATA_DIR, "settings.json")
RENAMES_FILE = os.path.join(DATA_DIR, "renames.json")
GLOSSARY_FILE = os.path.join(DATA_DIR, "glossary.json")
BACKUP_DIR = os.path.join(DATA_DIR, "backup")


def load_settings() -> dict:
    try:
        with open(SETTINGS_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def save_settings(data: dict) -> None:
    tmp = SETTINGS_FILE + ".tmp"
    try:
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=1)
        os.replace(tmp, SETTINGS_FILE)
    except OSError:
        pass


# Каталоги моделей для двух бэкендов.
# MLX читает только weights.npz / weights.safetensors, поэтому репозитории
# с model.safetensors ему не подходят. faster-whisper работает с CTranslate2.
MODEL_CATALOG_MLX = [
    {
        "id": "mlx-community/whisper-large-v3-turbo-q4",
        "label": "Large v3 turbo, сжатая (4 бита)",
        "size_mb": 442,
        "note": "Рекомендуется для 8 ГБ: почти как полная, но в 3.6 раза меньше памяти",
    },
    {
        "id": "mlx-community/whisper-medium-mlx-4bit",
        "label": "Medium, сжатая (4 бита)",
        "size_mb": 489,
        "note": "Слабее и медленнее turbo, выигрыш по памяти небольшой",
    },
    {
        "id": "mlx-community/whisper-small-mlx-q4",
        "label": "Small, сжатая (4 бита)",
        "size_mb": 187,
        "note": "Самая лёгкая. Заметно хуже распознаёт имена и термины",
    },
]

MODEL_CATALOG_FASTER = [
    {
        "id": "deepdml/faster-whisper-large-v3-turbo-ct2",
        "label": "Large v3 turbo (процессор)",
        "size_mb": 1543,
        "note": "Лучшее качество на процессоре. Скорость зависит от машины",
    },
    {
        "id": "Systran/faster-whisper-small",
        "label": "Small (процессор)",
        "size_mb": 461,
        "note": "Быстрее, но хуже распознаёт имена и термины",
    },
    {
        "id": "Systran/faster-whisper-base",
        "label": "Base (процессор)",
        "size_mb": 145,
        "note": "Самая лёгкая, качество на русском невысокое",
    },
    {
        "id": "Systran/faster-whisper-tiny",
        "label": "Tiny — «нано» (процессор)",
        "size_mb": 39,
        "note": "Самая быстрая на слабых машинах, но качество на русском низкое — для черновых расшифровок",
    },
]

MODEL_CATALOG = MODEL_CATALOG_MLX if BACKEND == "mlx" else MODEL_CATALOG_FASTER
DEFAULT_MODEL = MODEL_CATALOG[0]["id"] if MODEL_CATALOG else ""

_settings = load_settings()
MODEL = (os.environ.get("WHISPER_MODEL")
         or _settings.get("model")
         or DEFAULT_MODEL)
# если сохранена модель от другого бэкенда (например, MLX при faster-whisper),
# берём модель по умолчанию для текущего бэкенда
if MODEL_CATALOG and not any(m["id"] == MODEL for m in MODEL_CATALOG):
    MODEL = DEFAULT_MODEL
LANGUAGE = os.environ.get("WHISPER_LANGUAGE", "ru")


def _int_env(name: str, default: int, minimum: int = 0) -> int:
    try:
        return max(minimum, int(os.environ.get(name, str(default))))
    except (TypeError, ValueError):
        return max(minimum, default)


def _float_env(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, str(default)))
    except (TypeError, ValueError):
        return default


CHUNK_SECONDS = _int_env("WHISPER_CHUNK", 600, 60)
CHUNK_OVERLAP = min(_int_env("WHISPER_OVERLAP", 15), 120)
MIN_CHUNK_SECONDS = _int_env("WHISPER_MIN_CHUNK", 120, 30)

# Silero VAD: определяет, где в записи речь, а где тишина/шум/музыка.
VAD_MODEL = os.path.join(BASE_DIR, "models", "silero_vad.onnx")
VAD_ENABLED = os.environ.get("WHISPER_VAD", "1") != "0"
VAD_THRESHOLD = _float_env("WHISPER_VAD_THRESHOLD", 0.5)
VAD_MIN_SPEECH = _float_env("WHISPER_VAD_MIN_SPEECH", 0.25)     # сек
VAD_MIN_SILENCE = _float_env("WHISPER_VAD_MIN_SILENCE", 0.35)  # сек
VAD_MIN_SPEECH_RATIO = _float_env("WHISPER_VAD_MIN_SPEECH_RATIO", 0.005)
VAD_SNAP_WINDOW = _float_env("WHISPER_VAD_SNAP_WINDOW", 90.0)
VAD_WIN = 512          # новых отсчётов за шаг (16 кГц = 32 мс)
VAD_CTX = 64           # контекст, который ждёт модель Silero v5
VAD_HOP = VAD_WIN / 16000.0
SILENCE_DBFS = _float_env("WHISPER_SILENCE_DBFS", -65.0)
SILENCE_PEAK_DBFS = _float_env("WHISPER_SILENCE_PEAK_DBFS", -45.0)
STALL_SECONDS = _int_env("WHISPER_STALL_SECONDS", 2700, 300)
EXTRACT_TIMEOUT = _int_env("WHISPER_EXTRACT_TIMEOUT", 900, 60)
JOB_TOTAL_TIMEOUT = _int_env("WHISPER_JOB_TIMEOUT", 21600, 600)
MAX_TRACKS = 4
MAX_CHUNKS = 10000
MAX_UPLOAD_BYTES = _int_env("WHISPER_MAX_UPLOAD_GB", 64) * (1024 ** 3)
MAX_NAME_BYTES = 100
MAX_PROMPT_CHARS = 500
MAX_JOBS_RESPONSE = 300
UPLOAD_KEEP_DAYS = 3

RUNNING_STATES = ("queued", "probe", "extract", "transcribe", "save")

for _d in (DATA_DIR, UPLOAD_DIR, WORK_DIR, OUTPUT_DIR):
    os.makedirs(_d, exist_ok=True)

# --------------------------------------------------------------------------
# Ресурсы машины
# --------------------------------------------------------------------------
def available_memory_bytes() -> int | None:
    """Сколько памяти реально свободно (важно на машинах с 8 ГБ)."""
    try:
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_AVPHYS_PAGES")
    except (ValueError, OSError, AttributeError):
        pass
    if sys.platform.startswith("win"):
        return _windows_memory()[1]
    if sys.platform == "darwin":
        # на macOS SC_AVPHYS_PAGES нет — считаем по vm_stat
        try:
            out = subprocess.run(["vm_stat"], capture_output=True, text=True,
                                 timeout=5).stdout
            m = re.search(r"page size of (\d+) bytes", out)
            page = int(m.group(1)) if m else 4096
            pages = 0
            for key in ("Pages free", "Pages inactive", "Pages speculative",
                        "Pages purgeable"):
                mm = re.search(key + r":\s+(\d+)", out)
                if mm:
                    pages += int(mm.group(1))
            if pages:
                return page * pages
        except Exception:
            pass
    return None


def _windows_memory() -> tuple[int | None, int | None]:
    """(всего, свободно) в байтах для Windows — там os.sysconf нет."""
    try:
        import ctypes

        class _MemoryStatusEx(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_ulong),
                ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]

        st = _MemoryStatusEx()
        st.dwLength = ctypes.sizeof(_MemoryStatusEx)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(st)):
            return int(st.ullTotalPhys), int(st.ullAvailPhys)
    except Exception:
        pass
    return None, None


def total_memory_bytes() -> int | None:
    try:
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    except (ValueError, OSError, AttributeError):
        pass
    if sys.platform.startswith("win"):
        return _windows_memory()[0]
    return None


# --------------------------------------------------------------------------
# ffmpeg / ffprobe
# --------------------------------------------------------------------------
def _find_ffmpeg() -> str | None:
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return None


FFMPEG = _find_ffmpeg()
FFPROBE = shutil.which("ffprobe")
# mlx_whisper вызывает буквально «ffmpeg» из PATH, поэтому резервный бинарник
# без PATH бесполезен — учитываем это в сообщении об ошибке.
if FFMPEG and not shutil.which("ffmpeg"):
    os.environ["PATH"] = os.path.dirname(FFMPEG) + os.pathsep + os.environ.get("PATH", "")


def audio_track_indices(path: str) -> list[int]:
    """Индексы аудиодорожек в файле. Пустой список — звука нет."""
    if not FFPROBE:
        return []
    try:
        out = subprocess.run(
            [FFPROBE, "-v", "error", "-select_streams", "a",
             "-show_entries", "stream=index", "-of", "csv=p=0", path],
            capture_output=True, text=True, timeout=120,
        )
        tracks = []
        for line in out.stdout.splitlines():
            line = line.strip().rstrip(",")
            if line.isdigit():
                tracks.append(int(line))
        return tracks
    except Exception:
        return []


def probe_duration(path: str) -> float:
    """Длительность файла в секундах (0.0, если не удалось узнать)."""
    if FFPROBE:
        try:
            out = subprocess.run(
                [FFPROBE, "-v", "error", "-show_entries", "format=duration",
                 "-of", "default=nw=1:nk=1", path],
                capture_output=True, text=True, timeout=120,
            )
            for line in out.stdout.splitlines():
                line = line.strip()
                if line:
                    try:
                        return float(line)
                    except ValueError:
                        pass
        except Exception:
            pass
    if FFMPEG:
        try:
            out = subprocess.run(
                [FFMPEG, "-hide_banner", "-i", path],
                capture_output=True, text=True, timeout=120,
            )
            m = re.search(r"Duration:\s*(\d+):(\d+):(\d+(?:\.\d+)?)", out.stderr or "")
            if m:
                h, mnt, s = int(m.group(1)), int(m.group(2)), float(m.group(3))
                return h * 3600 + mnt * 60 + s
        except Exception:
            pass
    return 0.0


def wav_level_dbfs(path: str) -> tuple[float, float] | None:
    """(RMS дБFS, пик дБFS) для 16-битного WAV. Читает блоками — память не растёт."""
    try:
        import numpy as np
    except Exception:
        return None
    try:
        with wave.open(path, "rb") as w:
            if w.getsampwidth() != 2:
                return None
            sum_sq = 0.0
            peak = 0
            total = 0
            while True:
                raw = w.readframes(1_000_000)
                if not raw:
                    break
                arr = np.frombuffer(raw, dtype="<i2").astype(np.float32)
                sum_sq += float(arr.dot(arr))
                m = float(np.abs(arr).max()) if arr.size else 0.0
                if m > peak:
                    peak = m
                total += int(arr.size)
            del arr
            if total == 0:
                return None
            rms = math.sqrt(sum_sq / total) / 32768.0
            pk = peak / 32768.0
            rms_db = 20 * math.log10(rms) if rms > 0 else -120.0
            peak_db = 20 * math.log10(pk) if pk > 0 else -120.0
            return rms_db, peak_db
    except Exception:
        return None


def is_silent_level(level: tuple[float, float] | None) -> bool:
    """Тишина только если И средне-тихо, И пики низкие.

    Двойное условие защищает от ложного отбрасывания тихой, но настоящей речи.
    Если уровень измерить не удалось — считаем, что звук есть.
    """
    if level is None:
        return False
    rms_db, peak_db = level
    return rms_db < SILENCE_DBFS and peak_db < SILENCE_PEAK_DBFS


def _run_ffmpeg_extract(src: str, start: float, dur: float | None, out: str,
                        track: int | None) -> tuple[int, str]:
    """Одна попытка извлечения. Возвращает (размер файла или -1, текст ошибки)."""
    cmd = [FFMPEG, "-nostdin", "-hide_banner", "-loglevel", "error", "-y",
           "-analyzeduration", "100M", "-probesize", "100M"]
    if start and start > 0:
        cmd += ["-ss", f"{start:.3f}"]
    cmd += ["-i", src]
    if dur:
        cmd += ["-t", f"{dur:.3f}"]
    if track is not None:
        cmd += ["-map", f"0:a:{track}"]
    cmd += ["-vn", "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", out]

    if os.path.exists(out):
        try:
            os.remove(out)
        except OSError:
            pass
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=EXTRACT_TIMEOUT)
    except subprocess.TimeoutExpired:
        return -1, "ffmpeg не успел извлечь звук за отведённое время"
    except (OSError, ValueError) as e:
        return -1, str(e)
    if r.returncode != 0:
        return -1, (r.stderr or "").strip()
    size = os.path.getsize(out) if os.path.exists(out) else 0
    return size, ""


def extract_chunk(src: str, start: float, dur: float | None, out: str,
                  tracks: list[int] | None) -> tuple[int, bool, int]:
    """Извлекает кусок аудио 16 кГц моно.

    Возвращает (размер в байтах, признак тишины, номер использованной дорожки).
    Перебирает дорожки: если первая пустая или тихая — пробует следующую.
    """
    order: list[int | None] = list(tracks) if tracks else []
    if not order:
        order = [None]
    if None not in order:
        order.append(None)  # запасной вариант: пусть ffmpeg выберет сам

    last_err = ""
    hard_errors = 0
    best_size = 0
    best_silent = False
    best_track = -1

    for track in order:
        size, err = _run_ffmpeg_extract(src, start, dur, out, track)
        if size < 0:
            if err:
                last_err = err
            hard_errors += 1
            continue
        if size > best_size:
            best_size = size
        if size < 2000:
            continue
        if not is_silent_level(wav_level_dbfs(out)):
            return size, False, (track if track is not None else -1)
        # тихая дорожка — запомним и попробуем следующую
        best_silent = True
        best_track = track if track is not None else -1

    if best_size == 0 and hard_errors >= len(order):
        raise RuntimeError(
            "Не удалось прочитать звук из файла. " + last_err[:300]
        )
    return best_size, best_silent, best_track


# --------------------------------------------------------------------------
# Словарь терминов и имён
# --------------------------------------------------------------------------
# terms — правильные написания: они уходят в подсказку распознаванию.
# replacements — пары «как слышится → как правильно»; это основы слов,
# поэтому «гигел → Гегел» исправит и «Гигелем», и «Гигеля».
# Это только пример-заготовка: имена и термины своей встречи добавьте в
# интерфейсе приложения. Рабочий словарь хранится в data/glossary.json и
# в репозиторий не попадает.
DEFAULT_GLOSSARY = {
    "terms": [
        "экспедиция", "модератор", "фасилитатор", "подгруппа", "лидер команды",
        "мастер-план", "философия", "антиутопия", "утопия",
    ],
    "replacements": [
        ["гигел", "Гегел"],
        ["фацилитатор", "фасилитатор"],
        ["фасцилитатор", "фасилитатор"],
        ["касселизатор", "фасилитатор"],
        ["салитатор", "фасилитатор"],
        ["наиполее", "наиболее"],
        ["ликтофон", "диктофон"],
    ],
}


def load_glossary() -> dict:
    try:
        with open(GLOSSARY_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            terms = [str(x).strip() for x in (data.get("terms") or []) if str(x).strip()]
            repls = [[str(a).strip(), str(b).strip()]
                     for a, b in (data.get("replacements") or []) if str(a).strip()]
            return {"terms": terms, "replacements": repls}
    except (OSError, json.JSONDecodeError, TypeError, ValueError):
        pass
    return {"terms": list(DEFAULT_GLOSSARY["terms"]),
            "replacements": [list(p) for p in DEFAULT_GLOSSARY["replacements"]]}


def save_glossary(g: dict) -> None:
    tmp = GLOSSARY_FILE + ".tmp"
    try:
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(g, f, ensure_ascii=False, indent=1)
        os.replace(tmp, GLOSSARY_FILE)
    except OSError:
        pass


def ensure_glossary_file() -> None:
    if not os.path.exists(GLOSSARY_FILE):
        save_glossary(load_glossary())


def glossary_to_text(g: dict) -> str:
    lines = list(g.get("terms", []))
    for a, b in g.get("replacements", []):
        lines.append(f"{a} → {b}")
    return "\n".join(lines)


def text_to_glossary(text: str) -> dict:
    terms: list[str] = []
    repls: list[list[str]] = []
    for line in (text or "").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        sep = "→" if "→" in line else ("->" if "->" in line else
                                        ("=>" if "=>" in line else None))
        if sep:
            a, _, b = line.partition(sep)
            a, b = a.strip(), b.strip()
            if a and b:
                repls.append([a, b])
        else:
            terms.append(line)
    seen = set()
    unique = []
    for t in terms:
        key = t.lower()
        if key not in seen:
            seen.add(key)
            unique.append(t)
    return {"terms": unique, "replacements": repls}


def apply_glossary(text: str, glossary: dict | None = None) -> tuple[str, int]:
    """Заменяет частые ошибки на правильные слова. Возвращает (текст, число замен)."""
    g = glossary if glossary is not None else load_glossary()
    total = 0
    # длинные правила применяем первыми: иначе «наиполе» испортит «наиполее»
    rules = sorted((p for p in g.get("replacements", [])
                    if len(p) >= 2 and p[0] and p[1]),
                   key=lambda p: len(p[0]), reverse=True)
    for a, b in rules:
        pattern = re.compile(r"(?<![\w])" + re.escape(a), re.IGNORECASE)

        def repl(m, _b=b):
            found = m.group(0)
            if found[:1].isupper():
                return _b[:1].upper() + _b[1:]
            if _b[:1].isalpha():
                return _b[:1].lower() + _b[1:]
            return _b

        text, n = pattern.subn(repl, text)
        total += n
    return text, total


def glossary_prompt(glossary: dict | None = None) -> str:
    """Строка с правильными терминами для подсказки распознаванию."""
    g = glossary if glossary is not None else load_glossary()
    return " · ".join(g.get("terms", [])[:40])[:400]


def apply_glossary_to_job(jid: str) -> tuple[bool, str, int]:
    """Прогоняет словарь по уже готовому файлу. Оригинал сохраняет в data/backup."""
    with LOCK:
        job = JOBS.get(jid)
        out = job.get("output") if job else None
    if not out or not os.path.exists(out):
        return False, "Файл результата не найден", 0
    try:
        with open(out, "r", encoding="utf-8") as f:
            text = f.read()
    except OSError as e:
        return False, f"Не удалось прочитать файл: {e}", 0

    parts = text.split("=" * 70, 1)
    if len(parts) == 2:
        head, body = parts[0], parts[1]
        body, count = apply_glossary(body)
        new_text = head + "=" * 70 + body
    else:
        new_text, count = apply_glossary(text)

    if count == 0:
        return True, "Замен не потребовалось", 0

    try:
        os.makedirs(BACKUP_DIR, exist_ok=True)
        shutil.copy2(out, os.path.join(BACKUP_DIR, os.path.basename(out)))
    except OSError:
        pass
    try:
        tmp = out + ".part"
        with open(tmp, "w", encoding="utf-8") as f:
            f.write(new_text)
        os.replace(tmp, out)
    except OSError as e:
        return False, f"Не удалось сохранить: {e}", 0
    return True, os.path.basename(out), count


# --------------------------------------------------------------------------
# Переименование готовых файлов
# --------------------------------------------------------------------------
def load_renames() -> dict:
    try:
        with open(RENAMES_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def save_renames(data: dict) -> None:
    tmp = RENAMES_FILE + ".tmp"
    try:
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=1)
        os.replace(tmp, RENAMES_FILE)
    except OSError:
        pass


def clean_output_name(name: str) -> str:
    """Имя для готового файла: точки внутри сохраняем, расширение отбрасываем."""
    name = os.path.basename((name or "").replace("\\", "/")).strip()
    if name.lower().endswith(".txt"):
        name = name[:-4]
    name = re.sub(r'[<>:"|?*\x00-\x1f/\\]', "_", name).strip(" .")
    if not name:
        return ""
    return name.encode("utf-8")[:MAX_NAME_BYTES].decode("utf-8", "ignore").strip(" ._")


def rename_output(jid: str, new_name: str) -> tuple[bool, str]:
    """Переименовывает готовый .txt на диске. Возвращает (успех, имя/ошибку)."""
    with LOCK:
        job = JOBS.get(jid)
        if not job:
            return False, "Задача не найдена"
        old = job.get("output")
    if not old or not os.path.exists(old):
        return False, "Файл результата не найден на диске"

    stem = clean_output_name(new_name)
    if not stem:
        return False, "Имя файла не может быть пустым"
    target = os.path.join(OUTPUT_DIR, stem + ".txt")
    n = 1
    while os.path.exists(target) and os.path.realpath(target) != os.path.realpath(old):
        n += 1
        target = os.path.join(OUTPUT_DIR, f"{stem} ({n}).txt")

    try:
        if os.path.realpath(target) != os.path.realpath(old):
            os.replace(old, target)
    except OSError as e:
        return False, f"Не удалось переименовать: {e}"

    old_base = os.path.basename(old)
    new_base = os.path.basename(target)
    final_stem = os.path.splitext(new_base)[0]
    renames = load_renames()
    renames.pop(old_base, None)
    renames[new_base] = final_stem
    save_renames(renames)

    with LOCK:
        job["output"] = target
        job["name"] = new_base
        job["display_name"] = final_stem
    persist_jobs()
    return True, final_stem


# --------------------------------------------------------------------------
# Модели распознавания
# --------------------------------------------------------------------------
HF_HOME = os.environ.get("HF_HOME") or os.path.join(
    os.path.expanduser("~"), ".cache", "huggingface")
WEIGHT_FILES = ("weights.safetensors", "weights.npz")


def model_cache_dir(model_id: str) -> str:
    return os.path.join(HF_HOME, "hub", "models--" + model_id.replace("/", "--"))


def model_installed(model_id: str) -> bool:
    """Есть ли веса модели в локальном кеше (то есть работает ли она офлайн)."""
    snap = os.path.join(model_cache_dir(model_id), "snapshots")
    try:
        versions = os.listdir(snap)
    except OSError:
        return False
    for ver in versions:
        base = os.path.join(snap, ver)
        for wf in WEIGHT_FILES:
            path = os.path.join(base, wf)
            if os.path.exists(path):
                try:
                    if os.path.getsize(os.path.realpath(path)) > 1_000_000:
                        return True
                except OSError:
                    return True
    return False


def model_disk_size(model_id: str) -> int:
    """Сколько модель занимает на диске (общие blobs считаем один раз)."""
    total = 0
    seen: set[str] = set()
    for root, _dirs, files in os.walk(model_cache_dir(model_id)):
        for name in files:
            try:
                real = os.path.realpath(os.path.join(root, name))
            except OSError:
                continue
            if real in seen:
                continue
            seen.add(real)
            try:
                total += os.path.getsize(real)
            except OSError:
                pass
    return total


def catalog_entry(model_id: str) -> dict | None:
    for m in MODEL_CATALOG:
        if m["id"] == model_id:
            return m
    return None


def set_current_model(model_id: str) -> None:
    """Переключает модель на лету и запоминает выбор."""
    global MODEL, _faster_model, _faster_model_id
    MODEL = model_id
    s = load_settings()
    s["model"] = model_id
    save_settings(s)
    # освобождаем предыдущую модель из памяти: на 8 ГБ это существенно
    try:
        from mlx_whisper.transcribe import ModelHolder

        ModelHolder.model = None
        ModelHolder.model_path = None
    except Exception:
        pass
    with _model_lock:
        _faster_model = None
        _faster_model_id = None


DOWNLOADS: dict[str, dict] = {}
_download_lock = threading.Lock()


def download_status(model_id: str, size_mb: int) -> dict:
    with _download_lock:
        st = dict(DOWNLOADS.get(model_id) or {})
    if st.get("status") == "downloading":
        have = model_disk_size(model_id)
        expected = max(1, size_mb) * 1024 * 1024
        st["progress"] = min(99, int(100 * have / expected)) if have else 0
        st["elapsed"] = int(time.time() - st.get("started", time.time()))
        st["have_mb"] = int(have / 1024 / 1024)
    return st


def start_download(model_id: str) -> bool:
    with _download_lock:
        cur = DOWNLOADS.get(model_id)
        if cur and cur.get("status") == "downloading":
            return False
        DOWNLOADS[model_id] = {"status": "downloading",
                               "started": time.time(), "error": ""}
    threading.Thread(target=_download_worker, args=(model_id,), daemon=True).start()
    return True


def _download_worker(model_id: str) -> None:
    saved = {}
    # скачивание — единственный момент, когда приложению нужен интернет
    for k in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE"):
        if k in os.environ:
            saved[k] = os.environ.pop(k)
    status, err = "done", ""
    try:
        from huggingface_hub import snapshot_download

        snapshot_download(
            repo_id=model_id,
            allow_patterns=["*.json", "*.txt", "*.md", "*.npz",
                            "*.safetensors", ".gitattributes"],
        )
        if not model_installed(model_id):
            status, err = "error", "Файлы скачались, но веса модели не найдены."
    except Exception as e:
        status, err = "error", f"{type(e).__name__}: {e}"
    finally:
        for k, v in saved.items():
            os.environ[k] = v
    with _download_lock:
        started = (DOWNLOADS.get(model_id) or {}).get("started", time.time())
        DOWNLOADS[model_id] = {"status": status, "error": err,
                               "started": started, "finished": time.time()}
    if status == "done":
        set_current_model(model_id)


# --------------------------------------------------------------------------
# Вспомогательные функции
# --------------------------------------------------------------------------
def safe_stem(name: str) -> str:
    stem = os.path.splitext(os.path.basename(name))[0]
    stem = stem.replace("/", "_").replace("\\", "_").replace("\x00", "")
    stem = re.sub(r'[<>:"|?*\x00-\x1f]', "_", stem).strip(" .")
    if not stem:
        return "audio"
    raw = stem.encode("utf-8")[:MAX_NAME_BYTES]
    stem = raw.decode("utf-8", "ignore").strip(" ._")
    return stem or "audio"


def fmt_ts(t: float) -> str:
    t = max(0.0, t)
    h = int(t // 3600)
    m = int((t % 3600) // 60)
    s = int(t % 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


# --------------------------------------------------------------------------
# Фильтр галлюцинаций
# --------------------------------------------------------------------------
# Whisper «уверенно» выдумывает эти фразы на тишине, шуме и музыке.
HALLUCINATION_PATTERNS = (
    "продолжение следует",
    "субтитры сделал",
    "субтитры подготовил",
    "редактор субтитров",
    "dimatorzok",
    "dima torzok",
    "подписывайтесь на канал",
    "ставьте лайки",
    "спасибо за просмотр",
)


def _norm_text(text: str) -> str:
    t = text.lower()
    t = re.sub(r"[^\w\s]", " ", t, flags=re.UNICODE)
    return re.sub(r"\s+", " ", t).strip()


def _words(text: str) -> list[str]:
    return _norm_text(text).split()


def _containment(shorter: list[str], longer: list[str]) -> float:
    """Какая доля слов короткого фрагмента встречается в длинном."""
    if not shorter:
        return 0.0
    pool = set(longer)
    return sum(1 for w in shorter if w in pool) / len(shorter)


def dedup_segments(segments: list[dict]) -> list[dict]:
    """Убирает дубли, возникающие на стыках кусков.

    Иногда на границе кусков один и тот же фрагмент речи распознаётся дважды —
    один раз коротко и без знаков, второй раз полностью. Оставляем более
    информативный вариант, порядок делаем хронологическим.
    """
    ordered = sorted(segments, key=lambda s: (s["start"], s["end"]))
    out: list[dict] = []
    for seg in ordered:
        if out:
            prev = out[-1]
            overlaps = seg["start"] < prev["end"] and seg["end"] > prev["start"]
            if overlaps:
                a, b = _words(prev["text"]), _words(seg["text"])
                dup = False
                if a and b:
                    if a == b:
                        dup = True
                    elif len(a) >= 3 and len(b) >= 3:
                        shorter, longer = (b, a) if len(b) <= len(a) else (a, b)
                        dup = _containment(shorter, longer) >= 0.65
                if dup:
                    if (len(seg["text"]), seg["end"] - seg["start"]) > \
                       (len(prev["text"]), prev["end"] - prev["start"]):
                        out[-1] = seg
                    continue
        out.append(seg)
    return out


def is_hallucination(seg: dict) -> bool:
    """Отбрасывает классический «мусор» Whisper на не-речи."""
    text = (seg.get("text") or "").strip()
    if not text:
        return True
    norm = _norm_text(text)
    if not norm:
        return True
    if any(p in norm for p in HALLUCINATION_PATTERNS):
        return True
    # одна и та же «фраза» из повторов одного слова
    words = norm.split()
    if len(words) >= 4 and len(set(words)) == 1:
        return True
    # классический признак зацикливания: плохая логвероятность + аномальное сжатие
    alp = seg.get("avg_logprob")
    cr = seg.get("compression_ratio")
    if isinstance(alp, float) and isinstance(cr, float):
        if alp < -1.0 and cr > 2.4:
            return True
    return False


# --------------------------------------------------------------------------
# Очередь задач
# --------------------------------------------------------------------------
LOCK = threading.Lock()
JOBS: dict[str, dict] = {}
ORDER: list[str] = []
QUEUE: Queue = Queue()
_persist_lock = threading.Lock()


def _serialize_job(j: dict) -> dict:
    keys = ("id", "name", "size", "status", "progress", "message", "error",
            "error_detail", "created", "started", "finished", "duration",
            "output", "timestamps", "prompt", "src", "chunk_index", "chunk_total")
    return {k: j.get(k) for k in keys}


def persist_jobs() -> None:
    with LOCK:
        data = [_serialize_job(JOBS[j]) for j in ORDER if j in JOBS][-200:]
    tmp = JOBS_FILE + ".tmp"
    try:
        with _persist_lock:
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump({"jobs": data}, f, ensure_ascii=False)
            os.replace(tmp, JOBS_FILE)
    except OSError:
        pass


def upd(job: dict, **kw) -> None:
    with LOCK:
        if job.get("cancel") and "status" in kw and kw["status"] != "error":
            return
        job.update(kw)
        if "progress" in kw or "status" in kw:
            job["progress_ts"] = time.time()
        need_save = "status" in kw
    if need_save:
        persist_jobs()


def public_job(j: dict) -> dict:
    started = j.get("started")
    progress = j.get("progress", 0)
    eta = None
    if started and progress and 5 < progress < 100:
        elapsed = time.time() - started
        eta = int(elapsed * (100 - progress) / max(1, progress - 5))
    return {
        "id": j["id"],
        "name": j.get("name", ""),
        "display_name": j.get("display_name") or j.get("name", ""),
        "file_stem": (os.path.splitext(os.path.basename(j["output"]))[0]
                      if j.get("output") else None),
        "size": j.get("size", 0),
        "status": j.get("status", "queued"),
        "progress": progress,
        "message": j.get("message", ""),
        "error": j.get("error", ""),
        "error_detail": (j.get("error_detail") or "")[:1200],
        "created": j.get("created"),
        "started": started,
        "finished": j.get("finished"),
        "duration": j.get("duration"),
        "chunk_index": j.get("chunk_index"),
        "chunk_total": j.get("chunk_total"),
        "eta": eta,
        "download": f"/api/download?id={j['id']}" if j.get("status") == "done" else None,
        "can_retry": bool(j.get("src")) and os.path.exists(j.get("src") or ""),
    }


def register_outputs() -> None:
    """Показывает готовые .txt от прошлых запусков."""
    try:
        names = sorted(
            (n for n in os.listdir(OUTPUT_DIR) if n.lower().endswith(".txt")),
            key=lambda n: os.path.getmtime(os.path.join(OUTPUT_DIR, n)),
            reverse=True,
        )
    except OSError:
        return
    renames = load_renames()
    for n in names:
        path = os.path.join(OUTPUT_DIR, n)
        try:
            size = os.path.getsize(path)
            mtime = os.path.getmtime(path)
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                head = f.read(300)
        except OSError:
            continue
        # битые/пустые/недописанные файлы не показываем как готовые
        if size < 40 or "Файл:" not in head:
            continue
        display = n
        m = re.search(r"^Файл:\s*(.+)$", head, re.MULTILINE)
        if m:
            display = m.group(1).strip()
        if renames.get(n):  # имя, заданное пользователем, важнее
            display = renames[n]
        jid = "out_" + uuid.uuid5(uuid.NAMESPACE_URL, path).hex[:12]
        with LOCK:
            if jid in JOBS:
                continue
            JOBS[jid] = {
                "id": jid, "name": n, "display_name": display, "size": size,
                "status": "done", "progress": 100, "message": "Готово",
                "error": "", "created": mtime, "started": None,
                "finished": mtime, "duration": None, "output": path,
                "external": True,
            }
            ORDER.append(jid)


def load_jobs() -> None:
    """Восстанавливает задачи после перезапуска. Незавершённые — как прерванные."""
    try:
        with open(JOBS_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError):
        return
    for j in data.get("jobs", []):
        if not isinstance(j, dict) or not j.get("id"):
            continue
        if j.get("status") == "done":
            continue  # готовые и так подхватит register_outputs
        jid = j["id"]
        src = j.get("src")
        has_src = bool(src) and os.path.exists(src)
        j["status"] = "error"
        j["message"] = "Задача прервана"
        j["error"] = ("Приложение было закрыто или перезапущено во время обработки. "
                      "Можно запустить заново.")
        j["error_detail"] = ""
        j["cancel"] = True
        j.pop("external", None)
        with LOCK:
            if jid in JOBS:
                continue
            JOBS[jid] = j
            ORDER.append(jid)


def cleanup_work_dir() -> None:
    try:
        names = os.listdir(WORK_DIR)
    except OSError:
        return
    for n in names:
        p = os.path.join(WORK_DIR, n)
        try:
            if os.path.isdir(p):
                shutil.rmtree(p, ignore_errors=True)
            else:
                os.remove(p)
        except OSError:
            pass


def cleanup_old_uploads(keep: set[str]) -> None:
    """Удаляет старые загрузки, чтобы не копить мусор. Свежие не трогает."""
    cutoff = time.time() - UPLOAD_KEEP_DAYS * 86400
    try:
        names = os.listdir(UPLOAD_DIR)
    except OSError:
        return
    for n in names:
        p = os.path.join(UPLOAD_DIR, n)
        if p in keep:
            continue
        try:
            if os.path.getmtime(p) < cutoff:
                os.remove(p)
        except OSError:
            pass


def cleanup_parts() -> None:
    """Убирает недописанные файлы результатов."""
    try:
        names = os.listdir(OUTPUT_DIR)
    except OSError:
        return
    for n in names:
        if n.endswith(".part") or n.endswith(".tmp"):
            try:
                os.remove(os.path.join(OUTPUT_DIR, n))
            except OSError:
                pass


def enqueue(filename: str, tmp_path: str, size: int, timestamps: bool,
            prompt: str) -> str:
    jid = uuid.uuid4().hex[:12]
    job = {
        "id": jid, "name": filename, "size": size, "status": "queued",
        "progress": 0, "message": "В очереди", "error": "", "error_detail": "",
        "created": time.time(), "started": None, "finished": None,
        "duration": None, "output": None, "timestamps": timestamps,
        "prompt": prompt, "src": tmp_path, "progress_ts": time.time(),
        "chunk_index": None, "chunk_total": None,
    }
    with LOCK:
        JOBS[jid] = job
        ORDER.append(jid)
    persist_jobs()
    QUEUE.put(jid)
    return jid


def retry_job(jid: str) -> bool:
    with LOCK:
        job = JOBS.get(jid)
        if not job or job.get("status") not in ("error",):
            return False
        src = job.get("src")
        if not src or not os.path.exists(src):
            return False
        job.update(status="queued", progress=0, message="В очереди", error="",
                   error_detail="", started=None, finished=None, output=None,
                   duration=None, cancel=False, progress_ts=time.time(),
                   chunk_index=None, chunk_total=None)
    persist_jobs()
    QUEUE.put(jid)
    return True


def write_txt(job: dict, segments: list[dict], duration: float) -> str:
    stem = safe_stem(job["name"])
    out_path = os.path.join(OUTPUT_DIR, stem + ".txt")
    n = 1
    while os.path.exists(out_path):
        n += 1
        out_path = os.path.join(OUTPUT_DIR, f"{stem} ({n}).txt")
    now = time.strftime("%Y-%m-%d %H:%M")

    lines = [
        f"Файл: {job['name']}",
        f"Дата транскрибации: {now}",
        f"Модель: {MODEL} (язык: {LANGUAGE})",
    ]
    if duration:
        lines.append(f"Длительность: {fmt_ts(duration)}")
    header = "\n".join(lines) + "\n" + "=" * 70 + "\n\n"

    part_path = out_path + ".part"
    try:
        with open(part_path, "w", encoding="utf-8") as f:
            f.write(header)
            if job.get("timestamps", True):
                for s in segments:
                    text = s["text"].strip()
                    if text:
                        f.write(f"[{fmt_ts(s['start'])}] {text}\n")
            else:
                buf: list[str] = []
                for s in segments:
                    text = s["text"].strip()
                    if not text:
                        continue
                    buf.append(text)
                    if len(" ".join(buf)) > 900:
                        f.write(" ".join(buf) + "\n\n")
                        buf = []
                if buf:
                    f.write(" ".join(buf) + "\n")
        os.replace(part_path, out_path)  # результат появляется только целиком
    except OSError:
        try:
            os.remove(part_path)
        except OSError:
            pass
        raise
    return out_path


# --------------------------------------------------------------------------
# Обработка задачи
# --------------------------------------------------------------------------
MODEL_READY = False


@contextlib.contextmanager
def _quiet_stderr():
    """Глушит служебные полосы tqdm от mlx_whisper, сохраняя текст для отладки."""
    buf = io.StringIO()
    old = sys.stderr
    sys.stderr = buf
    try:
        yield buf
    finally:
        sys.stderr = old


def human_error(exc: BaseException, extra: str = "") -> str:
    msg = f"{type(exc).__name__}: {exc} {extra}".lower()
    if not FFMPEG and "ffmpeg" in msg:
        return "Не найден ffmpeg. Установите его командой: brew install ffmpeg"
    if "no space left" in msg or getattr(exc, "errno", None) == 28:
        return "На диске закончилось место. Освободите место и нажмите «Повторить»."
    if isinstance(exc, OSError) and getattr(exc, "errno", None) == 63:
        return "Слишком длинное имя файла — переименуйте его короче."
    if "file name too long" in msg:
        return "Слишком длинное имя файла — переименуйте его короче."
    if "does not contain any stream" in msg or "output file does not contain" in msg:
        return "В файле нет звуковой дорожки — только видео или служебные данные."
    if "invalid data found" in msg or "corrupt" in msg or "moov atom not found" in msg:
        return "Файл повреждён или скачан не полностью — звук не читается."
    if "localentrynotfound" in msg or "cached snapshot" in msg:
        return ("Модель распознавания не найдена в кеше, а интернет отключён. "
                "Подключите интернет и запустите с WHISPER_ALLOW_NET=1.")
    if IMPORT_ERROR:
        return ("Не удалось загрузить библиотеку распознавания. Запустите "
                "setup.sh (macOS) или setup.ps1 (Windows), затем приложение "
                "через «Запустить Whisper.command».")
    if "ffmpeg" in msg:
        return "ffmpeg не смог прочитать этот файл — возможно, он повреждён."
    return "Не удалось обработать файл."


_faster_model = None
_faster_model_id = None
_model_lock = threading.Lock()


def _get_faster_model(model_id: str):
    """Загружает и кэширует модель faster-whisper (CTranslate2, процессор)."""
    global _faster_model, _faster_model_id
    with _model_lock:
        if _faster_model is None or _faster_model_id != model_id:
            device = os.environ.get("WHISPER_DEVICE", "cpu")
            compute = os.environ.get("WHISPER_COMPUTE_TYPE", "int8")
            _faster_model = faster_whisper.WhisperModel(
                model_id, device=device, compute_type=compute)
            _faster_model_id = model_id
        return _faster_model


def _transcribe_mlx(chunk_path: str, prompt: str) -> list[dict]:
    kwargs = dict(
        path_or_hf_repo=MODEL,
        language=LANGUAGE,
        task="transcribe",
        condition_on_previous_text=False,
        verbose=False,
        word_timestamps=False,
        temperature=(0.0, 0.2, 0.4, 0.6, 0.8, 1.0),
    )
    if prompt:
        kwargs["initial_prompt"] = prompt
    res = mlx_whisper.transcribe(chunk_path, **kwargs)
    return list(res.get("segments") or [])


def _transcribe_faster(chunk_path: str, prompt: str) -> list[dict]:
    model = _get_faster_model(MODEL)
    segments, _info = model.transcribe(
        chunk_path,
        language=LANGUAGE,
        task="transcribe",
        beam_size=_int_env("WHISPER_BEAM", 5, 1),
        condition_on_previous_text=False,
        initial_prompt=prompt or None,
        temperature=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
        vad_filter=False,      # тишину отсекаем сами, через Silero VAD
        word_timestamps=False,
    )
    out: list[dict] = []
    for s in segments:         # это генератор — читаем его здесь же
        out.append({
            "start": float(getattr(s, "start", 0.0) or 0.0),
            "end": float(getattr(s, "end", 0.0) or 0.0),
            "text": (getattr(s, "text", "") or "").strip(),
            "avg_logprob": getattr(s, "avg_logprob", None),
            "compression_ratio": getattr(s, "compression_ratio", None),
        })
    return out


def transcribe_chunk(chunk_path: str, prompt: str) -> tuple[list[dict], str]:
    """Распознаёт кусок и возвращает сегменты в общем формате.

    Внутри — выбранный бэкенд: MLX (Apple Silicon) или faster-whisper (CPU).
    """
    prompt = (prompt or "").strip()[:MAX_PROMPT_CHARS]
    with _quiet_stderr() as buf:
        try:
            if BACKEND == "faster":
                segs = _transcribe_faster(chunk_path, prompt)
            else:
                segs = _transcribe_mlx(chunk_path, prompt)
        except BaseException:
            raise RuntimeError((buf.getvalue() or "").strip()[-400:]
                               or "сбой распознавания")
    return segs, buf.getvalue() or ""


_vad_lock = threading.Lock()
_vad_session = None


def vad_available() -> bool:
    return VAD_ENABLED and os.path.exists(VAD_MODEL)


def _get_vad_session():
    """Ленивая загрузка Silero VAD (модель ~2 МБ, память ~50 МБ)."""
    global _vad_session
    if not vad_available():
        return None
    with _vad_lock:
        if _vad_session is None:
            try:
                import onnxruntime as ort

                opts = ort.SessionOptions()
                opts.log_severity_level = 3
                opts.inter_op_num_threads = 1
                opts.intra_op_num_threads = 1
                _vad_session = ort.InferenceSession(
                    VAD_MODEL, sess_options=opts,
                    providers=["CPUExecutionProvider"])
            except Exception:
                _vad_session = False
        return _vad_session or None


def vad_probs(wav_path: str) -> list[float]:
    """Вероятность речи каждые 32 мс для WAV 16 кГц моно."""
    sess = _get_vad_session()
    if sess is None:
        return []
    try:
        import numpy as np

        with wave.open(wav_path, "rb") as w:
            if w.getsampwidth() != 2 or w.getframerate() != 16000:
                return []
            sr = np.array(16000, dtype=np.int64)
            state = np.zeros((2, 1, 128), dtype=np.float32)
            ctx = np.zeros(VAD_CTX, dtype=np.float32)
            probs: list[float] = []
            tail = np.zeros(0, dtype=np.float32)
            while True:
                raw = w.readframes(VAD_WIN * 400)
                if not raw:
                    break
                block = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
                block = np.concatenate([tail, block])
                steps = (len(block) - VAD_WIN) // VAD_WIN
                for k in range(steps):
                    chunk = block[k * VAD_WIN:(k + 1) * VAD_WIN]
                    inp = np.concatenate([ctx, chunk])[None, :]
                    out, state = sess.run(None, {"input": inp, "state": state,
                                                 "sr": sr})
                    ctx = chunk[-VAD_CTX:]
                    probs.append(float(out[0][0]))
                tail = block[steps * VAD_WIN:]
            return probs
    except Exception:
        return []


def speech_intervals(probs: list[float]) -> list[tuple[float, float]]:
    """Из вероятностей делает отрезки речи, склеивая короткие паузы."""
    raw: list[list[float]] = []
    start = None
    for i, p in enumerate(probs):
        t = i * VAD_HOP
        if p >= VAD_THRESHOLD and start is None:
            start = t
        elif p < VAD_THRESHOLD and start is not None:
            raw.append([start, t])
            start = None
    if start is not None:
        raw.append([start, len(probs) * VAD_HOP])

    merged: list[list[float]] = []
    for a, b in raw:
        if merged and a - merged[-1][1] < VAD_MIN_SILENCE:
            merged[-1][1] = b
        else:
            merged.append([a, b])
    return [(a, b) for a, b in merged if b - a >= VAD_MIN_SPEECH]


def silence_gaps(intervals: list[tuple[float, float]],
                 total: float) -> list[tuple[float, float]]:
    gaps: list[tuple[float, float]] = []
    prev = 0.0
    for a, b in intervals:
        if a - prev > 0:
            gaps.append((prev, a))
        prev = max(prev, b)
    if total - prev > 0:
        gaps.append((prev, total))
    return gaps


def chunk_has_speech(wav_path: str, rms_silent: bool) -> bool:
    """Есть ли в куске речь. Если VAD недоступен — судим по громкости."""
    probs = vad_probs(wav_path)
    if not probs:
        return not rms_silent
    ratio = sum(1 for p in probs if p >= VAD_THRESHOLD) / len(probs)
    return ratio >= VAD_MIN_SPEECH_RATIO


def snap_to_silence(src: str, tracks: list[int], duration: float,
                    target: float, probe_path: str) -> float:
    """Ищет ближайшую паузу около target, чтобы не резать посреди фразы."""
    lo = max(0.0, target - VAD_SNAP_WINDOW)
    hi = min(duration, target + VAD_SNAP_WINDOW)
    if hi - lo < 10:
        return target
    size, _silent, _track = extract_chunk(src, lo, hi - lo, probe_path, tracks)
    if size < 2000:
        return target
    probs = vad_probs(probe_path)
    if not probs:
        return target
    intervals = speech_intervals(probs)
    gaps = [g for g in silence_gaps(intervals, len(probs) * VAD_HOP)
            if g[1] - g[0] >= VAD_MIN_SILENCE]
    best = None
    for a, b in gaps:
        mid = lo + (a + b) / 2.0
        dist = abs(mid - target)
        if best is None or dist < best[0]:
            best = (dist, mid)
    if best and best[0] <= VAD_SNAP_WINDOW:
        return best[1]
    return target


def find_chunk_boundaries(src: str, tracks: list[int], duration: float,
                          tmpdir: str) -> list[float]:
    """Границы кусков, сдвинутые к паузам. Без VAD — обычная равномерная нарезка."""
    if duration <= CHUNK_SECONDS:
        return [0.0, duration]
    if not vad_available():
        bounds = [float(x) for x in range(0, int(duration), CHUNK_SECONDS)]
        if bounds[-1] < duration:
            bounds.append(duration)
        return bounds

    bounds = [0.0]
    probe_path = os.path.join(tmpdir, "probe.wav")
    target = float(CHUNK_SECONDS)
    while target < duration - MIN_CHUNK_SECONDS:
        b = snap_to_silence(src, tracks, duration, target, probe_path)
        b = max(bounds[-1] + MIN_CHUNK_SECONDS, min(b, duration - 1.0))
        bounds.append(b)
        target = b + CHUNK_SECONDS
    bounds.append(duration)
    return bounds


def _remove(path: str) -> None:
    try:
        os.remove(path)
    except OSError:
        pass


def _absorb(segments: list[dict], new_segs: list[dict],
            glossary: dict | None = None) -> tuple[int, int]:
    """Добавляет фрагменты: словарь терминов, отсев галлюцинаций и дублей.

    Возвращает (сколько отброшено, сколько слов исправлено словарём).
    """
    dropped = 0
    fixed = 0
    for seg in new_segs:
        text, n = apply_glossary(seg["text"], glossary)
        if n:
            seg["text"] = text
            fixed += n
        if is_hallucination(seg):
            dropped += 1
            continue
        if (segments and _norm_text(segments[-1]["text"]) == _norm_text(seg["text"])
                and seg["start"] - segments[-1]["end"] < 2.0):
            continue
        segments.append(seg)
    return dropped, fixed


def _transcribe_window(job: dict, src: str, tracks: list[int], tmpdir: str,
                       idx: int, core_start: float, core_end: float,
                       label: str, prompt: str) -> tuple[list[dict], bool, bool]:
    """Обрабатывает одно окно. Возвращает (сегменты, была_тишина, пусто)."""
    global MODEL_READY
    ext_start = max(0.0, core_start - CHUNK_OVERLAP)
    ext_len = (core_end - core_start) + 2 * CHUNK_OVERLAP
    chunk_path = os.path.join(tmpdir, f"chunk_{idx:04d}.wav")

    size, rms_silent, track_no = extract_chunk(src, ext_start, ext_len,
                                               chunk_path, tracks)
    if size < 2000:
        _remove(chunk_path)
        return [], True, True       # за пределами файла или звука нет вовсе
    if not chunk_has_speech(chunk_path, rms_silent):
        _remove(chunk_path)
        return [], True, False      # тишина, шум или музыка

    if not MODEL_READY:
        upd(job, message=f"{label} · загрузка модели (первый раз 10–20 с)…")
    else:
        extra = f" · дорожка {track_no + 1}" if track_no > 0 else ""
        upd(job, message=f"{label} · распознавание{extra}")
    upd(job, status="transcribe")

    raw_segments, _ = transcribe_chunk(chunk_path, prompt)
    MODEL_READY = True
    _remove(chunk_path)  # куски на диске не копим

    out = []
    for s in raw_segments:
        seg = {
            "start": float(s.get("start", 0.0)) + ext_start,
            "end": float(s.get("end", 0.0)) + ext_start,
            "text": (s.get("text") or "").strip(),
            "avg_logprob": s.get("avg_logprob"),
            "compression_ratio": s.get("compression_ratio"),
        }
        # фрагмент относится к окну, в которое попало его НАЧАЛО
        if seg["start"] < core_start or seg["start"] >= core_end:
            continue
        out.append(seg)
    return out, False, False


def process_job(job: dict) -> None:
    src = job["src"]
    upd(job, status="probe", message="Анализ файла…", started=time.time(),
        progress=1)

    if IMPORT_ERROR:
        raise RuntimeError("Бэкенд распознавания недоступен: " + IMPORT_ERROR)

    tracks = audio_track_indices(src)
    if not tracks:
        raise RuntimeError("В файле нет звуковой дорожки — только видео или данные.")

    duration = probe_duration(src)
    known = duration > 0

    # словарь терминов и имён: правильные слова идут в подсказку распознаванию
    glossary = load_glossary()
    terms = glossary_prompt(glossary)
    user_prompt = (job.get("prompt") or "").strip()
    prompt = ((user_prompt + " · " + terms).strip(" ·") if terms else user_prompt)

    tmpdir = tempfile.mkdtemp(prefix="job_", dir=WORK_DIR)
    segments: list[dict] = []
    dropped = 0
    fixed = 0
    silent_chunks = 0
    ok = False
    try:
        if known:
            if vad_available() and duration > CHUNK_SECONDS:
                upd(job, status="extract", message="Поиск пауз в записи…", progress=3)
            bounds = find_chunk_boundaries(src, tracks, duration, tmpdir)
            windows = list(zip(bounds[:-1], bounds[1:]))
            total_chunks = max(1, len(windows))
        else:
            windows = []
            total_chunks = 0

        detail = f"Звук: {len(tracks)} дорожк(и)"
        if known:
            detail += f", длительность {fmt_ts(duration)}"
            if vad_available():
                detail += f" · кусков: {total_chunks}"
        upd(job, duration=duration or None, chunk_total=total_chunks or None,
            message=detail)

        if windows:
            for i, (a, b) in enumerate(windows):
                if job.get("cancel"):
                    raise RuntimeError("задача прервана")
                label = f"Фрагмент {i + 1} из {total_chunks}"
                upd(job, status="extract", message=f"Подготовка звука. {label}",
                    progress=5 + int(90 * i / total_chunks), chunk_index=i + 1)
                segs, was_silent, _empty = _transcribe_window(
                    job, src, tracks, tmpdir, i, a, b, label, prompt)
                if was_silent:
                    silent_chunks += 1
                else:
                    d, f = _absorb(segments, segs, glossary)
                    dropped += d
                    fixed += f
                upd(job, progress=5 + int(90 * (i + 1) / total_chunks))
        else:
            # длительность неизвестна — идём окнами до конца файла
            i = 0
            while i < MAX_CHUNKS:
                if job.get("cancel"):
                    raise RuntimeError("задача прервана")
                a = i * CHUNK_SECONDS
                b = a + CHUNK_SECONDS
                label = f"Фрагмент {i + 1} (длительность неизвестна)"
                upd(job, status="extract", message=label,
                    progress=min(95, 5 + i * 5), chunk_index=i + 1)
                segs, was_silent, empty = _transcribe_window(
                    job, src, tracks, tmpdir, i, a, b, label, prompt)
                if empty:
                    break  # дошли до конца файла
                if was_silent:
                    silent_chunks += 1
                else:
                    d, f = _absorb(segments, segs, glossary)
                    dropped += d
                    fixed += f
                upd(job, progress=min(95, 5 + (i + 1) * 5))
                i += 1

        upd(job, status="save", message="Сохранение результата…", progress=99)
        segments = dedup_segments(segments)
        out_path = write_txt(job, segments, duration)

        note = []
        if silent_chunks:
            note.append(f"пропущено тишины: {silent_chunks}")
        if dropped:
            note.append(f"отфильтровано служебных фраз: {dropped}")
        if fixed:
            note.append(f"словарь исправил слов: {fixed}")
        tail = (" (" + ", ".join(note) + ")") if note else ""
        upd(job, status="done", progress=100,
            message=f"Готово. Фрагментов: {len(segments)}{tail}",
            output=out_path, finished=time.time(), cancel=False)
        ok = True
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)
        if ok:
            _remove(src)


def _run_job_safe(job: dict) -> None:
    try:
        process_job(job)
    except BaseException as e:  # включая SystemExit из библиотек
        detail = ""
        try:
            if isinstance(e, RuntimeError):
                detail = str(e)[:800]
        except Exception:
            pass
        upd(job, status="error", message="Ошибка",
            error=human_error(e, detail), error_detail=detail,
            finished=time.time(), cancel=True)


def worker() -> None:
    while True:
        jid = QUEUE.get()
        with LOCK:
            job = JOBS.get(jid)
        if job is None:
            QUEUE.task_done()
            continue
        t = threading.Thread(target=_run_job_safe, args=(job,), daemon=True)
        t.start()
        # Ждём, но не бесконечно: если задача зависла, очередь не должна встать.
        while t.is_alive():
            t.join(10)
            if job.get("cancel"):
                break
        QUEUE.task_done()


def watchdog() -> None:
    """Помечает зависшие задачи ошибкой, чтобы очередь продолжала работать."""
    while True:
        time.sleep(30)
        now = time.time()
        changed = False
        with LOCK:
            for jid in ORDER:
                j = JOBS.get(jid)
                if not j or j.get("external"):
                    continue
                if j.get("status") in RUNNING_STATES and not j.get("cancel"):
                    if now - j.get("progress_ts", now) > STALL_SECONDS:
                        j.update(status="error", message="Обработка зависла",
                                 error="Обработка не отвечает слишком долго. "
                                       "Нажмите «Повторить».",
                                 error_detail="", finished=now, cancel=True)
                        changed = True
            if changed:
                pass
        if changed:
            persist_jobs()


# --------------------------------------------------------------------------
# HTTP-сервер
# --------------------------------------------------------------------------
STATIC = {
    "/": ("index.html", "text/html; charset=utf-8"),
    "/index.html": ("index.html", "text/html; charset=utf-8"),
    "/style.css": ("style.css", "text/css; charset=utf-8"),
    "/app.js": ("app.js", "application/javascript; charset=utf-8"),
}

SERVER_PORT = 0  # заполняется при запуске — нужен для проверки Host


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = "WhisperLocal/" + VERSION
    timeout = 300  # не держим мёртвые соединения

    # -- проверки запроса --------------------------------------------------
    def _allowed_hosts(self) -> set[str]:
        hosts = {"127.0.0.1", "localhost", "[::1]"}
        if SERVER_PORT:
            hosts |= {f"127.0.0.1:{SERVER_PORT}", f"localhost:{SERVER_PORT}",
                      f"[::1]:{SERVER_PORT}"}
        return hosts

    def _check_request(self) -> bool:
        """Защита от CSRF и DNS-rebinding: проверяем Host и Origin."""
        host = (self.headers.get("Host") or "").strip()
        if host and host not in self._allowed_hosts():
            self.close_connection = True
            self._error(403, "Запрос с чужого адреса отклонён")
            return False
        origin = (self.headers.get("Origin") or "").strip()
        if not origin:
            referer = (self.headers.get("Referer") or "").strip()
            if referer:
                origin = referer
        if origin:
            ok = False
            for h in self._allowed_hosts():
                if origin.startswith(f"http://{h}/") or origin == f"http://{h}":
                    ok = True
                    break
            if not ok:
                self.close_connection = True
                self._error(403, "Запрос со сторонней страницы отклонён")
                return False
        return True

    # -- утилиты -----------------------------------------------------------
    def _send(self, code: int, body: bytes, ctype: str,
              extra: dict | None = None) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        if self.close_connection:
            self.send_header("Connection", "close")
        for k, v in (extra or {}).items():
            self.send_header(k, v)
        self.end_headers()
        if self.command != "HEAD" and body:
            try:
                self.wfile.write(body)
            except (BrokenPipeError, ConnectionResetError):
                pass

    def _json(self, obj, code: int = 200) -> None:
        self._send(code, json.dumps(obj, ensure_ascii=False).encode("utf-8"),
                   "application/json; charset=utf-8")

    def _error(self, code: int, msg: str) -> None:
        self._json({"error": msg}, code)

    def log_message(self, fmt, *args):  # тише в консоли
        if (getattr(self, "path", "") or "").startswith("/api/jobs"):
            return
        sys.stderr.write("[whisper] %s\n" % (fmt % args))

    # -- GET ---------------------------------------------------------------
    def do_GET(self):
        if not self._check_request():
            return
        parsed = urlparse(self.path)
        path = parsed.path

        if path == "/favicon.ico":
            self.send_response(204)
            self.end_headers()
            return

        if path == "/api/info":
            self._json({
                "app": APP_NAME,
                "version": VERSION,
                "model": MODEL,
                "backend": BACKEND,
                "backend_label": BACKEND_LABEL,
                "language": LANGUAGE,
                "chunk": CHUNK_SECONDS,
                "overlap": CHUNK_OVERLAP,
                "silence_dbfs": SILENCE_DBFS,
                "silence_peak_dbfs": SILENCE_PEAK_DBFS,
                "vad": vad_available(),
                "ffmpeg": FFMPEG or "",
                "output_dir": OUTPUT_DIR,
                "ready": IMPORT_ERROR is None and FFMPEG is not None,
                "import_error": IMPORT_ERROR or "",
                "mem_total": total_memory_bytes(),
                "mem_available": available_memory_bytes(),
                "model_label": (catalog_entry(MODEL) or {}).get("label", MODEL),
                "platform": sys.platform,
            })
            return

        if path == "/api/glossary":
            g = load_glossary()
            self._json({"text": glossary_to_text(g),
                        "terms": len(g.get("terms", [])),
                        "replacements": len(g.get("replacements", []))})
            return

        if path == "/api/models":
            catalog = []
            for m in MODEL_CATALOG:
                catalog.append({
                    **m,
                    "installed": model_installed(m["id"]),
                    "current": m["id"] == MODEL,
                    "download": download_status(m["id"], m["size_mb"]),
                })
            self._json({"current": MODEL, "catalog": catalog})
            return

        if path == "/api/jobs":
            with LOCK:
                all_ids = list(ORDER)
                jobs = [public_job(JOBS[j]) for j in all_ids if j in JOBS]
            jobs.reverse()  # новые сверху
            total = len(jobs)
            if total > MAX_JOBS_RESPONSE:
                jobs = jobs[:MAX_JOBS_RESPONSE]
            self._json({"jobs": jobs, "total": total,
                        "truncated": total > len(jobs)})
            return

        if path == "/api/download":
            qs = parse_qs(parsed.query)
            jid = (qs.get("id") or [""])[0]
            with LOCK:
                job = JOBS.get(jid)
                out = job.get("output") if job else None
            if not out or not os.path.exists(out):
                self._error(404, "Файл не найден")
                return
            self._send_file(out, os.path.basename(out))
            return

        if path in STATIC:
            fname, ctype = STATIC[path]
            fpath = os.path.join(BASE_DIR, fname)
            if not os.path.exists(fpath):
                self._error(404, "Нет файла " + fname)
                return
            try:
                with open(fpath, "rb") as f:
                    self._send(200, f.read(), ctype)
            except OSError:
                self._error(404, "Нет файла " + fname)
            return

        self._error(404, "Не найдено")

    def do_HEAD(self):
        self.do_GET()

    def _send_file(self, path: str, download_name: str) -> None:
        try:
            size = os.path.getsize(path)
        except OSError:
            self._error(404, "Файл не найден")
            return
        from urllib.parse import quote

        self.send_response(200)
        self.send_header("Content-Type", "text/plain; charset=utf-8")
        self.send_header("Content-Length", str(size))
        self.send_header("Content-Disposition",
                         'attachment; filename="transcript.txt"; '
                         "filename*=UTF-8''" + quote(download_name, safe=""))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        if self.command == "HEAD":
            return
        try:
            with open(path, "rb") as f:
                shutil.copyfileobj(f, self.wfile)
        except (BrokenPipeError, ConnectionResetError, OSError):
            pass

    # -- POST --------------------------------------------------------------
    def do_POST(self):
        if not self._check_request():
            return
        parsed = urlparse(self.path)
        path = parsed.path

        if path == "/api/upload":
            self._handle_upload()
            return

        if path == "/api/glossary":
            body = self._read_body(limit=256 * 1024)
            try:
                data = json.loads(body or b"{}")
            except json.JSONDecodeError:
                data = {}
            if not isinstance(data, dict):
                data = {}
            g = text_to_glossary(str(data.get("text", "")))
            save_glossary(g)
            self._json({"ok": True, "terms": len(g["terms"]),
                        "replacements": len(g["replacements"])})
            return

        if path == "/api/apply-glossary":
            body = self._read_body(limit=64 * 1024)
            try:
                data = json.loads(body or b"{}")
            except json.JSONDecodeError:
                data = {}
            if not isinstance(data, dict):
                data = {}
            ok, result, count = apply_glossary_to_job(str(data.get("id", "")))
            if ok:
                self._json({"ok": True, "name": result, "count": count})
            else:
                self._error(400, result)
            return

        if path == "/api/rename":
            body = self._read_body(limit=64 * 1024)
            try:
                data = json.loads(body or b"{}")
            except json.JSONDecodeError:
                data = {}
            if not isinstance(data, dict):
                data = {}
            ok, result = rename_output(str(data.get("id", "")),
                                       str(data.get("name", "")))
            if ok:
                self._json({"ok": True, "name": result})
            else:
                self._error(400, result)
            return

        if path in ("/api/models/select", "/api/models/download"):
            body = self._read_body(limit=64 * 1024)
            try:
                data = json.loads(body or b"{}")
            except json.JSONDecodeError:
                data = {}
            model_id = str(data.get("id", "")) if isinstance(data, dict) else ""
            entry = catalog_entry(model_id)
            if not entry:
                self._error(400, "Неизвестная модель")
                return
            if path == "/api/models/download":
                if model_installed(model_id):
                    set_current_model(model_id)
                    self._json({"ok": True, "installed": True})
                elif start_download(model_id):
                    self._json({"ok": True, "started": True}, 202)
                else:
                    self._error(409, "Скачивание уже идёт")
                return
            if not model_installed(model_id):
                self._error(409, "Модель ещё не скачана")
                return
            set_current_model(model_id)
            self._json({"ok": True})
            return

        if path in ("/api/open-output", "/api/retry", "/api/reveal"):
            body = self._read_body(limit=64 * 1024)
            try:
                data = json.loads(body or b"{}")
            except json.JSONDecodeError:
                data = {}
            if not isinstance(data, dict):
                data = {}

            if path == "/api/open-output":
                self._open_path(OUTPUT_DIR)
                self._json({"ok": True})
                return

            if path == "/api/retry":
                if retry_job(str(data.get("id", ""))):
                    self._json({"ok": True})
                else:
                    self._error(404, "Повтор невозможен: исходный файл уже удалён")
                return

            with LOCK:
                job = JOBS.get(str(data.get("id", "")))
                out = job.get("output") if job else None
            if out and os.path.exists(out):
                self._open_path(os.path.dirname(out))
                self._json({"ok": True})
            else:
                self._error(404, "Файл не найден")
            return

        # неизвестный путь: обязательно закрываем соединение,
        # иначе невычитанное тело сломает следующий запрос
        self.close_connection = True
        self._error(404, "Не найдено")

    def _read_body(self, limit: int = 1024 * 1024) -> bytes:
        try:
            n = int(self.headers.get("Content-Length", "0"))
        except (TypeError, ValueError):
            self.close_connection = True
            return b""
        if n < 0 or n > limit:
            self.close_connection = True
            return b""
        return self.rfile.read(n) if n > 0 else b""

    def _handle_upload(self):
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except (TypeError, ValueError):
            self.close_connection = True
            self._error(400, "Некорректная длина тела запроса")
            return
        if length <= 0:
            self._error(400, "Пустой файл")
            return
        if length > MAX_UPLOAD_BYTES:
            self.close_connection = True
            self._error(413, "Файл слишком большой")
            return

        try:
            free = shutil.disk_usage(UPLOAD_DIR).free
        except OSError:
            free = None
        if free is not None and length + 512 * 1024 * 1024 > free:
            self.close_connection = True
            self._error(507, "На диске недостаточно места для этого файла")
            return

        raw_name = unquote(self.headers.get("X-Filename", "")).strip()
        name = os.path.basename(raw_name).replace("\x00", "") or "audio"
        if len(name.encode("utf-8", "ignore")) > 200:
            name = safe_stem(name) + os.path.splitext(name)[1][:10]
        timestamps = self.headers.get("X-Timestamps", "1") != "0"
        prompt = unquote(self.headers.get("X-Prompt", "")).strip()[:MAX_PROMPT_CHARS]

        tmp_path = os.path.join(UPLOAD_DIR, uuid.uuid4().hex[:12] + "__" + name)
        remaining = length
        try:
            with open(tmp_path, "wb") as f:
                while remaining > 0:
                    chunk = self.rfile.read(min(1024 * 1024, remaining))
                    if not chunk:
                        break
                    f.write(chunk)
                    remaining -= len(chunk)
        except (OSError, ValueError) as e:
            try:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
            except OSError:
                pass
            self.close_connection = True
            if getattr(e, "errno", None) == 28:
                self._error(507, "На диске закончилось место")
            else:
                self._error(500, "Не удалось сохранить файл")
            return

        if remaining > 0:
            try:
                os.remove(tmp_path)
            except OSError:
                pass
            self.close_connection = True
            self._error(400, "Загрузка прервалась — попробуйте ещё раз")
            return

        jid = enqueue(name, tmp_path, length, timestamps, prompt)
        self._json({"id": jid, "name": name}, 201)

    def _open_path(self, path: str) -> None:
        try:
            if sys.platform == "darwin":
                subprocess.Popen(["open", path])
            elif sys.platform.startswith("win"):
                os.startfile(path)  # type: ignore[attr-defined]
            else:
                subprocess.Popen(["xdg-open", path])
        except Exception:
            pass


class QuietServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    def handle_error(self, request, client_address):
        exc = sys.exc_info()[1]
        name = type(exc).__name__ if exc else "ошибка"
        sys.stderr.write(f"[whisper] клиент отключился ({name})\n")


# --------------------------------------------------------------------------
# Запуск
# --------------------------------------------------------------------------
INSTANCE_PORT = _int_env("WHISPER_INSTANCE_PORT", 47821, 1024)
_instance_socket = None


def acquire_single_instance() -> bool:
    """Не даёт запустить второй экземпляр приложения.

    Работает одинаково на macOS, Windows и Linux: занимаем локальный порт.
    Пока процесс жив, порт занят — повторный запуск это увидит.
    """
    global _instance_socket
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        s.bind(("127.0.0.1", INSTANCE_PORT))
        s.listen(1)
    except OSError:
        try:
            s.close()
        except OSError:
            pass
        return False
    _instance_socket = s
    return True


def existing_instance_url() -> str:
    try:
        with open(PORT_FILE, "r", encoding="utf-8") as f:
            return f"http://127.0.0.1:{int(f.read().strip())}/"
    except (OSError, ValueError):
        return "http://127.0.0.1:8765/"


def find_port(preferred: int = 8765) -> int:
    # стараемся занять тот же порт, что и в прошлый раз: тогда ссылка в браузере,
    # которую пользователь оставил открытой, продолжит работать
    try:
        with open(PORT_FILE, "r", encoding="utf-8") as f:
            last = int(f.read().strip())
        if 1024 <= last <= 65535:
            preferred = last
    except (OSError, ValueError):
        pass
    for p in range(preferred, preferred + 50):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(("127.0.0.1", p))
                return p
            except OSError:
                continue
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def main() -> None:
    global SERVER_PORT
    print("=" * 68)
    print(f"  {APP_NAME}  v{VERSION}")
    print("=" * 68)
    print(f"  Бэкенд   : {BACKEND_LABEL}")
    print(f"  Модель   : {MODEL}")
    print(f"  Язык     : {LANGUAGE}")
    print(f"  Фрагмент : {CHUNK_SECONDS} с (перекрытие {CHUNK_OVERLAP} с)")
    print("  Паузы    : " + ("Silero VAD — куски режутся по тишине"
                             if vad_available() else
                             "VAD недоступен, тишина по громкости"))
    print(f"  ffmpeg   : {FFMPEG or 'НЕ НАЙДЕН'}")
    print(f"  Результат: {OUTPUT_DIR}")
    free = available_memory_bytes()
    if free is not None:
        print(f"  Свободно : {free / 1024 ** 3:.1f} ГБ памяти")
        if free < 2 * 1024 ** 3:
            print("  ВНИМАНИЕ: мало свободной памяти. Закройте лишние программы.")
    if IMPORT_ERROR:
        print("\n  ВНИМАНИЕ:", IMPORT_ERROR)
        print("  Убедитесь, что запускаете через ../_venv/bin/python")

    # если выбранной модели нет в кеше — переключаемся на модель по умолчанию
    # (для текущего бэкенда: MLX и faster-whisper используют разные модели)
    if not model_installed(MODEL):
        fallback = DEFAULT_MODEL
        if MODEL != fallback and model_installed(fallback):
            print(f"\n  Модель {MODEL} не найдена в кеше — беру {fallback}")
            set_current_model(fallback)
        else:
            print(f"\n  ВНИМАНИЕ: модель {MODEL} не найдена в кеше.")
            print("  Подключите интернет и скачайте её в интерфейсе приложения.")
    print("-" * 68)

    if not acquire_single_instance():
        url = existing_instance_url()
        print("\n  Приложение уже запущено — второй экземпляр не нужен.")
        print(f"  Открываю работающее окно: {url}")
        if os.environ.get("WHISPER_NO_BROWSER") != "1":
            webbrowser.open(url)
        return

    ensure_glossary_file()
    cleanup_parts()
    cleanup_work_dir()
    load_jobs()
    register_outputs()
    with LOCK:
        keep = {j.get("src") for j in JOBS.values() if j.get("src")}
    cleanup_old_uploads(keep)

    threading.Thread(target=worker, daemon=True).start()
    threading.Thread(target=watchdog, daemon=True).start()

    port = find_port()
    SERVER_PORT = port
    url = f"http://127.0.0.1:{port}/"
    try:
        httpd = QuietServer(("127.0.0.1", port), Handler)
    except OSError as e:
        print(f"\n  Не удалось занять порт {port}: {e}")
        return

    try:
        with open(PORT_FILE, "w", encoding="utf-8") as f:
            f.write(str(port))
    except OSError:
        pass

    print(f"  Интерфейс открыт: {url}")
    print("  Это окно должно оставаться открытым, пока идёт работа.")
    print("  Закройте его, чтобы выключить приложение.")
    print("=" * 68)

    if os.environ.get("WHISPER_NO_BROWSER") != "1":
        threading.Timer(0.8, lambda: webbrowser.open(url)).start()
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nОстановлено.")
    finally:
        httpd.server_close()


if __name__ == "__main__":
    main()
