"""Сравнение всех моделей на одном файле с одинаковыми параметрами (beam=5, VAD)."""
import os
import time
from faster_whisper import WhisperModel

AUDIO = os.path.join(os.environ["LOCALAPPDATA"], "Temp", "giga_test.mp3")
MODELS = [
    "Systran/faster-whisper-tiny",
    "Systran/faster-whisper-base",
    "Systran/faster-whisper-small",
    "deepdml/faster-whisper-large-v3-turbo-ct2",
]

print(f"файл: {AUDIO}", flush=True)
print(f"{'модель':45} {'load':>8} {'transcribe':>11} {'символов':>9}", flush=True)
for mid in MODELS:
    try:
        t0 = time.time()
        m = WhisperModel(mid, device="cpu", compute_type="int8")
        load = time.time() - t0
        t1 = time.time()
        segs, info = m.transcribe(AUDIO, language="ru", beam_size=5, vad_filter=True)
        text = " ".join(s.text.strip() for s in segs)
        dt = time.time() - t1
        print(f"{mid.split('/')[-1]:45} {load:7.1f}s {dt:10.1f}s {len(text):8}", flush=True)
    except Exception as e:
        print(mid, "ОШИБКА:", str(e)[:100], flush=True)
