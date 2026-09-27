"""Честный замер на РЕАЛЬНОЙ речи: все модели, одинаковые параметры (beam=5, VAD, int8)."""
import os
import time
from faster_whisper import WhisperModel

AUDIO = os.path.join(os.environ["LOCALAPPDATA"], "Temp", "ru_wiki_speech.ogg")
MODELS = [
    "Systran/faster-whisper-tiny",
    "Systran/faster-whisper-base",
    "Systran/faster-whisper-small",
    "deepdml/faster-whisper-large-v3-turbo-ct2",
]

print(f"файл: {AUDIO}", flush=True)
results = []
for mid in MODELS:
    try:
        t0 = time.time()
        m = WhisperModel(mid, device="cpu", compute_type="int8")
        load = time.time() - t0
        t1 = time.time()
        segs, info = m.transcribe(AUDIO, language="ru", beam_size=5, vad_filter=True)
        text = " ".join(s.text.strip() for s in segs)
        dt = time.time() - t1
        dur = info.duration
        rtf = dt / dur
        results.append((mid.split("/")[-1], dt, rtf, len(text), text))
        print(f"\n=== {mid.split('/')[-1]} ===", flush=True)
        print(f"аудио: {dur:.1f} с | распознавание: {dt:.1f} с | RTF: {rtf:.2f}", flush=True)
        print(f"символов: {len(text)}", flush=True)
        print(f"фрагмент: {text[:220]}", flush=True)
    except Exception as e:
        print(mid, "ОШИБКА:", str(e)[:150], flush=True)

print("\n\n========== ИТОГО (по скорости) ==========", flush=True)
base_dt = None
for name, dt, rtf, n, _ in sorted(results, key=lambda r: r[1]):
    print(f"{name:45} {dt:7.1f} с  RTF {rtf:.2f}  {n} символов", flush=True)
