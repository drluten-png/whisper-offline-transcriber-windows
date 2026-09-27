"""Замер turbo на том же файле, что и tiny/base/small (2:55)."""
import os
import time
from faster_whisper import WhisperModel

AUDIO = os.path.join(os.environ["LOCALAPPDATA"], "Temp", "giga_test.mp3")
MODEL_ID = "deepdml/faster-whisper-large-v3-turbo-ct2"

print("качаю turbo (~1.5 ГБ, разово)...", flush=True)
t0 = time.time()
m = WhisperModel(MODEL_ID, device="cpu", compute_type="int8")
print(f"модель готова за {time.time() - t0:.0f} с", flush=True)

for beam in (5, 1):
    t0 = time.time()
    segs, info = m.transcribe(AUDIO, language="ru", beam_size=beam, vad_filter=True)
    text = " ".join(s.text.strip() for s in segs)
    dt = time.time() - t0
    print(f"turbo beam={beam} | {dt:.1f} с | символов {len(text)}", flush=True)
    print("   ", text[:130].replace("\n", " "), flush=True)
