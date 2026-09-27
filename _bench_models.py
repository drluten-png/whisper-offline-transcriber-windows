# -*- coding: utf-8 -*-
"""Замер скорости распознавания разными моделями на одном файле."""
import os
import sys
import time

sys.path.insert(0, '.')
os.environ.setdefault('WHISPER_NO_BROWSER', '1')

from faster_whisper import WhisperModel

AUDIO = os.path.join(os.environ['LOCALAPPDATA'], 'Temp', 'giga_test.mp3')
if not os.path.exists(AUDIO):
    AUDIO = r"C:\Users\drlut\Downloads\Giga chat_suno v6.mp3"

for model_id in ['Systran/faster-whisper-base', 'Systran/faster-whisper-small']:
    try:
        t0 = time.time()
        m = WhisperModel(model_id, device='cpu', compute_type='int8')
        t_load = time.time() - t0
        for beam in (5, 1):
            t0 = time.time()
            segs, info = m.transcribe(AUDIO, language='ru', beam_size=beam,
                                      vad_filter=True)
            text = ' '.join(s.text.strip() for s in segs)
            dt = time.time() - t0
            print('%-42s beam=%d | загрузка %.1fс | распознавание %.1fс | текст %d симв.'
                  % (model_id.split('/')[-1], beam, t_load, dt, len(text)))
            print('   пример:', text[:130].replace('\n', ' '))
    except Exception as e:
        print(model_id, 'ОШИБКА:', str(e)[:150])
