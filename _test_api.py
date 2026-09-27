# -*- coding: utf-8 -*-
"""Полный прогон API Whisper-транскрайбера (Windows-проверка)."""
import json
import os
import urllib.request
import urllib.error
import time

PORT = open(os.path.join(os.path.dirname(__file__), 'data', 'port.txt')).read().strip()
BASE = 'http://127.0.0.1:%s' % PORT


def req(method, path, body=None, headers=None, timeout=60):
    h = {'Content-Type': 'application/json'}
    if headers:
        h.update(headers)
    data = None
    if body is not None:
        data = body if isinstance(body, bytes) else json.dumps(body).encode('utf-8')
    r = urllib.request.Request(BASE + path, data=data, headers=h, method=method)
    try:
        with urllib.request.urlopen(r, timeout=timeout) as resp:
            raw = resp.read()
            try:
                return resp.status, json.loads(raw.decode('utf-8'))
            except Exception:
                return resp.status, raw[:200]
    except urllib.error.HTTPError as e:
        return e.code, e.read()[:300].decode('utf-8', 'replace')
    except Exception as e:
        return 0, str(e)[:200]


results = []
def check(name, ok, detail=''):
    results.append((name, ok, detail))
    print(('  OK  ' if ok else ' FAIL ') + name + (' — ' + str(detail)[:90] if detail else ''))


print('=== API ENDPOINTS ===')
code, info = req('GET', '/api/info')
check('GET /api/info', code == 200 and info.get('version'), 'v%s, backend %s, platform %s' % (info.get('version'), info.get('backend'), info.get('platform')) if isinstance(info, dict) else info)

code, jobs = req('GET', '/api/jobs')
check('GET /api/jobs', code == 200 and 'jobs' in jobs, 'задач: %s' % jobs.get('total') if isinstance(jobs, dict) else jobs)

code, g = req('GET', '/api/glossary')
check('GET /api/glossary', code == 200, 'терминов: %s' % g.get('terms') if isinstance(g, dict) else g)

code, m = req('GET', '/api/models')
check('GET /api/models', code == 200, (str(m)[:80] if code != 200 else 'каталог получен'))

code, st = req('POST', '/api/glossary', {'text': 'Проверка словаря: гигел, Дигория, фасилитатор'})
check('POST /api/glossary (текст→словарь)', code == 200 and st.get('ok'), st)

print()
print('=== БЕЗОПАСНОСТЬ ===')
code, _ = req('GET', '/api/jobs', headers={'Host': 'evil.example.com'})
check('Отказ чужому Host', code == 403, 'код %s' % code)
code, _ = req('GET', '/api/jobs', headers={'Origin': 'http://evil.example.com'})
check('Отказ чужому Origin', code == 403, 'код %s' % code)

print()
print('=== ОШИБОЧНЫЕ СЦЕНАРИИ ===')
code, e1 = req('POST', '/api/upload', b'', headers={'Content-Type': 'application/octet-stream', 'X-Filename': 'empty.wav'})
check('Пустой файл отклонён', code in (400, 413), 'код %s: %s' % (code, e1))

code, dl = req('GET', '/api/download?id=nonexistent123')
check('Скачивание несуществующего', code in (400, 404), 'код %s' % code)

code, rn = req('POST', '/api/rename', {'id': 'несуществующий', 'name': 'x'})
check('Переименование несуществующего', code in (400, 404), 'код %s' % code)

code, rt = req('POST', '/api/retry', {'id': 'несуществующий'})
check('Повтор несуществующего', code in (400, 404), 'код %s' % code)

print()
ok = sum(1 for _, o, _ in results if o)
print('=== ИТОГ: %d/%d ===' % (ok, len(results)))
for n, o, d in results:
    if not o:
        print('  ПРОВАЛ:', n, '—', d)
