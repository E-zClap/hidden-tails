"""Download the public original rejected snapshot for a read-only support audit."""
from pathlib import Path
import hashlib
import json
import time
import urllib.request

ROOT = Path(__file__).resolve().parent
URL = 'https://www.kaggle.com/api/v1/datasets/download/wordsforthewise/lending-club/rejected_2007_to_2018Q4.csv.gz?datasetVersionNumber=3'
TARGET = ROOT / 'rejected_2007_to_2018Q4.csv.gz'
if TARGET.exists():
    raise SystemExit('Refusing to overwrite an existing downloaded source.')
started = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
request = urllib.request.Request(URL)
digest = hashlib.sha256()
count = 0
with urllib.request.urlopen(request, timeout=60) as response:
    expected = response.headers.get('Content-Length')
    content_type = response.headers.get('Content-Type')
    # The resolved signed storage URL is deliberately not persisted.
    with TARGET.open('xb') as handle:
        while True:
            block = response.read(4 * 1024 * 1024)
            if not block:
                break
            handle.write(block)
            digest.update(block)
            count += len(block)
            if count % (32 * 1024 * 1024) == 0:
                print(f'Downloaded {count:,} bytes', flush=True)
if expected and count != int(expected):
    raise RuntimeError(f'Content length mismatch: {count} != {expected}')
metadata = {
    'source_url': URL,
    'source_dataset': 'wordsforthewise/lending-club',
    'filename': TARGET.name,
    'retrieval_started_utc': started,
    'retrieval_completed_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
    'bytes': count,
    'http_content_length': expected,
    'http_content_type': content_type,
    'sha256': digest.hexdigest(),
    'leading_magic_hex': TARGET.open('rb').read(8).hex(),
    'purpose': 'Read-only historical support/measurement audit; no model fitting',
}
(ROOT / 'download_provenance.json').write_text(json.dumps(metadata, indent=2) + '\n', encoding='utf-8')
print(json.dumps(metadata, indent=2), flush=True)
