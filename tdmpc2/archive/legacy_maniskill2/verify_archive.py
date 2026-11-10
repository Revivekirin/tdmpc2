"""Verify every original experimental artifact against the pre-migration snapshot."""
import csv
import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

def verify():
    checked = 0
    for row in csv.DictReader((Path(__file__).parent / 'migration_manifest.csv').open()):
        if not row['sha256']:
            continue  # Already absent before this task.
        for key in ('original_path', 'proposed_destination'):
            path = ROOT / row[key]
            assert path.is_file(), f'Missing {path}'
            assert hashlib.sha256(path.read_bytes()).hexdigest() == row['sha256'], f'Changed {path}'
        checked += 1
    print(f'PASS: {checked} original files accessible with unchanged SHA-256 at both paths')

if __name__ == '__main__':
    verify()
