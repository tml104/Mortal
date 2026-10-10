"""用正式入口比较实现前快照与当前版本；仅生成自动清理的临时 chunks。"""

import argparse
from contextlib import redirect_stdout
from dataclasses import replace
import gc
import hashlib
import io
import json
import os
from pathlib import Path
import resource
import shlex
import sys
import tempfile
import time


REPOSITORY = Path('/home/tml104/Mortal')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--baseline-root', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    baseline = args.baseline_root is not None
    sys.path.insert(0, str(REPOSITORY / 'mortal'))
    if baseline:
        sys.path.insert(0, str(args.baseline_root / 'mortal'))
    import zstandard as zstd
    from precompute_pipeline.common.settings import load_settings
    from precompute_pipeline.precompute.precompute import _build_generator, run_precompute

    manifest = REPOSITORY / '.scratch/new-precompute-pipeline/precompute-performance-inputs.json'
    selected = json.loads(manifest.read_text())[:128]
    original = load_settings(REPOSITORY / 'mortal/precompute_pipeline/config.toml').precompute
    with (
        tempfile.TemporaryDirectory(prefix='precompute-acceptance-') as local,
        tempfile.TemporaryDirectory(prefix='precompute-acceptance-', dir='/mnt/d') as drive,
    ):
        root = Path(local)
        for split in ('train', 'val'):
            (root / 'input' / split).mkdir(parents=True)
        for index, filename in enumerate(selected):
            (root / 'input/train' / f'{index:06}.json.gz').symlink_to(filename)
        settings = replace(
            original,
            train_dir=root / 'input/train', val_dir=root / 'input/val',
            logical_root=root / 'logical', progress_file=root / 'logical/progress.json',
            c_root=root / 'c', d_root=Path(drive) / 'd',
            c_budget_bytes=2**30, d_budget_bytes=2**30,
            rayon_threads=4 if baseline else 16,
            torch_threads=4 if baseline else 1,
            file_batch_size=4,
        )
        generator = _build_generator(settings)
        generator.load_files(selected[:4])
        gc.collect()
        output = io.StringIO()
        cpu_start = time.process_time()
        with redirect_stdout(output):
            started = time.monotonic()
            state = run_precompute(settings, generator=generator)
            elapsed = time.monotonic() - started
        cpu_seconds = time.process_time() - cpu_start
        peak_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2
        records = [
            dict(item.split('=', 1) for item in shlex.split(line)[1:])
            for line in output.getvalue().splitlines()
            if line.startswith('[precompute] ')
        ]
        digests = []
        for filename in sorted((settings.d_root / 'train').glob('chunk_*.mpc.zst')):
            payload = zstd.ZstdDecompressor().decompress(filename.read_bytes())
            digests.append(hashlib.sha256(payload).hexdigest())
        last = records[-1] if records else {}
        stages = {key: float(last[key + '_s']) for key in ('rust','grp','python','encode','commit')} if last else {}
        result = {
            'variant': 'baseline' if baseline else 'implemented',
            'rayon_threads': settings.rayon_threads,
            'torch_threads': settings.torch_threads,
            'file_batch_size': settings.file_batch_size,
            'files': len(selected),
            'samples': state['splits']['train']['sample_count'],
            'chunks': state['splits']['train']['chunk_count'],
            'seconds': elapsed,
            'samples_per_second': state['splits']['train']['sample_count'] / elapsed,
            'cpu_percent': cpu_seconds / elapsed / (os.cpu_count() or 1) * 100,
            'peak_rss_gib': peak_rss,
            'stages_seconds': stages,
            'compressed_bytes': sum(state['volume_bytes'].values()),
            'payload_sha256': digests,
            'log_lines': len(records),
            'last_log': last,
        }
        if not baseline:
            assert last['event'] == 'complete'
            assert last['files'] == '0/0'
            assert all(value > 0 for value in stages.values())
            assert len([r for r in records if r['event'] == 'chunk_committed']) == result['chunks']
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open('a') as handle:
            handle.write(json.dumps(result) + '\n')
        print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
