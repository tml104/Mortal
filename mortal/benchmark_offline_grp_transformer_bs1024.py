import argparse
import gc
import json
import os
import random
import time

from config import config
from train_offline_grp_transformer import create_data_loader, load_file_list


CANDIDATES = {
    'A': {'workers': 2, 'file_batch_size': 6, 'rayon_threads': 6},
    'B': {'workers': 3, 'file_batch_size': 4, 'rayon_threads': 4},
    'C': {'workers': 4, 'file_batch_size': 3, 'rayon_threads': 3},
}


def read_meminfo():
    values = {}
    with open('/proc/meminfo', encoding='ascii') as f:
        for line in f:
            key, value = line.split(':', 1)
            values[key] = int(value.strip().split()[0]) * 1024
    return values


def process_rss(pid):
    try:
        with open(f'/proc/{pid}/status', encoding='ascii') as f:
            for line in f:
                if line.startswith('VmRSS:'):
                    return int(line.split()[1]) * 1024
    except FileNotFoundError:
        return 0
    return 0


def cpu_snapshot():
    with open('/proc/stat', encoding='ascii') as f:
        fields = [int(value) for value in f.readline().split()[1:]]
    idle = fields[3] + fields[4]
    return sum(fields), idle


def benchmark_candidate(name, files, warmup_batches, measure_batches, seed):
    candidate = CANDIDATES[name]
    config['dataset']['file_batch_size'] = candidate['file_batch_size']
    os.environ['RAYON_NUM_THREADS'] = str(candidate['rayon_threads'])
    loader = create_data_loader(files, [], seed, num_workers=candidate['workers'])
    iterator = iter(loader)
    worker_pids = [worker.pid for worker in getattr(iterator, '_workers', ())]

    try:
        for _ in range(warmup_batches):
            next(iterator)

        start_cpu = cpu_snapshot()
        started = time.perf_counter()
        samples = 0
        batches = 0
        min_available = read_meminfo()['MemAvailable']
        min_swap_free = read_meminfo()['SwapFree']
        max_worker_rss = 0
        elapsed = None
        end_cpu = None
        for batch in iterator:
            if measure_batches > 0 and batches >= measure_batches:
                continue
            samples += batch[0].shape[0]
            batches += 1
            meminfo = read_meminfo()
            min_available = min(min_available, meminfo['MemAvailable'])
            min_swap_free = min(min_swap_free, meminfo['SwapFree'])
            max_worker_rss = max(
                max_worker_rss,
                sum(process_rss(pid) for pid in worker_pids),
            )
            if measure_batches > 0 and batches == measure_batches:
                elapsed = time.perf_counter() - started
                end_cpu = cpu_snapshot()
        if elapsed is None:
            elapsed = time.perf_counter() - started
            end_cpu = cpu_snapshot()
    finally:
        del iterator, loader
        gc.collect()

    total_delta = end_cpu[0] - start_cpu[0]
    idle_delta = end_cpu[1] - start_cpu[1]
    cpu_percent = 100 * (1 - idle_delta / total_delta)
    return {
        'candidate': name,
        **candidate,
        'batches': batches,
        'samples': samples,
        'elapsed_seconds': elapsed,
        'samples_per_second': samples / elapsed,
        'cpu_percent': cpu_percent,
        'min_mem_available_gib': min_available / 1024 ** 3,
        'min_swap_free_gib': min_swap_free / 1024 ** 3,
        'max_worker_rss_gib': max_worker_rss / 1024 ** 3,
    }


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--candidates', nargs='+', choices=sorted(CANDIDATES), default=['A', 'B'])
    parser.add_argument('--files', type=int, default=384)
    parser.add_argument('--warmup-batches', type=int, default=4)
    parser.add_argument('--measure-batches', type=int, default=128)
    return parser.parse_args()


def main():
    args = parse_args()
    config['grp_transformer']['state_file'] = config['grp_transformer']['source_state_file']
    files = load_file_list(config['dataset'])
    seed = config['control']['epoch_seed']
    random.Random(seed).shuffle(files)
    files = files[:args.files]

    results = {}
    for name in args.candidates:
        if (
            name == 'C'
            and 'B' in results
            and results['B']['min_mem_available_gib'] < 4
        ):
            print(json.dumps({'candidate': 'C', 'skipped': 'candidate B left less than 4 GiB available'}))
            continue
        result = benchmark_candidate(
            name,
            files,
            args.warmup_batches,
            args.measure_batches,
            seed,
        )
        results[name] = result
        print(json.dumps(result, sort_keys=True), flush=True)


if __name__ == '__main__':
    main()
