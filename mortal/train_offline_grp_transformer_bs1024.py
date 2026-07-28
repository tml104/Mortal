import logging
import os
from pathlib import Path
import random
import sys
import tomllib


EXPECTED_BATCH_SIZE = 1024
OUTPUT_DIR = Path('artifacts/mortal_transformer_offline_bs1024_v1').resolve()


def load_launcher_config():
    config_file = os.environ.get('MORTAL_CFG')
    if not config_file:
        raise RuntimeError('MORTAL_CFG must point to the bs1024 config')
    with open(config_file, 'rb') as f:
        return Path(config_file).resolve(), tomllib.load(f)


CONFIG_FILE, LAUNCH_CONFIG = load_launcher_config()
RAYON_NUM_THREADS = int(LAUNCH_CONFIG['dataset']['rayon_num_threads'])
MODEL_INIT_SEED = int(LAUNCH_CONFIG['control']['epoch_seed'])
os.environ['RAYON_NUM_THREADS'] = str(RAYON_NUM_THREADS)

import numpy as np
import torch

import train_offline_grp_transformer as trainer


def validate_launcher_config():
    if LAUNCH_CONFIG['control']['batch_size'] != EXPECTED_BATCH_SIZE:
        raise ValueError(f'batch_size must be {EXPECTED_BATCH_SIZE}')
    if RAYON_NUM_THREADS <= 0:
        raise ValueError('dataset.rayon_num_threads must be positive')

    output_paths = (
        LAUNCH_CONFIG['control']['state_file'],
        LAUNCH_CONFIG['control']['completion_file'],
        LAUNCH_CONFIG['control']['run_log_file'],
        LAUNCH_CONFIG['control']['tensorboard_dir'],
        LAUNCH_CONFIG['grp_transformer']['state_file'],
    )
    output_prefix = f'{OUTPUT_DIR}{os.sep}'
    for filename in output_paths:
        resolved = Path(filename).resolve()
        if resolved != OUTPUT_DIR and not str(resolved).startswith(output_prefix):
            raise ValueError(f'bs1024 output escapes its run directory: {resolved}')


def install_reproducibility_contract():
    original = trainer.build_implementation_manifest

    def build_implementation_manifest():
        manifest = original()
        manifest['files'][Path(__file__).name] = {
            'path': str(Path(__file__).resolve()),
            'sha256': trainer.sha256_file(__file__),
        }
        manifest['runtime']['model_init_seed'] = MODEL_INIT_SEED
        manifest['runtime']['rayon_num_threads'] = RAYON_NUM_THREADS
        return manifest

    trainer.build_implementation_manifest = build_implementation_manifest


def seed_model_initialization():
    random.seed(MODEL_INIT_SEED)
    np.random.seed(MODEL_INIT_SEED)
    torch.manual_seed(MODEL_INIT_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(MODEL_INIT_SEED)


def main():
    validate_launcher_config()
    install_reproducibility_contract()
    seed_model_initialization()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    try:
        trainer.main()
    finally:
        if torch.cuda.is_available() and torch.cuda.is_initialized():
            logging.info(
                'CUDA peak memory: allocated=%.3f GiB reserved=%.3f GiB',
                torch.cuda.max_memory_allocated() / 1024 ** 3,
                torch.cuda.max_memory_reserved() / 1024 ** 3,
            )


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        logging.exception('Training interrupted; resume from the latest completed file shard')
        sys.exit(130)
