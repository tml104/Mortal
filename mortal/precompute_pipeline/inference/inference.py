"""只读取新 Conv2d checkpoint 的普通 mjai 协议入口。"""

import argparse
from pathlib import Path
import sys

MORTAL_DIR = Path(__file__).resolve().parents[2]
if str(MORTAL_DIR) not in sys.path:
    sys.path.insert(0, str(MORTAL_DIR))

from libriichi.mjai import Bot
from precompute_pipeline.common.policy import create_engine, policy_settings
from precompute_pipeline.common.runtime import read_toml, configure_threads, optional_threads


def run_inference(settings, player_id, input_stream=None, output_stream=None):
    """stdout/输出流仅用于协议响应，不导入旧全局训练配置。"""
    if type(player_id) is not int or not 0 <= player_id < 4:
        raise ValueError('player_id 必须位于 0..3')
    engine, _ = create_engine(settings)
    bot = Bot(engine, player_id)
    input_stream = sys.stdin if input_stream is None else input_stream
    output_stream = sys.stdout if output_stream is None else output_stream
    for line in input_stream:
        line = line.strip()
        if line and (reaction := bot.react(line)):
            print(reaction, file=output_stream, flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description='使用原生 Conv2d 模型进行普通 mjai 推理')
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('player_id', type=int, choices=range(4))
    args = parser.parse_args(argv)
    raw = read_toml(args.config)['inference']
    configure_threads(optional_threads(raw, 'torch_threads'), optional_threads(raw, 'rayon_threads'))
    run_inference(policy_settings(raw), args.player_id)


if __name__ == '__main__':
    main()
