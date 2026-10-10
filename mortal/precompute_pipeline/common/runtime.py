"""新训练、推理与评测共用的配置基本类型及运行线程设置。"""

from pathlib import Path
import tomllib


def read_toml(filename):
    """独立入口只解析各自配置，不触发旧全局 config 导入。"""
    with Path(filename).open('rb') as stream:
        return tomllib.load(stream)


def positive_int(section, key, default):
    value = section.get(key, default)
    if type(value) is not int or value <= 0:
        raise ValueError(f'{key} 必须是正整数')
    return value


def boolean(section, key, default):
    value = section.get(key, default)
    if type(value) is not bool:
        raise ValueError(f'{key} 必须是布尔值')
    return value


def path_value(section, key):
    value = section.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f'{key} 必须是非空路径')
    return Path(value).expanduser().resolve()


def optional_threads(section, key):
    return positive_int(section, key, None) if key in section else None


def configure_threads(torch_threads=None, rayon_threads=None, compile_threads=None):
    """对应运行时首次使用前设置线程；省略即沿用库默认。"""
    import os
    if rayon_threads is not None:
        os.environ['RAYON_NUM_THREADS'] = str(rayon_threads)
    if torch_threads is not None:
        import torch
        torch.set_num_threads(torch_threads)
    if compile_threads is not None:
        import torch._inductor.config
        torch._inductor.config.compile_threads = compile_threads
