"""多个入口共用的文件发现与原子提交操作。"""

import json
import os
from pathlib import Path


def iter_dataset_files(root):
    """按相对 POSIX 路径返回目录中的 gzip mjai 牌谱。"""
    root = Path(root)
    files = (path for path in root.rglob('*.json.gz') if path.is_file())
    return sorted(files, key=lambda path: path.relative_to(root).as_posix())


def atomic_write_json(filename, value):
    """在目标目录原子替换小型 JSON 状态。"""
    filename = Path(filename)
    filename.parent.mkdir(parents=True, exist_ok=True)
    temporary = filename.with_name(f'{filename.name}.tmp-{os.getpid()}')
    try:
        with open(temporary, 'w', encoding='utf-8') as f:
            json.dump(value, f, ensure_ascii=False, indent=2, sort_keys=True)
            f.write('\n')
            f.flush()
            os.fsync(f.fileno())
        os.replace(temporary, filename)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_write_bytes(filename, value):
    """在目标文件系统写入 bytes 后原子发布文件。"""
    filename = Path(filename)
    filename.parent.mkdir(parents=True, exist_ok=True)
    temporary = filename.with_name(f'{filename.name}.tmp-{os.getpid()}')
    try:
        with open(temporary, 'wb') as f:
            f.write(value)
            f.flush()
            os.fsync(f.fileno())
        os.replace(temporary, filename)
    finally:
        temporary.unlink(missing_ok=True)
