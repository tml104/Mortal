"""为直接执行 Transformer 实验脚本准备路径和迁移兼容常量。

原始 Mortal Python 文件都位于 ``mortal/``，并使用 ``import config``、
``from model import Brain`` 这类平铺导入。脚本原先也在该目录中，执行
``python mortal/example.py`` 时，Python 会自动把脚本目录 ``mortal/`` 放进
``sys.path``，因此这些导入可以直接找到相邻模块。

Transformer 实验链路迁移到 ``mortal/transformer_pipeline/`` 后，直接执行脚本时
自动加入 ``sys.path`` 的变成了这个子目录，位于上一级的 ``config.py``、``model.py``
和 ``libriichi`` 扩展将无法找到。本模块必须在这些本地模块之前导入；导入时把
``mortal/`` 放回模块搜索路径，同时提供路径常量给数据隔离和实现摘要逻辑使用。

本模块还保存目录迁移前后的精确源码 SHA256。离线训练器和预计算缓存训练器用这些
常量确认旧断点只跨越了已知的目录迁移，而不是混入其他代码变化。这里不加载配置、
模型或数据，也不会启动训练。
"""

import sys
from pathlib import Path


# 所有目录都从当前文件的真实位置推导，不依赖进程的当前工作目录。这样即使调用方
# 使用绝对脚本路径，或先切换到其他目录，源码与数据目录的定位仍然保持一致。
PIPELINE_DIR = Path(__file__).resolve().parent  # mortal/transformer_pipeline/
MORTAL_DIR = PIPELINE_DIR.parent  # mortal/
REPOSITORY_ROOT = MORTAL_DIR.parent  # 仓库根目录

# ``sys.path`` 只接受字符串形式的目录。插入到索引 0 可确保优先使用当前仓库中的
# config、model、libriichi 等模块，而不是环境里碰巧存在的同名第三方模块。先检查可
# 避免多个实验模块重复导入本文件时不断插入相同目录。
mortal_path = str(MORTAL_DIR)
if mortal_path not in sys.path:
    sys.path.insert(0, mortal_path)

# 离线 Transformer-GRP 主策略训练的迁移前摘要。训练 checkpoint 的实现清单同时
# 记录文件路径和内容摘要；两个文件从 mortal/ 移到 transformer_pipeline/ 后，这两项
# 都会变化。验证器只接受这里列出的旧摘要，不能用任意旧版本触发迁移豁免。
PRE_PIPELINE_IMPLEMENTATION_SHA256 = {
    'train_offline_grp_transformer.py': (
        '3c585314559b0e90165124de2d6b9f33e16e9172ae04d9e2284f1b25566126cb'
    ),
    'dataloader_grp_transformer.py': (
        'e7a81f6b3b6aeb7f398dbd001143eca15cc6bd1023aba48656bc7365d75e9cba'
    ),
}

# 预计算缓存训练使用独立 checkpoint schema，只在运行契约中记录训练器自身摘要，
# 因而单独保存迁移前的 train_precomputed_cache.py 摘要。
PRE_PIPELINE_CACHE_TRAINER_SHA256 = (
    '0514017801fb712b55c706309d24837c16380cc882e2f541a8212b2ee9e528bf'
)

# 与上面的旧摘要成对记录本次迁移完成后的精确文件内容。恢复时必须同时命中旧值和
# 新值，并且配置、数据、运行时及其他实现文件完全一致，才会接受跨目录恢复。以后若
# 修改这些受保护文件，其摘要将自然变化，因此不会误用本次迁移豁免。
POST_PIPELINE_IMPLEMENTATION_SHA256 = {
    'train_offline_grp_transformer.py': (
        '40e4c11c3179791071fd99130db7651d5a73fa5794c01258e8644b5a6171bfcc'
    ),
    'dataloader_grp_transformer.py': (
        'f5347c088e19c4118701b2a739eb5549354f5020ac017ecb7ea17fc9e6dc2344'
    ),
}

# 预计算缓存训练器迁移完成后的对应摘要；用途与上面的 POST 映射相同。
POST_PIPELINE_CACHE_TRAINER_SHA256 = (
    'c268e0fd1a4130efb2fd82f0854b571633838fd8d48aead7050eb545858780a4'
)
