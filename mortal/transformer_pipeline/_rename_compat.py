"""仅为精确匹配的脚本改名版本保留历史恢复契约身份。

历史名称仅用于持久化契约，不是可执行入口。任何额外源码变化都会使用真实身份，
由原有恢复校验拒绝不兼容版本。映射不包含配置、权重、数据或运行时的豁免。
"""

import hashlib
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
# 新路径 -> (历史路径、改名后到改名前 SHA256 的精确映射)。
# 预计算脚本另记录实施前已有注释的工作区版本；提交版本和该版本各自保留原摘要。
RENAME_IDENTITIES = {
    'mortal/reward_calculator.py': (
        'mortal/reward_calculator.py',
        {
            '87b314ed4412210583e1731908ed7485daa79b95aaeb30117d9b5f8e6965f832':
                'fbe8e96c3ab6640683adf943cd494688b91881aff2f080093a3b3fda1a5d344d',
        },
    ),
    'mortal/transformer_pipeline/dataloader_with_grp_transformer.py': (
        'mortal/transformer_pipeline/dataloader_grp_transformer.py',
        {
            '196d4a8913d7bd36b471f676d0cd069e4fdf0b452e84f4225bf4e6309db97340':
                'f5347c088e19c4118701b2a739eb5549354f5020ac017ecb7ea17fc9e6dc2344',
        },
    ),
    'mortal/transformer_pipeline/precompute_grp_transformer_cache.py': (
        'mortal/transformer_pipeline/precompute_grp_transformer_cache.py',
        {
            'fb6cb019c2fcb426702dac1fd6e728c1d8cd7ea73662a97ccd4312ad9d1e1ff1':
                'cc6ce5c5bbd9a9c58e468000708daef31e3acec8bfdfefdb89832ef96cd50c39',
            '971b1392e7e8955c62e04d47cb4392be65a9132c3c312c0d59c4c169ec547179':
                '73e55f0c0dff1c4a5ae66aadef01ae71c9ac280283f5460f2dfa92ad8b01996b',
        },
    ),
    'mortal/transformer_pipeline/precomputed_cache.py': (
        'mortal/transformer_pipeline/precomputed_cache.py',
        {
            '93350138a5a9d6ce45161f5ea12b8494a0980cc7a1a23ddd9f717fa30389b877':
                'a6413ca8504d79b96d0ae6466852a92a0eb19b961973b3f1edfdc23cb010f46e',
        },
    ),
    'mortal/transformer_pipeline/train_mortal_offline_with_grp_transformer.py': (
        'mortal/transformer_pipeline/train_offline_grp_transformer.py',
        {
            '0b18234eb91bcc44d1bb1a2f96dbdef9e9f3787a4eb3af9ffe314c3aeba22195':
                '40e4c11c3179791071fd99130db7651d5a73fa5794c01258e8644b5a6171bfcc',
        },
    ),
    'mortal/transformer_pipeline/train_mortal_offline_with_grp_transformer_bs1024.py': (
        'mortal/transformer_pipeline/train_offline_grp_transformer_bs1024.py',
        {
            '7c4b569ca6411cc618bf33a778e6ddbddc21714148412800a6f2a215ecd3e539':
                '1d6c9edacbf467e8e72c67e62177a41a0500204080298712b06cab9109b77907',
        },
    ),
    'mortal/transformer_pipeline/train_with_precomputed_cache.py': (
        'mortal/transformer_pipeline/train_precomputed_cache.py',
        {
            '7eb99aff69e720d55982ac511ec01c83f2a0d1b3a672a87d1868257e44f6f9d8':
                'c268e0fd1a4130efb2fd82f0854b571633838fd8d48aead7050eb545858780a4',
        },
    ),
}


def source_identity(filename, name=None):
    """为已核验的纯改名源码返回历史身份，其他文件保留真实路径和摘要。"""
    filename = Path(filename).resolve()
    with filename.open('rb') as source:
        digest = hashlib.file_digest(source, 'sha256').hexdigest()
    try:
        relative = filename.relative_to(REPOSITORY_ROOT).as_posix()
    except ValueError:
        relative = None
    identity = RENAME_IDENTITIES.get(relative)
    if identity is not None and digest in identity[1]:
        old_path, digests = identity
        filename = REPOSITORY_ROOT / old_path
        return filename.name, {'path': str(filename), 'sha256': digests[digest]}
    return name or filename.name, {'path': str(filename), 'sha256': digest}
