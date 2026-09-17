"""脚本改名保留恢复契约，但不豁免未知源码修改。"""

import hashlib
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _rename_compat import source_identity


class ScriptRenameTests(unittest.TestCase):
    def test_renamed_trainer_keeps_checkpoint_identity(self):
        """改名后的训练入口仍能产生历史断点中的实现身份。"""
        filename = Path(__file__).resolve().parents[1] / 'train_mortal_offline_with_grp_transformer.py'
        name, record = source_identity(filename)
        self.assertEqual(name, 'train_offline_grp_transformer.py')
        self.assertEqual(Path(record['path']), filename.with_name(name))
        self.assertEqual(record['sha256'], '40e4c11c3179791071fd99130db7651d5a73fa5794c01258e8644b5a6171bfcc')

    def test_unrecognized_source_uses_real_identity(self):
        """同名的未知源码不能借重命名豁免绕过摘要校验。"""
        with tempfile.TemporaryDirectory() as directory:
            filename = Path(directory) / 'train_mortal_offline_with_grp_transformer.py'
            filename.write_bytes(b'unknown implementation')
            name, record = source_identity(filename)
            self.assertEqual(name, filename.name)
            self.assertEqual(record['path'], str(filename))
            self.assertEqual(record['sha256'], hashlib.sha256(filename.read_bytes()).hexdigest())


    def test_modified_known_file_loses_rename_exemption(self):
        """在副本中修改已登记源码，确保不是只凭路径放行。"""
        import importlib.util
        import shutil
        pipeline = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / 'mortal' / 'transformer_pipeline'
            target.mkdir(parents=True)
            helper = target / '_rename_compat.py'
            shutil.copyfile(pipeline / helper.name, helper)
            name = 'train_mortal_offline_with_grp_transformer.py'
            filename = target / name
            shutil.copyfile(pipeline / name, filename)
            spec = importlib.util.spec_from_file_location('isolated_rename_compat', helper)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            self.assertEqual(module.source_identity(filename)[0], 'train_offline_grp_transformer.py')
            with filename.open('a') as source:
                source.write('\n# unknown later change\n')
            actual_name, record = module.source_identity(filename)
            self.assertEqual(actual_name, name)
            self.assertEqual(record['sha256'], hashlib.sha256(filename.read_bytes()).hexdigest())


class RenameIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """使用真实模块的契约接口，不加载模型、训练样本或启动训练。"""
        import importlib.util
        import os
        if importlib.util.find_spec('torch') is None:
            raise unittest.SkipTest('需要 mortal 环境中的训练依赖')
        os.environ.setdefault('MORTAL_CFG', str(Path(__file__).resolve().parents[3] / 'config.offline_grp_transformer.toml'))
        import train_mortal_offline_with_grp_transformer as trainer
        import train_with_precomputed_cache as cached
        cls.trainer = trainer
        cls.cached = cached

    def test_offline_contract_accepts_legacy_and_rejects_other_changes(self):
        """历史实现清单可恢复；运行时、训练语义或额外源码变化必须拒绝。"""
        import copy
        trainer = self.trainer
        manifest = trainer.build_implementation_manifest()
        self.assertEqual(manifest['files']['train_offline_grp_transformer.py']['sha256'],
                         '40e4c11c3179791071fd99130db7651d5a73fa5794c01258e8644b5a6171bfcc')
        self.assertEqual(manifest['files']['dataloader_grp_transformer.py']['sha256'],
                         'f5347c088e19c4118701b2a739eb5549354f5020ac017ecb7ea17fc9e6dc2344')
        contract = {'implementation': manifest, 'batch_size': 1024}
        state = {'training_contract': contract, 'training_contract_sha256': trainer.sha256_json(contract)}
        trainer.validate_resume_training_contract(state, contract, trainer.sha256_json(contract))
        for field in ('source', 'runtime', 'semantics'):
            changed = copy.deepcopy(contract)
            if field == 'source':
                changed['implementation']['files']['model.py']['sha256'] = 'unknown'
            elif field == 'runtime':
                changed['implementation']['runtime']['torch'] = 'unknown'
            else:
                changed['batch_size'] = 512
            with self.subTest(field=field), self.assertRaises(RuntimeError):
                trainer.validate_resume_training_contract(state, changed, trainer.sha256_json(changed))

    def test_cache_trainer_contract_keeps_legacy_digest(self):
        """缓存训练断点沿用旧训练器摘要，并拒绝其他实现或输入变化。"""
        from types import SimpleNamespace
        cached = self.cached
        with tempfile.TemporaryDirectory() as directory:
            manifest_file = Path(directory) / 'manifest.json'
            checkpoint = Path(directory) / 'initial.pth'
            manifest_file.write_text('{}')
            checkpoint.write_bytes(b'fixture checkpoint identity only')
            args = SimpleNamespace(batch_size=1024, shuffle_seed=1, device='cpu')
            contract, digest = cached.build_run_contract(args, manifest_file,
                {'contract_sha256': 'cache', 'sample_count': 1, 'chunks': [{}]}, checkpoint)
            self.assertEqual(contract['trainer_sha256'],
                             'c268e0fd1a4130efb2fd82f0854b571633838fd8d48aead7050eb545858780a4')
            state = {'run_contract': contract, 'run_contract_sha256': digest}
            self.assertFalse(cached.validate_resume_run_contract(state, contract, digest))
            for key in ('trainer_sha256', 'manifest_sha256', 'config_sha256'):
                changed = dict(contract, **{key: 'unknown'})
                with self.subTest(key=key), self.assertRaises(RuntimeError):
                    cached.validate_resume_run_contract(state, changed, cached.sha256_json(changed))


if __name__ == '__main__':
    unittest.main()
