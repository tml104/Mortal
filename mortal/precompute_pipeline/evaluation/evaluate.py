"""新格式 Mortal 对 Mortal 一对三：连续种子分批，完整批次提交报告。"""

import argparse
from dataclasses import dataclass, asdict
from pathlib import Path
import sys
import tempfile

MORTAL_DIR = Path(__file__).resolve().parents[2]
if str(MORTAL_DIR) not in sys.path:
    sys.path.insert(0, str(MORTAL_DIR))

from libriichi.arena import OneVsThree
from precompute_pipeline.common.files import atomic_write_json
from precompute_pipeline.common.policy import PolicySettings, create_engine, policy_settings
from precompute_pipeline.common.runtime import read_toml, path_value, positive_int, boolean, optional_threads, configure_threads


@dataclass(frozen=True)
class EvaluationSettings:
    challenger: PolicySettings
    champion: PolicySettings
    output_dir: Path
    games: int = 400
    games_per_batch: int = 64
    seed_start: int = 10000
    seed_key: int = 0x2000
    save_logs: bool = False
    torch_threads: int | None = None
    rayon_threads: int | None = None

    def __post_init__(self):
        """每 seed 恰好四场，不舍弃不能组成轮座位组的场次。"""
        if any(type(value) is not int or value <= 0 or value % 4 for value in (self.games, self.games_per_batch)):
            raise ValueError('games 和 games_per_batch 必须是 4 的正倍数')
        if any(type(value) is not int or not 0 <= value < 2 ** 64 for value in (self.seed_start, self.seed_key)):
            raise ValueError('seed_start 和 seed_key 必须是 u64')
        if self.seed_start + self.games // 4 > 2 ** 64:
            raise ValueError('seed 范围溢出')


def load_settings(filename):
    section = read_toml(filename)['evaluation']
    return EvaluationSettings(
        challenger=policy_settings(section['challenger'], default_device='cuda:0', default_amp=True),
        champion=policy_settings(section['champion'], default_device='cuda:0', default_amp=True),
        output_dir=path_value(section, 'output_dir'),
        games=positive_int(section, 'games', 400), games_per_batch=positive_int(section, 'games_per_batch', 64),
        seed_start=section.get('seed_start', 10000), seed_key=section.get('seed_key', 0x2000),
        save_logs=boolean(section, 'save_logs', False),
        torch_threads=optional_threads(section, 'torch_threads'), rayon_threads=optional_threads(section, 'rayon_threads'),
    )


def _policy_report(settings, engine, metadata):
    report = asdict(settings)
    report['checkpoint'] = str(settings.checkpoint.resolve())
    report['device'] = str(engine.device)
    report['enable_amp'] = engine.enable_amp
    report['model_spec'] = metadata
    return report


def _ranking_metrics(rankings):
    total = sum(rankings)
    return {
        'games_completed': total, 'rankings': rankings,
        'rank_proportions': [value / total for value in rankings] if total else [0.0] * 4,
        'average_rank': sum(value * rank for value, rank in zip(rankings, (1, 2, 3, 4))) / total if total else None,
        'average_rank_points': sum(value * points for value, points in zip(rankings, (90, 45, 0, -135))) / total if total else None,
    }


def run_evaluation(settings, *, arena_factory=OneVsThree):
    """每次调用建立独立目录，中断不自动续跑或纳入未完成批次。"""
    configure_threads(settings.torch_threads, settings.rayon_threads)
    challenger, challenger_meta = create_engine(settings.challenger)
    champion, champion_meta = create_engine(settings.champion)
    settings.output_dir.mkdir(parents=True, exist_ok=True)
    run_dir = Path(tempfile.mkdtemp(prefix='run-', dir=settings.output_dir))
    report_file = run_dir / 'result.json'
    log_dir = run_dir / 'logs' if settings.save_logs else None
    if log_dir:
        log_dir.mkdir()
    report = {
        'status': 'incomplete', 'games_requested': settings.games,
        'games_per_batch': settings.games_per_batch,
        'seed_start': settings.seed_start, 'seed_key': settings.seed_key,
        'seeds_completed': 0, 'save_logs': settings.save_logs,
        'torch_threads': settings.torch_threads, 'rayon_threads': settings.rayon_threads,
        'challenger': _policy_report(settings.challenger, challenger, challenger_meta),
        'champion': _policy_report(settings.champion, champion, champion_meta),
        **_ranking_metrics([0, 0, 0, 0]),
    }
    atomic_write_json(report_file, report)
    print(f'event=evaluation_start games={settings.games} games_per_batch={settings.games_per_batch} result={report_file}', flush=True)
    seeds_total = settings.games // 4
    seeds_per_batch = settings.games_per_batch // 4
    for offset in range(0, seeds_total, seeds_per_batch):
        count = min(seeds_per_batch, seeds_total - offset)
        arena = arena_factory(disable_progress_bar=False, log_dir=str(log_dir) if log_dir else None)
        rankings = list(arena.py_vs_py(
            challenger=challenger, champion=champion,
            seed_start=(settings.seed_start + offset, settings.seed_key), seed_count=count,
        ))
        if len(rankings) != 4 or any(value < 0 for value in rankings) or sum(rankings) != count * 4:
            raise RuntimeError('arena 返回的排名数与完整 seed 批次不一致')
        accumulated = [left + right for left, right in zip(report['rankings'], rankings, strict=True)]
        report.update(_ranking_metrics(accumulated))
        report['seeds_completed'] = offset + count
        report['status'] = 'complete' if offset + count == seeds_total else 'incomplete'
        atomic_write_json(report_file, report)
        print(
            f'event=evaluation_batch games={report["games_completed"]}/{settings.games} '
            f'rankings={accumulated} proportions={report["rank_proportions"]} '
            f'average_rank={report["average_rank"]:.6f} average_rank_points={report["average_rank_points"]:.6f} '
            f'status={report["status"]}', flush=True,
        )
    return report_file


def main(argv=None):
    parser = argparse.ArgumentParser(description='新 Conv2d 模型的独立一对三评测')
    parser.add_argument('--config', type=Path, required=True)
    args = parser.parse_args(argv)
    run_evaluation(load_settings(args.config))


if __name__ == '__main__':
    main()
