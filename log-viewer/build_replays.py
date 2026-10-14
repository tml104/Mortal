"""将评测 mjai 牌谱生成无需模型和训练环境的静态回放页面。"""

import argparse
import gzip
import html
import json
from pathlib import Path
import shutil


def build(log_dir, output_dir):
    """重新生成回放 HTML 和本地资源，保留输入牌谱及评测结果。"""
    source = Path(__file__).resolve().parent
    log_dir = log_dir.resolve()
    output_dir = output_dir.resolve()
    if output_dir == log_dir or output_dir in log_dir.parents:
        raise ValueError("回放输出目录不能覆盖牌谱目录")
    logs = sorted(log_dir.glob("*.json.gz"))
    if not logs:
        raise ValueError(f"没有找到牌谱：{log_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    for name in ("replay-state.js", "replay.js", "replay.css"):
        shutil.copyfile(source / name, output_dir / name)
    # 资源体积很小，复制可让导出的目录在 Windows 或其它机器上独立打开。
    assets = output_dir / "files"
    if assets.is_symlink():
        assets.unlink()
    shutil.copytree(source / "files" / "images", assets / "images", dirs_exist_ok=True)
    template = (source / "replay.html").read_text(encoding="utf-8")
    report_path = log_dir.parent / "result.json"
    report = json.loads(report_path.read_text(encoding="utf-8")) if report_path.exists() else {}
    challenger = report.get("challenger", {}).get("name")
    rows = []
    participants = set()
    for number, path in enumerate(logs, 1):
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            events = [json.loads(line) for line in handle if line.strip()]
        title = path.name.removesuffix(".json.gz")
        filename = title + ".html"
        payload = json.dumps(events, ensure_ascii=True, separators=(",", ":")).replace("<", "\\u003c")
        page = template.replace("__TITLE__", html.escape(title)).replace("__COUNT__", str(len(logs))).replace("__EVENTS__", payload)
        (output_dir / filename).write_text(page, encoding="utf-8")
        names = events[0]["names"]
        participants.update(names)
        challengers = [str(i) for i, name in enumerate(names) if name == challenger]
        rounds = sum(event["type"] == "start_kyoku" for event in events)
        rows.append(f'<tr><td>{number}</td><td><a href="{html.escape(filename, quote=True)}">{html.escape(title)}</a></td><td>{"、".join(challengers) or "—"}</td><td>{rounds}</td></tr>')
    description = "参评模型：" + html.escape("、".join(sorted(participants))) + "。原始牌谱和 Q 值未修改。"
    if challenger == "Mortal-epoch8-complete" and "Mortal-baseline-epoch3.46" in participants:
        description = "本次评测：epoch8 对阵三个相同 baseline，baseline 完成 3 个 epoch，第 4 个 epoch 进度 46.3397%。原始牌谱和 Q 值未修改。"
    index = f"""<!doctype html><html lang="zh-CN"><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>评测牌谱回放</title>
<style>body{{font:16px 'Microsoft YaHei',sans-serif;max-width:1000px;margin:30px auto;padding:0 20px;background:#101923;color:#eaf1f5}}a{{color:#7edbd2}}table{{border-collapse:collapse;width:100%}}td,th{{padding:8px;border-bottom:1px solid #304553;text-align:left}}</style>
<h1>评测牌谱回放 · {len(logs)} 个半庄</h1>
<p>中央显示场次、四家当前座次及分数、本场与供托；右侧显示实际动作、候选动作 Q 值和原始 JSON。</p>
<p>支持局数选择、时间轴、自动播放、切换视角及键盘左右键。适配 1280 × 720 窗口，长候选列表和 JSON 在各自面板内滚动。</p>
<p>{description}</p>
<table><thead><tr><th>编号</th><th>牌谱</th><th>待评模型开局玩家编号</th><th>小局数</th></tr></thead><tbody>{''.join(rows)}</tbody></table></html>"""
    (output_dir / "index.html").write_text(index, encoding="utf-8")
    print(f"已生成 {len(logs)} 个半庄回放：{output_dir / 'index.html'}")


def main():
    """从仓库根目录指定牌谱目录与回放输出目录。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--logs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    build(args.logs, args.output)


if __name__ == "__main__":
    main()
