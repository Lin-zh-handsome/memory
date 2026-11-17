"""Report complete Seed 1 results against the original fixed-seed M4 run."""
from pathlib import Path
from collections import Counter
from statistics import mean, stdev
import csv
import json

ROOT=Path(__file__).resolve().parent
BASELINE=ROOT.parent/"mn_validation/m4_cover_blocks"
EPOCHS=(50,100,150,200,250)
POS=("左","中","右")


def read(p):
    return json.loads(p.read_text())


def load(root,epoch):
    d=root/f"evaluation/M4/epoch_{epoch:03d}"
    summary=read(d/"summary.json")
    rows=[json.loads(s) for s in (d/"episodes.jsonl").read_text().splitlines()]
    assert summary["complete"] and summary["episodes"]==len(rows)==50
    assert [r["seed"] for r in rows]==list(range(100000,100050))
    return rows


def episode(r):
    trace=r["stage_transitions"]
    max_stage=max(s["stage"] for s in trace)
    next_stage=next((s for s in trace if s["stage"]>=4 or
             (s["stage"]==3 and s["invalid"])),None)
    order=r["initial_state"]["uncover_order"]
    exposed=[] if next_stage is None else [i for i,v in enumerate(next_stage["cover_pattern"]) if not v]
    return dict(seed=r["seed"],layout="".join(POS[i] for i in order),
                first_cover=exposed[0] if len(exposed)==1 else None,red=order[0],
                covered=max_stage>=3,first_correct=max_stage>=4,
                first_order_failure=max_stage==3 and r["failure_category"]=="invalid_uncover_order",
                failed_after_first=max_stage>=4 and not r["success"],success=r["success"],
                category=r["failure_category"],video=r["video"])


def summarize(rows):
    return {k:sum(r[k] for r in rows) for k in
            ("covered","first_correct","first_order_failure","failed_after_first","success")}


def compare(x,y):
    assert x["seed"]==y["seed"] and x["layout"]==y["layout"]
    return {"environment_seed":x["seed"],"layout_rgb":x["layout"],
            "seed0":x,"seed1":y,
            "both_first_wrong":x["first_order_failure"] and y["first_order_failure"],
            "both_failed":not x["success"] and not y["success"],
            "same_wrong_first":x["first_order_failure"] and y["first_order_failure"] and
                x["first_cover"] is not None and x["first_cover"]==y["first_cover"]}


def main():
    cfg=read(ROOT/"configs/protocol.json")
    calib=read(ROOT/"configs/event_threshold_calibration.json")
    basecfg=read(BASELINE/"configs/protocol.json")
    changed={k for k in cfg if cfg[k]!=basecfg[k]}
    assert changed=={"seed","event_thresholds"} and cfg["seed"]==calib["seed"]==1
    assert read(ROOT/"logs/M4_complete.json")["epoch"]==250
    allrows=[]
    layouts={}
    snapshot=[]
    for t in EPOCHS:
        x,y=([episode(r) for r in load(root,t)] for root in (BASELINE,ROOT))
        pairs=[compare(a,b) for a,b in zip(x,y)]
        allrows.extend({"epoch":t,**r} for r in pairs)
        a,b=summarize(x),summarize(y)
        stats={}
        for k in ("success","first_correct"):
            stats[k]={"seed_mean_pct":mean((a[k]*2,b[k]*2)),
                      "sample_sd_pp":stdev((a[k]*2,b[k]*2)),
                      "seed1_minus_seed0_pp":(b[k]-a[k])*2}
        snapshot.append({"epoch":t,"seed0":a,"seed1":b,**stats,
                         "same_wrong_first":sum(r["same_wrong_first"] for r in pairs)})
    best={k:max(snapshot,key=lambda r:r[k]["success"]) for k in ("seed0","seed1")}
    best_pct=[best[k][k]["success"]*2 for k in ("seed0","seed1")]
    def aggregate(rows):
        groups={}
        for l in sorted({r["layout_rgb"] for r in rows}):
            subs=[r for r in rows if r["layout_rgb"]==l]
            groups[l]={"n":len(subs),"seed0_first_correct":sum(r["seed0"]["first_correct"] for r in subs),
                       "seed1_first_correct":sum(r["seed1"]["first_correct"] for r in subs),
                       **{k:sum(r[k] for r in subs) for k in ("both_first_wrong","both_failed","same_wrong_first")}}
        return groups
    layouts={str(t):aggregate([r for r in allrows if r["epoch"]==t]) for t in EPOCHS}
    persistent=[]
    for s in range(100000,100050):
        records=[r for r in allrows if r["environment_seed"]==s]
        times=[r["epoch"] for r in records if r["same_wrong_first"]]
        if len(times)>=2:
            persistent.append({"seed":s,"layout":records[0]["layout_rgb"],"epochs_with_same_wrong_first":times})
    output={"checkpoints":snapshot,"paired_layouts":layouts,"repeated_wrong_seed_cases":persistent,
            "best_epochs":{k:r["epoch"] for k,r in best.items()},"best_pct":best_pct,
            "best_mean_pct":mean(best_pct),"best_sample_sd_pp":stdev(best_pct)}
    for file, val in (("comparison.json",output),("paired_episodes.json",allrows)):
        path=ROOT/file
        tmp=path.with_suffix(".tmp")
        tmp.write_text(json.dumps(val,ensure_ascii=False,indent=2)+"\n")
        tmp.replace(path)
    lines=["# M4 方块覆盖 · 两个训练种子","",
           f"两个训练种子分别完成 250 轮，各评测 250 回合。种子 0 最佳 {best_pct[0]}%，种子 1 最佳 {best_pct[1]}%。"
           f"两者最佳成绩均值 {mean(best_pct):.1f}%，样本标准差 {stdev(best_pct):.2f} 个百分点。"
           "每个检查点复用同样的 50 个环境种子 100000–100049。标准差只来自两个训练种子。",
           "",
           f"种子 1 复用原八条训练示范、0.2 分位数标定流程，得到视觉阈值 {calib['event_thresholds']['visual']:.9f}，历史阈值 {calib['event_thresholds']['history']:.9f}。","",
           "## 各检查点","",
           "| 轮次 | 种子 0 最终成功 | 种子 1 最终成功 | 成功率均值 ± 样本标准差 | 种子 0 首次揭对 | 种子 1 首次揭对 | 首次揭对均值 ± 样本标准差 |",
           "|---:|---:|---:|---:|---:|---:|---:|"]
    for t in snapshot:
        a,b=t['seed0'],t['seed1']; s=t['success']; f=t['first_correct']
        lines.append(f"| {t['epoch']} | {2*a['success']}%（{a['success']}/50） | {2*b['success']}%（{b['success']}/50） | "
                     f"{s['seed_mean_pct']:.1f}% ± {s['sample_sd_pp']:.2f} 个百分点 | {a['first_correct']}/50 | {b['first_correct']}/50 | "
                     f"{f['seed_mean_pct']:.1f}% ± {f['sample_sd_pp']:.2f} 个百分点 |")
    lines+=["","| 轮次 | 盖齐（0 / 1） | 首次揭盖顺序错误（0 / 1） | 首次揭对后失败（0 / 1） | 同一环境种子上揭了同一个错误盖子 |",
            "|---:|---:|---:|---:|---:|"]
    for t in snapshot:
        a,b=t["seed0"],t["seed1"]
        fields=" | ".join(f"{a[k]} / {b[k]}" for k in ("covered","first_order_failure","failed_after_first"))
        lines.append(f"| {t['epoch']} | {fields} | {t['same_wrong_first']} |")
    lines+=["","最佳检查点取五个完整检查点中的最高成功率。同分取较早轮次。"
            f"种子 0 最佳为第 {best['seed0']['epoch']} 轮，种子 1 最佳为第 {best['seed1']['epoch']} 轮。"
            f"两个最佳值相差 {abs(best_pct[1]-best_pct[0])} 个百分点。",
            "","## 配对失败和颜色布局","",
            "布局按红、绿、蓝对应的桌面位置记录。表中各行只计算第 250 轮的配对回合，其他检查点的同口径统计见 JSON。","",
            "| 红绿蓝位置 | 回合数 | 种子 0 首次揭对 | 种子 1 首次揭对 | 两者首次都揭错 | 两者揭了同一个错误盖子 | 两者最终失败 |",
            "|---|---:|---:|---:|---:|---:|---:|"]
    for name,r in layouts["250"].items():
        lines.append(f"| {name} | {r['n']} | {r['seed0_first_correct']} | {r['seed1_first_correct']} | {r['both_first_wrong']} | "
                     f"{r['same_wrong_first']} | {r['both_failed']} |")
    lines+=["",f"有 {len(persistent)} 个固定环境种子，在至少两个共同检查点上由两个训练种子揭开同一个错误盖子。"
            "这些是重复失败的观察记录。检查点复用环境，且训练种子仅有两个，暂无法估计跨种子失败的总体比例。",
            "","[配对明细](paired_episodes.json) · [布局和差异统计](comparison.json) · [标定记录](configs/event_threshold_calibration.json)"]
    (ROOT/"REPORT.md").write_text("\n".join(lines)+"\n")
    videos=["# 代表性视频索引","", "视频位于服务器相应评测目录。",""]
    for row in allrows:
        for seed in ("seed0","seed1"):
            r=row[seed]
            if r["video"]:
                videos.append(f"- {seed} 第 {row['epoch']} 轮，环境种子 {r['seed']}，{'成功' if r['success'] else '失败'}，{r['video']}")
    (ROOT/"videos.md").write_text("\n".join(videos)+"\n")


if __name__=="__main__":
    main()
