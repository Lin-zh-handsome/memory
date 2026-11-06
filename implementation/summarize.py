"""Aggregate only complete evaluations; do not manufacture missing rates."""
import csv
import json
from pathlib import Path
import argparse
import numpy as np


def write_csv(path,rows,fields):
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fields);w.writeheader();w.writerows(rows)


def summarize(root):
    root=Path(root);checkpoints=[];episodes=[]
    for p in root.glob('eval/*_seed*_epoch*.json'):
        r=json.loads(p.read_text())
        if r.get('episodes')==50:checkpoints.append(r)
    for p in root.glob('eval/*_seed*_epoch*_episodes.csv'):
        with p.open() as f:episodes.extend(csv.DictReader(f))
    checkpoints.sort(key=lambda x:(x['model'],x['train_seed'],x['epoch']))
    write_csv(root/'eval/checkpoint_results.csv',checkpoints,
              ['model','train_seed','epoch','successful_episodes','episodes','success_rate'])
    write_csv(root/'eval/episodes.csv',episodes,['model','train_seed','epoch','env_seed','success','steps','stages'])
    summary=[]
    for model in ['M0','M2']:
        scores=[]
        for seed in range(3):
            rows=[r for r in checkpoints if r['model']==model and r['train_seed']==seed]
            scores.append(max(r['success_rate'] for r in rows) if len(rows)==5 else None)
        complete=all(x is not None for x in scores)
        summary.append({'model':model,'seed0_best':scores[0],'seed1_best':scores[1],'seed2_best':scores[2],
             'mean':float(np.mean(scores)) if complete else None,
             'std_ddof1':float(np.std(scores,ddof=1)) if complete else None,
             'M2_minus_M0_percentage_points':None})
    if all(r['mean'] is not None for r in summary):
        summary[1]['M2_minus_M0_percentage_points']=100*(summary[1]['mean']-summary[0]['mean'])
    write_csv(root/'results_summary.csv',summary,list(summary[0]))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(7,4))
    for model,color in [('M0','#356cb0'),('M2','#de7833')]:
        for seed in range(3):
            rows=[r for r in checkpoints if r['model']==model and r['train_seed']==seed]
            if rows:ax.plot([r['epoch'] for r in rows],[100*r['success_rate'] for r in rows],marker='o',color=color,alpha=0.65,label=f'{model} seed {seed}')
    ax.set(xlabel='Epoch',ylabel='Closed-loop success (%)',ylim=(0,100),xticks=[50,100,150,200,250])
    if checkpoints:ax.legend()
    if checkpoints:
        fig.tight_layout();fig.savefig(root/'plots/success_by_epoch.png',dpi=180)
    plt.close(fig)
    if all(r['mean'] is not None for r in summary):
        fig,ax=plt.subplots(figsize=(5,4));ax.bar(['M0','M2'],[100*r['mean'] for r in summary],yerr=[100*r['std_ddof1'] for r in summary],capsize=5,color=['#356cb0','#de7833'])
        ax.set(ylabel='Best-of-5 success (%)',ylim=(0,100));fig.tight_layout();fig.savefig(root/'plots/model_comparison.png',dpi=180);plt.close(fig)
    fig,ax=plt.subplots(figsize=(7,4))
    for p in root.glob('logs/*_epochs.csv'):
        with p.open() as f:rows=list(csv.DictReader(f))
        if rows:ax.plot([int(r['epoch']) for r in rows],[float(r['loss']) for r in rows],label=p.stem)
    ax.set(xlabel='Epoch',ylabel='Training loss')
    if ax.lines:ax.legend()
    if ax.lines:
        fig.tight_layout();fig.savefig(root/'plots/loss.png',dpi=180)
    plt.close(fig)
    complete_runs=list(root.glob('logs/*_complete.json'))
    status={'complete_trainings':len(complete_runs),'complete_checkpoint_evaluations':len(checkpoints),
            'completed_episodes':len(episodes),'summary':summary}
    (root/'status.json').write_text(json.dumps(status,indent=2))
    checks_path=root/'logs/interface_checks.json'
    checks=json.loads(checks_path.read_text()) if checks_path.exists() else {}
    overhead={}
    if 'M0' in checks and 'M2' in checks:
        n0=checks['M0']['trainable_parameters'];n2=checks['M2']['trainable_parameters']
        head=checks['M2']['auxiliary_head_parameters']
        overhead.update(M0_parameters=n0,M2_training_parameters=n2,
                        M2_inference_parameters=n2-head,added_training_parameters=n2-n0,
                        added_inference_parameters=n2-head-n0)
    timings={}
    for model in ['M0','M2']:
        durations=[]
        for p in root.glob(f'logs/{model}_seed*_epochs.csv'):
            with p.open() as f:records=list(csv.DictReader(f))
            durations.extend(float(r['seconds']) for r in records if int(r['epoch'])>1)
        if durations:timings[model]=float(np.median(durations))
    overhead['median_formal_epoch_seconds']=timings
    overhead['timing_interpretation']='Observed wall time includes data loading and GPU sharing. Seed-0 models are authorized to run concurrently on one GPU; timing ratios are not isolated module compute overhead.'
    if len(timings)==2:overhead['observed_epoch_time_ratio_M2_to_M0']=timings['M2']/timings['M0']
    (root/'eval/overhead.json').write_text(json.dumps(overhead,indent=2))
    failures=[]
    for model in ['M0','M2']:
        failed=[r for r in episodes if r['model']==model and int(r['success'])==0]
        for key in ['grasp','insertion','mug_grasp','mug_place']:
            never=sum(not json.loads(r['stages']).get(key,False) for r in failed)
            failures.append({'model':model,'metric':key,'failed_episodes':len(failed),
                             'never_achieved_count':never,
                             'never_achieved_fraction':never/len(failed) if failed else None})
    write_csv(root/'eval/failure_stages.csv',failures,list(failures[0]))
    readme=root/'README_results.md'
    if readme.exists():
        content=readme.read_text().split('<!-- ACTUAL_RESULTS -->')[0]
        if len(complete_runs)==6 and len(checkpoints)==30:
            content=content.replace('本实验尚在执行，未产生完整闭环结果。未完成的数值保持空值，不能作为零成功率。',
                                    '六次正式训练及三十次检查点评测均已完成，真实结果见文末实际执行状态。')
        lines=['<!-- ACTUAL_RESULTS -->','\n## 实际执行状态\n',
               f'已完成训练：{len(complete_runs)}/6；已完成检查点评测：{len(checkpoints)}/30；已记录闭环回合：{len(episodes)}/1500。\n',
               '| 模型 | 种子0最佳 | 种子1最佳 | 种子2最佳 | 均值 ± 样本标准差 |',
               '|---|---|---|---|---|']
        def fmt(x):return '未完成' if x is None else f'{100*x:.2f}%'
        for r in summary:
            estimate='未完成' if r['mean'] is None else f"{100*r['mean']:.2f}% ± {100*r['std_ddof1']:.2f}%"
            lines.append(f"| {r['model']} | {fmt(r['seed0_best'])} | {fmt(r['seed1_best'])} | {fmt(r['seed2_best'])} | {estimate} |")
        delta=summary[1]['M2_minus_M0_percentage_points']
        if delta is not None:
            lines.append(f'\nM2 − M0：{delta:+.2f} 个百分点。')
            decision=('当前固定协议的结果支持继续研究该历史分支与辅助重构组合，下一步仍需独立评测或机制消融确认来源。' if delta>0 else
                      '当前固定协议未显示该组合的平均成功率收益；现有结果不支持直接扩大该实现，应先结合失败阶段和开销决定是否修改方法。')
            lines.append('\n'+decision+' 三个训练种子的 Best-of-5 结果不能视为独立测试集的无偏估计。')
        else:lines.append('\n结果尚不完整，不能判断 M2 的收益或研究方向。')
        lines.append('\n参数与计算开销（正式轮次耗时包含数据加载及共享算力竞争；种子0两模型同卡并行，耗时比不能直接解释为模块纯计算开销）：\n\n```json\n'+json.dumps(overhead,indent=2)+'\n```')
        lines.append('\n失败阶段计数见 `eval/failure_stages.csv`，记录失败回合中从未达到抓取胶囊、插入胶囊、抓杯、放杯等官方子条件的比例。多个子条件可能同时缺失，不能视为互斥归因。')
        readme.write_text(content+'\n'.join(lines)+'\n')
    print(json.dumps(status),flush=True)
    return status


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);args=p.parse_args();summarize(args.root)
