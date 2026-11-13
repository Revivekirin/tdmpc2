"""Focused, read-only figures from completed HalfCheetah and Reacher comparisons.

Usage: python -m experiments.nsgym.analysis.research_interpretation
       --results-root RESULTS_NSGYM --output NEW_DIRECTORY
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SEVERITIES = (.5, .75, 1., 1.25, 1.5, 2.)
COLORS = {'D': '#0072B2', 'D+Q': '#D55E00', 'Joint': '#009E73'}
PHYSICS = {'bthigh_damping':'bthigh damping (nominal 6)',
           'torso_mass':'torso mass (nominal 6.2502 kg)',
           'D0':'joint0 damping (nominal 1)', 'D1':'joint1 damping (nominal 1)',
           'M0':'body0 mass (nominal 0.035605 kg)',
           'M1':'body1 mass (nominal 0.035605 kg)'}


def interval(values):
    """Adaptation stream is the sampling unit; show the observed mean and seed bootstrap."""
    x = np.asarray(values, dtype=float)
    assert len(x) == 3 and np.isfinite(x).all(), f'Expected three finite streams: {x}'
    rng = np.random.default_rng(20261113)
    means = rng.choice(x, (10000, len(x)), replace=True).mean(axis=1)
    return float(x.mean()), *np.quantile(means, [.025, .975])


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream,'sha256').hexdigest()


def inputs(root):
    for path in (root/'dynamics-q-750k-v1/comparison/completion.json',
                 root/'reacher-dynamics-q-v1/comparison/completion.json',
                 root/'q-planning-mechanism-750k-v1/provenance/completion.json'):
        assert json.loads(path.read_text())['status']=='PASS', f'Incomplete source: {path}'
    hc = pd.read_csv(root/'dynamics-q-750k-v1/comparison/per_adaptation_seed_comparison.csv')
    re = pd.read_csv(root/'reacher-dynamics-q-v1/comparison/four_way_per_adaptation_seed.csv')
    re = re[re.kind == 'shifted'].copy()
    q = pd.read_csv(root/'q-planning-mechanism-750k-v1/tables/immediate_per_stream.csv')
    js = pd.read_csv(root/'q-planning-mechanism-750k-v1/tables/divergence_per_stream.csv')
    train = pd.read_csv(root/'reacher-dynamics-q-v1/comparison/update_diagnostics.csv')
    assert len(hc) == 36 and len(re) == 72 and len(q) == 144 and len(js) == 144
    assert set(hc.severity) == set(re.severity) == set(SEVERITIES)
    assert re.groupby(['family','severity']).size().eq(3).all()
    assert hc.groupby(['parameter','severity']).size().eq(3).all()
    return hc, re, q, js, train


def stream_table(hc, re):
    a = pd.DataFrame(dict(task='HalfCheetah', family=hc.parameter, severity=hc.severity,
        seed=hc.adaptation_seed, d_return=hc.ttt_return-hc.frozen_return,
        dq_return=hc.dq_return-hc.frozen_return, joint_return=hc.joint_return-hc.frozen_return,
        dq_minus_d=hc.dq_return-hc.ttt_return,
        d_mse=100*(hc.frozen_latent_mse-hc.ttt_latent_mse)/hc.frozen_latent_mse,
        dq_mse=100*(hc.frozen_latent_mse-hc.dq_latent_mse)/hc.frozen_latent_mse,
        dq_minus_d_mse=100*(hc.ttt_latent_mse-hc.dq_latent_mse)/hc.frozen_latent_mse))
    b = pd.DataFrame(dict(task='Reacher', family=re.family, severity=re.severity,
        seed=re.adaptation_seed,
        d_return=re['Dynamics-only_episode_return']-re['Frozen_episode_return'],
        dq_return=re['Dynamics+Q_episode_return']-re['Frozen_episode_return'],
        joint_return=re['Joint RL_episode_return']-re['Frozen_episode_return'],
        dq_minus_d=re['Dynamics+Q_episode_return']-re['Dynamics-only_episode_return'],
        d_mse=100*(re.frozen_mse-re.dynamics_mse)/re.frozen_mse,
        dq_mse=100*(re.frozen_mse-re.dq_mse)/re.frozen_mse,
        dq_minus_d_mse=100*(re.dynamics_mse-re.dq_mse)/re.frozen_mse))
    return pd.concat([a,b],ignore_index=True)


def paired_plot(data, task, output):
    rows = data[data.task == task]
    families = list(rows.family.unique())
    fig, axes = plt.subplots(len(families), 2, figsize=(12, 3.1*len(families)), squeeze=False)
    for i, family in enumerate(families):
        sub = rows[rows.family == family]
        for j, (fields, title, ylabel) in enumerate([
            (('d_return','dq_return','joint_return'), 'Shifted return relative to Frozen', 'Return difference ↑'),
            (('d_mse','dq_mse'), 'Held-out latent error reduction vs Frozen', 'MSE reduction (%) ↑')]):
            ax = axes[i,j]
            for k, field in enumerate(fields):
                label = ('D','D+Q','Joint')[k]
                x, y, low, high = [], [], [], []
                for severity in SEVERITIES:
                    vals = sub.loc[sub.severity == severity,field]
                    mean, lo, hi = interval(vals)
                    x.append(severity); y.append(mean); low.append(mean-lo); high.append(hi-mean)
                    ax.scatter(np.full(3,severity + (k-1)*.015), vals, color=COLORS[label],s=12,alpha=.38)
                ax.errorbar(x,y,yerr=[low,high],fmt='o-',capsize=2,color=COLORS[label],label=label,lw=1.5)
            ax.axhline(0,color='black',lw=.8); ax.axvline(1,color='gray',ls=':',lw=.8)
            ax.set(xlabel='Physical multiplier k',ylabel=ylabel,title=f'{PHYSICS[family]}: {title}')
            ax.set_xticks(SEVERITIES); ax.grid(alpha=.15)
            if i==0: ax.legend(fontsize=8)
    budget=3000 if task=='HalfCheetah' else 10000
    fig.suptitle(f'{task} | {budget:,} adaptation transitions | paired held-out seeds | dots = adaptation streams (n=3) | bars = seed-bootstrap 95% CI\n'
        'k=1 is nominal physics; Joint encoder-changing MSE is not compared',fontsize=11)
    fig.tight_layout(rect=(0,0,1,.94))
    fig.savefig(output/f'{task.lower()}_return_and_latent.png',dpi=170);plt.close(fig)


def mechanism_plot(data, re, output):
    fig,axes=plt.subplots(2,2,figsize=(12,9))
    for i,task in enumerate(('HalfCheetah','Reacher')):
        sub=data[data.task==task]
        ax=axes[i,0]
        for family in sub.family.unique():
            x=[];y=[];lo=[];hi=[]
            for severity in SEVERITIES:
                v=sub.loc[(sub.family==family)&(sub.severity==severity),'dq_minus_d']
                m,l,h=interval(v);x.append(severity);y.append(m);lo.append(m-l);hi.append(h-m)
            ax.errorbar(x,y,yerr=[lo,hi],marker='o',capsize=2,label=family)
        ax.axhline(0,color='black',lw=.8);ax.set(title=f'{task}: incremental Q control effect',xlabel='Physical multiplier k',ylabel='D+Q − Dynamics-only return ↑')
        ax.legend(fontsize=8);ax.grid(alpha=.15)
        ax=axes[i,1]
        for family in sub.family.unique():
            rr=sub[sub.family==family]
            ax.scatter(rr.dq_minus_d_mse,rr.dq_minus_d,label=family,s=24,alpha=.7)
        ax.axhline(0,color='black',lw=.8);ax.axvline(0,color='black',lw=.8)
        ax.set(title=f'{task}: stream-level prediction vs control',xlabel='Additional MSE reduction from Q step (% Frozen MSE) ↑',ylabel='D+Q − Dynamics-only return ↑')
        ax.legend(fontsize=8);ax.grid(alpha=.15)
    fig.suptitle('Same severity and evaluation seeds within each task; n=3 adaptation streams per condition\n'
        'HalfCheetah: 3,000 transitions; Reacher: 10,000. Quadrants are associations, not Q causality.',fontsize=11)
    fig.tight_layout(rect=(0,0,1,.93));fig.savefig(output/'incremental_q_prediction_vs_control.png',dpi=170);plt.close(fig)

    # Reacher return is the sum of recorded distance reward and control reward.
    fig,axes=plt.subplots(2,2,figsize=(12,7),sharex=True)
    for ax,family in zip(axes.flat,re.family.unique()):
        rr=re[re.family==family]
        for col,label in [('reward_dist','Distance reward'),('reward_ctrl','Control reward'),('episode_return','Total return')]:
            x=[];y=[]
            for severity in SEVERITIES:
                q=rr[rr.severity==severity]
                v=q['Dynamics+Q_'+col]-q['Dynamics-only_'+col]
                x.append(severity);y.append(interval(v)[0])
            ax.plot(x,y,'o-',label=label)
        ax.axhline(0,color='black',lw=.8);ax.set(title=PHYSICS[family],xlabel='Physical multiplier k',ylabel='D+Q − Dynamics-only reward ↑');ax.grid(alpha=.15)
    axes[0,0].legend(fontsize=8);fig.suptitle('Reacher reward mechanism: distance and action penalty sum to total return; seed means (n=3)')
    fig.tight_layout(rect=(0,0,1,.95));fig.savefig(output/'reacher_reward_decomposition.png',dpi=170);plt.close(fig)


def halfcheetah_reward(root, output):
    """Pair D and D+Q rollouts by both adaptation and evaluation seed."""
    droot=root/'dynamics-only-750k-v1/final-snapshot-v2/shifted'
    qroot=root/'dynamics-q-750k-v1/shifted'
    records=[]
    for qpath in sorted(qroot.glob('*/shifted.csv')):
        name=qpath.parent.name
        dpath=droot/name.replace('dynamics-q','decision')/'shifted.csv'
        assert dpath.is_file(), dpath
        q=pd.read_csv(qpath);d=pd.read_csv(dpath)
        merged=q.merge(d,on=['parameter','severity','adaptation_seed','evaluation_seed'],validate='one_to_one')
        assert len(merged)==20 and (merged.episode_length_x==1000).all()
        assert (merged.episode_length_y==1000).all()
        for _,r in merged.iterrows():
            records.append(dict(parameter=r.parameter,severity=r.severity,adaptation_seed=r.adaptation_seed,
                evaluation_seed=r.evaluation_seed,
                velocity_contribution=1000*(r.forward_velocity-r.decision_velocity),
                control_contribution=-(r.control_cost-r.decision_control_cost),
                return_difference=r.episode_return-r.decision_return))
    df=pd.DataFrame(records)
    assert len(df)==720
    assert np.allclose(df.velocity_contribution+df.control_contribution,df.return_difference,atol=.03), 'Return decomposition mismatch'
    streams=df.groupby(['parameter','severity','adaptation_seed'],as_index=False)[['velocity_contribution','control_contribution','return_difference']].mean()
    streams.to_csv(output/'halfcheetah_reward_components.csv',index=False)
    fig,axes=plt.subplots(1,2,figsize=(12,4))
    for ax,param in zip(axes,streams.parameter.unique()):
        sub=streams[streams.parameter==param]
        for field,label in [('velocity_contribution','Forward velocity × 1000 steps'),
                            ('control_contribution','Negative control cost'),('return_difference','Total return')]:
            ax.plot(SEVERITIES,[interval(sub.loc[sub.severity==s,field])[0] for s in SEVERITIES],
                    'o-',label=label)
        ax.axhline(0,color='black',lw=.8);ax.set(title=PHYSICS[param],xlabel='Physical multiplier k',ylabel='D+Q − Dynamics-only return components ↑');ax.grid(alpha=.15)
    axes[0].legend(fontsize=8)
    fig.suptitle('HalfCheetah reward mechanism | 20 paired held-out episodes per adaptation stream; n=3 streams\n'
        'Return = 1000 × forward velocity − episode control cost (verified within rounding)')
    fig.tight_layout(rect=(0,0,1,.88));fig.savefig(output/'halfcheetah_reward_decomposition.png',dpi=170);plt.close(fig)


def q_plots(q, js, train, output):
    # These are held-out fixed-observation HalfCheetah measurements.
    fig,axes=plt.subplots(2,2,figsize=(12,8),sharex=True)
    for i,family in enumerate(q.parameter.unique()):
        a,b=axes[i]
        for right,color in [('DQ',COLORS['D+Q']),('J',COLORS['Joint'])]:
            sub=js[(js.parameter==family)&(js.left=='F')&(js.right==right)]
            values=[interval(sub[sub.severity==s].js)[0] for s in SEVERITIES]
            a.plot(SEVERITIES,values,'o-',color=color,label=f'Frozen → {right}')
        a.set(title=f'{PHYSICS[family]}: categorical JS divergence',ylabel='JS divergence (nats) ↑',xlabel='Physical multiplier k')
        for method,color in [('F','black'),('D',COLORS['D']),('DQ',COLORS['D+Q']),('J',COLORS['Joint'])]:
            sub=q[(q.parameter==family)&(q.method==method)]
            b.plot(SEVERITIES,[interval(sub[sub.severity==s].entropy)[0] for s in SEVERITIES],
                'o-',color=color,label=method)
        b.set(title=f'{PHYSICS[family]}: categorical entropy',ylabel='Entropy (nats)',xlabel='Physical multiplier k')
        a.grid(alpha=.15);b.grid(alpha=.15)
    axes[0,0].legend(fontsize=8);axes[0,1].legend(fontsize=8)
    fig.suptitle('HalfCheetah | fixed held-out observations and actions | points average 3 adaptation streams\n'
        'JS is distribution change, not improved calibration or return. Frozen and D share unchanged Q. k=1 is duplicated nominal physics.',fontsize=10)
    fig.tight_layout(rect=(0,0,1,.93));fig.savefig(output/'halfcheetah_q_distribution.png',dpi=170);plt.close(fig)

    # Reacher has only online training-batch summaries, not matched held-out logits.
    fig,axes=plt.subplots(2,2,figsize=(12,7))
    train=train.copy();train['q_gap']=train.q_prediction-train.q_target
    for ax,(field,title) in zip(axes.flat,[('q_gap','Q prediction − target'),('q_entropy','Categorical entropy'),
        ('q_disagreement','Ensemble disagreement'),('q_loss','Q training loss')]):
        for family in train.family.unique():
            values=train[train.family==family].groupby('transition')[field].median()
            ax.plot(values.index,values,label=family,lw=1)
        ax.axhline(0,color='gray',lw=.6);ax.set(title=title,xlabel='Adaptation transitions',ylabel='Median across online updates');ax.grid(alpha=.15)
    axes[0,0].legend(fontsize=8)
    fig.suptitle('Reacher D+Q | online training batches only | median across conditions/streams\n'
        'These values are NOT a held-out Q distribution or directly comparable with HalfCheetah JS.',fontsize=11)
    fig.tight_layout(rect=(0,0,1,.93));fig.savefig(output/'reacher_q_training_diagnostics.png',dpi=170);plt.close(fig)


def report(data, output):
    lines=['# HalfCheetah–Reacher 적응 결과 해석','',
        '완료된 비교 파일만 읽어 다시 그렸다. 새 학습, 모델 추론, 환경 rollout은 실행하지 않았다. '
        '동일 held-out 평가 seed를 짝지었고, 각 물리 조건의 적응 stream 3개를 통계 단위로 사용했다.',
        '', '## 핵심 결론','',
        '두 환경 모두 Dynamics-only와 D+Q는 Frozen보다 one-step latent MSE를 낮췄다. 그러나 return 개선은 같은 방향이 아니다. '
        'HalfCheetah D+Q는 Frozen 대비 28/36 stream에서 return이 높고, Reacher D+Q는 9/72 stream에서만 높다. '
        '이는 각 환경의 **적응 stream별 기술적 개수**이며 독립 학습모델 36개/72개를 뜻하지 않는다. '
        '예측 정확도 개선만으로 제어 성능 개선을 주장할 수 없다.',
        '', 'HalfCheetah는 두 파라미터 모두 k=2에서 D+Q−Dynamics-only 평균 return이 양수다 '
        '(bthigh damping +487, torso mass +707). 하지만 전체 조건에서는 18/36 stream만 D+Q가 Dynamics-only를 이겼다. '
        'Reacher는 D+Q−Dynamics-only가 29/72 stream에서 양수이고, 질량 family M0/M1에서는 평균이 음수다. '
        'Reacher의 return 차이는 절대값 약 0.01 이하인 조건이 많으므로 작은 차이를 실용적 개선으로 읽지 말아야 한다.',
        '', 'Q step이 Dynamics-only 대비 추가 latent MSE를 줄인 stream은 HalfCheetah 16/36, Reacher 35/72다. '
        '그중 return도 함께 오른 stream은 각각 6/36, 17/72다. '
        'HalfCheetah의 Frozen→D+Q categorical JS divergence는 0이 아니지만, 이 값은 분포가 달라졌다는 뜻뿐이다. '
        'Q target 정확도, 보정(calibration), 실제 후보 action의 우열은 이 그림만으로 판단할 수 없다.',
        '', '## 그림 읽는 법','',
        '- `halfcheetah_return_and_latent.png`, `reacher_return_and_latent.png`: 왼쪽 shifted return 차이가 0보다 크면 Frozen보다 좋은 제어다. 오른쪽 MSE 감소율이 0보다 크면 고정 transition bank에서 one-step latent 예측이 더 정확하다. 작은 점은 적응 stream, 오차 막대는 stream 3개의 bootstrap 구간이다. Joint는 encoder가 달라 같은 latent 목표로 MSE를 비교할 수 없다.',
        '- `incremental_q_prediction_vs_control.png`: D+Q−Dynamics-only는 Q update를 추가한 **절차 전체**의 차이다. 산점도의 오른쪽 위는 MSE와 return이 모두 개선, 오른쪽 아래는 MSE만 개선된 사례다. 동일 seed 비교이지만 Q mismatch에 대한 인과 증거는 아니다.',
        '- `reacher_reward_decomposition.png`: distance reward와 control reward(음수 action penalty)를 합치면 total return이다. 두 성분이 상쇄하면 reaching은 좋아져도 전체 return은 악화될 수 있다.',
        '- `halfcheetah_reward_decomposition.png`: return 차이를 1,000 × 평균 전진속도 차이 − episode control-cost 차이로 나눴다. 이 데이터에서 큰 return 변화는 주로 속도 변화에서 온다.',
        '- `halfcheetah_q_distribution.png`: 고정된 held-out 관측·행동에서 categorical Q 분포의 JS divergence를 비교한다. Frozen→Dynamics-only JS=0은 Q를 업데이트하지 않았기 때문에 예상되는 결과다. D+Q와 Joint의 JS가 0보다 커도 Q가 더 정확하다는 뜻은 아니다. Entropy는 분포 폭이지 불확실성 보정이나 정책 성능의 증거가 아니다.',
        '- `reacher_q_training_diagnostics.png`: Q prediction−target, entropy, ensemble disagreement, Q loss는 **온라인 학습 batch**의 기록이다. Reacher에는 같은 held-out 관측·행동에서 저장한 Q logits/probabilities가 없어 HalfCheetah와 같은 JS 비교는 계산할 수 없다.',
        '', '## 관측된 방향 (조건별 값은 CSV 참고)','']
    for task in ('HalfCheetah','Reacher'):
        sub=data[data.task==task]
        for family in sub.family.unique():
            rr=sub[sub.family==family]
            improved=int((rr.dq_minus_d>0).sum());mse=int((rr.dq_minus_d_mse>0).sum())
            lines.append(f'- {task} {family}: D+Q improves return over Dynamics-only in {improved}/{len(rr)} streams; '
                f'additional MSE reduction in {mse}/{len(rr)}. Mean D+Q−D return {rr.dq_minus_d.mean():+.5g} '
                f'(task-specific return units).')
    lines += ['', '## Limits and required data','',
        '- Each severity has only three adaptation streams; bootstrap intervals are descriptive, not significance tests. k=1 in multiple physical families is the same nominal physics and must not be counted as independent tasks.',
        '- The two tasks have different return and latent scales. Do not pool raw differences or categorical Q magnitudes across tasks.',
        '- HalfCheetah adaptation used 3,000 transitions; Reacher used 10,000. Cross-task panels are parallel descriptive views, not a matched-budget comparison.',
        '- Q distribution comparison on Reacher requires saved *held-out* matched state/action bank identifiers, evaluation seeds, categorical support, per-head Q logits or normalized probabilities, and source/snapshot hashes for Frozen, Dynamics-only, D+Q, and Joint. `reacher-dynamics-q-v1/comparison/update_diagnostics.csv` contains only online batch aggregates.',
        '- HalfCheetah Joint fixed-encoder MSE is unavailable because Joint changes its encoder. Existing self-consistency MSE does not share a latent target with Frozen/D/D+Q.',
        '- Reacher nominal retention is available, but this figure set focuses on shifted control. HalfCheetah D+Q nominal retention was NOT_EVALUATED; do not infer it from k=1 shifted rows.',
        '', '## Sources','',
        '- HalfCheetah: `dynamics-q-750k-v1/comparison/per_adaptation_seed_comparison.csv`; `q-planning-mechanism-750k-v1/tables/immediate_per_stream.csv` and `divergence_per_stream.csv`.',
        '- Reacher: `reacher-dynamics-q-v1/comparison/four_way_per_adaptation_seed.csv` and `update_diagnostics.csv`.',
        '- Exact plotted adaptation-stream values: `research_stream_effects.csv` and `halfcheetah_reward_components.csv`.']
    (output/'RESEARCH_INTERPRETATION.md').write_text('\n'.join(lines)+'\n')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--results-root',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():raise ValueError(f'Output exists: {a.output}')
    hc,re,q,js,train=inputs(a.results_root)
    a.output.mkdir(parents=True)
    data=stream_table(hc,re)
    data.to_csv(a.output/'research_stream_effects.csv',index=False)
    source_paths=[a.results_root/p for p in (
        'dynamics-q-750k-v1/comparison/per_adaptation_seed_comparison.csv',
        'reacher-dynamics-q-v1/comparison/four_way_per_adaptation_seed.csv',
        'reacher-dynamics-q-v1/comparison/update_diagnostics.csv',
        'q-planning-mechanism-750k-v1/tables/immediate_per_stream.csv',
        'q-planning-mechanism-750k-v1/tables/divergence_per_stream.csv')]
    (a.output/'SOURCE_INPUTS.json').write_text(json.dumps({str(p.relative_to(a.results_root)):
        digest(p) for p in source_paths},indent=2))
    for task in ('HalfCheetah','Reacher'):paired_plot(data,task,a.output)
    mechanism_plot(data,re,a.output)
    halfcheetah_reward(a.results_root,a.output)
    q_plots(q,js,train,a.output)
    report(data,a.output)
    print(f'Wrote 7 figures and report to {a.output}')


if __name__=='__main__':main()
