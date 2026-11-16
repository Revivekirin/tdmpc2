"""Offline evidence report. Missing closed-loop measurements remain explicit."""
import html
import shutil
import zipfile
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from ..metrics import read,write,require,sha256
from ..config import TASKS
from ..provenance import verified_artifact
from .aggregate import return_bridge


def integrate(root,out,swap_source=None):
    require(not out.exists() or {p.name for p in out.iterdir()} <= {'reports','validation'},'Report output must be new or contain only pre-report audits')
    verified={}
    for name in ('q-terminal-planning-v1','q-terminal-planning-presentation-v2','q-decision-relevance-v1'):
        verified[name]=verified_artifact(root,name)
    for sub in ('tables','figures/main','figures/supplementary','provenance'):(out/sub).mkdir(parents=True)
    v2=verified['q-terminal-planning-presentation-v2'][0]
    historical=pd.read_csv(v2/'tables/paired_heldout_returns.csv')
    require(not historical.duplicated(['task','family','severity','adaptation_seed','evaluation_seed','method']).any(),'Duplicate historical return key')
    long=historical.rename(columns={'family':'shift_family','return':'value'}).copy()
    long['metric']='episode_return';long['unit']='environment return';long['shift_schedule']='fixed'
    long['protocol']='historical-fixed-eval';long['checkpoint_sha256']=long.task.map({t:c['sha'] for t,c in TASKS.items()})
    long['source_file']=str(v2/'tables/paired_heldout_returns.csv');long['evidence_level']='measured_environment'
    long['is_measured_environment_outcome']=True;long['is_counterfactual_model_inference']=False
    long.to_csv(out/'tables/canonical_returns.csv',index=False)
    assets=[]
    for name,(source,record) in verified.items():
        write(out/'provenance'/f'{name}.json',dict(path=str(source),**record))
        for f in source.glob('figures/main/*'):
            if f.suffix in ('.png','.svg'):
                target=out/'figures/main'/f'{name}-{f.name}';shutil.copy2(f,target)
                if f.suffix=='.png':assets.append(target.relative_to(out).as_posix())
        for f in source.glob('tables/*.csv'):shutil.copy2(f,out/'tables'/f'{name}-{f.name}')
    gaps=['Historical diagnostics are not Q calibration, MPC regret, or a causal proof of control recovery.',
          'Full cross-family physical/prediction figure migration is not validated in this report. Existing source tables are preserved.',
          'Numerical migration of all historical inference streams has not been rerun; targeted regression is separate.']
    measured=False
    if swap_source is not None:
        done=read(swap_source/'completion.json')
        require(done['status']=='PASS' and not done['smoke'] and done['streams']==108,'Full verified Q-swap batch required')
        provenance=read(swap_source/'provenance.json')
        for path,record in provenance['sources'].items():require(sha256(Path(path))==record['sha256'],'Q-swap source changed')
        episode_files=sorted(swap_source.glob('*/episodes.csv'))
        require(len(episode_files)==108,'Missing Q-swap stream files')
        for path in episode_files:
            child=read(path.parent/'completion.json')
            require(child['status']=='PASS' and child['episodes']==40,'Incomplete child evaluation')
            for relative,digest in child['files'].items():require(sha256(path.parent/relative)==digest,'Changed Q-swap output')
        episodes=pd.concat([pd.read_csv(p) for p in episode_files],ignore_index=True)
        require(len(episodes)==4320,'Incomplete Q-swap episode matrix')
        paired,summary=return_bridge(episodes,historical)
        paired.to_csv(out/'tables/critic_return_bridge.csv',index=False);summary.to_csv(out/'tables/critic_contribution.csv',index=False)
        for task,frame in summary.groupby('task'):
            fig,axes=plt.subplots(1,2,figsize=(12,4))
            for ax,metric in zip(axes,['critic','predictor']):
                for family,g in frame[frame.metric==metric].groupby('family'):
                    g=g.sort_values('severity');ax.plot(g.severity,g['mean'],marker='o',label=family);ax.fill_between(g.severity,g.ci_low,g.ci_high,alpha=.15)
                ax.axhline(0,color='gray');ax.set(title=f'{task}: {metric} substitution',xlabel='severity multiplier',ylabel='Paired actual episode-return difference');ax.legend()
            fig.suptitle('3 adaptation streams; 20 paired episodes/stream; descriptive hierarchical 95% interval')
            fig.tight_layout()
            for ext in ('png','svg'):fig.savefig(out/'figures/main'/f'{task}-closed-loop-bridge.{ext}',dpi=170)
            plt.close(fig);assets.append(f'figures/main/{task}-closed-loop-bridge.png')
        write(out/'provenance/q-swap.json',provenance);measured=True
    else:gaps.append('Closed-loop critic intervention NOT_RUN: no new measured critic contribution or return bridge.')
    sections=[('Research question','When is dynamics-only adaptation sufficient, and when does critic adaptation improve actual control?'),
      ('Theory','For a fixed policy, equal rewards and discounted kernels: ΔQπ = γ(I−γK₁π)⁻¹(K₁π−K₀π)Q₀π. This allows value mismatch; it does not prove adaptation improves control.'),
      ('Why TD-MPC2','Latent dynamics, reward, policy prior and terminal critic jointly determine short-horizon model-predictive scores.'),
      ('Adaptation configuration','Frozen, Dynamics-only, Dynamics+Q and Joint differ in components and update budgets. Equal interaction counts do not imply equal optimization budgets. Joint own-encoder error is not shared-latent accuracy.'),
      ('Physical validation','Use original fixed physics wrappers and readback. Numeric scales differ between mass, damping and tasks.'),
      ('Prediction and control','Prediction errors and measured returns are distinct outcomes. No inference from lower latent MSE alone to improved return.'),
      ('Terminal Q','Gₘ(U)=Rₘᴴ(U)+γᴴQₘ(zᴴₘ,aᴴ). ΔG = ΔR_pred + ΔQ_pred + ΔQ_critic under the documented common-action intervention. This is a predicted score decomposition.'),
      ('Candidate preference','For otherwise fixed scores and positive baseline margin Mᵢⱼ, γᴴ(δQⱼ−δQᵢ)>Mᵢⱼ reverses the candidate ordering. Diagnostic candidates are not historical MPPI execution.'),
      ('Native MPPI','Matched random draws and actual native code test planner sensitivity, not true action quality or planning regret.'),
      ('Closed-loop intervention','ΔJ_critic=J(dDQ,QDQ)−J(dDQ,Q0). '+('Measured paired episodes included.' if measured else 'NOT_RUN; no return benefit is inferred.')),
      ('Predictor/critic bridge','J(dDQ,QDQ)−J(dD,Q0)=[J(dDQ,Q0)−J(dD,Q0)]+[J(dDQ,QDQ)−J(dDQ,Q0)]. This is a path-dependent substitution decomposition, not unique attribution.'),
      ('Cross-task interpretation','Keep HalfCheetah and Reacher on separate return/Q scales. Shared nominal family labels are not independent nominal tasks.'),
      ('Limitations','\n'.join(gaps)),('Reproduction','python -m experiments.nsgym.analysis.control_q.run --help. All figures/tables remain tied to original source hashes; offline assets only.')]
    md=['# Integrated control–critic evidence','', '**Status: PARTIAL**','']
    for title,text in sections:md.extend(['## '+title,'',text,''])
    for asset in assets:md.append(f'![{Path(asset).stem}]({asset})')
    (out/'report.md').write_text('\n'.join(md))
    body='<h1>Integrated control–critic evidence</h1><p><strong>PARTIAL</strong></p><nav>'+''.join(f'<a href="#s{i}">{html.escape(t)}</a> · ' for i,(t,_) in enumerate(sections))+'</nav>'
    for i,(title,text) in enumerate(sections):body+=f'<section id="s{i}"><h2>{html.escape(title)}</h2><p>{html.escape(text)}</p></section>'
    for asset in assets:body+=f'<figure><img src="{asset}" alt="{html.escape(Path(asset).stem)}"><figcaption>{html.escape(Path(asset).stem)}</figcaption></figure>'
    (out/'report.html').write_text('<!doctype html><html lang="en"><meta charset="utf-8"><title>Control–critic research</title><style>body{max-width:1100px;margin:40px auto;font:17px/1.6 system-ui;padding:0 24px}img{max-width:100%}nav{font-size:14px}section{border-top:1px solid #ddd}p{white-space:pre-line}</style>'+body+'</html>')
    require(all((out/a).is_file() for a in assets),'Broken figure link')
    write(out/'validation.json',dict(status='PARTIAL',offline_assets=True,figure_links_valid=True,closed_loop_measured=measured,gaps=gaps))
    write(out/'completion.json',dict(status='PARTIAL',historical_return_rows=len(long),closed_loop_measured=measured,limitations=gaps))
    target=out.with_suffix('.zip');require(not target.exists(),'ZIP already exists')
    with zipfile.ZipFile(target,'w',zipfile.ZIP_DEFLATED) as z:
        for f in sorted(out.rglob('*')):
            if f.is_file():z.write(f,f.relative_to(out))
    with zipfile.ZipFile(target) as z:require(z.testzip() is None,'ZIP corruption')
