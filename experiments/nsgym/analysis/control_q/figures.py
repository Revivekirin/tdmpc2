"""The five research figures and three restrained explanatory appendices."""
from __future__ import annotations

from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .config import COLORS, GRID, LABEL, TASKS
from .data import paired_terminal_arrays
from .metrics import require

METHODS=('F','D','DQ','J')
PAIR_LABEL={'D → DQ':'Dynamics-only → D+Q','DQ_F → DQ':'Frozen → adapted critic'}


def _save(fig,path):
    path.parent.mkdir(parents=True,exist_ok=True)
    fig.tight_layout()
    fig.savefig(path,dpi=180,bbox_inches='tight')
    plt.close(fig)


def _axis(ax,title,ylabel,zero=True):
    ax.set(title=title,xlabel='Physical severity multiplier',ylabel=ylabel)
    ax.set_xticks(GRID)
    ax.axvline(1,color='0.6',ls=':',lw=.8)
    if zero:ax.axhline(0,color='0.7',lw=.6)
    ax.grid(alpha=.16)


def _curves(ax,frame,metric,by='method',selected=None):
    d=frame[frame.metric==metric]
    for key,g in d.groupby(by):
        if selected and key not in selected:continue
        g=g.sort_values('severity')
        require(len(g)==6 and np.allclose(g.severity.to_numpy(),GRID),f'Incomplete {metric}: {key}')
        y=g['mean'].to_numpy(float)
        lo=g.ci_low.to_numpy(float);hi=g.ci_high.to_numpy(float)
        require(np.isfinite(np.r_[y,lo,hi]).all(),f'Nonfinite {metric}: {key}')
        label=LABEL.get(key,PAIR_LABEL.get(key,key))
        color=COLORS.get(key,None)
        ax.plot(GRID,y,'o-',ms=3,label=label,color=color)
        ax.fill_between(GRID,lo,hi,alpha=.12,color=color)


def _terminal_table(arrays,discount_by):
    """Reaggregate the same fixed two Q heads for every method and condition."""
    from .metrics import aggregate
    rows=[];validations=[];examples={};candidate_rows=[]
    for (task,family,k,seed),path in sorted(arrays.items()):
        a=paired_terminal_arrays(path)
        require(all(v.shape==(80,) for v in a['q'].values()),f'Terminal shape: {path}')
        if k==2 and seed==TASKS[task]['adaptation'][0]:examples[(task,family)]=a
        delta=a['terminal']['DQ']-a['terminal']['DQ_F']
        before=a['score']['DQ_F'];after=a['score']['DQ']
        d=delta
        discount=discount_by[(task,family,k,seed)]
        for key,left,right in (('predictor_D','D','F'),('predictor_D_to_DQ','DQ_F','D'),
                               ('total_DQ_minus_F','DQ','F')):
            require(np.allclose(discount*(a['q'][left]-a['q'][right]),a['effects'][key],
                                atol=3e-4,rtol=2e-5),f'Terminal decomposition parity: {path}/{key}')
        require(np.allclose(discount*(a['q']['DQ']-a['q']['DQ_F']),a['critic_effect'],
                            atol=3e-4,rtol=2e-5),f'Critic-only tape parity: {path}')
        # For every strict pairwise baseline preference, a sign reversal must
        # satisfy the exact candidate-wise score difference identity.
        margin=before[:,:,None]-before[:,None,:]
        induced=d[:,None,:]-d[:,:,None]
        reversal=(margin>1e-8)&((after[:,:,None]-after[:,None,:])< -1e-8)
        require(np.all(induced[reversal]>margin[reversal]-5e-4),
                f'Candidate reversal identity: {path}')
        validations.append(dict(task=task,family=family,severity=k,adaptation_seed=seed,
                                candidate_states=len(before),candidate_count=before.shape[1],
                                ranking_reversals=int(reversal.sum()),
                                max_score_identity_error=float(np.max(np.abs(after-before-d))),
                                frozen_d_immediate_equal=True))
        scalar_delta=d/discount
        for state in range(len(before)):
            old_winner=int(np.argmax(before[state]));new_winner=int(np.argmax(after[state]))
            candidate_rows.append(dict(task=task,family=family,severity=k,adaptation_seed=seed,
                evaluation_seed=int(a['seeds'][state//4]),state_index=state,candidates=32,
                mean_signed_delta_q=float(scalar_delta[state].mean()),
                candidate_dependent_sd=float(scalar_delta[state].std()),
                baseline_score_margin=float(before[state,old_winner]-before[state,new_winner]),
                top1_reversal=int(old_winner!=new_winner)))
        for ei,eval_seed in enumerate(a['seeds']):
            sl=slice(ei*4,(ei+1)*4)
            base=float(a['q']['F'][sl].mean())
            for method in METHODS:
                rows.append(dict(task=task,family=family,severity=k,adaptation_seed=seed,
                                 evaluation_seed=int(eval_seed),method=method,
                                 terminal_delta=float(a['q'][method][sl].mean()-base)))
            rows.append(dict(task=task,family=family,severity=k,adaptation_seed=seed,
                             evaluation_seed=int(eval_seed),method='Qswap',
                             terminal_delta=float(a['critic_effect'][sl].mean())))
    summary=aggregate(pd.DataFrame(rows),['task','family','severity','method'],['terminal_delta'])
    return pd.DataFrame(rows),summary,pd.DataFrame(validations),examples,pd.DataFrame(candidate_rows)


def _example(figdir,task,family,a,discount,metadata):
    # Predeclared explanatory rule: k=2, smallest adaptation seed, earliest
    # state with a top-1 Q-swap reversal; if none, the first state. This
    # selects a mechanism example, never a favorable return condition.
    stem=f'{task}/{family}/appendix'
    qf=a['q']['F'];q_dq=a['q']['DQ']
    fig,ax=plt.subplots(figsize=(5.6,3.5))
    for method in ('D','DQ','J'):
        change=np.sort(a['q'][method]-qf)
        ax.plot(change,np.arange(1,len(change)+1)/len(change),label=LABEL[method],color=COLORS[method])
    ax.set(xlabel='Matched-state terminal scalar Q − Frozen',ylabel='Empirical fraction',
           title='Fixed heads 0,1; first adaptation stream, k=2')
    ax.legend();_save(fig,figdir/stem/'A1_matched_scalar_q.png')
    changing=np.flatnonzero(a['score']['DQ_F'].argmax(axis=1)!=a['score']['DQ'].argmax(axis=1))
    state=int(changing[0]) if len(changing) else 0
    before=a['score']['DQ_F'][state];after=a['score']['DQ'][state]
    delta=(a['terminal']['DQ'][state]-a['terminal']['DQ_F'][state])/discount
    require(np.allclose(before+a['terminal']['DQ'][state]-a['terminal']['DQ_F'][state],after,
                        atol=5e-4,rtol=2e-5),'Representative score identity')
    winner=int(np.argmax(before));new_winner=int(np.argmax(after))
    margin=float(before[winner]-before[new_winner])
    fig,ax=plt.subplots(figsize=(6.2,3.5))
    ax.plot(np.arange(32),delta,'o-',ms=3,label='Q swap only')
    ax.scatter(winner,delta[winner],c='#0072B2',s=75,zorder=3,label='Baseline top-1')
    if new_winner!=winner:
        ax.scatter(new_winner,delta[new_winner],c='#D55E00',s=75,zorder=3,label='After Q swap top-1')
    ax.set(xlabel='Common diagnostic candidate ID',ylabel='Terminal scalar ΔQ',
           title=f'State {state}: winner {winner} → {new_winner}; score margin {margin:.3g}')
    ax.legend();_save(fig,figdir/stem/'A2_action_dependent_terminal_q.png')
    fig,axes=plt.subplots(1,3,figsize=(12,3.4))
    for method in ('DQ_F','DQ'):
        color=COLORS[method]
        for ax,values in zip(axes,(a['rewards'][method][state],a['terminal'][method][state],a['score'][method][state])):
            ax.plot(np.arange(32),values-np.median(values),'o-',ms=2.5,label=LABEL[method],color=color)
    for ax,title in zip(axes,('Predicted reward','Discounted terminal Q','Total score')):
        ax.set(xlabel='Same candidate ID',ylabel='Centered score component',title=title)
    axes[0].legend();_save(fig,figdir/stem/'A3_candidate_score_example.png')
    metadata.append(dict(task=task,family=family,severity=2.,adaptation_seed=TASKS[task]['adaptation'][0],
                         evaluation_seed=int(a['seeds'][state//4]),state_index=state,candidates=32,
                         state_rule='earliest top-1 reversal, else first state',
                         original_winner=winner,swapped_winner=new_winner,
                         baseline_margin=margin,head_pair='0,1'))


def render(root,output,tables,arrays,paths,swap=None):
    """Generate one progression per family; never run a model or simulator."""
    out=output/'figures';out.mkdir(parents=True,exist_ok=True)
    discount_table=pd.read_csv(paths['terminal']/'tables/terminal_effect_decomposition_per_seed.csv')
    grouped=discount_table.groupby(['task','family','severity','adaptation_seed']).discount_power
    require(grouped.nunique().eq(1).all(),'Inconsistent saved discount powers')
    discount_by={key:float(group.iloc[0]) for key,group in grouped}
    per_seed,terminal,checks,examples,candidate_rows=_terminal_table(arrays,discount_by)
    candidate_rows.to_csv(output/'candidate_q_change_per_state.csv',index=False)
    candidate_rows.groupby(['task','family','severity','adaptation_seed','evaluation_seed'])[
        ['mean_signed_delta_q','candidate_dependent_sd','baseline_score_margin','top1_reversal']
    ].mean().reset_index().to_csv(output/'candidate_q_change_per_seed.csv',index=False)
    # The saved decomposition used the identical fixed head pair but a
    # different deterministic bootstrap key. Reuse its verified interval
    # after asserting the independently reconstructed condition means.
    archived=tables['effects'][tables['effects'].metric=='critic_DQ']
    for row in archived.itertuples():
        mask=(terminal.task==row.task)&(terminal.family==row.family)&(terminal.severity==row.severity)&\
             (terminal.method=='Qswap')&(terminal.metric=='terminal_delta')
        require(mask.sum()==1 and np.isclose(float(terminal.loc[mask,'mean'].iloc[0]),row.mean,atol=1e-9),
                f'Q-swap terminal mean parity: {row.task}/{row.family}/{row.severity}')
        for key in ('mean','median','ci_low','ci_high'):
            terminal.loc[mask,key]=getattr(row,key)
    per_seed.to_csv(output/'terminal_per_seed.csv',index=False)
    terminal.to_csv(output/'terminal_fixed_head_summary.csv',index=False)
    checks.to_csv(output/'candidate_identity_checks.csv',index=False)
    metadata=[];source_map=[];coverage=[]
    if swap is not None:
        from .reporting.aggregate import return_bridge
        from .metrics import aggregate
        source,episodes=swap
        historical=pd.read_csv(paths['returns']/'tables/paired_heldout_returns.csv')
        bridge,bridge_summary=return_bridge(episodes,historical)
        bridge.to_csv(output/'closed_loop_paired_episodes.csv',index=False)
        bridge_summary.to_csv(output/'closed_loop_critic_summary.csv',index=False)
        empirical=pd.concat([historical[historical.method=='D'].rename(columns={'return':'episode_return'}),
                             episodes],ignore_index=True)
        empirical=empirical[empirical.method.isin(('D','DQ_F','DQ'))]
        control_swap=aggregate(empirical,['task','family','severity','method'],['episode_return'])
    else:bridge_summary=None;control_swap=None
    for task,t in TASKS.items():
        for family in t['families']:
            base=out/task/family;base.mkdir(parents=True,exist_ok=True)
            def sub(name):return tables[name][(tables[name].task==task)&(tables[name].family==family)]
            control=sub('control');ranking=sub('ranking');native=sub('native');term=terminal[(terminal.task==task)&(terminal.family==family)]
            # 01: measured return and fixed-encoder prediction share a physical
            # severity axis, but never a numerical y axis.
            fig,axes=plt.subplots(1,2,figsize=(11,3.8))
            _curves(axes[0],control,'return',selected=METHODS)
            _axis(axes[0],'A. Measured held-out control','Episode return',zero=False)
            _curves(axes[1],control,'fixed_encoder_latent_mse',selected=('F','D','DQ'))
            _axis(axes[1],'B. Fixed-bank prediction','One-step latent MSE',zero=False)
            axes[1].set_yscale('log');axes[0].legend(fontsize=7);axes[1].legend(fontsize=7)
            fig.suptitle(f'{task} | {t["families"][family]} | fixed physical shift')
            _save(fig,base/'01_prediction_control.png')
            # 02: fixed heads 0,1 for both panels. The Q-swap panel is discounted.
            fig,axes=plt.subplots(1,2,figsize=(11,3.8))
            _curves(axes[0],term,'terminal_delta',selected=METHODS)
            _axis(axes[0],'A. Predicted terminal Q − Frozen','Scalar Q change, heads 0,1')
            _curves(axes[1],term,'terminal_delta',selected=('Qswap',))
            _axis(axes[1],'B. Critic-only terminal score change','Discounted score change, heads 0,1')
            axes[0].legend(fontsize=7);axes[1].legend(['DQ predictor + adapted Q − Frozen Q'],fontsize=7)
            fig.suptitle(f'{task} | {t["families"][family]} | common physical starts and action tape')
            _save(fig,base/'02_terminal_q.png')
            fig,axes=plt.subplots(1,2,figsize=(11,3.8))
            for ax,metric,title,ylabel in ((axes[0],'argmax_disagreement','A. Top-1 preference changes','Fraction'),
                                           (axes[1],'spearman','B. Candidate rank agreement','Spearman ρ')):
                frame=ranking[ranking.phase=='fixed_total_score']
                _curves(ax,frame,metric,by='comparison',selected=PAIR_LABEL)
                _axis(ax,title,ylabel);ax.legend(fontsize=7)
            fig.suptitle(f'{task} | {t["families"][family]} | same 32 diagnostic candidates')
            _save(fig,base/'03_candidate_ranking.png')
            last=int(native.iteration.max())
            frame=native[(native.iteration==last)&(native.comparison=='DQ_F → DQ')]
            fig,ax=plt.subplots(figsize=(6.5,3.8))
            _curves(ax,frame,'final_action_l2',by='comparison')
            _axis(ax,'Native MPPI: critic-only proposed-action change','Proposed first-action L2')
            fig.suptitle(f'{task} | {t["families"][family]} | proposals not executed')
            ax.legend(fontsize=7);_save(fig,base/'04_native_mppi.png')
            if bridge_summary is not None:
                fig,axes=plt.subplots(1,2,figsize=(11,3.8))
                group=control_swap[(control_swap.task==task)&(control_swap.family==family)]
                _curves(axes[0],group,'episode_return',selected=('D','DQ_F','DQ'))
                _axis(axes[0],'A. Actual closed-loop control','Measured episode return',zero=False)
                axes[0].legend(fontsize=7)
                effect=bridge_summary[(bridge_summary.task==task)&(bridge_summary.family==family)&
                                      (bridge_summary.metric=='critic')].sort_values('severity')
                require(len(effect)==6,'Closed-loop critic severity coverage')
                axes[1].plot(effect.severity,effect['mean'],'o-',label='D+Q critic − Frozen critic')
                axes[1].fill_between(effect.severity,effect.ci_low,effect.ci_high,alpha=.15)
                _axis(axes[1],'B. Paired critic-only effect','Measured return difference')
                axes[1].legend(fontsize=7)
                fig.suptitle(f'{task} | {t["families"][family]} | matched held-out evaluation')
                _save(fig,base/'05_closed_loop_q_swap.png')
            a=examples[(task,family)]
            match=discount_table[(discount_table.task==task)&(discount_table.family==family)&(discount_table.severity==2)]
            require(len(match)==60 and match.discount_power.nunique()==1,'Discount protocol mismatch')
            _example(out,task,family,a,float(match.discount_power.iloc[0]),metadata)
            for number,sources in [('01',[paths['decision']/'tables/actual_control_prediction_evidence.csv',
                                          paths['decision']/'tables/control_prediction_summary.csv']),
                                   ('02',[paths['terminal']/'q_arrays',
                                          paths['terminal']/'tables/terminal_effect_decomposition_per_seed.csv']),
                                   ('03',[paths['decision']/'tables/fixed_summary.csv',paths['terminal']/'q_arrays']),
                                   ('04',[paths['decision']/'tables/native_summary.csv',
                                          paths['decision']/'tables/native_per_state.csv'])]:
                source_map.append(dict(task=task,family=family,figure=number,source='; '.join(map(str,sources)),
                                       severity_count=6,adaptation_seeds=3,evaluation_seeds=20))
            if bridge_summary is not None:
                source_map.append(dict(task=task,family=family,figure='05',source=str(source/'*/episodes.csv'),
                                       severity_count=6,adaptation_seeds=3,evaluation_seeds=20))
            coverage.extend(dict(task=task,family=family,severity=k,adaptation_seeds=3,evaluation_seeds=20,
                                 figures='01,02,03,04'+(',05' if bridge_summary is not None else '')) for k in GRID)
    pd.DataFrame(metadata).to_csv(output/'REPRESENTATIVE_SELECTION.csv',index=False)
    pd.DataFrame(source_map).to_csv(output/'FIGURE_SOURCE_MAP.csv',index=False)
    pd.DataFrame(coverage).to_csv(output/'FIGURE_COVERAGE.csv',index=False)
    return dict(png_count=len(list(out.rglob('*.png'))),families=6,conditions=36,
                inference_streams=len(checks),candidate_reversals=int(checks.ranking_reversals.sum()),
                max_score_identity_error=float(checks.max_score_identity_error.max()),
                closed_loop_critic_effect='MEASURED' if bridge_summary is not None else 'NOT_MEASURED')
