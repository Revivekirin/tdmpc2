"""Migrated historical computations; numerical formulas preserved."""
from __future__ import annotations
import hashlib
import json
import time
import zipfile
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from experiments.nsgym.analysis.control_q.config import COLORS, GRID, LABEL, METHODS, TASKS
from experiments.nsgym.analysis.reacher_dq_figures import sha256
from experiments.nsgym.analysis.control_q.metrics import aggregate, centered_components, probability_window, read, require, write
from experiments.nsgym.analysis.control_q.models import Sources


def figures(out, summaries, array_paths):
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    count=0
    def save(fig,area,name):
        nonlocal count
        p=out/'figures'/area;p.mkdir(parents=True,exist_ok=True)
        for ext in ('png','svg'):fig.savefig(p/f'{name}.{ext}',dpi=180,bbox_inches='tight')
        plt.close(fig);count+=1
    def lines(ax,df,metric,field='method'):
        if df.empty:return
        for method,g in df[df.metric==metric].groupby(field):
            g=g.sort_values('severity');color=COLORS.get(method,dict(zip(['F → D','D → DQ','F → J','DQ → J','DQ_F → DQ'],['#0072B2','#D55E00','#009E73','#CC79A7','#9467BD'])).get(method))
            ax.plot(g.severity,g['mean'],marker='o',ms=4,label=LABEL.get(method,method).replace('D+Q predictor / Frozen Q','D+Q predictor, F critic'),color=color,ls='--' if method=='D' else '-')
            ax.fill_between(g.severity,g.ci_low,g.ci_high,alpha=.12,color=color)
        ax.set(xlabel='Severity multiplier k',ylabel=metric.replace('_',' '));ax.set_xticks(GRID);ax.grid(alpha=.2)
        ax.axvline(1,ls=':',color='grey',lw=.8)
    for task,t in TASKS.items():
        for family,parameter in t['families'].items():
            select=lambda d:d[(d.task==task)&(d.family==family)] if not d.empty else d
            imm=select(summaries['immediate']);term=select(summaries['terminal']);eff=select(summaries['effects'])
            rank=select(summaries['ranking_summary']);comp=select(summaries['components'])
            if imm.empty:continue
            for area,name,df,metrics,field in [
                ('immediate_q','A',imm,['signed_delta','absolute_delta','head_std'],'method'),
                ('immediate_q','A_supplement',imm,['entropy','js_frozen'],'method'),
                ('terminal_q','B',term[term.method.isin(METHODS)] if not term.empty else term,['scalar_q','signed_delta'],'method')]:
                if df.empty:continue
                fig,axes=plt.subplots(1,len(metrics),figsize=(5*len(metrics),4),squeeze=False)
                for ax,metric in zip(axes[0],metrics):lines(ax,df,metric,field)
                axes[0,0].legend(fontsize=8)
                fig.suptitle(f'{task} | {parameter} | {name} — '+('shared-input immediate Q' if area=='immediate_q' else 'multi-state executed-tape terminal Q'))
                fig.text(.5,-.03,f'Descriptive paired hierarchical 95% intervals; up to {int(df.n_adaptation_streams.max())} adaptation streams × {int(df.n_evaluation_seeds.max())} eval seeds. k=1 labels alias nominal physics. Q movement is not accuracy.',ha='center',fontsize=8)
                fig.tight_layout();save(fig,area,f'{name}_{task}_{family}')
            if eff.empty:continue
            fig,ax=plt.subplots(figsize=(8,4))
            for metric,label in [('predictor_D','Frozen → D predictor'),('predictor_D_to_DQ','D → D+Q predictor'),('critic_DQ','Frozen Q → D+Q Q')]:
                g=eff[eff.metric==metric].sort_values('severity')
                ax.plot(g.severity,g['mean'],marker='o',label=label);ax.fill_between(g.severity,g.ci_low,g.ci_high,alpha=.12)
            ax.axhline(0,color='grey',lw=.8);ax.set(xlabel='Severity k',ylabel='Discounted terminal contribution change',title=f'{task} | {parameter} | C — additive controlled decomposition')
            ax.legend(fontsize=8);fig.tight_layout();save(fig,'terminal_q',f'C_{task}_{family}')
            fig,axes=plt.subplots(2,3,figsize=(15,8))
            controlled=rank[rank.protocol=='controlled'];native=rank[rank.protocol=='native_terminal_sensitivity']
            for ax,df,metric,field in [(axes[0,0],controlled,'top_k_jaccard','comparison'),(axes[0,1],controlled,'spearman','comparison'),
                (axes[0,2],native,'spearman','comparison'),(axes[1,0],comp,'mean_candidate_reward','method'),
                (axes[1,1],comp,'mean_candidate_terminal','method'),(axes[1,2],comp,'margin','method')]:lines(ax,df,metric,field)
            axes[0,2].set_title('Separate native terminal stochastic sensitivity')
            axes[0,0].legend(fontsize=7);axes[1,0].legend(fontsize=8)
            fig.suptitle(f'{task} | {parameter} | D — fixed diagnostic candidates (not historical MPPI)')
            fig.tight_layout();save(fig,'planning',f'D_{task}_{family}')
    # Predeclared examples: severity 2, lowest adaptation seed, earliest eval/time.
    for p in array_paths:
        if '-k2-' not in p.name or not any(f'seed{t["adaptation"][0]}' in p.name for t in TASKS.values()):continue
        with np.load(p) as data:
            if 'F_terminal_probability' not in data:continue
            fig,axes=plt.subplots(1,2,figsize=(12,4))
            for ax,kind in zip(axes,('immediate','terminal')):
                for m in METHODS:
                    prob=data[f'{m}_{kind}_probability'][:,0]
                    for head in prob:ax.plot(data['symlog_support'],head,color=COLORS[m],alpha=.12,lw=.7)
                    ax.plot(data['symlog_support'],prob.mean(0),color=COLORS[m],label=LABEL[m],ls='--' if m=='D' else '-',lw=1.5)
                ax.set(xlabel='Shared symlog support bin',ylabel='Categorical probability mass',title=kind+' Q: per-head faint, ensemble mean bold')
            axes[0].legend();fig.suptitle(p.stem+' | E — predeclared example, not population evidence')
            fig.tight_layout();save(fig,'terminal_q',f'E_{p.stem}')
    return count


def finish(out, rows, validations, src, gaps, configs, args, elapsed):
    finish_started=time.monotonic()
    frames={k:pd.DataFrame(v) for k,v in rows.items()}
    common=['task','family','severity']
    metrics=['scalar_q','head_std','entropy','js_frozen','signed_delta','absolute_delta']
    summaries={}
    for name,group,met in [('immediate',common+['method'],metrics),('terminal',common+['method'],metrics),
        ('effects',common,['predictor_D','predictor_D_to_DQ','critic_DQ','total_DQ_minus_F']),
        ('ranking_summary',common+['comparison','protocol'],['spearman','kendall','top_k_overlap','top_k_jaccard','top_action_l2','argmax_disagreement']),
        ('components',common+['method'],['selected_reward','selected_terminal','mean_candidate_reward','mean_candidate_terminal','margin'])]:
        summaries[name]=aggregate(frames[name],group,met)
    names={'immediate':'immediate_scalar_q','terminal':'terminal_q_by_condition','effects':'terminal_effect_decomposition',
           'ranking_summary':'ranking_summary','components':'candidate_score_components'}
    for name,stem in names.items():
        frames[name].to_csv(out/'tables'/f'{stem}_per_seed.csv',index=False)
        summaries[name].to_csv(out/'tables'/f'{stem}.csv',index=False)
    frames['rankings'].to_csv(out/'tables/fixed_candidate_rankings.csv',index=False)
    nfig=figures(out,summaries,list((out/'q_arrays').glob('*.npz')))
    changed=src.verify();require(not changed,f'Original files changed: {changed}')
    expected=4 if args.smoke else 108
    complete=[v for v in validations if v.get('status')=='PASS']
    status='PASS' if len(complete)==expected and not gaps else 'PASS_WITH_COVERAGE_GAPS' if complete or not frames['immediate'].empty else 'BLOCKED'
    validation=dict(status='PASS' if len(complete)==expected else 'PARTIAL',streams=validations,
        source_hashes_unchanged=True,environment_transitions=0,optimizer_steps=0,
        smoke=args.smoke,expected_complete_streams=expected,complete_streams=len(complete))
    write(out/'NUMERICAL_VALIDATION.json',validation);write(out/'SOURCE_INPUTS.json',src.files)
    pd.DataFrame(src.inventory).to_csv(out/'tables/SOURCE_INVENTORY.csv',index=False)
    write(out/'protocol.json',dict(states_per_evaluation_seed=args.states,immediate_states=16,candidates=args.candidates,
        starting_rule='linspace(0, episode_length - planning_horizon - 1, states), integer indices',
        terminal_action='saved Frozen action at start+H, identical for all models and all candidates at a start',
        primary_head_pair=[0,1],terminal_distribution_summary='mean of individually decoded heads; decomposition/planning use fixed pair 0,1',
        candidate_origin='new diagnostic Gaussian(0,max_std) clipped to [-1,1], no actor prior/warmstart; not historical MPPI candidates',
        candidate_seed='sha256(task,family,severity,evaluation_seed,timestep,candidates) first 4 bytes; independent of adaptation seed',
        sensitivity='same exogenous candidates; native pi terminal action and random two-head average sampled separately per planning state; paired RNG repeats, separate protocol',
        sensitivity_repeats=args.sensitivity_repeats,configurations=configs,
        actual_candidate_returns='NOT_MEASURED',planning_regret='NOT_MEASURED',q_calibration='NOT_MEASURED'))
    lines=['# Q terminal and fixed-candidate planning analysis','',
        f'Status: {status}. Completed inference streams: {len(complete)}/{expected}; figures: {nfig}.',
        'One stream = one adaptation seed at one task/family/severity; each complete full stream uses 20 shared evaluation seeds.',
        'Immediate Frozen = D is a checked structural invariant. Native per-head scalar decoding precedes ensemble aggregation.',
        'Terminal A/B/C/D and Joint use common physical initial observations, saved action tapes and the same saved terminal action.',
        'Primary candidate scores use fixed heads 0,1; terminal distribution figures use all-head scalar means. These are explicitly different aggregations.',
        'Native planner sensitivity uses stochastic actor terminal actions and random two-head averaging on the same candidate bank. It is not a historical MPPI replay.',
        'The native MPPI loop additionally samples/refines candidates and samples an elite; predicted top-1 is a diagnostic, not the actual historical chosen action.',
        'Terminal decomposition is algebraically exact on each common tape; Joint uses its own encoder and is never subjected to incompatible latent swaps.',
        '', '## Observed severity-2 changes']
    for task in TASKS:
        e=summaries['effects'];r=summaries['ranking_summary']
        if e.empty:continue
        for family in TASKS[task]['families']:
            v=e[(e.task==task)&(e.family==family)&(e.severity==2)]
            if v.empty:continue
            d=dict(zip(v.metric,v['mean']))
            s=r[(r.task==task)&(r.family==family)&(r.severity==2)&(r.protocol=='controlled')&(r.comparison=='D → DQ')]
            rr=dict(zip(s.metric,s['mean']))
            lines.append(f'- {task} {family}: discounted predictor D={d.get("predictor_D",float("nan")):.6g}; D→DQ predictor={d.get("predictor_D_to_DQ",float("nan")):.6g}; critic={d.get("critic_DQ",float("nan")):.6g}; D vs DQ Spearman={rr.get("spearman",float("nan")):.4g}, top-K Jaccard={rr.get("top_k_jaccard",float("nan")):.4g}.')
    limitations=['# Coverage and limitations','',
        'No environment transitions, optimizer steps, retraining, or historical-result mutations.',
        f'All intervals are descriptive hierarchical bootstraps: states/candidates/heads first aggregate within each evaluation seed; resample adaptation streams and paired evaluation seeds. This run uses {1 if args.smoke else 3} adaptation streams and {2 if args.smoke else 20} evaluation seeds per condition.',
        'Frozen is one shared model, not three independently trained baselines. Methods share evaluation seeds; their online training trajectories differ.',
        'Severity 1 labels alias nominal physics and are never pooled as independent tasks. Raw Q magnitudes are never pooled across tasks.',
        'Fixed common terminal actions/pair isolate controlled changes but differ from native stochastic terminal selection. Sensitivity is separately tabulated.',
        'Reward/terminal decomposition and ranking differences do not establish physical candidate accuracy, causal control recovery, Q calibration, or optimal actions.',
        'Actual candidate returns and planning regret remain unmeasured. Saved actual Gaussian MPPI candidates with full matching provenance were not available; diagnostic candidates are explicitly generated.',
        'Joint latents are not compared to Frozen latents. Immediate entropy reflects categorical encoding, not calibrated predictive uncertainty.',
        'A complete run inventories historical archives, but recomputes new statewise diagnostics from verified checkpoints. Existing tables are not used as invented raw samples.',
        'Zero tied-score variance yields undefined rank correlations, recorded as missing with counts, never replaced by a numerical score.',
        '', '## Gaps']+[json.dumps(g) for g in gaps]
    (out/'reports/Q_TERMINAL_PLANNING_ANALYSIS.md').write_text('\n'.join(lines)+'\n')
    (out/'reports/COVERAGE_AND_LIMITATIONS.md').write_text('\n'.join(limitations)+'\n')
    conditions=len({(v['task'],v['family'],v['severity']) for v in complete})
    done=dict(status=status,conditions=conditions,complete_inference_streams=len(complete),expected_streams=expected,
        figures=nfig,numerical_validation=validation['status'],environment_transitions=0,optimizer_steps=0,seconds=elapsed+time.monotonic()-finish_started,inference_inventory_seconds=elapsed)
    write(out/'completion.json',done)
    # Summaries, figures and small selected probability/score arrays only.
    archive=out.with_suffix('.zip')
    require(not archive.exists(),f'Archive already exists: {archive}')
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
        for p in out.rglob('*'):
            if p.is_file() and 'q_arrays' not in p.parts and not p.name.endswith('_per_seed.csv') and p.name!='fixed_candidate_rankings.csv':z.write(p,str(p.relative_to(out)))
        for p in sorted((out/'q_arrays').glob('*-k2-*')):
            if any(f'seed{t["adaptation"][0]}' in p.name for t in TASKS.values()):
                import io
                with np.load(p) as data:
                    small={n:data[n] for n in ('symlog_support','candidate_ids') if n in data}
                    for m in METHODS:
                        for kind in ('immediate','terminal'):
                            key=f'{m}_{kind}_probability'
                            if key in data:small[key]=data[key][:,:1]
                        if m+'_scores' in data:small[m+'_scores']=data[m+'_scores'][:1]
                buffer=io.BytesIO();np.savez_compressed(buffer,**small);z.writestr('representative_arrays/'+p.name,buffer.getvalue())
    print(json.dumps(dict(**done,output=str(out),zip=str(archive)),indent=2),flush=True)
    return status


def presentation(root, source, out):
    """Render v2 from completed v1 arrays and verified actual control summaries."""
    src=Sources();src.track(Path(__file__),'presentation code')
    from experiments.nsgym.analysis.control_q.models import track_implementation
    track_implementation(src)
    def js(p): return read(src.track(p,'completion/validation/provenance'))
    def csv(p): return pd.read_csv(src.track(p,'verified summary'))
    done=js(source/'completion.json'); validation=js(source/'NUMERICAL_VALIDATION.json')
    require(done['status']=='PASS' and done['complete_inference_streams']==108,'Complete v1 required')
    require(validation['status']=='PASS' and len(validation['streams'])==108,'Validated v1 required')
    previous=js(source/'SOURCE_INPUTS.json');protocol=js(source/'protocol.json')
    require(protocol['states_per_evaluation_seed']==4 and protocol['candidates']==32 and protocol['primary_head_pair']==[0,1], 'Unsupported v1 presentation protocol')
    effect=csv(source/'tables/terminal_effect_decomposition.csv')
    effect_seed=csv(source/'tables/terminal_effect_decomposition_per_seed.csv')
    rank=csv(source/'tables/ranking_summary.csv');rank=rank[rank.protocol=='controlled']
    names=['predictor_D','predictor_D_to_DQ','critic_DQ','total_DQ_minus_F']
    residual=effect_seed[names[:3]].sum(axis=1)-effect_seed[names[3]]
    require(np.max(np.abs(residual))<1e-8,'Per-seed additive identity')
    pivot=effect.pivot(index=['task','family','severity'],columns='metric',values='mean')
    require(np.max(np.abs(pivot[names[:3]].sum(axis=1)-pivot[names[3]]))<1e-8,'Summary additive identity')
    controls=[]
    for task,folder,filename in [('halfcheetah','dynamics-q-750k-v1','four_method_paired_comparison.csv'),
                                  ('reacher','reacher-dynamics-q-v1','four_way_per_seed.csv')]:
        base=root/folder/'comparison';require(js(base/'completion.json')['status']=='PASS','Control comparison incomplete')
        path=base/filename;data=csv(path)
        # v1 inventoried these canonical comparison tables; validate their recorded content hash.
        records=[v for p,v in previous.items() if p.endswith('/'+folder+'/comparison/'+filename)]
        require(len(records)==1 and records[0]['sha256']==sha256(path),'Canonical control changed since v1')
        if task=='halfcheetah':
            data['family']=data.parameter.map({'bthigh_damping':'damping','torso_mass':'mass'})
            cols={'F':'frozen_return','D':'ttt_return','DQ':'dq_return','J':'joint_return'}
            contrast='dq_minus_ttt'
        else:
            data=data[data.kind=='shifted'].copy()
            cols={m:n+'_episode_return' for m,n in [('F','Frozen'),('D','Dynamics-only'),('DQ','Dynamics+Q'),('J','Joint RL')]}
            contrast='dq_minus_dynamics'
        require(np.allclose(data[cols['DQ']]-data[cols['D']],data[contrast]),'Paired control contrast mismatch')
        require(not data.duplicated(['family','severity','adaptation_seed','evaluation_seed']).any(),'Duplicate paired controls')
        for (family,k),g in data.groupby(['family','severity']):
            require(family in TASKS[task]['families'] and k in GRID,'Unexpected control condition')
            require(len(g)==60 and sorted(g.adaptation_seed.unique())==TASKS[task]['adaptation'] and
                    sorted(g.evaluation_seed.unique())==TASKS[task]['evaluation'],'Control seed coverage')
        require(len(data)==len(TASKS[task]['families'])*6*60,'Control grid incomplete')
        for m,col in cols.items():
            part=data[['family','severity','adaptation_seed','evaluation_seed']].copy()
            part['task']=task;part['method']=m;part['return']=data[col].to_numpy();controls.append(part)
        part=data[['family','severity','adaptation_seed','evaluation_seed']].copy()
        part['task']=task;part['method']='DQ-D';part['return']=data[contrast].to_numpy();controls.append(part)
    controls=pd.concat(controls,ignore_index=True)
    returns=aggregate(controls,['task','family','severity','method'],['return'])
    out.mkdir(parents=True)
    for name in ['figures','supplementary','tables','validation']:(out/name).mkdir()
    returns.to_csv(out/'tables/heldout_return_summary.csv',index=False)
    controls.to_csv(out/'tables/paired_heldout_returns.csv',index=False)
    write(out/'validation/v1_NUMERICAL_VALIDATION.json',validation)
    write(out/'validation/v1_protocol.json',protocol)
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    figures=[]
    def save(fig,name,main=False,foot=''):
        if foot: fig.text(.5,.015,foot,ha='center',fontsize=9)
        fig.tight_layout(rect=(0,.055,1,.95))
        for ext in ['png','svg']:
            p=out/('figures' if main else 'supplementary')/(name+'.'+ext)
            fig.savefig(p,dpi=180,bbox_inches='tight')
        plt.close(fig);figures.append(name)
    def bars(ax,frame,groups,series,field,labels,colors):
        width=.78/len(series)
        for j,s in enumerate(series):
            rows=[]
            for group in groups:
                row=frame[(frame.family==group)&(frame[field]==s)]
                require(len(row)==1,f'Missing/ambiguous bar: {group} {s}')
                rows.append(row.iloc[0])
            x=np.arange(len(groups))+(j-(len(series)-1)/2)*width
            means=np.array([r['mean'] for r in rows]);lo=np.array([r.ci_low for r in rows]);hi=np.array([r.ci_high for r in rows])
            ax.bar(x,means,width,label=labels[j],color=colors[j],alpha=.9)
            ax.vlines(x,lo,hi,color='black',lw=1);ax.plot(x,means,'_',color='black')
        ax.set_xticks(np.arange(len(groups)),groups);ax.axhline(0,color='.4',lw=.7);ax.grid(axis='y',alpha=.15)
    foot='Paired descriptive 95% bootstrap intervals; 3 adaptation streams × 20 held-out seeds. Frozen is a shared reference.'
    fig,axs=plt.subplots(2,2,figsize=(13,8))
    for row,task in enumerate(TASKS):
        d=returns[(returns.task==task)&(returns.severity==2)]
        families=list(TASKS[task]['families'])
        bars(axs[row,0],d,families,list(METHODS),'method',[LABEL[m] for m in METHODS],[COLORS[m] for m in METHODS])
        bars(axs[row,1],d,families,['DQ-D'],'method',['D+Q − D'],[COLORS['DQ']])
        axs[row,0].set_title(task+' | actual held-out return');axs[row,1].set_title(task+' | paired return difference')
        axs[row,0].set_ylabel('Episode return (higher is better)');axs[row,1].set_ylabel('D+Q − D return')
    axs[0,0].legend(ncol=4,fontsize=9);fig.suptitle('1. Control recovery at severity 2× — measured environment outcomes',fontsize=15)
    save(fig,'01_control_recovery',True,foot)
    effects_labels=['F → D\npredictor','D → D+Q\npredictor','Frozen → D+Q\ncritic','Total\nF → D+Q']
    fig,axs=plt.subplots(1,2,figsize=(12,5))
    for ax,family in zip(axs,['damping','mass']):
        d=effect[(effect.task=='halfcheetah')&(effect.family==family)&(effect.severity==2)]
        for j,(metric,label) in enumerate(zip(names,effects_labels)):
            r=d[d.metric==metric].iloc[0];ax.bar(j,r['mean'],color=['#0072B2','#CC79A7','#D55E00','#333333'][j])
            ax.vlines(j,r.ci_low,r.ci_high,color='black');ax.text(j+.09,r['mean'],f"{r['mean']:+.2f}",ha='left',va='bottom' if r['mean']>=0 else 'top',fontsize=9)
        ax.set_xticks(range(4),effects_labels);ax.axhline(0,color='.4',lw=.8);ax.set_title('HalfCheetah | '+family+' 2×');ax.set_ylabel('Discounted terminal score change')
    fig.suptitle('2. Predicted discounted terminal score change; not Q accuracy',fontsize=14)
    save(fig,'02_terminal_value_decomposition',True,'Same terminal tape/action; fixed Q heads 0,1; γ^H = 0.995³. Total = three signed contributions; n=3.')
    pairs=['F → D','D → DQ','DQ_F → DQ','F → J'];pairlabels=['Frozen → D','D → D+Q','Q swap only','Frozen → Joint'];paircolors=['#0072B2','#D55E00','#9467bd','#009E73']
    fig,axs=plt.subplots(1,2,figsize=(13,5))
    for ax,task in zip(axs,TASKS):
        d=rank[(rank.task==task)&(rank.severity==2)&(rank.metric=='argmax_disagreement')]
        bars(ax,d,list(TASKS[task]['families']),pairs,'comparison',pairlabels,paircolors)
        ax.set_title(task+' | severity 2×');ax.set_ylabel('Fraction of states with a different top-1');ax.set_ylim(0,1 if task=='halfcheetah' else .15)
    axs[0].legend(fontsize=9,ncol=2);fig.suptitle('3. Candidate preference change',fontsize=15)
    save(fig,'03_candidate_preference_change',True,'Same 32 diagnostic candidates, not historical MPPI execution. Fixed terminal action and Q heads 0,1; n=3.')
    for task,t in TASKS.items():
        for family in t['families']:
            fig,axs=plt.subplots(1,3,figsize=(14,4))
            for m in METHODS:
                d=returns[(returns.task==task)&(returns.family==family)&(returns.method==m)].sort_values('severity')
                axs[0].plot(d.severity,d['mean'],'o-',label=LABEL[m],color=COLORS[m]);axs[0].fill_between(d.severity,d.ci_low,d.ci_high,color=COLORS[m],alpha=.12)
            d=returns[(returns.task==task)&(returns.family==family)&(returns.method=='DQ-D')].sort_values('severity')
            axs[1].plot(d.severity,d['mean'],'o-',color=COLORS['DQ']);axs[1].fill_between(d.severity,d.ci_low,d.ci_high,color=COLORS['DQ'],alpha=.15);axs[1].axhline(0,color='.5',lw=.7)
            for name,label in zip(names,effects_labels):
                d=effect[(effect.task==task)&(effect.family==family)&(effect.metric==name)].sort_values('severity')
                axs[2].plot(d.severity,d['mean'],'o-',label=label.replace('\n',' '));axs[2].fill_between(d.severity,d.ci_low,d.ci_high,alpha=.1)
            axs[0].legend(fontsize=8);axs[2].legend(fontsize=7);axs[2].axhline(0,color='.5',lw=.7)
            for ax,title in zip(axs,['Actual episode return','Paired D+Q − D return','Predicted terminal contribution (not accuracy)']):ax.set_title(title);ax.set_xlabel('Severity multiplier')
            fig.suptitle(task+' | '+family+' | full severity grid');save(fig,f'control_and_effects_{task}_{family}',foot=foot)
            fig,axs=plt.subplots(1,3,figsize=(13,4))
            for ax,metric,title in zip(axs,['argmax_disagreement','top_k_jaccard','spearman'],['Top-1 disagreement','Top-5 Jaccard','Spearman correlation']):
                for pair,label,color in zip(pairs,pairlabels,paircolors):
                    d=rank[(rank.task==task)&(rank.family==family)&(rank.comparison==pair)&(rank.metric==metric)].sort_values('severity')
                    ax.plot(d.severity,d['mean'],'o-',label=label,color=color);ax.fill_between(d.severity,d.ci_low,d.ci_high,color=color,alpha=.12)
                ax.set_title(title);ax.set_xlabel('Severity multiplier');ax.set_ylim(-.02,1.02)
            axs[0].legend(fontsize=8);fig.suptitle(task+' | '+family+' | candidate preference');save(fig,f'ranking_{task}_{family}',foot='Same 32 diagnostic candidates, not historical MPPI execution. Descriptive 95% intervals; n=3.')
    # Read every v1 array once, retain only state-level spread rows and small predeclared examples.
    spreads=[];heads=[];checks=[];candidate_checks={};examples={}
    for task,t in TASKS.items():
        for family in t['families']:
            for k in GRID:
                for seed in t['adaptation']:
                    p=src.track(source/'q_arrays'/f'{task}-{family}-k{k:g}-seed{seed}.npz','v1 saved Q/score arrays')
                    with np.load(p) as a:
                        require(a['F_scores'].shape==(80,32) and len(a['evaluation_seeds'])==20,'Unexpected diagnostic dimensions')
                        require(np.array_equal(a['F_immediate_probability'],a['D_immediate_probability']) and np.array_equal(a['F_immediate_scalar'],a['D_immediate_scalar']),'Immediate F=D invariant')
                        candidate_digest=hashlib.sha256(a['candidates'].tobytes()).hexdigest()
                        require(candidate_digest==str(a['candidate_input_sha256']),'Candidate hash mismatch')
                        key=(task,family,k)
                        require(candidate_checks.setdefault(key,candidate_digest)==candidate_digest,'Candidate bank differs across adaptation seeds')
                        err=np.max(np.abs(a['effect_predictor_D']+a['effect_predictor_D_to_DQ']+a['effect_critic_DQ']-a['effect_total_DQ_minus_F']))
                        require(err<1e-8,'Raw terminal identity')
                        for m in [*METHODS,'DQ_F']:
                            parts=[a[m+'_rewards'],a[m+'_terminal_scores'],a[m+'_scores']]
                            centered_components(*parts)
                            for component,x in zip(['reward','discounted_terminal','total'],parts):
                                for i in range(80):
                                    spreads.append(dict(task=task,family=family,severity=k,adaptation_seed=seed,evaluation_seed=int(a['evaluation_seeds'][i//4]),
                                        timestep=int(a['starts'][i]),method=m,component=component,standard_deviation=float(x[i].std()),
                                        interquartile_range=float(np.diff(np.quantile(x[i],[.25,.75]))[0]),range=float(np.ptp(x[i]))))
                        checks.append(dict(task=task,family=family,severity=k,adaptation_seed=seed,additive_max_error=float(err),status='PASS'))
                        if k==2 and seed==t['adaptation'][0]:
                            examples[(task,family)]={key:a[key].copy() for key in a.files if 'probability' in key or 'scalar' in key or key in ['symlog_support','evaluation_seeds','starts'] or key.endswith(('_scores','_rewards')) and 'native' not in key}
                    print(f'READ {task}/{family}/k{k:g}/seed{seed}',flush=True)
    spread=pd.DataFrame(spreads);spread.to_csv(out/'tables/per_state_score_spread.csv',index=False)
    group=['task','family','severity','method','component','adaptation_seed','evaluation_seed']
    spread_summary=aggregate(spread.groupby(group)[['standard_deviation','interquartile_range','range']].mean().reset_index(),group[:5],['standard_deviation','interquartile_range','range'])
    spread_summary.to_csv(out/'tables/score_spread_summary.csv',index=False)
    for (task,family),a in examples.items():
        fig,axs=plt.subplots(1,3,figsize=(14,4.5))
        for m in [*METHODS,'DQ_F']:
            parts=[a[m+'_rewards'][0],a[m+'_terminal_scores'][0],a[m+'_scores'][0]]
            for ax,x in zip(axs,centered_components(*parts)):
                ax.plot(range(32),x,'o-',ms=3,lw=.8,label=LABEL[m],color=COLORS[m],alpha=.8)
            winner=int(a[m+'_scores'][0].argmax());axs[2].scatter(winner,centered_components(*parts)[2][winner],marker='*',s=130,color=COLORS[m],edgecolor='black',zorder=5)
        for ax,title in zip(axs,['Reward sum − within-state median','Discounted terminal Q − median','Total score − median (★ winner)']):ax.set_title(title);ax.set_xlabel('Common diagnostic candidate ID');ax.axhline(0,color='.5',lw=.7)
        axs[0].legend(fontsize=7);fig.suptitle(f'{task} | {family} 2× | first adaptation seed, first held-out seed, t={a["starts"][0]}')
        save(fig,f'centered_candidate_scores_{task}_{family}',foot='Each component centered separately; centered components need not add to centered total. Original scores/rankings unchanged.')
        for kind in ['immediate','terminal']:
            support=a['symlog_support'];packs=np.stack([a[m+'_'+kind+'_probability'][:,0] for m in METHODS]);window=probability_window(support,packs)
            for full in [False,True]:
                fig,axs=plt.subplots(1,3,figsize=(14,4.5))
                for m in METHODS:
                    probs=a[m+'_'+kind+'_probability'][:,0];q=a[m+'_'+kind+'_scalar'][:,0];delta=q-a['F_'+kind+'_scalar'][:,0]
                    for head in probs:axs[0].plot(support,head,color=COLORS[m],alpha=.13,lw=.8)
                    axs[0].plot(support,probs.mean(0),label=LABEL[m],color=COLORS[m],lw=1.6,ls='--' if m=='D' else '-')
                    axs[1].plot(range(len(q)),q,'o',label=LABEL[m],color=COLORS[m],ls='--' if m=='D' else '-')
                    axs[2].plot(range(len(q)),delta,'o',color=COLORS[m],ls='--' if m=='D' else '-')
                    if not full:
                        for h in range(len(q)):heads.append(dict(task=task,family=family,severity=2,kind=kind,method=m,head=h,scalar_q=float(q[h]),delta_vs_frozen=float(delta[h]),evaluation_seed=int(a['evaluation_seeds'][0]),adaptation_seed=TASKS[task]['adaptation'][0],sample_index=0))
                if not full:axs[0].set_xlim(*window)
                axs[0].set_title('Categorical probability | '+('full support' if full else 'shared 99.8% mass window'));axs[0].set_xlabel('Shared symlog support');axs[0].legend(fontsize=8)
                axs[1].set_title('Native-decoded scalar Q per head');axs[2].set_title('Per-head scalar Q − Frozen');axs[2].axhline(0,color='.5',lw=.7)
                for ax in axs[1:]:ax.set_xlabel('Q head index');ax.set_xticks(range(5))
                fig.suptitle(f'{task} | {family} 2× | {kind} | predeclared first seed / first sample')
                save(fig,f'{kind}_{task}_{family}'+('_full_support' if full else '_zoom'),foot='All 5 heads shown; faint individual probabilities, bold head mean. Immediate F = D exactly. Distribution movement is not calibration.')
    pd.DataFrame(heads).to_csv(out/'tables/representative_per_head_q.csv',index=False)
    key=returns[returns.severity==2].assign(category='actual_return')
    key=pd.concat([key,rank[(rank.severity==2)&rank.comparison.isin(pairs)].assign(category='candidate_preference'),effect[effect.severity==2].assign(category='terminal_decomposition')],ignore_index=True)
    key.to_csv(out/'tables/severity2_numeric_results.csv',index=False)
    short=[]
    for task,t in TASKS.items():
        for family in t['families']:
            def pick(frame,**filters):
                for k,v in filters.items():frame=frame[frame[k]==v]
                require(len(frame)==1,'Ambiguous key summary');return frame.iloc[0]
            rr=pick(returns,task=task,family=family,severity=2,method='DQ-D')
            dr=pick(rank,task=task,family=family,severity=2,comparison='D → DQ',metric='argmax_disagreement')
            qr=pick(rank,task=task,family=family,severity=2,comparison='DQ_F → DQ',metric='argmax_disagreement')
            short.append(dict(task=task,family=family,paired_return_DQ_minus_D=rr['mean'],return_ci_low=rr.ci_low,return_ci_high=rr.ci_high,
                              top1_disagreement_D_to_DQ=dr['mean'],top1_disagreement_Q_swap=qr['mean']))
    pd.DataFrame(short).to_csv(out/'tables/key_severity2_results.csv',index=False)
    require(not src.verify(),'Inputs changed during presentation generation')
    write(out/'SOURCE_INPUTS.json',src.files)
    write(out/'NUMERICAL_VALIDATION.json',dict(status='PASS',streams=checks,canonical_hashes_match_v1=True,actual_return_seed_pairing=True,
        source_hashes_unchanged=True,summary_additive_max_error=float(np.max(np.abs(residual))),environment_transitions=0,optimizer_steps=0))
    lines=['# Q terminal/planning: presentation v2','',
        '## Read the three main figures in order','',
        '1. **Control recovery:** actual held-out episode return at severity 2×, plus paired D+Q−D return. Higher is better, including negative Reacher returns. These are measured environment outcomes from completed canonical evaluations.',
        '2. **Terminal value decomposition:** HalfCheetah signed predicted contributions on the same saved tape and terminal action. The first three bars sum to the total. A negative predicted score change can coexist with a positive measured return change.',
        '3. **Candidate preference change:** the fraction of starting states where the predicted best candidate differs. Q swap compares the same D+Q predictor/reward with Frozen versus adapted critic. Disagreement does not determine which choice is better.','',
        '## Severity 2× results','',
        '| Task | Shift | Paired return D+Q−D [95% CI] | D→D+Q top-1 change | Q-swap-only change |',
        '|---|---|---:|---:|---:|']
    for r in short:lines.append(f"| {r['task']} | {r['family']} | {r['paired_return_DQ_minus_D']:+.5g} [{r['return_ci_low']:+.5g}, {r['return_ci_high']:+.5g}] | {r['top1_disagreement_D_to_DQ']:.1%} | {r['top1_disagreement_Q_swap']:.1%} |")
    lines.extend(['','## Reading the observed pattern','',
        'At severity 2×, HalfCheetah D+Q exceeds D in measured return for damping and mass, with descriptive paired intervals above zero. Meanwhile the Frozen→D+Q predicted terminal change on the common tape is negative in both families. The sign of a terminal score shift is therefore not a return-improvement metric.',
        'The controlled Q swap changes the top candidate in 20.4% of damping states and 47.9% of mass states. It demonstrates sensitivity of these fixed candidates to critic parameters, but does not establish that the changed critic caused the measured recovery. Joint has the highest measured return for mass 2×, whereas D+Q has the highest for damping 2×.',
        'Reacher has small D+Q−D point estimates and all four severity-2 paired intervals include zero. D→D+Q top-1 changes are also small (0–3.8%). This evidence does not establish a clear Reacher control benefit. Larger Joint preference changes do not by themselves establish better control.',
        '', '### Per-state candidate score spread at severity 2× (D+Q)', '',
        'Mean of within-state population standard deviations across 32 candidates, then across states/seeds. Reward and terminal scores use the same task units. This describes variation among candidates, not uncertainty about true returns.', '',
        '| Task | Family | Reward SD | Discounted terminal SD | Total score SD |',
        '|---|---|---:|---:|---:|'])
    for task,t in TASKS.items():
        for family in t['families']:
            d=spread_summary[(spread_summary.task==task)&(spread_summary.family==family)&(spread_summary.severity==2)&(spread_summary.method=='DQ')&(spread_summary.metric=='standard_deviation')].set_index('component')['mean']
            lines.append(f"| {task} | {family} | {d['reward']:.5g} | {d['discounted_terminal']:.5g} | {d['total']:.5g} |")
    lines.extend(['','## Experimental meaning','',
        '- **Parameters:** Frozen is unchanged. D changes the predictor. D+Q changes predictor and critic. Joint changes encoder, predictor, reward, critic and actor. Existing v1 component hashes and validation are preserved under validation/. Joint is a whole-model comparison; its latents are not passed to a Frozen critic.',
        '- **Predicted preference:** 4 starting states per held-out seed, 32 newly generated Gaussian candidate trajectories per state. Common Frozen executed action tapes are used for terminal probes; controlled candidates use a common terminal action and fixed Q heads 0,1. These candidates are not historical MPPI trajectories or newly executed actions.',
        '- **Measured control:** independent completed held-out rollouts, matched by adaptation seed and evaluation seed. The return panel uses actual returns, never predicted candidate scores. Equal interaction budgets do not imply equal optimizer budgets.',
        '- **Not established:** causal control recovery, Q calibration, correct candidate ranking, actual historical MPPI action changes, or planning regret. No simulator or optimizer runs in v2.','',
        '## Score spread and categorical distributions','',
        'Candidate reward, discounted terminal Q and total score are each centered by their own per-state median. Their centered values need not add: the median of a sum is not generally the sum of medians. Raw reward + terminal score = total is checked. Per-state population standard deviation, IQR and range across all 32 candidates are reported in per_state_score_spread.csv; uncertainty summaries first average the four states within each evaluation seed. Large mean terminal Q alone does not establish that terminal Q dominates ranking.',
        'Examples are predeclared: severity 2×, lowest adaptation seed, lowest evaluation seed, first stored starting state. No examples are selected by outcome. Stars mark each method’s predicted best candidate. Supplementary Q plots use the union of all methods/heads’ central 99.8% probability support with one-bin padding. Full support remains available. Native-decoded per-head scalar Q and Frozen-relative differences are shown separately. All-head descriptive plots are distinct from fixed-two-head planning scores. Entropy/JS are distribution descriptors, not calibrated uncertainty; their original v1 supplementary plots remain available.',
        '', '## Statistics and coverage','',
        'All 36 condition labels and 108 matched adaptation streams are retained; each has 20 evaluation seeds. Deterministic paired hierarchical bootstrap resamples adaptation streams and shared evaluation seeds after within-seed aggregation. Three adaptation streams support descriptive intervals only. Frozen is one shared reference, not three independently trained models. Severity 1 labels alias nominal physics and are not pooled. HalfCheetah and Reacher use separate scales. Reacher return rows are shifted only; nominal-retention rows are excluded.',
        '', '## Provenance','',f'V1 source: `{source}`',f'Results root: `{root}`',
        'SOURCE_INPUTS.json records hashes of every consumed table/array and the implementation. Canonical return table hashes match the v1 inventory. NUMERICAL_VALIDATION.json records additive checks, seed coverage, source immutability and zero simulator/optimizer calls. The compact ZIP excludes v1 arrays/checkpoints and detailed per-state/episode exports; they remain available in the source/output folders.'])
    (out/'PRESENTATION_Q_RESULTS.md').write_text('\n'.join(lines)+'\n')
    write(out/'completion.json',dict(status='PASS',conditions=36,streams=108,main_figures=3,supplementary_figures=len(figures)-3,environment_transitions=0,optimizer_steps=0))
    with zipfile.ZipFile(out.with_suffix('.zip'),'w',compression=zipfile.ZIP_DEFLATED) as z:
        for p in sorted(out.rglob('*')):
            if p.is_file() and p.name not in ['per_state_score_spread.csv','paired_heldout_returns.csv']:z.write(p,str(p.relative_to(out)))
    print(f'PRESENTATION PASS: {out}',flush=True)
