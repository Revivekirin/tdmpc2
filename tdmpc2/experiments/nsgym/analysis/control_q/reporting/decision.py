"""Migrated historical computations; numerical formulas preserved."""
from __future__ import annotations
import time
import zipfile
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from experiments.nsgym.analysis.control_q.config import COLORS, LABEL, METHODS, TASKS
from experiments.nsgym.analysis.control_q.metrics import aggregate, centered_components, probability_window, require, write
from experiments.nsgym.analysis.control_q.reporting.figures import figures


def decision_finish(root,source,v2,out,frames,examples,src,checks,configs,started):
    plt.rcParams.update({'font.size':12,'axes.titlesize':13,'axes.labelsize':12,'legend.fontsize':10,'axes.spines.top':False,'axes.spines.right':False})
    common=['task','family','severity'];summaries={}
    metrics=['mean_shift','mean_absolute_shift','action_dependent_sd','centered_pearson','spearman','kendall','top_k_overlap','argmax_disagreement','max_margin','best_second_gap','top_action_l2']
    specs={'matched':(['method'],['scalar_q','head_std','entropy','js_frozen','signed_delta','absolute_delta']),
           'action':(['phase','comparison'],metrics),'fixed':(['phase','comparison'],metrics),
           'native':(['comparison','iteration'],['mean_l2','std_l2','elite_overlap','top1_id_disagreement','candidates_diverged','candidate_l2','final_action_l2','above_001','above_005','above_010']),
           'shared_native':(['iteration'],metrics)}
    for name,(extra,ms) in specs.items():
        df=frames[name];ms=[m for m in ms if m in df.columns]
        group=common+extra+['adaptation_seed','evaluation_seed']
        aggregated=df.groupby(group,dropna=False)[ms].mean().reset_index()
        summaries[name]=aggregate(aggregated,common+extra,ms)
        summaries[name].to_csv(out/'tables'/f'{name}_summary.csv',index=False)
    # Native final action is identical across iteration rows: retain only last iteration before joining.
    n=frames['native'];n=n[n.iteration==n.iteration.max()]
    n[n.comparison=='DQ_F → DQ'].to_csv(out/'tables/native_qswap_final_action.csv',index=False)
    returns=pd.read_csv(src.track(v2/'tables/heldout_return_summary.csv','verified v2 returns'))
    paired=pd.read_csv(src.track(v2/'tables/paired_heldout_returns.csv','verified v2 paired returns'))
    # Recheck against completed canonical per-episode tables already hashed by inventory.
    joint=[]
    for task,folder,filename in [('halfcheetah','dynamics-q-750k-v1','four_method_paired_comparison.csv'),('reacher','reacher-dynamics-q-v1','four_way_per_seed.csv')]:
        raw=pd.read_csv(root/folder/'comparison'/filename)
        if task=='halfcheetah':
            raw['family']=raw.parameter.map({'bthigh_damping':'damping','torso_mass':'mass'});cols={'F':'frozen_return','D':'ttt_return','DQ':'dq_return','J':'joint_return'}
            pred={'F':'frozen_latent_mse','D':'ttt_latent_mse','DQ':'dq_latent_mse'}
        else:
            raw=raw[raw.kind=='shifted'];cols={m:v+'_episode_return' for m,v in [('F','Frozen'),('D','Dynamics-only'),('DQ','Dynamics+Q'),('J','Joint RL')]};pred={'F':'frozen_mse','D':'dynamics_mse','DQ':'dq_mse'}
        for m,col in cols.items():
            df=raw[['family','severity','adaptation_seed','evaluation_seed']].copy();df['task']=task;df['method']=m;df['return']=raw[col].to_numpy()
            p=paired[(paired.task==task)&(paired.method==m)]
            merged=df.merge(p,on=common+['method','adaptation_seed','evaluation_seed'],validate='one_to_one',suffixes=('_raw','_v2'))
            require(len(merged)==len(df) and np.allclose(merged.return_raw,merged.return_v2),'Canonical return mismatch')
            if m in pred:df['fixed_encoder_latent_mse']=raw[pred[m]].to_numpy()
            if task=='reacher':
                prefix=col.removesuffix('_episode_return');df['reaching_distance']=raw[prefix+'_distance_mean'].to_numpy();df['reward_ctrl']=raw[prefix+'_reward_ctrl'].to_numpy()
            joint.append(df)
    evidence=pd.concat(joint,ignore_index=True);evidence.to_csv(out/'tables/actual_control_prediction_evidence.csv',index=False)
    aggregate(evidence,['task','family','severity','method'],['return','fixed_encoder_latent_mse','reaching_distance','reward_ctrl']).to_csv(out/'tables/control_prediction_summary.csv',index=False)
    groups=common+['adaptation_seed']
    control=paired[paired.method=='DQ-D'].groupby(groups)['return'].mean().rename('measured_return_DQ_minus_D')
    fixed=frames['fixed'][frames['fixed'].comparison=='DQ_F → DQ'].groupby(groups)['argmax_disagreement'].mean().rename('fixed_qswap_disagreement')
    native=n[n.comparison=='DQ_F → DQ'].groupby(groups)[['final_action_l2','above_005']].mean().rename(columns={'final_action_l2':'native_proposed_action_l2','above_005':'native_fraction_above_005'})
    links=pd.concat([control,fixed,native],axis=1).reset_index();require(len(links)==108 and not links.isna().any().any(),'Link coverage')
    links.to_csv(out/'tables/return_planning_per_stream.csv',index=False)
    # Retain metadata and original source validation without copying large arrays.
    effects=pd.read_csv(src.track(source/'tables/terminal_effect_decomposition.csv','v1 terminal decomposition'))
    figures=[]
    def save(fig,name,main=False,note=''):
        fig.text(.5,.014,note,ha='center',fontsize=8)
        fig.tight_layout(rect=(0,.06,1,.94))
        for ext in ['png','svg']:fig.savefig(out/'figures'/('main' if main else 'supplementary')/(name+'.'+ext),dpi=170,bbox_inches='tight')
        plt.close(fig);figures.append(name)
    def plotlines(ax,df,metric,by='comparison'):
        for label,g in df[df.metric==metric].groupby(by):
            g=g.sort_values('severity');ax.plot(g.severity,g['mean'],'o-',label=label);ax.fill_between(g.severity,g.ci_low,g.ci_high,alpha=.12)
        ax.set_xlabel('Severity multiplier');ax.set_ylabel(metric.replace('_',' '));ax.legend(fontsize=7)
    # Slide 1: no cross-task scalar pooling.
    fig,axs=plt.subplots(2,2,figsize=(16,9))
    for row,task in enumerate(TASKS):
        family='mass' if task=='halfcheetah' else 'M0';a,new=examples[(task,family)]
        f=a['F_immediate_scalar'].mean(0)
        for m in ['D','DQ','J']:axs[row,0].scatter(f,a[m+'_immediate_scalar'].mean(0),s=6,alpha=.4,label=LABEL[m],color=COLORS[m])
        lo,hi=axs[row,0].get_xlim();axs[row,0].plot([lo,hi],[lo,hi],':',color='black');axs[row,0].set_xlabel('Frozen decoded Q');axs[row,0].set_ylabel('Adapted decoded Q');axs[row,0].legend()
        d=summaries['action'];d=d[(d.task==task)&(d.severity==2)&(d.phase=='immediate_action_value')&(d.comparison=='F → J')]
        dq=summaries['action'];dq=dq[(dq.task==task)&(dq.severity==2)&(dq.phase=='immediate_action_value')&(dq.comparison=='D → DQ')]
        for offset,metric in [(-.18,'mean_absolute_shift'),(.18,'action_dependent_sd')]:
            dd=dq[dq.metric==metric].set_index('family').loc[list(TASKS[task]['families'])]
            x=np.arange(len(dd))+offset;axs[row,1].bar(x,dd['mean'],.34,label=metric.replace('_',' '));axs[row,1].vlines(x,dd.ci_low,dd.ci_high,color='black')
        axs[row,1].set_xticks(np.arange(len(dd)),dd.index);axs[row,1].set_ylabel('Immediate Q change across 32 actions');axs[row,1].legend(fontsize=8)
        axs[row,0].set_title(f'{task} | {family} 2× | first stream, all matched inputs')
        axs[row,1].set_title(task+' | D → D+Q at 2×; all streams')
    fig.suptitle('1. Critic update changes which actions are valued',fontsize=17)
    save(fig,'01_critic_action_value',True,'Left: recorded behavior actions, descriptive example. Right: 32 Gaussian diagnostic actions/state, paired 95% CI, n=3. Q changes are not accuracy.')
    fig,axs=plt.subplots(1,2,figsize=(16,9))
    for ax,family in zip(axs,['damping','mass']):
        d=effects[(effects.task=='halfcheetah')&(effects.family==family)&(effects.severity==2)]
        for j,metric in enumerate(['predictor_D','predictor_D_to_DQ','critic_DQ','total_DQ_minus_F']):
            r=d[d.metric==metric].iloc[0];ax.bar(j,r['mean'],color=['#0072B2','#CC79A7','#D55E00','#333333'][j]);ax.vlines(j,r.ci_low,r.ci_high,color='black')
        ax.set_xticks(range(4),['F → D\npredictor','D → DQ\npredictor','Frozen → DQ\ncritic','Total']);ax.axhline(0,color='black',lw=.7);ax.set_title(family+' 2×');ax.set_ylabel('Predicted discounted terminal score change')
    fig.suptitle('2. Dynamics versus critic effects on terminal Q',fontsize=17)
    save(fig,'02_terminal_decomposition',True,'Common executed tape/terminal action; Q heads 0,1; n=3, paired descriptive 95% CI. Signed changes are not Q accuracy or causal recovery.')
    fig,axs=plt.subplots(2,2,figsize=(16,9))
    for row,task in enumerate(TASKS):
        for col,(name,metric,title) in enumerate([('fixed','argmax_disagreement','Fixed 32-candidate Q-swap disagreement'),('native','final_action_l2','Native planner Q-swap proposed-action L2')]):
            df=summaries[name];df=df[(df.task==task)&(df.severity==2)&(df.comparison=='DQ_F → DQ')&(df.metric==metric)]
            if name=='native':df=df[df.iteration==df.iteration.max()]
            df=df.set_index('family').loc[list(TASKS[task]['families'])]
            ax=axs[row,col];ax.bar(np.arange(len(df)),df['mean'],color='#9467bd');ax.vlines(np.arange(len(df)),df.ci_low,df.ci_high,color='black');ax.set_xticks(np.arange(len(df)),df.index);ax.set_ylabel('Fraction' if col==0 else 'Action L2 (normalized action units)');ax.set_title(task+' | '+title)
    fig.suptitle('3. Q updates change predicted MPPI action preferences',fontsize=17)
    save(fig,'03_mppi_preferences',True,'2× shifts; n=3. Fixed bank: 20 eval seeds × 4 states. Native: 2 predeclared states/stream, replayed randomness. Proposed actions not executed.')
    fig,axs=plt.subplots(2,2,figsize=(16,9))
    for row,task in enumerate(TASKS):
        d=links[(links.task==task)&(links.severity==2)]
        for family,g in d.groupby('family'):
            for col,xname in enumerate(['fixed_qswap_disagreement','native_proposed_action_l2']):
                sx=summaries['fixed' if col==0 else 'native']
                sx=sx[(sx.task==task)&(sx.family==family)&(sx.severity==2)&(sx.comparison=='DQ_F → DQ')&(sx.metric==('argmax_disagreement' if col==0 else 'final_action_l2'))]
                if col==1:sx=sx[sx.iteration==sx.iteration.max()]
                rx=sx.iloc[0];ry=returns[(returns.task==task)&(returns.family==family)&(returns.severity==2)&(returns.method=='DQ-D')].iloc[0]
                mean=np.array([rx['mean'],ry['mean']]);low=np.array([rx.ci_low,ry.ci_low]);high=np.array([rx.ci_high,ry.ci_high])
                axs[row,col].errorbar(mean[0],mean[1],xerr=[[max(0,mean[0]-low[0])],[max(0,high[0]-mean[0])]],
                    yerr=[[max(0,mean[1]-low[1])],[max(0,high[1]-mean[1])]],fmt='D',color='black',capsize=3,ms=5)
            axs[row,0].scatter(g.fixed_qswap_disagreement,g.measured_return_DQ_minus_D,label=family,s=45)
            axs[row,1].scatter(g.native_proposed_action_l2,g.measured_return_DQ_minus_D,label=family,s=45)
        for col in range(2):
            ax=axs[row,col];ax.axhline(0,color='black',lw=.7);ax.set_ylabel('Measured held-out return D+Q − D');ax.legend();ax.set_title(task+' | severity 2×; each dot = adaptation stream')
            ax.set_xlabel('Fixed Q-swap top-1 disagreement fraction' if col==0 else 'Native proposed-action L2 (not executed)')
    fig.suptitle('4. Do preference changes coincide with actual control recovery?',fontsize=17)
    save(fig,'04_return_coincidence',True,'Dots: 3 stream means/condition; diamonds: mean + marginal hierarchical 95% CI. Return: 20 eval seeds; native: 2 states. No causal or action-quality claim.')
    for (task,family),(a,new) in examples.items():
        fig,axs=plt.subplots(1,3,figsize=(16,9))
        for m in METHODS:
            delta=a[m+'_immediate_scalar'].mean(0)-a['F_immediate_scalar'].mean(0);ordered=np.sort(delta)
            axs[0].plot(ordered,np.arange(1,len(delta)+1)/len(delta),label=LABEL[m],color=COLORS[m])
            axs[1].plot(range(5),a[m+'_immediate_scalar'][:,0],'o-',label=LABEL[m],color=COLORS[m])
            p=a[m+'_immediate_probability'][:,0]
            for head in p:axs[2].plot(a['symlog_support'],head,color=COLORS[m],alpha=.12)
            axs[2].plot(a['symlog_support'],p.mean(0),color=COLORS[m])
        axs[2].set_xlim(*probability_window(a['symlog_support'],np.stack([a[m+'_immediate_probability'][:,0] for m in METHODS])))
        axs[0].set_xlabel('Statewise signed scalar Q − Frozen');axs[0].set_ylabel('Empirical fraction');axs[0].legend()
        axs[1].set_xlabel('Head');axs[1].set_ylabel('Native decoded Q');axs[2].set_xlabel('Symlog support');axs[2].set_ylabel('Categorical probability')
        fig.suptitle(f'{task} {family} 2× | matched Frozen behavior bank');save(fig,f'matched_{task}_{family}',note='Predeclared first stream. ECDF descriptive across inputs; heads/timesteps are not independent runs. Immediate F=D exactly; entropy is not epistemic uncertainty.')
        for phase,label in [('immediate_action_value','Immediate all-head Q'),('terminal_action_value','Terminal fixed-head Q')]:
            fig,axs=plt.subplots(1,3,figsize=(16,9))
            ref=new['F_action_q'].mean(0)[0] if phase.startswith('immediate') else a['F_terminal_scores'][0]
            for m in ['D','DQ','J']:
                q=new[m+'_action_q'].mean(0)[0] if phase.startswith('immediate') else a[m+'_terminal_scores'][0]
                axs[0].scatter(ref,q,label=LABEL[m],color=COLORS[m]);axs[1].plot(q-np.median(q),'o-',label=LABEL[m],color=COLORS[m]);axs[2].plot(q-ref,'o-',color=COLORS[m])
            lo,hi=axs[0].get_xlim();axs[0].plot([lo,hi],[lo,hi],':',color='black');axs[0].legend()
            for ax,x,y in zip(axs,['Frozen Q','Common action/candidate ID','Common action/candidate ID'],['Adapted Q','Q − action-bank median','Q − Frozen']):ax.set_xlabel(x);ax.set_ylabel(y)
            fig.suptitle(f'{task} {family} 2× | {label}'+(' (discounted)' if phase.startswith('terminal') else ''))
            save(fig,f'{phase}_{task}_{family}',note='First held-out state, first adaptation stream. Fixed diagnostic candidate bank; not historical MPPI execution. No Q accuracy claim.')
        fig,axs=plt.subplots(1,3,figsize=(16,9))
        for m in ['F','D','DQ_F','DQ','J']:
            parts=centered_components(a[m+'_rewards'][0],a[m+'_terminal_scores'][0],a[m+'_scores'][0])
            for ax,p in zip(axs,parts):ax.plot(p,'o-',ms=3,label=LABEL[m],color=COLORS[m])
            win=int(a[m+'_scores'][0].argmax());axs[2].scatter(win,parts[2][win],marker='*',s=180,color=COLORS[m],edgecolor='black')
        for ax,title in zip(axs,['Centered predicted reward','Centered discounted terminal Q','Centered total; star = top-1']):ax.set_title(title);ax.set_xlabel('Same candidate ID')
        axs[0].legend(fontsize=8);fig.suptitle(f'{task} {family} 2× | controlled candidates')
        save(fig,f'candidate_components_{task}_{family}',note='Fixed diagnostic candidate bank; not historical MPPI execution. Components centered separately. Common terminal action and heads 0,1.')
        fig,axs=plt.subplots(1,3,figsize=(16,9))
        d=summaries['action'];d=d[(d.task==task)&(d.family==family)&(d.phase=='terminal_action_value')]
        for ax,metric in zip(axs,['action_dependent_sd','spearman','top_k_overlap']):plotlines(ax,d,metric)
        fig.suptitle(f'{task} {family} | all severities retained');save(fig,f'severity_{task}_{family}',note='Within-seed state averages, then paired hierarchical bootstrap, n=3. k=1 aliases not pooled; action dependence is not correctness.')
    deltas={}
    for p in sorted((source/'q_arrays').glob('*.npz')):
        with np.load(p) as a:
            for suffix,label in [('scores','score'),('rewards','reward'),('terminal_scores','terminal')]:
                deltas[p.stem+'_'+label]=a['DQ_'+suffix]-a['DQ_F_'+suffix]
    np.savez_compressed(out/'arrays/fixed_candidate_qswap_deltas.npz',**deltas)
    # Retain condition-level marginal intervals alongside paired stream means.
    ret=returns[returns.method=='DQ-D'];native_summary=summaries['native'];native_summary=native_summary[(native_summary.comparison=='DQ_F → DQ')&(native_summary.iteration==native_summary.iteration.max())]
    ret.to_csv(out/'tables/paired_return_summary.csv',index=False)
    rows=[]
    for task,t in TASKS.items():
        for family in t['families']:
            def get(df,metric):return df[(df.task==task)&(df.family==family)&(df.severity==2)&(df.metric==metric)].iloc[0]
            rr=get(ret,'return');nn=get(native_summary,'final_action_l2');ff=get(summaries['fixed'][summaries['fixed'].comparison=='DQ_F → DQ'],'argmax_disagreement')
            rows.append(dict(task=task,family=family,return_delta=rr['mean'],return_low=rr.ci_low,return_high=rr.ci_high,
                             fixed_disagreement=ff['mean'],native_action_l2=nn['mean'],native_action_low=nn.ci_low,native_action_high=nn.ci_high))
    pd.DataFrame(rows).to_csv(out/'tables/key_results_2x.csv',index=False)
    messages=['# Decision-relevant critic diagnostics','',
        'All phase 1 matched-behavior probabilities/scalars, terminal decompositions and 32-candidate scores reuse v1. New inference evaluates immediate common action banks and native planning only. No simulator, agent constructor, optimizer or adaptation runs.','',
        '## Severity 2× observations','',
        '| Task | Shift | Actual return DQ−D [95% CI] | Fixed Q-swap top-1 change | Native proposed action L2 [95% CI] |',
        '|---|---|---:|---:|---:|']
    for r in rows:messages.append(f"| {r['task']} | {r['family']} | {r['return_delta']:+.5g} [{r['return_low']:+.5g}, {r['return_high']:+.5g}] | {r['fixed_disagreement']:.1%} | {r['native_action_l2']:.5g} [{r['native_action_low']:.5g}, {r['native_action_high']:.5g}] |")
    messages.extend(['','## Offset versus action dependence at severity 2×','',
        'A constant change has zero action-dependent SD even if its mean is large. The table retains both quantities; no arbitrary binary offset/ranking threshold is applied. Terminal Q-swap holds DQ latents and terminal actions fixed; immediate D→DQ uses matched physical states and the same 32 first actions. Values are decoded Q units within each task (terminal values here are undiscounted).', '',
        '| Task | Family | Comparison | Mean signed shift | Mean absolute shift | SD of shift across actions |',
        '|---|---|---|---:|---:|---:|'])
    for task,t in TASKS.items():
        for family in t['families']:
            for phase,pair in [('immediate_action_value','D → DQ'),('terminal_action_value','DQ_F → DQ')]:
                d=summaries['action'];d=d[(d.task==task)&(d.family==family)&(d.severity==2)&(d.phase==phase)&(d.comparison==pair)].set_index('metric')['mean']
                messages.append(f"| {task} | {family} | {phase}: {pair} | {d['mean_shift']:+.5g} | {d['mean_absolute_shift']:.5g} | {d['action_dependent_sd']:.5g} |")
    messages.extend(['','## What these measurements mean','',
        'Matched-behavior Q uses the recorded Frozen actions. Immediate multi-action probes instead use first actions from the existing clipped Gaussian diagnostic candidates (saved proposal std 2, action bounds [-1,1]). These are different action regimes: a larger Q shift on the diagnostic bank does not establish in-distribution accuracy or better control. A common offset is summarized by mean signed shift; variation of that shift across common actions is action-dependent SD. Centering removes a constant offset. Neither a larger mean Q nor entropy/JS identifies accuracy. F=D on identical immediate inputs is a required structural invariant. Joint uses its own encoder; its latent coordinates are never fed to another critic.',
        'Fixed Q-swap uses DQ predictor/reward and identical candidates, terminal actions and head pair, changing only critic parameters. The native shared-proposal analysis explicitly reuses each DQ iteration’s cached candidate tensors, terminal actions and head indices. Full iterative native planning shares base Gaussian/policy noises, head selection and final Gumbel randomness, while allowing Gaussian means/stds and candidate tensors to diverge. A matched slot ID is not an identical trajectory once candidates diverge; native elite-ID overlap and top-1-ID disagreement must be interpreted alongside candidate_l2 and final action distance.',
        'Native _plan is invoked unchanged, uncompiled on CPU, through a namespace without constructing an agent. Parity is against this native CPU reference, not a reconstruction of historical GPU rollouts. Each model is run once without instrumentation and once with instrumentation under recorded/replayed randomness. Final action and previous-mean tensors must match exactly. Learned policy-prior candidates are retained. Warm initialization is imposed identically from a saved tape; it is not reconstruction of historical warm starts. eval_mode=True still samples a final elite by native Gumbel selection; no additional exploration noise is added.',
        'Native proposed actions are model planning outputs; they were not executed. Actual control return comes exclusively from completed held-out evaluations. Planner probes use Frozen behavior-bank states, which need not be visited by the adapted closed-loop policies; shared evaluation seeds do not make the probe states causally identical to adapted trajectories. Reacher reaching distance and reward_ctrl are separate columns in actual_control_prediction_evidence.csv; reward_ctrl is signed control reward, not return or distance. Fixed-encoder latent MSE is retained for F/D/DQ only; Joint latent MSE has a changed representation and is omitted from that comparison.',
        'Fixed analyses use 20 paired evaluation seeds and four states each; native uses only two predetermined state/seed cases per stream. n=3 adaptation streams is the independent cluster count. Intervals resample adaptation streams and paired evaluation seeds after within-seed averaging; they are descriptive. Native cold and warm cases are different predeclared cases, not a causal warm-start contrast. Figure 4 diamonds show condition means with marginal hierarchical intervals for each axis, not a joint confidence region or causal association test. No raw Q pooling across tasks. k=1 aliases are not pooled.',
        'Not established: Q calibration, physical action quality, actual historical MPPI action changes, causal control recovery, Monte Carlo Q accuracy, or planning regret. Associations between planning changes and return are descriptive; all streams and negative effects are retained.'])
    (out/'reports/RESULTS_AND_LIMITATIONS.md').write_text('\n'.join(messages)+'\n')
    for filename,title,text in [
        ('Q_REPRESENTATION_ANALYSIS.md','Matched Q representation','Saved immediate categorical and native decoded per-head values are reused. matched_summary.csv separates signed and absolute movements. Supplementary ECDF/scatter/head/probability plots use first adaptation stream, severity 2×, no outcome selection. Distribution entropy is not epistemic uncertainty.'),
        ('ACTION_VALUE_ANALYSIS.md','Offsets and action-dependent changes','action_summary.csv distinguishes immediate all-head action Q and terminal fixed-head Q. Mean signed shifts, absolute shifts and action-dependent SD are separate. Spearman/Kendall, centered correlation and top-K/argmax measure preference changes, never correctness. fixed_summary.csv uses full score with unchanged reward in the Q swap. Undefined correlations for constant vectors are left undefined, not fabricated.'),
        ('PLANNER_INTERVENTION_ANALYSIS.md','Native MPPI intervention','Native configuration and subset are recorded in protocol.json. arrays/ records native priors/Gaussian candidates, means/stds before and after each iteration, rewards, terminal Q/actions/head pairs, scores, elite IDs/weights and final actions plus random tapes. Native parity is exact per model/state. Shared native candidate scoring replays cached DQ proposals and terminal actions rather than assuming RNG alone gives identical trajectories. Per-dimension action differences and thresholds 0.01/0.05/0.1 are in native_per_state.csv. Shared native scores are a separate protocol from the 32-candidate bank.')]:
        (out/'reports'/filename).write_text('# '+title+'\n\n'+text+'\n\nSee RESULTS_AND_LIMITATIONS.md for actual outcomes and interpretation.\n')
    require(not src.verify(),'Sources changed during analysis')
    write(out/'SOURCE_INPUTS.json',src.files)
    write(out/'NUMERICAL_VALIDATION.json',dict(status='PASS',checks=checks,streams=108,native_parity_comparisons=108*2*5,
        source_hashes_unchanged=True,F_D_immediate_invariant=True,headwise_native_decoder_parity=True,probability_normalization=True,same_physical_observations=True,controlled_candidate_identity=True,terminal_action_and_head_protocol_checked=True,additive_terminal_identity=True,total_score_identity=True,adaptation_evaluation_seed_pairing=True,cross_task_Q_pooling=False,native_states_per_stream=2,environment_interactions=0,optimizer_steps=0))
    write(out/'completion.json',dict(status='PASS',streams=108,conditions=36,native_states_per_stream=2,main_figures=4,
        supplementary_figures=len(figures)-4,seconds=time.monotonic()-started,environment_interactions=0,optimizer_steps=0))
    with zipfile.ZipFile(out.with_suffix('.zip'),'w',compression=zipfile.ZIP_DEFLATED) as z:
        for p in sorted(out.rglob('*')):
            if p.is_file() and 'arrays' not in p.parts and not p.name.endswith('_per_state.csv') and p.name!='actual_control_prediction_evidence.csv':z.write(p,str(p.relative_to(out)))
    print('COMPLETE PASS '+str(out),flush=True)
