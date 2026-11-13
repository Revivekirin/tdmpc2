"""Recompute saved condition tables; fail visibly on any unresolved numeric difference."""
from collections import Counter, defaultdict
from pathlib import Path
import json
import math
import numpy as np
from .schema import rows, write_csv, sha256
from .statistics import summary
from .q_statistics import summarize as q_summarize
from . import ttt, reacher


def records(path):
    return list(rows(path))


def numeric_rows(data):
    result=[]
    for row in data:
        converted={}
        for key,value in row.items():
            if value == '': converted[key]=None; continue
            try: converted[key]=float(value)
            except (ValueError,TypeError): converted[key]=value
        result.append(converted)
    return result


class Checks:
    def __init__(self): self.rows=[]; self.inputs={}
    def read(self,p):
        self.inputs[str(p.resolve())]=sha256(p)
        return records(p)
    def check(self,source,key,metric,expected,actual):
        status='EXACT';delta=''
        if expected in ('',None) and actual in ('',None): pass
        else:
            try:
                a,b=float(expected),float(actual)
                delta=b-a
                if math.isnan(a) and math.isnan(b):pass
                elif a==b:pass
                elif math.isclose(a,b,rel_tol=1e-9,abs_tol=1e-12):status='TOLERANCE'
                else:status='FAIL'
            except (TypeError,ValueError):
                if str(expected)!=str(actual):status='FAIL'
        self.rows.append(dict(source=str(source),identity=str(key),metric=metric,expected=expected,
                              actual=actual,difference=delta,status=status,rtol=1e-9,atol=1e-12))
    def table(self,path,actual,keys):
        expected=self.read(path)
        def ident(r):return tuple(str(float(r[k])) if _number(r[k]) else str(r[k]) for k in keys)
        index={ident(r):r for r in actual}
        self.check(path,'table','row_count',len(expected),len(actual))
        self.check(path,'table','unique_keys',len(actual),len(index))
        for r in expected:
            k=ident(r);found=index.get(k)
            if found is None:self.check(path,k,'row_present',True,False);continue
            for col,val in r.items():
                if col in keys:continue
                if col in found:self.check(path,k,col,val,found[col])


def _number(x):
    try:float(x);return True
    except (TypeError,ValueError):return False


def reduce_means(data,keys,metrics):
    groups=defaultdict(list)
    for r in data:groups[tuple(r[k] for k in keys)].append(r)
    return [dict(zip(keys,k),**{m:float(np.mean([float(r[m]) for r in rr])) for m in metrics},
                 n_evaluation_seeds=len({r['evaluation_seed'] for r in rr})) for k,rr in groups.items()]


def run(results,output,unified=None):
    output.mkdir(parents=True,exist_ok=False);c=Checks();gaps=[]
    # All four-method conditions, with Frozen evaluation seeds counted once.
    base=results/'dynamics-q-750k-v1/comparison'
    paired=c.read(base/'four_method_paired_comparison.csv')
    saved=c.read(base/'condition_summary.csv');metrics=list(dict.fromkeys(r['metric'] for r in saved))
    streams=reduce_means(paired,['parameter','severity','adaptation_seed'],metrics)
    c.table(base/'per_adaptation_seed_comparison.csv',streams,['parameter','severity','adaptation_seed'])
    cond=[]
    for r in saved:
        same=lambda x:x['parameter']==r['parameter'] and float(x['severity'])==float(r['severity'])
        rr=[x for x in streams if same(x)];m=r['metric']
        frozen=m.startswith('frozen_')
        values=list({x['evaluation_seed']:float(x[m]) for x in paired if same(x)}.values()) if frozen else [x[m] for x in rr]
        st=summary(values);n=st.pop('n');outliers=st.pop('outliers')
        cond.append(dict(parameter=r['parameter'],severity=r['severity'],metric=m,**st,
                         n_adaptation_seeds=0 if frozen else n,n_evaluation_seeds=20,
                         positive_streams=None if frozen else sum(v>0 for v in values),
                         min=min(values),max=max(values),tukey_outlier_count=outliers,
                         sampling_unit='evaluation_seed_shared_frozen_baseline' if frozen else 'adaptation_seed'))
    c.table(base/'condition_summary.csv',cond,['parameter','severity','metric'])
    # Seed uniqueness, every matched episode, and arithmetic of all saved contrasts.
    for task,path,keys,contrasts in [
        ('halfcheetah',base/'four_method_paired_comparison.csv',['parameter','severity','adaptation_seed','evaluation_seed'],
         {'dq_minus_ttt':('dq_return','ttt_return'),'dq_minus_frozen':('dq_return','frozen_return'),
          'joint_minus_dq':('joint_return','dq_return'),'ttt_minus_frozen':('ttt_return','frozen_return'),
          'joint_minus_frozen':('joint_return','frozen_return'),'dq_mse_improvement':('frozen_latent_mse','dq_latent_mse')}),
        ('reacher',results/'reacher-analysis-v3/tables/matched_comparison.csv',['family','severity','kind','adaptation_seed','evaluation_seed'],
         {'delta_D':('ttt_return','frozen_return'),'delta_J':('joint_return','frozen_return'),'delta_JD':('joint_return','ttt_return')})]:
        data=c.read(path);identities=[tuple(r[k] for k in keys) for r in data]
        c.check(path,'all','unique_seed_pairs',len(data),len(set(identities)))
        for i,r in enumerate(data):
            for m,(a,b) in contrasts.items():c.check(path,i,m,r[m],float(r[a])-float(r[b]))
    # Reacher's migrated original aggregation, all kinds/conditions/metrics.
    rb=results/'reacher-analysis-v3/tables'
    data=numeric_rows(c.read(rb/'matched_comparison.csv'))
    summaries,rs,coupling=reacher.summarize(data,{'evaluation_seeds':list(range(20))},.001,.001)
    c.table(rb/'condition_summary.csv',summaries,['family','severity','kind','metric'])
    c.table(rb/'per_adaptation_seed.csv',rs,['family','severity','kind','adaptation_seed'])
    # Reward decomposition uses evaluation episodes before adaptation-seed bootstrap.
    reward=numeric_rows(c.read(rb/'reward_decomposition.csv'))
    fields=['joint_delta_reward_dist','joint_delta_reward_ctrl','delta_reward_dist','delta_reward_ctrl']
    rr=reduce_means(reward,['family','severity','kind','adaptation_seed'],fields)
    c.table(rb/'reward_components_per_stream.csv',rr,['family','severity','kind','adaptation_seed'])
    reward_summary=[]
    for saved_row in c.read(rb/'joint_reward_summary.csv'):
        vals=[r[saved_row['metric']] for r in rr if r['kind']=='shifted' and r['family']==saved_row['family'] and float(r['severity'])==float(saved_row['severity'])]
        st=summary(vals,'reacher',single=True);st.pop('outliers')
        reward_summary.append(dict(family=saved_row['family'],severity=saved_row['severity'],metric=saved_row['metric'],**st))
    c.table(rb/'joint_reward_summary.csv',reward_summary,['family','severity','metric'])
    # Migrated distribution functions, every condition and quantile.
    tb=results/'dynamics-only-750k-v1/analysis-distribution-v3'
    for label,metric in [('prediction','mse_improvement'),('control','delta_return')]:
        stream=numeric_rows(c.read(tb/f'{label}_per_stream.csv'))
        c.table(tb/f'{label}_distribution.csv',ttt.summarize_streams(stream,metric),['parameter','severity'])
    # Saved individual Q head/observation -> evaluation seed -> adaptation stream.
    qb=results/'q-planning-mechanism-750k-v1 2/tables'
    specifications=[('immediate_q_metrics','immediate',['method'],['q','entropy','ensemble_std','boundary_mass','interhead_js','bellman_residual','bellman_soft_ce']),
        ('q_distribution_divergence','divergence',['left','right'],['js','q_difference']),
        ('terminal_q_metrics','terminal_q_metrics',['method'],['q','entropy','ensemble_std']),
        ('planning_score_components','planning_score_components',['method'],['reward_sum','terminal_contribution','total_score','ess']),
        ('candidate_ranking_comparison','candidate_ranking_comparison',['left','right'],['top_k_overlap','argmax_disagreement']),
        ('terminal_counterfactuals','terminal_counterfactuals',[],['dynamics_effect','q_head_effect','interaction'])]
    for raw,name,extra,fields in specifications:
        data=c.read(qb/f'{raw}.csv');keys=['parameter','severity']+extra
        qs,qc=q_summarize(data,keys,fields)
        c.table(qb/f'{name}_per_stream.csv',qs,keys+['adaptation_seed'])
        c.table(qb/f'{name}_summary.csv',qc,keys+['metric'])
    data=c.read(qb/'immediate_q_metrics.csv')
    for regime in ('action_saturation','observation_regime','velocity_regime','physical_response_regime'):
        keys=['parameter','severity','method',regime]
        _,qc=q_summarize(data,keys,['q','entropy','ensemble_std'])
        c.table(qb/f'{regime}_summary.csv',qc,keys+['metric'])
    # Physical raw replay-free reconstruction uses the migrated strict Frozen provenance checks.
    from . import physical
    pb=results/'physical-severity-750k-v1';ref=results/'physical-severity-750k-refinement-01'
    physical.verify(pb)
    frozen=physical.episodes(pb,'frozen');abrupt=physical.episodes(pb,'nonstationary-abrupt')
    physical.check_duplicates(pb,frozen,abrupt)
    probes,applied=physical.physical_seed_data(pb)
    latent=physical.physical_fixed_latent(pb)
    c.table(ref/'integrated_seed_summary.csv',physical.condition_rows(probes,applied,latent,frozen),['parameter','severity'])
    effort=physical.effort_data(pb,frozen,applied)
    c.table(ref/'action_effort_seed_summary.csv',effort,['parameter','severity','seed'])
    es=[]
    for r in c.read(ref/'action_effort_condition_summary.csv'):
        vals=[x[r['metric']] for x in effort if x['parameter']==r['parameter'] and float(x['severity'])==float(r['severity'])]
        st=physical.stats(vals);st['seed_n']=st.pop('n')
        es.append(dict(parameter=r['parameter'],severity=r['severity'],metric=r['metric'],**st))
    c.table(ref/'action_effort_condition_summary.csv',es,['parameter','severity','metric'])
    fixed_error, _, fixed_regime, _ = physical.latent_seed_data(pb/'heldout-latent/latent_errors.csv')
    abrupt_error, _, abrupt_regime, bins = physical.latent_seed_data(pb/'nonstationary-abrupt-latent/latent_errors.csv',onset=250,bins=True)
    timeline=physical.time_table(bins,physical.velocity_bins(pb),applied)
    c.table(ref/'shift_aligned_seed_bins.csv',timeline,['parameter','severity','shift_relative_bin_start'])
    seed_rows,seed_summary,regimes=physical.scatter_data(frozen,abrupt,fixed_error,abrupt_error,fixed_regime,abrupt_regime,applied)
    c.table(ref/'per_seed_error_vs_return.csv',seed_rows,['schedule','parameter','severity','seed'])
    c.table(ref/'per_seed_error_return_summary.csv',seed_summary,['schedule','parameter','severity'])
    c.table(ref/'action_regime_counts.csv',regimes,['schedule','parameter','severity'])
    # Entire saved Q-learning update traces, not only latest updates.
    from . import diagnostics
    db=results/'dynamics-q-750k-v1/followup-analysis-v1'
    updates=numeric_rows(c.read(db/'all_update_diagnostics.csv'));groups=defaultdict(list)
    for r in updates:groups[(r['parameter'],r['severity'],r['adaptation_seed'])].append(r)
    ds=[dict(parameter=k[0],severity=k[1],adaptation_seed=k[2],**diagnostics.trace_summary(rr)) for k,rr in groups.items()]
    c.table(db/'training_trace_summary.csv',ds,['parameter','severity','adaptation_seed'])
    # HalfCheetah nominal checkpoint validation, same historical 90210 bootstrap.
    from statistics import mean,stdev
    nb=results/'nominal-convergence-audit-01';data=c.read(nb/'nominal_checkpoint_validation.csv')
    saved=json.loads((nb/'nominal_checkpoint_summary.json').read_text())
    c.inputs[str((nb/'nominal_checkpoint_summary.json').resolve())]=sha256(nb/'nominal_checkpoint_summary.json')
    for r in saved:
        rr=[x for x in data if int(x['transitions'])==r['transitions']];values=[float(x['episode_return']) for x in rr]
        st=summary(values,'budgets');observed=dict(episodes=len(rr),mean_return=mean(values),std_return=stdev(values))
        for metric in ('return_per_environment_step','forward_velocity','control_cost','control_cost_per_step','episode_length','action_abs_mean','action_abs_max','action_rms','action_saturation_fraction'):
            observed[metric]=mean(float(x[metric]) for x in rr)
        for metric in ('terminated','truncated'):observed[metric]=mean(x[metric]=='True' for x in rr)
        for metric,value in observed.items():c.check(nb,r['transitions'],metric,r[metric],value)
        for i,key in enumerate(('ci_low','ci_high')):c.check(nb,r['transitions'],key,r['return_ci95'][i],st[key])
    # Registered raw-to-summary paths and their source fingerprints.
    for directory in ('physical','physical-latent','frozen','heldout-latent','nonstationary-nominal','nonstationary-fixed','nonstationary-abrupt','nonstationary-abrupt-latent'):
        for source in (pb/directory).glob('*.csv'):c.inputs[str(source.resolve())]=sha256(source)
    # Do not certify unavailable raw sources just because a derived table exists.
    gaps.extend(['Reacher raw physical probe records not registered for reconstruction; saved physical summaries retained.',
                 'Nominal training JSONL and full budget evaluator stages are not present in the registered exports; checkpoint validation is covered.',
                 'Q decoding from original distribution logits requires raw arrays/checkpoints not present in the transferred Q tables.',
                 'Reacher Dynamics+Q not complete locally; excluded.',
                 'Runtime-imported legacy aggregators remain deferred; regression PASS is not full migration completion.'])
    if unified is not None:
        hc=c.read(results/'dynamics-q-750k-v1/comparison/condition_summary.csv')
        rc=c.read(results/'reacher-analysis-v3/tables/condition_summary.csv')
        mapping={'dynamics_only':('ttt_minus_frozen','delta_D'),'dynamics_q':('dq_minus_frozen',None),'joint':('joint_minus_frozen','delta_J')}
        for r in c.read(unified/'comparisons.csv'):
            idx=0 if r['task']=='halfcheetah' else 1
            metric=mapping[r['method']][idx];table=hc if idx==0 else rc
            candidates=[x for x in table if x['parameter']==r['shift_parameter'] and float(x['severity'])==float(r['severity_multiplier']) and x['metric']==metric and (idx==0 or x['kind']=='shifted')]
            c.check(unified/'comparisons.csv',str(r),'matched_reference_rows',1,len(candidates))
            if len(candidates)==1:
                for field in ('mean','median','std','ci_low','ci_high'):
                    c.check(unified/'comparisons.csv',(r['task'],r['shift_parameter'],r['severity_multiplier'],r['method']),field,candidates[0][field],r[field])
    fields=['source','identity','metric','expected','actual','difference','status','rtol','atol']
    write_csv(output/'REGRESSION_COMPARISONS.csv',c.rows,fields)
    counts=Counter(r['status'] for r in c.rows)
    (output/'regression_inputs.json').write_text(json.dumps(c.inputs,indent=2))
    status='FAIL' if counts['FAIL'] else 'PASS_WITH_COVERAGE_GAPS'
    (output/'completion.json').write_text(json.dumps(dict(status=status,counts=counts,gaps=gaps),indent=2))
    (output/'UNIFIED_REGRESSION_REPORT.md').write_text('# Numerical regression\n\n'+f'Status: {status}. Checks: {dict(counts)}.\n\n'
        'Comparisons use all conditions in the registered completed tables, not selected headline results. '
        'Exact and tolerance matches are separate. Unresolved mismatches are FAIL. '
        'Counts are field comparisons, not experiments. Frozen evaluation aliases are deduplicated in the original sampling unit.\n\n'
        '## Coverage gaps\n\n'+'\n'.join('- '+g for g in gaps)+'\n')
    print(status,dict(counts))
    return counts['FAIL']==0
