"""Seed-level aggregation and paired comparisons, isolated by checkpoint and schedule."""
from collections import defaultdict
from statistics import mean
from .statistics import summary

CONDITION=('task','cohort_id','shift_family','shift_parameter','severity_multiplier','schedule','kind','metric')


def summarize_records(records):
    grouped=defaultdict(list)
    for r in records:
        key=tuple(r[k] for k in CONDITION)+(r['method'],r['adaptation_budget'])
        grouped[key].append(r)
    result=[]
    for key,rr in sorted(grouped.items()):
        streams=defaultdict(list)
        for r in rr:
            unit=r['adaptation_seed'] if r['method']!='frozen' and r['adaptation_seed'] else r['evaluation_seed']
            streams[unit].append(r['value'])
        values=[mean(v) for _,v in sorted(streams.items())]
        result.append(dict(zip(CONDITION+('method','adaptation_budget'),key),**summary(values,'q' if rr[0]['kind'].startswith('q_') else key[0]),
            sampling_unit='adaptation_seed' if any(r['adaptation_seed'] for r in rr) else 'evaluation_or_calibration_seed',
            evaluation_records=len(rr),checkpoint_id=rr[0].get('checkpoint_id',''),evidence='DERIVED_ONLY' if any(r['evidence']=='DERIVED_ONLY' for r in rr) else 'RAW_REPRODUCED',
            source_records=';'.join(sorted(set(r['source_record_id'] for r in rr))),
            nominal_alias=str(float(rr[0]['severity_multiplier'])==1),
            nominal_parameter=rr[0]['nominal_parameter'],applied_parameter_value=rr[0]['applied_parameter_value']))
    return result


def paired_improvements(evaluations,methods=None,kind='shifted'):
    # Frozen references are shared by evaluation seed, never replicated as independent training runs.
    baseline={}; adapted=defaultdict(list)
    for r in evaluations:
        if r['metric']!='episode_return' or r['kind']!=kind:continue
        key=tuple(r[k] for k in ('task','cohort_id','shift_family','shift_parameter','severity_multiplier','schedule','evaluation_seed'))
        if r['method']=='frozen':baseline[key]=r
        elif methods is None or r['method'] in methods:adapted[(r['method'],r['adaptation_budget'],*key[:-1],r['adaptation_seed'])].append(r)
    streams=[]
    for key,rr in sorted(adapted.items()):
        deltas=[];sources=[];seeds=[]
        for r in rr:
            bk=tuple(r[k] for k in ('task','cohort_id','shift_family','shift_parameter','severity_multiplier','schedule','evaluation_seed'))
            if bk not in baseline:continue
            deltas.append(r['value']-baseline[bk]['value']);sources += [r['source_record_id'],baseline[bk]['source_record_id']];seeds.append(r['evaluation_seed'])
        # Partial pairing is excluded, not quietly reduced to the intersection.
        if not deltas or len(deltas)!=len(rr):continue
        r=rr[0]
        streams.append({k:r[k] for k in ('task','cohort_id','shift_family','shift_parameter','severity_multiplier','schedule','method','adaptation_budget','adaptation_seed')} | dict(kind=kind,evaluation_seeds=len(deltas),seed_set=','.join(sorted(seeds)),mean_return_improvement=mean(deltas),source_records=';'.join(sorted(set(sources)))))
    groups=defaultdict(list)
    for r in streams:groups[tuple(r[k] for k in ('task','cohort_id','shift_family','shift_parameter','severity_multiplier','schedule','method','adaptation_budget'))].append(r)
    comparisons=[]
    for key,rr in sorted(groups.items()):
        if len({r['seed_set'] for r in rr})!=1:continue
        values=[r['mean_return_improvement'] for r in rr]
        comparisons.append(dict(zip(('task','cohort_id','shift_family','shift_parameter','severity_multiplier','schedule','method','adaptation_budget'),key),kind=kind,baseline='frozen',**summary(values,key[0]),n_adaptation_streams=len(rr),evaluation_seeds_per_stream=rr[0]['evaluation_seeds'],improved_streams=sum(v>0 for v in values),nominal_duplicate=str(float(key[4])==1),source_records=';'.join(sorted(set(r['source_records'] for r in rr)))))
    return streams,comparisons


def prediction_improvements(records):
    frozen={};out=[]
    keys=('task','cohort_id','shift_parameter','severity_multiplier','schedule','evaluation_seed')
    for r in records:
        if r['kind']=='fixed_bank' and r['metric']=='latent_mse' and r['method']=='frozen':frozen[tuple(r[k] for k in keys)]=r
    for r in records:
        key=tuple(r[k] for k in keys)
        if r['kind']!='fixed_bank' or r['metric']!='latent_mse' or r['method']=='frozen' or key not in frozen:continue
        b=frozen[key];out.append({**r,'frozen_mse':b['value'],'adapted_mse':r['value'],'mse_improvement':b['value']-r['value'],'baseline_source':b['source_record_id']})
    return out


def prediction_control_pairs(streams,pairs):
    group=defaultdict(list)
    keys=('task','cohort_id','shift_parameter','severity_multiplier','schedule','method','adaptation_budget','adaptation_seed')
    for r in pairs:group[tuple(r[k] for k in keys)].append(r)
    out=[]
    for r in streams:
        rr=group.get(tuple(r[k] for k in keys),[])
        if not rr or set(x['evaluation_seed'] for x in rr)!=set(r['seed_set'].split(',')):continue
        mse=mean(x['mse_improvement'] for x in rr)
        out.append({**r,'mse_improvement':mse,'prediction_seeds':len(rr),'prediction_sources':';'.join(x['source_record_id'] for x in rr),
            'quadrant':('prediction_improved' if mse>0 else 'prediction_worsened')+'/'+('control_improved' if r['mean_return_improvement']>0 else 'control_worsened')})
    return out


def method_contrasts(records):
    """Pair adapted methods on the same adaptation and evaluation seeds."""
    keys=('task','cohort_id','shift_family','shift_parameter','severity_multiplier','schedule','adaptation_budget','adaptation_seed','evaluation_seed')
    index=defaultdict(dict)
    for r in records:
        if r['kind']=='shifted' and r['metric']=='episode_return' and r['method']!='frozen':index[tuple(r[k] for k in keys)][r['method']]=r
    streams=defaultdict(list);lineage=defaultdict(set);seeds=defaultdict(set)
    for key,observed in index.items():
        for method in observed:seeds[key[:-1]+(method,)].add(key[-1])
    for key,observed in index.items():
        for left,right in (('dynamics_only','dynamics_q'),('dynamics_only','joint'),('dynamics_q','joint')):
            if left not in observed or right not in observed:continue
            if seeds[key[:-1]+(left,)]!=seeds[key[:-1]+(right,)]:continue
            k=key[:-1]+(left,right)
            streams[k].append(observed[right]['value']-observed[left]['value'])
            lineage[k].update((observed[left]['source_record_id'],observed[right]['source_record_id']))
    groups=defaultdict(list)
    for k,values in streams.items():groups[k[:-3]+k[-2:]].append((mean(values),len(values),lineage[k]))
    result=[]
    for k,values in sorted(groups.items()):
        result.append(dict(zip(keys[:-2]+('baseline','method'),k),**summary([x[0] for x in values],k[0]),evaluation_seeds_per_stream=min(x[1] for x in values),improved_streams=sum(x[0]>0 for x in values),source_records=';'.join(sorted(set().union(*(x[2] for x in values))))))
    return result


def normalized_comparisons(records,comparisons):
    keys=('task','cohort_id','shift_parameter','severity_multiplier','schedule')
    baseline=defaultdict(dict)
    for r in records:
        if r['method']=='frozen' and r['kind']=='shifted' and r['metric']=='episode_return':baseline[tuple(r[k] for k in keys)][r['evaluation_seed']]=r['value']
    result=[]
    for r in comparisons:
        values=baseline[tuple(r[k] for k in keys)].values()
        if not values:continue
        scale=abs(mean(values))
        result.append({**r,'normalized_improvement':r['mean']/scale if scale>1e-12 else '',
            'normalization':'paired_delta / abs(unique-seed Frozen mean)','frozen_mean_return':mean(values)})
    return result
