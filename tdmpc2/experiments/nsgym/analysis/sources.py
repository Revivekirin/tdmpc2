"""Raw-schema adapters for the shared discovery catalog (no historical run paths)."""
from __future__ import annotations
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
import numpy as np
from .discovery import METHODS, first, load, passed, task_of
from .schema import sha256

SOURCES=load(Path(__file__).with_name('config.json')).get('tasks',{})
CONTROL=('episode_return','forward_velocity','control_cost','reward_dist','reward_ctrl','distance_initial',
         'distance_mean','distance_final','distance_min','action_abs','action_saturation','actuator_force_rms',
         'actuator_force_saturation','force_limited_count','episode_length','return_per_step')
ALIASES={'action_abs':('action_abs_mean',),'actuator_force_rms':('force_rms',),
         'actuator_force_saturation':('force_saturation',),'return_per_step':('reward_per_step','return_per_environment_step')}


def number(x):
    try:
        v=float(x)
        return v if math.isfinite(v) else None
    except (ValueError,TypeError):return None


def normalized(x):
    n=number(x)
    return format(n,'.12g') if n is not None else str(x or '')


class Records:
    def __init__(self,catalog,filters=None):
        self.c=catalog; self.filters=filters or {}; self.rows=[]; self.duplicates=[];self.checks=[];self.training=[]
        self.provenance={};self.index={};self.conflicts=set()
    def check(self,path,metric,expected,actual,rtol=1e-9,atol=1e-12):
        a,b=number(expected),number(actual)
        status='EXACT' if a==b and a is not None else ('TOLERANCE' if a is not None and b is not None and math.isclose(a,b,rel_tol=rtol,abs_tol=atol) else 'FAIL')
        self.checks.append(dict(source=self.c.rel(path),metric=metric,expected=expected,actual=actual,status=status,rtol=rtol,atol=atol))
    def base(self,p,row,meta,done,method=None):
        ident={**meta.get('identity',{}),**done.get('identity',{}),**row}
        task=task_of(meta,ident,self.c.registry)
        if not task:return None
        training=self.c.training_for(meta,done)
        checkpoint=first(meta,'checkpoint_sha256','expected_checkpoint_sha256','model_sha256') or row.get('checkpoint_sha256','')
        if training:checkpoint=first(training['meta'],'checkpoint_sha256','model_sha256') or checkpoint
        cfg=self.c.registry[task]
        parameter=ident.get('parameter',ident.get('mass_parameter','nominal'))
        family=ident.get('family') or cfg['parameters'].get(parameter,parameter)
        method=method or METHODS.get(ident.get('method'),'frozen')
        budget=ident.get('budget',done.get('budget',''))
        if budget=='':budget=first(training['done'] if training else {},'transitions','adaptation_transitions',default=0)
        schedule=ident.get('schedule',first(meta,'schedule',default='fixed'))
        if schedule=='nominal':schedule='fixed'
        seed=ident.get('evaluation_seed',ident.get('seed',''))
        raw_dependencies=self.c.rel(p.parent/'steps.csv') if (p.parent/'steps.csv').is_file() else ''
        record=dict(raw_dependencies=raw_dependencies,task=task,environment_id=cfg['environment_id'],checkpoint_id=checkpoint,
            checkpoint_sha256=checkpoint,method=method,shift_family=family,shift_parameter=parameter,
            severity_multiplier=normalized(ident.get('severity',ident.get('mass_scale',1))),schedule=schedule,
            adaptation_seed=normalized(ident.get('adaptation_seed','')),evaluation_seed=normalized(seed),
            adaptation_budget=normalized(budget),optimizer_steps=normalized(ident.get('optimizer_steps',ident.get('gradient_updates',''))),
            nominal_parameter=ident.get('nominal_parameter',''),applied_parameter_value=ident.get('actual_parameter',ident.get('actual_parameter_min','')),
            model_hash=done.get('snapshot_sha256',meta.get('snapshot_sha256',row.get('snapshot_sha256',''))),
            source_run_id=self.c.rel(p.parent),source_path=self.c.rel(p),completion_status='PASS',
            cohort_id=checkpoint or 'UNVERIFIED_CHECKPOINT',training_path=training['path'] if training else '',
            kind=ident.get('kind',ident.get('mode','shifted')),update_granularity=first(meta,'update_granularity',default=''))
        if not record['checkpoint_sha256']:
            self.c.add_issue(p,'Missing checkpoint identity; cannot pair with other runs','MISSING')
        return record
    def emit(self,base,metric,value,p,rowid,evidence='RAW_REPRODUCED',kind=None):
        v=number(value)
        if v is None:
            if value not in (None,'') and str(value).lower() in ('nan','inf','-inf','infinity'):self.c.add_issue(p,f'Nonfinite {metric} at {rowid}')
            return
        if base is None:return
        for f,k in (('task','task'),('method','method'),('shift_family','shift_family'),('severity','severity_multiplier'),('schedule','schedule')):
            allowed=self.filters.get(f)
            if allowed and base[k] not in allowed:
                dependencies={'DQ-full':{'dynamics_q'},'DQ-dyn-only':{'dynamics_q'},'TTT+DQ-Q':{'dynamics_only','dynamics_q'},'F+DQ-Q':{'frozen','dynamics_q'}}
                required=set()
                for part in base[k].split('→'):
                    required.update(dependencies.get(part,{METHODS.get(part,part)}))
                if f!='method' or not required<=allowed:return
        row={**base,'metric':metric,'value':v,'evidence':evidence,'kind':kind or base['kind'],
             'source_record_id':f'{self.c.rel(p)}:{rowid}','source_sha256':self.c.digest(p)}
        key=tuple(row[k] for k in ('task','cohort_id','method','shift_parameter','severity_multiplier','schedule','adaptation_seed','evaluation_seed','adaptation_budget','kind','metric','model_hash'))
        # Frozen aliases and repeated budgets share one nominal model and one evaluation seed.
        if row['method']=='frozen':
            key=tuple(row[k] for k in ('task','cohort_id','method','shift_parameter','severity_multiplier','schedule','evaluation_seed','kind','metric'))
            row.update(adaptation_seed='',adaptation_budget='0',model_hash=row['checkpoint_sha256'])
        old=self.index.get(key)
        if old:
            if math.isclose(old['value'],v,rel_tol=1e-7,abs_tol=1e-10):
                self.duplicates.append(dict(source=row['source_record_id'],canonical=old['source_record_id'],reason='same measurement identity/value'))
                return
            # Nominal-retention episodes for Frozen may be repeated for each shifted parameter; keep contexts separate above.
            self.c.add_issue(p,f'Conflicting duplicated metric {metric} with {old["source_record_id"]}')
            self.conflicts.add(key);return
        self.index[key]=row;self.rows.append(row)
    def finish(self):
        # Dynamics+Q enters scientific comparison only with paired prediction and
        # control for the same trained snapshot and held-out seed set.
        dq=defaultdict(list)
        for r in self.rows:
            if r['method']=='dynamics_q' and not r['kind'].startswith('q_'):
                dq[tuple(r[k] for k in ('task','cohort_id','shift_parameter','severity_multiplier','adaptation_seed','model_hash'))].append(r)
        excluded=set()
        for key,rr in dq.items():
            pred={r['evaluation_seed'] for r in rr if r['kind']=='fixed_bank' and r['metric']=='latent_mse'}
            ctrl={r['evaluation_seed'] for r in rr if r['kind']=='shifted' and r['metric']=='episode_return'}
            if not pred or not ctrl:
                # A running held-out batch can have completed children in only
                # one mode. Its partial results are pending, not invalid data.
                for r in rr:excluded.add(id(r))
                self.c.excluded.append(dict(path=rr[0]['source_run_id'],status='INCOMPLETE',reason='Dynamics+Q awaits both completed held-out batches'))
            elif pred!=ctrl or any(not r['training_path'] for r in rr):
                for r in rr:excluded.add(id(r))
                self.c.issues.append(dict(path=rr[0]['source_run_id'],status='MISSING',reason='Dynamics+Q needs completed training and matched prediction/control seeds for this snapshot'))
        self.rows=[r for r in self.rows if id(r) not in excluded]
        bad={id(self.index[k]) for k in self.conflicts}
        self.rows=[r for r in self.rows if id(r) not in bad]
        nominal={}
        for r in self.rows:
            if number(r['nominal_parameter']) is not None:nominal[(r['task'],r['cohort_id'],r['shift_parameter'])]=(r['nominal_parameter'],r['source_record_id'])
        for r in self.rows:
            key=(r['task'],r['cohort_id'],r['shift_parameter'])
            if r['nominal_parameter']=='' and key in nominal:
                r['nominal_parameter'],r['parameter_reference']=nominal[key]
        return self


def _validate_eval(rec,p,meta,done,rows):
    c=rec.c;c.verify_file(p,done)
    if not rows:raise ValueError('Empty completed evaluation')
    expected=done.get('rows',done.get('episodes_or_banks'))
    if expected is not None and len(rows)!=expected:raise ValueError(f'Expected {expected} rows, got {len(rows)}')
    snapshot=c.resolve(done.get('snapshot',meta.get('snapshot')))
    expected_hash=done.get('snapshot_sha256',meta.get('snapshot_sha256'))
    if snapshot and expected_hash and c.digest(snapshot)!=expected_hash:raise ValueError('Snapshot hash mismatch')
    training=c.training_for(meta,done)
    if training and training['status']!='PASS':raise ValueError('Adaptation training is not complete')
    if training and expected_hash and not snapshot:
        choices=list((c.root/training['path']).glob('snapshot-*.pt'))
        budget=first(training['done'],'transitions','adaptation_transitions',default=0)
        choices=sorted(choices,key=lambda x:not x.stem.endswith(f'{int(budget):07d}'))
        if not any(c.digest(x)==expected_hash for x in choices):raise ValueError('No snapshot bytes match completed evaluation')
    if training and training['method']=='dynamics_q':
        batch=c.json.get(p.parent.parent/'completion.json',{})
        if not passed(batch):raise ValueError('Dynamics+Q held-out batch not complete')
    seeds=first(meta,'evaluation_seeds',default=[]) or done.get('evaluation_seeds',[])
    # Two kinds may share seeds; uniqueness is checked per kind.
    grouped=defaultdict(list)
    for row in rows:grouped[(row.get('kind','shifted'),row.get('parameter'),row.get('severity'))].append(str(row.get('evaluation_seed',row.get('seed',''))))
    for ss in grouped.values():
        if len(ss)!=len(set(ss)):raise ValueError('Duplicate evaluation seed inside one completed stream')
        if seeds and set(map(normalized,ss))!=set(map(normalized,seeds)):raise ValueError('Held-out seed set differs from declared evaluation seeds')
    if training:
        train_seeds=first(training['meta'],'adaptation_seeds',default=[]) or [training['identity'].get('adaptation_seed')]
        if set(map(normalized,train_seeds)) & set(s for ss in grouped.values() for s in ss):raise ValueError('Training/evaluation seed overlap')


def final_rows(rows):
    return [r for r in rows if (r.get('method')!='frozen' or float(r.get('budget',0))==0)
            and (not r.get('declared_final_budget') or r.get('method')=='frozen' or float(r['budget'])==float(r['declared_final_budget']))]


def reconstruct_steps(rec,p,rows):
    steps=p.parent/'steps.csv'
    if not steps.is_file():return {}
    grouped=defaultdict(list)
    for n,r in enumerate(rec.c.table(steps),2):
        if not all(k in r for k in ('reward','seed')):return {}
        key=(r.get('parameter'),normalized(r.get('severity')),normalized(r['seed']),r.get('kind','shifted'))
        grouped[key].append((n,r))
    result={}
    for row in rows:
        key=(row.get('parameter'),normalized(row.get('severity')),normalized(row.get('seed',row.get('evaluation_seed'))),row.get('kind','shifted'))
        values=grouped.get(key,[])
        if not values:continue
        if len(values)!=int(row['episode_length']):raise ValueError('Raw step count differs from episode length')
        metrics={'episode_return':sum(float(r['reward']) for _,r in values)}
        for field in ('reward_dist','reward_ctrl'):
            if field in values[0][1]:metrics[field]=sum(float(r[field]) for _,r in values)
        for field,raw in [('distance_mean','distance'),('action_abs','action_abs'),('action_saturation','action_saturation'),('actuator_force_rms','force_rms')]:
            if raw in values[0][1]:metrics[field]=float(np.mean([float(r[raw]) for _,r in values]))
        for field,v in metrics.items():
            old=row.get(field,row.get('force_rms') if field=='actuator_force_rms' else None)
            if old is not None:rec.check(p,f'{key}:{field}',old,v,rtol=1e-6)
        result[key]=metrics
    rec.c.digest(steps)
    return result


def control(rec,p,meta,done):
    rows=list(rec.c.table(p))
    if not final_rows(rows):return
    _validate_eval(rec,p,meta,done,rows)
    raw=reconstruct_steps(rec,p,final_rows(rows)) if 'episode_length' in rows[0] else {}
    for i,row in enumerate(rows,2):
        if str(row.get('partial_smoke','False')).lower()=='true' or not final_rows([row]):continue
        key=(row.get('parameter'),normalized(row.get('severity')),normalized(row.get('seed',row.get('evaluation_seed'))),row.get('kind','shifted'))
        row={**row,**raw.get(key,{})}
        method='joint' if 'joint_return' in row else ('dynamics_only' if 'decision_return' in row else METHODS.get(row.get('method'),'frozen'))
        base=rec.base(p,row,meta,done,method)
        if not base:continue
        if 'budget' in row and method!='frozen' and row.get('declared_final_budget') and float(row['budget'])!=float(row['declared_final_budget']):continue
        for metric in CONTROL:
            prefix={'joint':'joint_','dynamics_only':'decision_'}.get(method,'')
            names=[prefix+metric,metric,*ALIASES.get(metric,())]
            if metric=='episode_return':names=['joint_return','decision_return','episode_return']
            if metric=='action_saturation':names=[prefix+'action_component_saturation',*names]
            v=next((row[k] for k in names if row.get(k) not in (None,'')),None)
            rec.emit(base,metric,v,p,i)
        if row.get('onpolicy_latent_mse'):rec.emit(base,'latent_mse',row['onpolicy_latent_mse'],p,i,kind='onpolicy_prediction')
        # Saved references are verified against independently discovered Frozen rows by deduplication.
        for key in ('frozen_return','frozen_episode_return'):
            if row.get(key):rec.emit({**base,'method':'frozen'},'episode_return',row[key],p,i,'DERIVED_ONLY')


def prediction(rec,p,meta,done):
    rows=list(rec.c.table(p))
    if not final_rows(rows):return
    _validate_eval(rec,p,meta,done,rows)
    for i,row in enumerate(rows,2):
        if not final_rows([row]):continue
        method='joint' if 'joint_self_consistency_mse' in row else ('dynamics_only' if 'decision_latent_mse' in row else METHODS.get(row.get('method'),'frozen'))
        base=rec.base(p,row,meta,done,method)
        if not base:continue
        if row.get('declared_final_budget') and method!='frozen' and float(row['budget'])!=float(row['declared_final_budget']):continue
        mse=next((row[k] for k in ('joint_self_consistency_mse','decision_latent_mse','latent_mse') if k in row),None)
        kind='joint_self_consistency' if method=='joint' else 'fixed_bank'
        rec.emit(base,'latent_mse',mse,p,i,kind=kind)
        if method!='joint' and row.get('frozen_latent_mse'):
            rec.emit({**base,'method':'frozen'},'latent_mse',row['frozen_latent_mse'],p,i,kind='fixed_bank')
        for k in ('low_action_n','mid_action_n','high_action_n'):
            if k in row:rec.emit(base,k,row[k],p,i,kind='action_regime_counts')
        if row.get('bank_sha256') and row.get('bank_file'):
            candidates=rec.c.by_name.get(Path(row['bank_file']).name,[])
            if candidates and not any(rec.c.digest(x)==row['bank_sha256'] for x in candidates):raise ValueError('Fixed transition bank hash mismatch')


def frozen(rec,p,meta,done):
    rows=list(rec.c.table(p))
    reconstruct_steps(rec,p,rows)
    for i,row in enumerate(rows,2):
        base=rec.base(p,row,meta,done,'frozen')
        if not base:continue
        base['kind']='shifted'
        for metric in CONTROL:
            v=next((row[k] for k in (metric,*ALIASES.get(metric,())) if k in row),None)
            rec.emit(base,metric,v,p,i)
        if row.get('latent_mse'):rec.emit(base,'latent_mse',row['latent_mse'],p,i,kind='fixed_bank')
        data=rec.c.resolve(str(p.parent/row.get('data_file',''))) if row.get('data_file') else None
        if data and data.suffix=='.npz':
            with np.load(data,allow_pickle=False) as a:
                if 'rewards' in a:
                    rec.check(p,f'episode_return[{i}]',row['episode_return'],float(a['rewards'].sum()))
                if 'actions' in a:
                    rec.check(p,f'action_abs[{i}]',row.get('action_abs_mean',row.get('action_abs')),float(np.abs(a['actions']).mean()),rtol=1e-6)
                    force=a['actuator_force'][1:] if 'actuator_force' in a else None
                    if force is not None:rec.emit(base,'actuator_force_rms',float(np.sqrt(np.mean(force**2))),data,'actuator_force[1:]')


def physical(rec,p,meta,done):
    groups=defaultdict(list)
    for i,row in enumerate(rec.c.table(p),2):
        if row.get('numerically_valid',row.get('numerical_valid','True'))!='True':raise ValueError('Invalid physical probe')
        base=rec.base(p,row,meta,done,'physical_probe')
        if not base:continue
        metrics=[k for k in row if k.endswith('_l2') or k in ('delta_q_rad','delta_qvel_rad_s','delta_tip_m','delta_accel_rad_s2','joint_acceleration_difference','root_acceleration_difference','nominal_latent_mse','shifted_latent_mse')]
        if 'qvel_delta_0' in row:
            dv=np.array([float(row[f'qvel_delta_{j}']) for j in range(9)])
            for metric,part in [('joint_velocity_l2',dv[3:]),('root_velocity_l2',dv[:2])]:
                rec.check(p,f'{metric}[{i}]',row[metric],float(np.linalg.norm(part)))
                row[metric]=float(np.linalg.norm(part))
        # Recompute Reacher state responses from saved vectors.
        for metric,field in [('delta_q_rad','q'),('delta_qvel_rad_s','qvel'),('delta_tip_m','tip')]:
            if f'nominal_{field}' in row:
                actual=float(np.linalg.norm(np.array(json.loads(row[f'shifted_{field}']))-np.array(json.loads(row[f'nominal_{field}']))))
                rec.check(p,f'{metric}[{i}]',row[metric],actual)
                row[metric]=actual
        for metric in metrics:
            if number(row[metric]) is not None:
                key=(base['task'],base['shift_parameter'],base['severity_multiplier'],base['evaluation_seed'],metric)
                groups[key].append((base,float(row[metric]),i))
    for key,values in groups.items():
        rec.emit(values[0][0],key[-1],np.mean([x[1] for x in values]),p,f'{values[0][2]}..{values[-1][2]}',kind='physical')


def latent(rec,p,meta,done):
    groups=defaultdict(list);bases={}
    for i,row in enumerate(rec.c.table(p),2):
        if int(row.get('horizon',1))!=1:continue
        identity=tuple(row.get(k,'') for k in ('parameter','severity','seed','schedule'))
        if identity not in bases:bases[identity]=rec.base(p,row,meta,done,'frozen')
        base=bases[identity]
        if not base:continue
        stage='latent_calibration' if row.get('source')=='nominal_primary_policy' else 'fixed_bank' if base['schedule']=='fixed' else 'onpolicy_prediction'
        for metric in ('latent_mse','reward_squared_error'):
            if number(row.get(metric)) is None:continue
            key=(base['shift_parameter'],base['severity_multiplier'],base['evaluation_seed'],stage,metric)
            groups[key].append((base,float(row[metric]),i))
    for key,items in groups.items():rec.emit(items[0][0],key[-1],np.mean([x[1] for x in items]),p,f'{items[0][2]}..{items[-1][2]}',kind=key[-2])


def nominal(rec,p,meta,done):
    for i,row in enumerate(rec.c.table(p),2):
        cp=rec.c.resolve(row.get('checkpoint'))
        if cp and (cp/'model.pt').is_file():meta={**meta,'checkpoint_sha256':rec.c.digest(cp/'model.pt')}
        if 'torso_mass' in row or row.get('mass_parameter')=='torso_mass':meta={**meta,'task':'halfcheetah'}
        row={**row,'parameter':'nominal','family':'nominal','severity':1}
        base=rec.base(p,row,meta,done,'frozen')
        if not base:continue
        base['cohort_id']=row.get('checkpoint_sha256',row.get('model_sha256',base['cohort_id']))
        base['checkpoint_sha256']=base['cohort_id'];base['checkpoint_id']=str(row.get('transitions',row.get('training_step',base['checkpoint_id'])))
        for metric in CONTROL:
            v=next((row[k] for k in (metric,*ALIASES.get(metric,())) if k in row),None)
            rec.emit(base,metric,v,p,i,kind='nominal_validation' if 'transitions' in row or 'training_step' in row else 'nominal_inspection')


def training_log(rec,p,meta,done):
    task=task_of(meta,{},rec.c.registry)
    if not task:return
    bins=defaultdict(list);counts=0;last=None
    with p.open() as stream:
        for line in stream:
            if not line.strip():continue
            row=json.loads(line);counts+=1
            step=row.get('global_step',row.get('transition',row.get('step',row.get('transitions',row.get('environment_transitions',row.get('decision',counts))))))
            last=step
            metrics=row.get('metrics',row)
            for k,v in metrics.items():
                if any(s in k for s in ('loss','grad_norm','entropy','pi_scale','q_target','q_mean','target_mean','gradient_norm')) and number(v) is not None:
                    bins[(int(step)//50000,k)].append(float(v))
    for (bin_id,metric),values in bins.items():
        rec.training.append(dict(task=task,source_path=rec.c.rel(p),source_sha256=rec.c.digest(p),metric=metric,
            transition_bin_start=bin_id*50000,n=len(values),mean=float(np.mean(values)),median=float(np.median(values)),std=float(np.std(values)),last_record_step=last,log_records=counts))


def collect(catalog,filters=None):
    rec=Records(catalog,filters)
    selected=[]
    for p in catalog.files:
        if p.suffix not in ('.csv','.jsonl'):continue
        name=p.name
        schema,example=catalog.classify_table(p) if p.suffix=='.csv' else ('training',{})
        adapter={'control':control,'prediction':prediction,'physical':physical,'nominal':nominal,
                 'frozen':frozen,'latent':latent,'training':training_log}.get(schema)
        if p.suffix=='.jsonl' and p.name!='training.jsonl':adapter=None
        if adapter is None:continue
        meta=catalog.context(p.parent);marker,done=catalog.marker(p.parent);status,reason=catalog.gate(p.parent)
        task=task_of(meta,{},catalog.registry)
        if adapter==nominal and not marker:
            integ=load(p.parent/'nominal_integrity.json')
            if isinstance(integ,list) and integ and all(v.get('learned_state_before')==v.get('learned_state_after') and v.get('optimizer_updates_during_evaluation')==0 for v in integ):status='PASS';reason='nominal integrity record'
        if filters and filters.get('task') and task_of(meta,example if p.suffix=='.csv' else {},catalog.registry) not in filters['task']:continue
        analysis=(filters or {}).get('analysis')
        if analysis in ('q_distribution','planning'):continue
        if adapter==training_log and status=='INCOMPLETE' and task_of(meta,{},catalog.registry):
            training_log(rec,p,meta,done)
            for r in rec.training:
                if r['source_path']==catalog.rel(p):r['completion_status']='INCOMPLETE_DESCRIPTIVE_ONLY'
        if status!='PASS':
            catalog.excluded.append(dict(path=catalog.rel(p),status=status,reason=reason));continue
        if adapter in (control,prediction) and METHODS.get((meta.get('identity') or {}).get('method'))=='dynamics_q':
            batch=catalog.json.get(p.parent.parent/'completion.json',{})
            if not passed(batch):
                catalog.excluded.append(dict(path=catalog.rel(p),status='INCOMPLETE',reason='Dynamics+Q held-out batch not complete'))
                continue
        selected.append((p,adapter,meta,done))
    # Independent baseline records first, then adapted evaluation references.
    selected.sort(key=lambda x:(x[1] not in (frozen,nominal),str(x[0])))
    for n,(p,adapter,meta,done) in enumerate(selected):
        if n%250==0:print(f'Ingest {n}/{len(selected)}: {p.name}',flush=True)
        before=len(rec.rows)
        try:adapter(rec,p,meta,done)
        except (ValueError,KeyError,OSError,TypeError) as exc:
            catalog.add_issue(p,str(exc));rec.rows=[r for r in rec.rows if r['source_path']!=catalog.rel(p)]
            rec.index={k:r for k,r in rec.index.items() if r['source_path']!=catalog.rel(p)}
            continue
        catalog.experiments.append(dict(path=catalog.rel(p.parent),task=task_of(meta,{},catalog.registry),stage=adapter.__name__,status='PASS',reason='validated schema and metadata',records=len(rec.rows)-before))
    from .q_statistics import ingest_archives
    if not filters or not filters.get('task') or 'halfcheetah' in filters['task']:ingest_archives(rec)
    return rec.finish()
