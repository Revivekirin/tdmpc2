"""Metadata-led discovery of saved experiments; never execute a model or environment."""
from __future__ import annotations
import csv
import hashlib
import json
import os
from pathlib import Path
from .schema import sha256

METHODS = {'frozen':'frozen', 'decision':'dynamics_only', 'episode':'dynamics_only',
           'dynamics_only':'dynamics_only', 'dynamics_q_118':'dynamics_q',
           'reacher_dynamics_q_v1':'dynamics_q', 'dynamics_q':'dynamics_q',
           'full_joint_rl':'joint', 'joint':'joint', 'F':'frozen','D':'dynamics_only','DQ':'dynamics_q','J':'joint'}

def load(path):
    try: return json.loads(Path(path).read_text())
    except (OSError, ValueError): return {}

def walk_values(d, key):
    if isinstance(d, dict):
        if key in d: yield d[key]
        for v in d.values(): yield from walk_values(v,key)
    elif isinstance(d,list):
        for v in d: yield from walk_values(v,key)

def first(d, *keys, default=''):
    for key in keys:
        for value in walk_values(d,key):
            if value is not None and value != '': return value
    return default

def task_of(meta, row, registry):
    text=' '.join(str(first(meta,k)) for k in ('task','environment_id','env_id')).lower()
    parameter=str(row.get('parameter',row.get('mass_parameter','torso_mass' if 'torso_mass' in row else first(meta,'parameter'))))
    for task,cfg in registry.items():
        if any(t.lower() in text for t in cfg['environment_aliases']) or parameter in cfg['parameters']: return task
    return ''

def passed(d):
    return d.get('status') == 'PASS' or d.get('integration') == 'PASS'

class Catalog:
    def __init__(self, root, registry, output=None):
        self.root=Path(root).resolve(); self.registry=registry
        self.files=[]; self.json={}; self.hashes={}; self.issues=[]; self.experiments=[]; self.excluded=[]
        excluded=Path(output).resolve() if output else None
        for folder, dirs, names in os.walk(self.root,followlinks=False):
            folder=Path(folder)
            # Detect this pipeline's outputs by its marker, regardless of directory name.
            marker=load(folder/'manifest.json')
            generated=(folder/'SOURCE_PROVENANCE.json').exists() or (folder/'UNIFIED_RESEARCH_REPORT.md').exists() or marker.get('no_training_or_simulator')
            if generated or folder==excluded or folder.name in ('.git','__pycache__','unified-analysis'):
                dirs[:]=[];continue
            dirs[:]=sorted(d for d in dirs if not (folder/d).is_symlink())
            for name in sorted(names):
                p=folder/name
                if p.is_symlink() or not p.is_file():continue
                self.files.append(p)
                if name in ('manifest.json','analysis_manifest.json','metadata.json','completion.json','audit.json','analysis_config.json','selected_reacher_checkpoint.json','nominal_integrity.json','integrity.json'):
                    try:self.json[p]=json.loads(p.read_text())
                    except (ValueError,OSError) as exc:self.issues.append(dict(path=self.rel(p),status='INVALID',reason=str(exc)))
        for marker,done in self.json.items():
            if marker.name!='completion.json' or not isinstance(done,dict) or not passed(done):continue
            names=done.get('completed',[])+done.get('skipped',[])
            if names and all(isinstance(n,str) for n in names):
                missing=[n for n in names if not (marker.parent/n/'completion.json').is_file()]
                if missing:self.add_issue(marker,f'Completed child records absent: {missing[:5]}')
                if done.get('expected') is not None and len(set(names))!=done['expected']:self.add_issue(marker,'Completed count differs from expected')
        self.initial_stats={p:(p.stat().st_size,p.stat().st_mtime_ns) for p in self.files}
        self.by_name={}
        for p in self.files:self.by_name.setdefault(p.name,[]).append(p)
        self.training={}; self.inventory=[]
        for p in self.files:
            role='binary' if p.suffix in ('.pt','.npz','.npy') else 'other'
            if p.suffix=='.json':role='provenance'
            if p.suffix=='.jsonl':role='training_log' if p.name=='training.jsonl' else 'measurement_log'
            if p.suffix=='.csv':role='table'
            self.inventory.append(dict(path=self.rel(p),bytes=p.stat().st_size,role=role))
        for p in self.by_name.get('manifest.json',[]):
            meta=self.json[p]; done=self.json.get(p.parent/'completion.json',{})
            if 'identity' not in meta:continue
            # Training manifests have configuration and a local trajectory/snapshot, evaluation manifests reference a snapshot.
            training=('mode' not in meta and (any(k in done for k in ('transitions','metrics','dynamics_steps')) or (p.parent/'training.jsonl').exists() or any(p.parent.glob('snapshot-*.pt'))))
            if not training:continue
            task=task_of(meta,meta['identity'],registry); method=METHODS.get(meta['identity'].get('method'),'')
            if not task or not method:continue
            status,reason=self.gate(p.parent)
            if status=='PASS' and done.get('snapshot_sha256') and not any(p.parent.glob('snapshot-*.pt')):
                status,reason='EXCLUDED','Completed training copy has no saved snapshot bytes'
            entry=dict(path=self.rel(p.parent),task=task,method=method,stage='adaptation',status=status,reason=reason,
                       identity=meta['identity'],meta=meta,done=done)
            self.training[p.parent]=entry
            self.experiments.append({k:v for k,v in entry.items() if k not in ('meta','done')})

        for record,p in zip(self.inventory,self.files):
            if p.suffix=='.csv':
                schema,row=self.classify_table(p)
                meta=self.context(p.parent);status,reason=self.gate(p.parent)
                record.update(role='raw_measurement' if schema and schema!='derived' else 'derived_summary' if schema=='derived' else 'unclassified_table',schema=schema,task=task_of(meta,row,self.registry),status=status)
                if schema and schema!='derived':
                    self.experiments.append(dict(path=self.rel(p),task=record['task'],stage=schema,status=status,reason=reason))

    def classify_table(self,p):
        try:
            with p.open(newline='') as f:
                rd=csv.DictReader(f); row=next(rd,{})
        except (OSError,UnicodeError,csv.Error):return '',{}
        keys=set(row)
        meta=self.context(p.parent)
        # Batch-level concatenations repeat completed leaf streams. Prefer those
        # leaf records with snapshot provenance; do not count the export again.
        local_done=self.json.get(p.parent/'completion.json',{})
        children=local_done.get('completed',[]) if isinstance(local_done,dict) else []
        if 'adaptation_seed' in keys and not meta.get('identity') and children and any(isinstance(n,str) and (p.parent/n/'completion.json').is_file() for n in children):return 'derived',row
        # Aggregate exports are references, never raw measurements.
        if ('mean' in keys and 'metric' in keys) or 'source_record_id' in keys or {'frozen_return','ttt_return'}<=keys:return 'derived',row
        if 'inputs' in meta and not meta.get('mode'):return 'derived',row
        if {'nominal_q','shifted_q'}<=keys or {'root_velocity_l2','source_step'}<=keys:return 'physical',row
        if {'horizon','episode_step','latent_mse'}<=keys:return 'latent',row
        if {'head','timestep','q'}<=keys or 'candidate_sha256' in keys or {'left','right','js'}<=keys:return 'q_measurement',row
        if 'checkpoint_sha256' in keys and ('transitions' in keys or 'mass_scale' in keys) and 'episode_return' in keys:return 'nominal',row
        if p.name=='reacher_checkpoint_validation.csv':return 'nominal',row
        if (('latent_mse' in keys and ('samples' in keys or 'transitions' in keys or 'frozen_latent_mse' in keys)) or 'decision_latent_mse' in keys or 'joint_self_consistency_mse' in keys) and ('seed' in keys or 'evaluation_seed' in keys):return 'prediction',row
        if ('episode_return' in keys or 'decision_return' in keys or 'joint_return' in keys) and ('seed' in keys or 'evaluation_seed' in keys):
            if 'adaptation_seed' in keys:return 'control',row
            if 'data_file' in keys or p.name in ('episodes.csv','reacher_frozen_severity.csv'):return 'frozen',row
        return '',row

    def rel(self,p):return str(Path(p).relative_to(self.root))
    def digest(self,p):
        p=Path(p)
        if p not in self.hashes:self.hashes[p]=sha256(p)
        return self.hashes[p]
    def resolve(self,value):
        if not value:return None
        p=Path(str(value))
        if p.is_absolute() and p.exists():p=p.resolve()
        if p.is_absolute() and p.is_relative_to(self.root) and p.exists():return p
        parts=p.parts
        # Relocation uses the results-root name, not a historical experiment directory.
        for i in range(len(parts)):
            if parts[i]==self.root.name:
                candidate=self.root.joinpath(*parts[i+1:])
                if candidate.exists():return candidate
        candidate=self.root/p
        if candidate.exists():return candidate
        return None
    def context(self,folder):
        merged={}; ancestors=[]
        folder=Path(folder)
        while folder.is_relative_to(self.root):
            ancestors.append(folder)
            if folder==self.root:break
            folder=folder.parent
        for parent in reversed(ancestors):
            for name in ('metadata.json','manifest.json','analysis_manifest.json'):
                merged.update(self.json.get(parent/name,{}))
        return merged
    def marker(self,folder):
        p=Path(folder)
        while p.is_relative_to(self.root):
            for c in (p/'completion.json',p/'provenance/completion.json',p/'integrity.json'):
                if c in self.json:return c,self.json[c]
            if p==self.root:break
            p=p.parent
        return None,{}
    def gate(self,folder):
        meta=self.context(folder)
        if any(v is True for k in ('smoke','synthetic','cpu_fixture','partial_smoke') for v in walk_values(meta,k)) or first(meta,'command','stage')=='smoke':
            return 'EXCLUDED','smoke/synthetic metadata'
        # Filename is only an additional conservative exclusion, never positive evidence.
        if any('smoke' in p.lower() or 'pilot' in p.lower() or p.startswith('archive-') for p in Path(folder).relative_to(self.root).parts):
            return 'EXCLUDED','bounded pilot/smoke/archive path'
        marker,done=self.marker(folder)
        if not passed(done):return 'INCOMPLETE',f'no PASS completion: {self.rel(marker) if marker else "absent"}'
        for flag in ('model_unchanged','model_and_adapter_unchanged','replay_evaluation_isolated'):
            if done.get(flag) is False:return 'INVALID',f'{flag}=false'
        return 'PASS','completed saved record'
    def verify_file(self,p,done):
        if done.get('csv_sha256') and p.suffix=='.csv' and self.digest(p)!=done['csv_sha256']:
            raise ValueError(f'CSV hash mismatch: {self.rel(p)}')
    def training_for(self,meta,done):
        snap=self.resolve(done.get('snapshot',meta.get('snapshot')))
        if snap and snap.parent in self.training:return self.training[snap.parent]
        digest=done.get('snapshot_sha256',meta.get('snapshot_sha256'))
        if digest:
            matches=[v for v in self.training.values() if v['status']=='PASS' and v['done'].get('snapshot_sha256')==digest]
            if matches and len({json.dumps(v['identity'],sort_keys=True) for v in matches})==1:return sorted(matches,key=lambda v:v['path'])[0]
        return None
    def training_batch_complete(self,training):
        if not training or training['status']!='PASS':return False
        stream=self.root/training['path']
        batch=self.json.get(stream.parent/'completion.json',{})
        completed=batch.get('completed',[]) if isinstance(batch,dict) else []
        return passed(batch) and stream.name in completed and batch.get('expected')==len(set(completed))
    def table(self,p):
        with p.open(newline='') as stream:yield from csv.DictReader(stream)
    def add_issue(self,p,reason,status='INVALID'):
        self.issues.append(dict(path=self.rel(p),status=status,reason=reason))
