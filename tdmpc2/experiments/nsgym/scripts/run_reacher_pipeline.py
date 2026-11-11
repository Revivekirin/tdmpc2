#!/usr/bin/env python3
"""Reacher R0–R3: independent stages; no exhaustive experiment runs implicitly."""
import argparse
from pathlib import Path
import json
import time
import reacher_pipeline as r


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage',choices=['discover','nominal','inventory','probe','frozen','smoke','adapt','predict','control','aggregate'])
    p.add_argument('--config',default=str(r.DEFAULT));p.add_argument('--output',required=True)
    p.add_argument('--search-root',default=str(r.ROOT/'results/nsgym'));p.add_argument('--run',help='Explicit discovered training run when multiple exist')
    p.add_argument('--discovery');p.add_argument('--selection',help='R0 selected_reacher_checkpoint.json; immutable checkpoint identity')
    p.add_argument('--inventory');p.add_argument('--probe');p.add_argument('--frozen');p.add_argument('--smoke')
    p.add_argument('--snapshot');p.add_argument('--runs',help='Adaptation matrix root for prediction/control evaluation')
    p.add_argument('--inputs',nargs='+',help='Read-only stage directories for aggregation')
    p.add_argument('--family',choices=['D0','D1','M0','M1'],help='Adapt stage filter only');p.add_argument('--parameter',help='Adapt stage filter only');p.add_argument('--severity',type=float,help='Adapt stage filter only')
    p.add_argument('--method',choices=['all','frozen','decision'],default='all')
    p.add_argument('--synthetic',action='store_true',help='Only smoke: CPU fixture, never scientific performance')
    p.add_argument('--dry-run',action='store_true')
    a=p.parse_args();c=r.config(a.config)
    if a.synthetic and a.stage!='smoke':p.error('--synthetic is smoke-only')
    if a.stage!='adapt' and any(x is not None for x in (a.family,a.parameter,a.severity)):p.error('Condition filters are supported only by adapt; other stages keep the full grid')
    if a.family and a.parameter and c['families'][a.family]!=a.parameter:p.error('family/parameter mismatch')
    if a.severity is not None and a.severity not in c['severities']:p.error('Severity outside preregistered grid')
    for stage,fields in {'nominal':['discovery'],'probe':['selection','inventory'],'frozen':['selection','probe'],'adapt':['selection','frozen','smoke'],'predict':['selection','frozen'],'control':['selection','frozen']}.items():
        if a.stage==stage:
            for key in fields:
                if not getattr(a,key):p.error(f'--{key} required for {stage}')
    if a.stage=='smoke' and not a.synthetic and not a.selection:p.error('--selection required for real smoke')
    if a.stage in ('predict','control') and bool(a.snapshot)==bool(a.runs):p.error('Provide exactly one of --snapshot or --runs')
    if a.stage=='aggregate' and not a.inputs:p.error('--inputs required')
    manifest=dict(stage=a.stage,**r.provenance(c),arguments=vars(a))
    if a.dry_run:print(json.dumps(manifest,indent=2));return
    s=r.selected(a.selection,c) if a.selection else None
    if s:manifest['checkpoint_sha256']=s['model_sha256']
    if a.stage=='probe':r.gate(a.inventory,'inventory',c)
    if a.stage=='frozen':r.gate(a.probe,'probe',c,s)
    if a.stage in ('adapt','predict','control'):r.gate(a.frozen,'frozen',c,s)
    if a.stage=='adapt':
        sm=r.gate(a.smoke,'smoke',c,s);assert not sm['arguments']['synthetic']
        budget_protocol=r.read(Path(a.smoke)/'budget_protocol.json')
        assert budget_protocol['status']=='READY' and budget_protocol['final_budget'], 'No runtime-feasible preregistered budget'
    out=Path(a.output);out.mkdir(parents=True,exist_ok=False);r.write(out/'manifest.json',manifest)
    try:
        if a.stage=='discover':
            result=r.discover(a.search_root,c,out,a.run)
            if result['status']=='BLOCKED':
                r.write(out/'completion.json',dict(status='BLOCKED',reason=result['reason']));print(json.dumps(result,indent=2));return
        elif a.stage=='nominal':r.nominal(c,a.discovery,out)
        elif a.stage=='inventory':r.inventory(c,out)
        elif a.stage=='probe':
            agent,_=r.load(s['checkpoint'],c);r.probe(c,out,agent)
            from analyze_reacher_pipeline import physical_plots
            physical_plots(out,out,c)
        elif a.stage=='frozen':
            r.frozen(c,s,out)
            from analyze_reacher_pipeline import frozen_plots
            frozen_plots(out,out,c,Path(a.probe))
        elif a.stage in ('smoke','adapt'):
            conds=r.conditions(c)
            if a.stage=='smoke':conds=[x for x in conds if x['severity']==1.5 and x['family'] in ('D0','M0')]
            else:
                conds=[x for x in conds if (not a.family or x['family']==a.family) and (not a.parameter or x['parameter']==a.parameter) and (a.severity is None or x['severity']==a.severity)]
            assert conds
            methods=c['methods'] if a.method=='all' else [a.method]
            if a.stage=='smoke' and methods!=c['methods']:raise ValueError('Smoke verifies both methods')
            seeds=c['adaptation_seeds'][:1] if a.stage=='smoke' else c['adaptation_seeds']
            budgets=[0,100] if a.stage=='smoke' else r.read(Path(a.smoke)/'budget_protocol.json')['budgets']
            if a.stage=='smoke':
                # Real physics all 24 conditions, including exact counterfactual replay.
                physics=out/'physics';physics.mkdir();r.inventory(c,physics);r.probe(c,physics)
            results=[]
            for cond in conds:
                for method in methods:
                    for seed in seeds:
                        name=f'{r.tag(cond)}-{method}-seed{seed}'
                        result=r.adapt_one(c,s,out/name,cond,method,seed,budgets,a.synthetic);results.append(result)
                        print('PASS',name,flush=True)
            if a.stage=='smoke':
                seconds_per_transition=max(x['wall_seconds']/100 for x in results)
                nstreams=len(r.conditions(c))*len(c['methods'])*len(c['adaptation_seeds'])
                feasible=[b for b in c['snapshot_budgets'][1:] if b*seconds_per_transition*nstreams<=24*3600]
                final=max(feasible) if feasible and not a.synthetic else None
                r.write(out/'budget_protocol.json',dict(budgets=[b for b in c['snapshot_budgets'] if final is not None and b<=final],final_budget=final,
                    status='READY' if final else 'BLOCKED',synthetic=a.synthetic,seconds_per_transition=seconds_per_transition,matrix_streams=nstreams,
                    estimated_adaptation_hours=final*seconds_per_transition*nstreams/3600 if final else None,
                    rule=c['budget_rule'],evaluation_cost='Not included; held-out evaluation launched separately'))
        elif a.stage in ('predict','control'):
            snapshots=[Path(a.snapshot)] if a.snapshot else sorted(Path(a.runs).glob('*/snapshot-*.pt'))
            assert snapshots,'No snapshots'
            if a.runs:assert r.read(Path(a.runs)/'completion.json')['status']=='PASS'
            for snapshot in snapshots:
                target=out/(snapshot.parent.name+'-'+snapshot.stem);target.mkdir()
                r.evaluate_snapshot(c,s,snapshot,target,a.frozen,a.stage)
                print('PASS',target.name,flush=True)
        elif a.stage=='aggregate':
            from analyze_reacher_pipeline import aggregate
            aggregate([Path(x) for x in a.inputs],out,c)
        if a.stage=='adapt':
            assert budgets and budgets[-1]>0
        r.done(out)
    except Exception as ex:
        r.write(out/'failure.json',dict(error=type(ex).__name__,message=str(ex),time=time.time()));raise

if __name__=='__main__':main()
