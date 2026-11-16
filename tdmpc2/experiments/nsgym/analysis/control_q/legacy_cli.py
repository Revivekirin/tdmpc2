"""Migrated historical computations; numerical formulas preserved."""
from __future__ import annotations
import argparse
import time
from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use('Agg')
from experiments.nsgym.analysis.control_q.config import SCRIPTS
from experiments.nsgym.analysis.control_q.diagnostics.planning import run
from experiments.nsgym.analysis.control_q.diagnostics.terminal_value import cached_immediate, infer_job
from experiments.nsgym.analysis.control_q.metrics import require, write
from experiments.nsgym.analysis.control_q.models import Sources, inventory
from experiments.nsgym.analysis.control_q.reporting.figures import finish, presentation


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results-root',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--decision-source',type=Path,help='Decision relevance including native environment-free planning from completed v1')
    parser.add_argument('--presentation-source',type=Path,help='Render presentation v2 from completed v1 arrays; no model inference')
    parser.add_argument('--device',default='cpu');parser.add_argument('--smoke',action='store_true')
    parser.add_argument('--states',type=int,default=4);parser.add_argument('--candidates',type=int,default=32)
    parser.add_argument('--sensitivity-repeats',type=int,default=3)
    args=parser.parse_args();require(args.states>=2 and args.candidates>=5 and args.sensitivity_repeats>=2,'Insufficient multi-state/candidate/sensitivity sample')
    root=args.results_root.resolve();out=args.output.resolve();require(not out.exists(),'Output directory must be new')
    require(not out.with_suffix('.zip').exists(),'Output ZIP must be new')
    if args.decision_source:
        from .diagnostics.planning import run
        run(root,args.decision_source.resolve(),out)
        return
    if args.presentation_source:
        presentation(root,args.presentation_source.resolve(),out)
        return
    out.mkdir(parents=True)
    for folder in ('reports','tables','figures','q_arrays'):(out/folder).mkdir()
    import torch
    torch.set_num_threads(2);started=time.monotonic();src=Sources()
    for p in [Path(__file__),SCRIPTS/'q_planning_core.py',SCRIPTS/'mechanism_forward.py',SCRIPTS/'training_support.py',SCRIPTS.parent/'analysis/reacher_dq_figures.py',
              SCRIPTS.parents[2]/'tdmpc2/tdmpc2.py',SCRIPTS.parents[2]/'tdmpc2/common/world_model.py',SCRIPTS.parents[2]/'tdmpc2/common/math.py']:
        src.track(p,'analysis/native source')
    jobs,configs,banks,gaps=inventory(root,src,args.smoke)
    from experiments.nsgym.analysis.control_q.models import track_implementation
    track_implementation(src)
    pd.DataFrame(src.inventory).to_csv(out/'tables/SOURCE_INVENTORY.csv',index=False)
    write(out/'SOURCE_INPUTS.json',src.files)
    print(f'Inventory complete: {sum(j["valid"] for j in jobs)}/{len(jobs)} selected streams; {len(gaps)} gaps',flush=True)
    rows={k:[] for k in ('immediate','terminal','effects','rankings','ranking_summary','components')};validations=[]
    for job in jobs:
        if not job['valid']:
            try:rows['immediate'].extend(cached_immediate(root,job,configs[job['task']],src,args,out))
            except Exception as exc:gaps.append(dict(task=job['task'],reason='Immediate cache unavailable: '+str(exc)))
            continue
        try:
            result,validation=infer_job(root,job,configs[job['task']],banks,src,args,out)
            for name,value in result.items():rows[name].extend(value)
            validations.append(validation)
            print(f'PASS {job["task"]} {job["family"]} k={job["severity"]:g} seed={job["adaptation_seed"]}',flush=True)
        except Exception as exc:
            import traceback
            traceback.print_exc();identity={n:job[n] for n in ('task','family','severity','adaptation_seed')}
            gaps.append(dict(**identity,reason=str(exc)));validations.append(dict(**identity,status='FAIL',error=str(exc)))
            if isinstance(exc,(ImportError,OSError)):
                try:rows['immediate'].extend(cached_immediate(root,job,configs[job['task']],src,args,out))
                except Exception as cache_error:gaps.append(dict(**identity,reason='Immediate cache unavailable: '+str(cache_error)))
            if args.smoke:break
    status=finish(out,rows,validations,src,gaps,configs,args,time.monotonic()-started)
    if status!='PASS':raise SystemExit(2)
