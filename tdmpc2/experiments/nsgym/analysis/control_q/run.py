"""Single public CLI for NS-Gym control–critic research stages."""
import argparse
import sys
from pathlib import Path
from .metrics import write, require
from .provenance import audit,verified_artifact


def infer_q(root,out,device,smoke=False):
    from .legacy_cli import main as inference
    previous_argv=sys.argv
    try:
        sys.argv=[previous_argv[0],'--results-root',str(root),'--output',str(out),'--device',device]+(['--smoke'] if smoke else [])
        inference()
    finally:
        sys.argv=previous_argv


def infer_planner(root,source,out):
    from .reporting.figures import presentation
    from .diagnostics.planning import run
    source,_=verified_artifact(root,'q-terminal-planning-v1',source)
    v2=out.parent/(out.name+'-presentation')
    require(not v2.exists() and not v2.with_suffix('.zip').exists(),'Presentation output must be new')
    presentation(root,source,v2)
    run(root,source,out,presentation_source=v2)
    return v2


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage',choices=['audit','q-diagnostics','planner-diagnostics','q-swap-eval','integrate-report','all'])
    p.add_argument('--results-root',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--device',default='cpu')
    p.add_argument('--smoke',action='store_true')
    p.add_argument('--smoke-result',type=Path)
    p.add_argument('--swap-source',type=Path)
    p.add_argument('--q-source',type=Path,help='Completed Q inference from this run')
    p.add_argument('--presentation-source',type=Path)
    p.add_argument('--planner-source',type=Path)
    p.add_argument('--validate-only',action='store_true')
    p.add_argument('--reuse-validated',action='store_true')
    a=p.parse_args();root=a.results_root.resolve();out=a.output.resolve()
    if a.validate_only or a.stage=='audit':
        if not audit(root,out):raise SystemExit(2)
        return
    if a.stage in ('q-diagnostics','planner-diagnostics'):
        name='q-terminal-planning-v1' if a.stage=='q-diagnostics' else 'q-decision-relevance-v1'
        if a.reuse_validated:
            source,record=verified_artifact(root,name)
            require(not out.exists(),'Output must be new');out.mkdir(parents=True)
            write(out/'validated_cache.json',dict(path=str(source),**record))
            write(out/'completion.json',dict(status='PASS',reused=True,environment_interactions=0))
        elif a.stage=='planner-diagnostics':
            require(a.q_source is not None,'Fresh planner diagnostics require --q-source')
            infer_planner(root,a.q_source.resolve(),out)
        else:
            infer_q(root,out,a.device,a.smoke)
        return
    if a.stage=='q-swap-eval':
        require(a.device.startswith('cuda'),'Use --device cuda:0 for historical evaluator parity')
        require(a.device=='cuda:0','Select physical GPU with CUDA_VISIBLE_DEVICES; logical device must be cuda:0')
        from .evaluation import evaluate
        evaluate(root,out,a.smoke,a.smoke_result)
    elif a.stage=='integrate-report':
        from .reporting.html import integrate
        integrate(root,out,a.swap_source,a.q_source,a.presentation_source,a.planner_source)
    else:
        require(not out.exists(),'Output must be new');out.mkdir(parents=True)
        require(audit(root,out/'audit'),'Audit failed; expansion stopped')
        infer_q(root,out/'q-diagnostics',a.device)
        presentation_source=infer_planner(root,out/'q-diagnostics',out/'planner-diagnostics')
        require(a.device=='cuda:0','All includes physical evaluation: --device cuda:0 is required')
        from .evaluation import evaluate
        evaluate(root,out/'smoke',True)
        if a.smoke:return
        evaluate(root,out/'q-swap',False,out/'smoke')
        from .reporting.html import integrate
        integrate(root,out/'report',out/'q-swap',out/'q-diagnostics',presentation_source,out/'planner-diagnostics')

if __name__=='__main__':main()
