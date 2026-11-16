"""Single public CLI for NS-Gym control–critic research stages."""
import argparse
import sys
from pathlib import Path
from .metrics import write, require
from .provenance import audit,verified_artifact


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage',choices=['audit','q-diagnostics','planner-diagnostics','q-swap-eval','integrate-report','all'])
    p.add_argument('--results-root',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--device',default='cpu')
    p.add_argument('--smoke',action='store_true')
    p.add_argument('--smoke-result',type=Path)
    p.add_argument('--swap-source',type=Path)
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
            source,_=verified_artifact(root,'q-terminal-planning-v1')
            from .diagnostics.planning import run
            run(root,source,out)
        else:
            from .legacy_cli import main as inference
            sys.argv=[sys.argv[0],'--results-root',str(root),'--output',str(out),'--device',a.device]+(['--smoke'] if a.smoke else [])
            inference()
        return
    if a.stage=='q-swap-eval':
        require(a.device.startswith('cuda'),'Use --device cuda:0 for historical evaluator parity')
        require(a.device=='cuda:0','Select physical GPU with CUDA_VISIBLE_DEVICES; logical device must be cuda:0')
        from .evaluation import evaluate
        evaluate(root,out,a.smoke,a.smoke_result)
    elif a.stage=='integrate-report':
        from .reporting.html import integrate
        integrate(root,out,a.swap_source)
    else:
        require(not out.exists(),'Output must be new');out.mkdir(parents=True)
        require(audit(root,out/'audit'),'Audit failed; expansion stopped')
        for name in ('q-terminal-planning-v1','q-decision-relevance-v1'):verified_artifact(root,name)
        require(a.device=='cuda:0','All includes physical evaluation: --device cuda:0 is required')
        from .evaluation import evaluate
        evaluate(root,out/'smoke',True)
        if a.smoke:return
        evaluate(root,out/'q-swap',False,out/'smoke')
        from .reporting.html import integrate
        integrate(root,out/'report',out/'q-swap')

if __name__=='__main__':main()
