"""Single read-only entry point for final NS-Gym research figures."""
from __future__ import annotations

import argparse
from pathlib import Path

from .data import ARTIFACTS, load_inputs, load_swap, paired_terminal_arrays
from .metrics import require, write, read, sha256
from .provenance import verified_artifact


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode',choices=('figures','report','all'),default='all')
    parser.add_argument('--results-root',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--swap-source',type=Path,help='Optional complete held-out Q-swap evaluation')
    parser.add_argument('--smoke',action='store_true',help='Validate one saved array; no model/simulator')
    parser.add_argument('--validate-only',action='store_true',help='Verify hashes and coverage; draw nothing')
    a=parser.parse_args(argv)
    root=a.results_root.resolve();out=a.output.resolve()
    if a.mode=='report' and not (a.smoke or a.validate_only):
        from .report import guide
        require((out/'FIGURE_VALIDATION.json').is_file(),'Report requires completed figures in --output')
        paths={key:str(verified_artifact(root,name)[0]) for key,name in ARTIFACTS.items()}
        if (out/'SOURCE_ARTIFACTS.json').is_file():paths.update(read(out/'SOURCE_ARTIFACTS.json'))
        guide(out,read(out/'FIGURE_VALIDATION.json'),paths)
        print(f'PASS report: {out}')
        return
    require(not out.exists(),'Output directory must be new')
    paths,tables,arrays,provenance=load_inputs(root)
    out.mkdir(parents=True)
    source_paths={key:str(value) for key,value in paths.items()}
    write(out/'SOURCE_ARTIFACTS.json',source_paths)
    write(out/'SOURCE_PROVENANCE.json',provenance)
    if a.validate_only or a.smoke:
        if a.smoke:paired_terminal_arrays(arrays[('halfcheetah','damping',2.,42000001)])
        write(out/'completion.json',dict(status='PASS',mode='smoke' if a.smoke else 'validate-only',
                                         verified_artifacts=3,conditions=36,simulator_calls=0))
        print(f'PASS source validation: {out}')
        return
    summary_dir=out/'tables';summary_dir.mkdir()
    for name in ('control','ranking','native','effects','returns'):
        tables[name].to_csv(summary_dir/f'{name}_summary.csv',index=False)
    from .figures import render
    from .report import code_audit,guide,validate_figures
    swap=load_swap(root,a.swap_source)
    if swap is not None:source_paths['q_swap']=str(swap[0]);write(out/'SOURCE_ARTIFACTS.json',source_paths)
    result=render(root,out,tables,arrays,paths,swap)
    for key,record in provenance.items():
        for relative,digest in record['outputs'].items():
            path=paths[key]/relative
            require(path.is_file() and sha256(path)==digest,f'Source changed during rendering: {path}')
    code_audit(out)
    validation=validate_figures(out,result)
    guide(out,validation,source_paths)
    print(f'PASS: {validation["figure_png_count"]} PNG, 36 conditions, 108 streams; {out}')


if __name__=='__main__':main()
