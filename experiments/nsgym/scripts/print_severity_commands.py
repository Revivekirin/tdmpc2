"""Print commands validated by the actual stage parser and versioned protocol."""
import argparse
import json
import shlex
from pathlib import Path
from severity_protocol import parser as stage_parser, resolve, load


def commands(root, checkpoint=None):
    root=Path(root);cp=str(checkpoint or load('physical_severity_grid.yaml')['primary_checkpoint'])
    def cmd(stage,name,*extra):
        return [stage,'--checkpoint',cp,'--output',str(root/name),*map(str,extra)]
    jobs=[('setup',cmd('manifest','preregistered')),
          ('setup',cmd('environment','environment','--smoke')),
          ('setup',cmd('inventory','inventory','--smoke'))]
    for schedule in load('nonstationary_schedules.yaml')['schedules']:
        jobs.append(('smoke',cmd('validate','validate-'+schedule,'--smoke','--schedule',schedule)))
    jobs += [
        ('physical',cmd('bank','bank')),
        ('physical',cmd('physical','physical','--input',root/'bank')),
        ('physical',cmd('summarize','physical-summary','--input',root/'physical')),
        ('diagnostics',cmd('latent','physical-latent','--input',root/'physical')),
        ('diagnostics',cmd('summarize','physical-latent-summary','--input',root/'physical-latent')),
        ('frozen',cmd('frozen','frozen','--gate',root/'physical')),
        ('frozen',cmd('summarize','frozen-summary','--input',root/'frozen','--seed-split','evaluation','--physical-reference',root/'physical')),
        ('frozen',cmd('latent','heldout-latent','--input',root/'frozen','--seed-split','evaluation')),
        ('frozen',cmd('summarize','heldout-latent-summary','--input',root/'heldout-latent','--seed-split','evaluation')),
        ('planning',cmd('planning','planning','--input',root/'physical','--severity','1.5','--gate',root/'frozen'))]
    for schedule in load('nonstationary_schedules.yaml')['schedules']:
        phase='nonstationary' if schedule in ('nominal','fixed','abrupt') else 'optional'
        extra=['--severity','1'] if schedule=='nominal' else []
        if phase=='optional':extra+=['--allow-optional-large']
        name='nonstationary-'+schedule
        jobs += [(phase,cmd('nonstationary',name,'--schedule',schedule,'--gate',root/'frozen',*extra)),
                 (phase,cmd('summarize',name+'-summary','--input',root/name,'--seed-split','evaluation')),
                 (phase,cmd('latent',name+'-latent','--input',root/name,'--seed-split','evaluation')),
                 (phase,cmd('summarize',name+'-latent-summary','--input',root/(name+'-latent'),'--seed-split','evaluation'))]
    return jobs


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output-root',type=Path,default=Path('results/nsgym/physical-severity-750k-v1'))
    p.add_argument('--checkpoint',type=Path,help='750k checkpoint directory; use an absolute path when results live outside this project root')
    p.add_argument('--phase',choices=['all','setup','smoke','physical','diagnostics','planning','frozen','nonstationary','optional'],default='all')
    p.add_argument('--python',default='python')
    p.add_argument('--dry-run',action='store_true',help='Append --dry-run to each generated stage command')
    args=p.parse_args()
    print('# Execute from the project root containing experiments/ and tdmpc2/ in tdmpc2-nsgym. Commands are printed, never executed.')
    print('# Run phases separately: physical results review -> frozen -> frozen results review -> nonstationary.')
    print('# optional S3-S5 commands require separate experiment approval. Every output directory must be new.')
    for phase,argv in commands(args.output_root,args.checkpoint):
        if phase=='optional' and args.phase!='optional':continue
        if args.phase not in ('all',phase):continue
        resolved=resolve(stage_parser().parse_args(argv))
        print(f'\n# Phase {phase}; budget {json.dumps(resolved["budget"])}')
        print(shlex.join([args.python,'experiments/nsgym/scripts/run_severity.py',*argv,*(['--dry-run'] if args.dry_run else [])]))


if __name__=='__main__':main()
