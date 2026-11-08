"""Regenerate plots/tables into a NEW directory; leave source results untouched."""
import argparse
from pathlib import Path
import shutil
from research_utils import output_dir,write_json,metadata,digest


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--kind',choices=['physical','training','frozen'],required=True)
    parser.add_argument('--input',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    files={'physical':['raw.npz','checks.json','metadata.json'],
           'training':['training.jsonl','evaluation.csv'],
           'frozen':['metadata.json','episodes.jsonl','integrity.json']}[args.kind]
    for name in files:
        if not (args.input/name).is_file():raise FileNotFoundError(args.input/name)
    out=output_dir(args.output)
    for name in files:shutil.copy2(args.input/name,out/name)
    write_json(out/'plot_provenance.json',metadata(dict(kind=args.kind,input=str(args.input),
               source_hashes={n:digest(args.input/n) for n in files})))
    if args.kind=='physical':
        from validate_physics import summarize
    elif args.kind=='training':
        from train_nominal import plot_training as summarize
    else:
        from evaluate_frozen import summarize
    summarize(out)
    print(out)

if __name__=='__main__':main()
