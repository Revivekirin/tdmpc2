"""Content-based discovery and cache validation with explicit relocation."""
from pathlib import Path
import csv
from .metrics import read, write, require, sha256


def discover(root, name):
    found=[p.parent for p in root.rglob('completion.json') if p.parent.name==name and read(p).get('status')=='PASS']
    require(len(found)==1,f'Expected one completed {name}; found {len(found)}')
    return found[0]


def relocate(path, root):
    p=Path(path)
    if p.is_file():return p
    marker='/results/nsgym/'
    require(marker in str(p),'Unresolvable historical source: '+str(p))
    relative=Path(str(p).split(marker,1)[1])
    direct=root/relative
    moved=root/'unified-analysis'/relative
    return moved if not direct.is_file() and moved.is_file() else direct


def verified_artifact(root, name, source=None):
    source=Path(source).resolve() if source is not None else discover(root,name)
    require(source.is_dir(), f'Missing {name} directory: {source}')
    require(read(source/'completion.json')['status']=='PASS', f'Incomplete {name}: {source}')
    validation=source/'NUMERICAL_VALIDATION.json'
    require(validation.is_file() and read(validation)['status']=='PASS','Missing numerical validation: '+name)
    records=read(source/'SOURCE_INPUTS.json');verified={};historical_code=[];optional_missing=[]
    inventory=source/'tables/SOURCE_INVENTORY.csv'
    optional=set()
    if inventory.is_file():
        with inventory.open() as f:optional={r['path'] for r in csv.DictReader(f) if r.get('required')=='False'}
    for path,info in records.items():
        if Path(path).suffix=='.py':
            historical_code.append(dict(path=path,sha256=info['sha256']))
            continue  # Historic implementation recorded, not confused with refactored code.
        p=relocate(path,root)
        if not p.is_file() and path in optional:
            optional_missing.append(path);continue
        require(p.is_file() and sha256(p)==info['sha256'],'Historical data changed or missing: '+str(p))
        verified[str(p)]=info['sha256']
    protocol=read(source/'protocol.json') if (source/'protocol.json').is_file() else None
    # Preserve original numerical validation and implementation identities alongside current content hashes.
    outputs={str(p.relative_to(source)):sha256(p) for p in source.rglob('*') if p.is_file() and p.suffix in ('.csv','.json','.npz')}
    return source,dict(sources=verified,outputs=outputs,historical_code=historical_code,optional_missing=optional_missing,protocol=protocol,
                       completion=read(source/'completion.json'),validation=read(validation))


def audit(root,out):
    from .models import Sources,inventory
    require(not out.exists(),'Audit output must be new');out.mkdir(parents=True)
    src=Sources();jobs,configs,banks,gaps=inventory(root,src,False)
    errors=list(gaps)
    # Earlier reports are output caches. A fresh run can derive them from the
    # validated checkpoints, snapshots and held-out transition banks above.
    write(out/'manifest.json',dict(sources=src.files,configs=configs,
                                   historical_q_outputs_required=False))
    write(out/'completion.json',dict(status='PARTIAL' if errors else 'PASS',streams=len(jobs),valid_streams=sum(j['valid'] for j in jobs),errors=errors,
                                     historical_q_outputs_required=False))
    return not errors
