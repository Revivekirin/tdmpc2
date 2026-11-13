"""Read-only migration inventory and conservative, exact-path cleanup preview."""
import ast
import hashlib
from collections import Counter
import json
from pathlib import Path
import shlex
from .schema import sha256, write_csv

PACKAGE=Path(__file__).parent
WORKSPACE=PACKAGE.parents[3]


def migration_inventory(output):
    records=json.loads((PACKAGE/'migration_inventory.json').read_text())
    for r in records:
        p=WORKSPACE/r['path']
        r['retired']=not p.exists()
        r['current_sha256']=sha256(p) if p.is_file() else ''
        unchanged=r['current_sha256']==r['sha256']
        if not unchanged and p.is_file() and r.get('function_ast_sha256'):
            nodes=[n for n in ast.walk(ast.parse(p.read_text())) if isinstance(n,ast.FunctionDef) and n.name==r['function']]
            unchanged=any(hashlib.sha256(ast.get_source_segment(p.read_text(),n).strip().encode()).hexdigest()==r['function_ast_sha256'] for n in nodes)
        r['runtime_unchanged']=unchanged if r['classification'] in ('KEEP_RUNTIME','DEFER_ACTIVE_JOB') else '' 
    write_csv(output/'CODE_MIGRATION_MATRIX.csv',records,list(records[0]))
    files={r['path']:r for r in records};retired=[r for r in files.values() if r['retired']]
    counts=Counter(r['classification'] for r in files.values())
    summary=dict(inspected_files=len(files),inspected_functions=sum(r['function']!='<module>' for r in records),
                 retired_files=len(retired),classifications=counts,
                 migrated_functions=sum(r['classification']=='MIGRATE_ANALYSIS' and bool(r['target']) for r in records),
                 protected_runtime_changed=sorted({r['path']+':'+r['function'] for r in records if r['runtime_unchanged'] is False}),
                 note='File movement alone does not reduce path count: retired entry points now share IO/bootstrap. Runtime aggregators are deferred.')
    (output/'MIGRATION_COUNTS.json').write_text(json.dumps(summary,indent=2))
    return summary


def tree_hashes(path):
    return {str(p.relative_to(path)):sha256(p) for p in sorted(path.rglob('*')) if p.is_file() and not p.is_symlink()}


def cleanup_preview(results,output):
    records=[];commands=[]
    for p in sorted(results.iterdir()):
        if not p.is_dir():continue
        files=[x for x in p.rglob('*') if x.is_file() and not x.is_symlink()]
        category='UNKNOWN';reason='No safe deletion certification';backup='';safe=False
        if 'reacher-dynamics-q' in p.name:category='IN_PROGRESS';reason='Remote experiment unfinished; retain all local preparation and provenance'
        elif any(x.suffix in ('.pt','.pth','.ckpt') for x in files):category='KEEP_CHECKPOINT';reason='Contains model or adaptation snapshots'
        elif any(x.suffix in ('.npz','.jsonl') for x in files):category='KEEP_SOURCE';reason='Contains raw transition bank or training records'
        elif any(x.name in ('episodes.csv','control.csv') for x in files):category='KEEP_EVALUATION';reason='Contains evaluation records'
        elif p.name in ('reacher-analysis-v3','dynamics-only-750k-v1','dynamics-q-750k-v1','q-planning-mechanism-750k-v1 2'):category='KEEP_SOURCE';reason='Registered analysis input; original remote data not assumed backed up'
        elif p.name=='unified-analysis':category='REGENERABLE_DERIVED';reason='Unified outputs; inspect manifests before approving any deletion; no independent backup certified'
        # Only a complete byte-identical sibling is a certified backup.
        if p.name.endswith(' 2') and (results/p.name[:-2]).is_dir():
            other=results/p.name[:-2]
            if tree_hashes(p)==tree_hashes(other):
                category='DUPLICATE';reason='Every relative file path and SHA256 agrees with retained sibling'
                backup=str(other.resolve());safe=True
                commands.append('rm -rf -- '+shlex.quote(str(p.resolve())))
        records.append(dict(path=str(p.resolve()),classification=category,files=len(files),bytes=sum(x.stat().st_size for x in files),
                            retained_backup=backup,verified_duplicate=safe,deletion_executed=False,reason=reason))
    write_csv(output/'LOCAL_CLEANUP_MANIFEST.csv',records,list(records[0]) if records else ['path'])
    text='# Local cleanup preview\n\nNo files were deleted. UNKNOWN, active data, checkpoints and source results remain protected.\n\n'
    text+='Only complete byte-identical directory duplicates have deletion commands below. Recheck hashes immediately before deletion; user approval is required.\n\n'
    text+=('```bash\n'+'\n'.join(commands)+'\n```\n') if commands else 'No directory is certified for deletion.\n'
    (output/'LOCAL_OUTPUT_CLEANUP.md').write_text(text)
