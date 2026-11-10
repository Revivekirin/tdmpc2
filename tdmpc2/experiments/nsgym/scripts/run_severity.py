"""Stage-specific, explicit frozen NS-Gym experiment entry point."""
import json
from pathlib import Path
from severity_protocol import parser,resolve


def prerequisites(c):
    if c['stage']=='nonstationary' and not c['smoke'] and any(
        x['schedule'] in ('gradual','reversible','repeated') for x in c['conditions']) and not c['allow_optional_large']:
        raise ValueError('S3-S5 large runs need separate experiment approval and --allow-optional-large; validation uses --smoke')
    if c['stage'] in ('physical','latent','planning','summarize'):
        if not c['input']:raise ValueError('--input is required')
        source=Path(c['input'])
        completion=json.loads((source/'completion.json').read_text())
        if completion['status']!='PASS':raise ValueError('Input stage did not pass')
    if c['stage'] in ('frozen','nonstationary','planning') and not c['smoke']:
        if not c['gate']:raise ValueError('--gate prerequisite result directory is required')
        gate=Path(c['gate']);m=json.loads((gate/'manifest.json').read_text())
        done=json.loads((gate/'completion.json').read_text())
        required='physical' if c['stage']=='frozen' and c['checkpoint_role']=='primary' else 'frozen'
        if done['status']!='PASS' or m['stage']!=required or m['smoke']:
            raise ValueError(f'Requires completed non-smoke {required} stage')
        if m['protocol_sha256']!=c['protocol_sha256']:
            raise ValueError('Prerequisite protocol differs')
        requested={(x['parameter'],x['severity']) for x in c['conditions']}
        measured={(x['parameter'],x['severity']) for x in m['conditions']}
        if not requested<=measured:raise ValueError('Requested conditions missing from prerequisite stage')
    if c['stage']=='frozen' and any(x['schedule']!='fixed' for x in c['conditions']):
        raise ValueError('Frozen generalization stage uses fixed schedule')


def main():
    p=parser();args=p.parse_args()
    try:c=resolve(args)
    except ValueError as exc:p.error(str(exc))
    print(json.dumps(c,indent=2),flush=True)
    if args.dry_run:return
    prerequisites(c)
    out=Path(c['output']);out.mkdir(parents=True,exist_ok=False)
    # Save the complete engineering grid BEFORE any checkpoint or shifted data is evaluated.
    (out/'manifest.json').write_text(json.dumps(c,indent=2)+'\n')
    try:
        if args.stage=='manifest':
            result=dict(transitions=0,status='PREREGISTERED_NOT_RUN')
        else:
            import severity_runtime as runtime
            runtime.js(out/'metadata.json',runtime.provenance(c))
            functions=dict(environment=runtime.environment_check,inventory=runtime.inventory,
                validate=runtime.validate,bank=lambda c,o:runtime.collect(c,o,bank=True),
                physical=runtime.physical,latent=runtime.latent,planning=runtime.planning,
                frozen=runtime.collect,nonstationary=runtime.collect)
            if args.stage=='summarize':
                from severity_analysis import summarize
                result=summarize(c,out)
            else:result=functions[args.stage](c,out)
        completion=dict(status='PASS',stage=args.stage,optimizer_updates=0,measurements=result,
                        outputs=sorted(str(x.relative_to(out)) for x in out.iterdir()))
        (out/'completion.json').write_text(json.dumps(completion,indent=2)+'\n')
    except Exception as exc:
        (out/'failure.json').write_text(json.dumps(dict(status='FAIL',error=repr(exc)),indent=2)+'\n')
        raise
    print(f'PASS: {out}',flush=True)


if __name__=='__main__':main()
