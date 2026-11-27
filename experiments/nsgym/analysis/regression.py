"""Raw → canonical → summary → displayed data checks, with discovered historical references."""
from collections import Counter,defaultdict
import json
import math
from pathlib import Path
import numpy as np
from .metrics import CONDITION
from .statistics import summary
from .schema import write_csv


def validate(catalog,rec,out,summaries,comparisons,unified=None):
    checks=list(rec.checks)
    def check(source,key,expected,actual,rtol=1e-9,atol=1e-12):
        try:
            a,b=float(expected),float(actual)
            status='EXACT' if a==b else 'TOLERANCE' if math.isclose(a,b,rel_tol=rtol,abs_tol=atol) else 'FAIL'
        except (TypeError,ValueError):status='EXACT' if expected==actual else 'FAIL'
        checks.append(dict(source=str(source),metric=key,expected=expected,actual=actual,status=status,rtol=rtol,atol=atol))
    # Independent re-reduction from saved canonical rows at the declared sampling unit.
    group=defaultdict(list)
    for r in rec.rows:group[tuple(r[k] for k in CONDITION)+(r['method'],r['adaptation_budget'])].append(r)
    for s in summaries:
        key=tuple(s[k] for k in CONDITION)+(s['method'],s['adaptation_budget']);rr=group[key];units=defaultdict(list)
        for r in rr:units[r['adaptation_seed'] or r['evaluation_seed']].append(float(r['value']))
        actual=summary([sum(v)/len(v) for _,v in sorted(units.items())],'q' if s['kind'].startswith('q_') else s['task'])
        for field in ('n','mean','median','std','ci_low','ci_high','outliers'):check('canonical_records.csv',str(key)+':'+field,s[field],actual[field])
    # Historical result tables are optional references, never ingestion inputs.
    index=defaultdict(list)
    for r in rec.rows:
        if r['kind'] in ('shifted','nominal','fixed_bank'):
            key=(r['shift_parameter'],r['severity_multiplier'],r['evaluation_seed'],r['method'],r['kind'],r['metric'])
            index[key].append(r)
    aliases={'frozen_return':'frozen','ttt_return':'dynamics_only','dq_return':'dynamics_q','joint_return':'joint'}
    refs=0
    for p in catalog.files:
        if p.suffix!='.csv':continue
        with p.open() as f:header=f.readline().strip().split(',')
        if not {'frozen_return','ttt_return','evaluation_seed','severity'}<=set(header):continue
        # Paired source measurements must actually be available in this selection.
        for n,row in enumerate(catalog.table(p),2):
            parameter=row.get('parameter') or catalog.registry['reacher']['parameters']
            if not isinstance(parameter,str):continue
            from .sources import normalized
            kind=row.get('kind','shifted')
            for col,method in aliases.items():
                if row.get(col) in ('',None):continue
                key=(parameter,normalized(row['severity']),normalized(row['evaluation_seed']),method,kind,'episode_return')
                matches=index.get(key,[])
                seed=normalized(row.get('adaptation_seed',''))
                matches=[r for r in matches if method=='frozen' or r['adaptation_seed']==seed]
                if len(matches)==1:
                    check(catalog.rel(p),f'row{n}:{col}',row[col],matches[0]['value']);refs+=1
        catalog.digest(p)
    # Discover condition-level reference schemas independently of historical run IDs.
    seen_refs=set()
    contrast_methods={'ttt_minus_frozen':'dynamics_only','dq_minus_frozen':'dynamics_q','joint_minus_frozen':'joint','delta_D':'dynamics_only','delta_J':'joint'}
    q_kinds={'immediate_summary.csv':'q_immediate','divergence_summary.csv':'q_divergence','terminal_q_metrics_summary.csv':'q_terminal','planning_score_components_summary.csv':'q_planning','candidate_ranking_comparison_summary.csv':'q_ranking','terminal_counterfactuals_summary.csv':'q_counterfactual'}
    from .discovery import METHODS
    from .sources import normalized
    for p in catalog.files:
        if p.suffix!='.csv':continue
        with p.open() as f:header=f.readline().strip().split(',')
        if not {'mean','median','ci_low','ci_high','metric','severity','parameter'}<=set(header):continue
        digest=catalog.digest(p)
        if digest in seen_refs:continue
        seen_refs.add(digest)
        for i,row in enumerate(catalog.table(p),2):
            if row.get('kind','shifted')!='shifted' and p.name not in q_kinds:continue
            metric=row['metric'];method=contrast_methods.get(metric)
            if method:
                candidates=[r for r in comparisons if r['method']==method and r['shift_parameter']==row['parameter'] and r['severity_multiplier']==normalized(row['severity']) and r['schedule']=='fixed']
            elif p.name in q_kinds:
                method=METHODS.get(row.get('method'),row.get('method',row.get('left','')+'→'+row.get('right','')))
                candidates=[r for r in summaries if r['method']==method and r['kind']==q_kinds[p.name] and r['metric']==metric and r['shift_parameter']==row['parameter'] and r['severity_multiplier']==normalized(row['severity'])]
            else:continue
            if len(candidates)==1:
                for field in ('mean','median','std','ci_low','ci_high'):
                    if row.get(field) not in ('',None):check(catalog.rel(p),f'row{i}:{field}',row[field],candidates[0][field]);refs+=1
    # Verify exactly what plotting consumed, including every plotted condition and value.
    data_path=out/'figure_data.json'
    if data_path.exists():
        data=json.loads(data_path.read_text())
        for figure in data:
            check(figure['figure'],'file_exists',True,(out/figure['figure']).is_file())
            for r in figure['rows']:
                if figure['value_column']=='mean_return_improvement':continue
                candidates=comparisons if 'baseline' in r else summaries
                keys=('task','cohort_id','shift_parameter','severity_multiplier','schedule','method','adaptation_budget','kind')
                found=[s for s in candidates if all(str(s.get(k,''))==str(r.get(k,'')) for k in keys) and s.get('metric')==r.get('metric')]
                check(figure['figure'],'matching_summary',1,len(found))
                if len(found)==1:check(figure['figure'],'displayed_mean',found[0]['mean'],r['mean'])
    if unified:
        marker=Path(unified)/'SOURCE_PROVENANCE.json'
        if not marker.is_file():check(str(marker),'provenance_present',True,False)
        else:
            for relative,digest in json.loads(marker.read_text())['files'].items():
                p=catalog.root/relative
                check(relative,'source_hash_unchanged',digest,catalog.digest(p) if p.is_file() else 'MISSING')
    counts=Counter(r['status'] for r in checks)
    evidence=Counter(r['evidence'] for r in rec.rows)
    result=dict(status='FAIL' if counts['FAIL'] else 'PASS_WITH_DOCUMENTED_DIFFERENCES' if counts['NUMERICAL_DIFFERENCE'] else 'PASS',counts=dict(counts),
        raw_reproduced_metrics=evidence['RAW_REPRODUCED'],derived_only_metrics=evidence['DERIVED_ONLY'],
        historical_reference_checks=refs,missing_sources=sum(x['status']=='MISSING' for x in catalog.issues),
        invalid_sources=sum(x['status']=='INVALID' for x in catalog.issues))
    fields=list(dict.fromkeys(k for r in checks for k in r)) or ['status'];write_csv(out/'RAW_TO_REPORT_CHECKS.csv',checks,fields)
    (out/'RAW_TO_REPORT_REGRESSION.md').write_text('# Raw-to-report regression\n\n'+json.dumps(result,indent=2)+'\n\n'
        'Counts are scalar checks and canonical metric records, not independent experimental replicates. '
        'All canonical summaries are re-reduced by seed. Figure inputs are checked against summaries; historical paired tables are optional numerical references. '
        'Float64 CSV arithmetic uses rtol=1e-9, atol=1e-12. Recorded GPU float32 Q quantities redecoded on CPU use explicitly recorded tolerances (rtol=3e-5, atol=2e-4); these are not model inference. '
        'Cancellation-sensitive Q differences outside direct tolerance but within 32 float32 epsilon × underlying Q scale are classified NUMERICAL_DIFFERENCE, not exact/tolerance matches. Original recorded float32 measurements remain the statistical inputs after CPU validation; max differences and bounds are recorded. '
        'Missing stored Bellman targets and unsupported original arrays remain derived-only.\n')
    return result
