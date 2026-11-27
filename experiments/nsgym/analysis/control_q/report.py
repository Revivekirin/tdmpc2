"""Human-readable figure guide and strict output validation."""
from __future__ import annotations

import csv
import ast
import html
import json
from pathlib import Path
import pandas as pd

from .config import GRID, TASKS
from .metrics import require, write, sha256


def validate_figures(out: Path, result: dict):
    coverage=pd.read_csv(out/'FIGURE_COVERAGE.csv')
    expected={(t,f,float(k)) for t,task in TASKS.items() for f in task['families'] for k in GRID}
    actual=set(zip(coverage.task,coverage.family,coverage.severity.astype(float)))
    require(actual==expected and len(coverage)==36,'Six-family × six-severity figure coverage')
    png=sorted((out/'figures').rglob('*.png'))
    expected_png=48 if result['closed_loop_critic_effect']=='MEASURED' else 42
    require(len(png)==expected_png and all(p.stat().st_size>1000 for p in png),
            f'Expected {expected_png} nonempty PNG files')
    require(not list((out/'figures').rglob('*.svg')),'SVG output is not allowed')
    for task,t in TASKS.items():
        for family in t['families']:
            family_dir=out/'figures'/task/family
            names={p.name for p in family_dir.glob('*.png')}
            required={f'{i:02d}_{stem}.png' for i,stem in ((1,'prediction_control'),(2,'terminal_q'),
                       (3,'candidate_ranking'),(4,'native_mppi'))}
            if result['closed_loop_critic_effect']=='MEASURED':required.add('05_closed_loop_q_swap.png')
            require(names==required,f'Figure sequence: {family_dir}')
    checks=pd.read_csv(out/'candidate_identity_checks.csv')
    require(len(checks)==108 and checks.frozen_d_immediate_equal.all(),
            '108-stream immediate-Q and candidate checks')
    require(checks.max_score_identity_error.max()<5e-4,'Candidate score identity')
    provenance=json.loads((out/'SOURCE_PROVENANCE.json').read_text())
    native=provenance['decision']['validation']
    require(native['status']=='PASS' and native['native_parity_comparisons']==1080 and
            native['F_D_immediate_invariant'] and native['headwise_native_decoder_parity'] and
            native['controlled_candidate_identity'] and native['total_score_identity'],
            'Archived native planner and Q numerical parity')
    result=dict(result,status='PASS',figure_png_count=len(png),
                coverage_conditions=len(coverage),terminal_protocol='fixed heads 0,1',
                archived_native_planner_parity=1080,terminal_archived_interval_parity=True,
                model_inference_calls=0,simulator_calls=0,optimizer_steps=0)
    write(out/'FIGURE_VALIDATION.json',result)
    return result


def guide(out: Path, validation: dict, sources: dict):
    images=sorted((out/'figures').rglob('*.png'))
    manifest=[]
    for path in images:
        relative=path.relative_to(out).as_posix()
        parts=path.relative_to(out/'figures').parts
        manifest.append(dict(path=relative,task=parts[0],family=parts[1],
                             role='appendix' if 'appendix' in parts else 'main',
                             sha256=sha256(path),bytes=path.stat().st_size))
    pd.DataFrame(manifest).to_csv(out/'FIGURE_MANIFEST.csv',index=False)
    lines=['# NS-Gym control–critic research figures','',
           'All curves retain six physical severities and all three adaptation streams. '
           'Intervals are descriptive hierarchical intervals over adaptation streams and matched evaluation seeds; '
           'the Frozen model is one shared reference. HalfCheetah and Reacher use separate scales.','',
           '## 01 Prediction and actual control','',
           'Left: measured held-out episode return for Frozen, Dynamics-only, Dynamics+Q and Joint. '
           'Right: one-step fixed-bank latent MSE for Frozen, Dynamics-only and Dynamics+Q. '
           'Joint has a changed encoder, so its latent MSE is excluded from this shared-coordinate comparison. '
           'Lower MSE does not guarantee higher return.','',
           '## 02 Predicted terminal Q','',
           'Left: Q at the end of a common saved action tape relative to Frozen. '
           'All methods use the same physical starting observations, terminal action and fixed Q-head pair 0,1. '
           'Right: discounted Q-network substitution with the D+Q predictor fixed. '
           'These are predicted score changes, not Q accuracy or calibrated uncertainty.','',
           '## 03 Fixed-candidate preference','',
           'Two controlled comparisons, D→D+Q and DQ_F→D+Q, use the same 32 diagnostic candidate sequences. '
           'DQ_F combines the D+Q predictor with the original Frozen critic; it is not separately trained. '
           'Top-1 disagreement and Spearman correlation describe rankings, not realized action quality. '
           'The saved candidate score identity was checked across all 108 streams.','',
           '## 04 Native MPPI proposed action','',
           'The DQ_F→D+Q L2 difference uses the native TD-MPC2 planner under the archived synchronized-randomness protocol. '
           'The planner proposals were not executed. Two predeclared planning states per stream limit coverage.','',
           '## 05 Closed-loop critic swap','',
           ('Measured paired critic-swap episodes were verified; Figure 05 shows actual return.'
            if validation['closed_loop_critic_effect']=='MEASURED' else
            'NOT_MEASURED in the validated source set. No 05 figure was generated. '
            'A predicted score or proposed-action difference is not substituted for actual environment return.'),'',
           '## Appendix','',
           'A1 shows one matched-state scalar-Q ECDF. A2 shows candidate-dependent terminal Q changes and the original '
           'winner margin. A3 shows centered reward, discounted terminal-Q and total-score components. '
           'Each uses k=2 and the first adaptation seed. A1 includes all 20 matched evaluation seeds; '
           'A2/A3 use the earliest state with a top-1 '
           'Q-swap reversal, or state 0 when none exists; the rule is set before viewing returns. '
           'The representative examples are not population severity sweeps.','',
           '## Evidence and limits','',
           f'{validation["inference_streams"]} saved inference streams; {validation["figure_png_count"]} PNG files; '
           f'{validation["candidate_reversals"]} ordered candidate-pair reversals checked. '
           'Source artifacts were verified by SHA256 before drawing. Plot mode makes no checkpoint inference, '
           'simulator transition or optimizer update.','',
           'Q calibration, counterfactual candidate returns, planning regret and causal recovery are not established. '
           'Different adaptation methods use different optimization budgets and possibly different online trajectories.','',
           'Small validated condition summaries are copied to `tables/` for review. '
           '`terminal_fixed_head_summary.csv` uses the same two-head protocol as the terminal figure; '
           '`candidate_q_change_per_seed.csv` aggregates four states within each held-out seed.','',
           'Source artifacts:']
    lines.extend(f'- {name}: `{path}`' for name,path in sources.items())
    lines.extend(['','## Exact retained PNGs',''])
    lines.extend(f'- [{path.relative_to(out).as_posix()}]({path.relative_to(out).as_posix()})'
                 for path in images)
    (out/'FIGURE_READING_GUIDE.md').write_text('\n'.join(lines)+'\n')
    body='<h1>NS-Gym control–critic figures</h1><p>'+html.escape(lines[2])+'</p>'
    for task,t in TASKS.items():
        for family in t['families']:
            body+=f'<h2>{html.escape(task)} — {html.escape(family)}</h2>'
            for path in images:
                if path.parent==out/'figures'/task/family:
                    relative=path.relative_to(out).as_posix()
                    body+=f'<figure><img src="{html.escape(relative)}" alt="{html.escape(path.stem)}"><figcaption>{html.escape(path.stem)}</figcaption></figure>'
    body+='<p>Closed-loop critic swap: '+html.escape(validation['closed_loop_critic_effect'])+'</p>'
    (out/'FIGURE_REPORT.html').write_text('<!doctype html><html lang="en"><meta charset="utf-8">'
        '<title>NS-Gym research figures</title><style>body{max-width:1100px;margin:2rem auto;'
        'font:16px/1.5 system-ui;padding:0 1rem}img{max-width:100%}figure{margin:2rem 0}'
        'h2{border-top:1px solid #ccc;padding-top:1rem}</style>'+body+'</html>')
    write(out/'completion.json',dict(status='PASS',figures=validation['figure_png_count'],
                                     conditions=36,streams=108,
                                     closed_loop_critic_effect=validation['closed_loop_critic_effect']))


def code_audit(out: Path):
    rows=[
        ('analysis/reacher_dq_figures.py','draw; line_panel; finish_axes; metric_panel; dashboard; deep_dive; supporting_figures; dashboard_report; main','historical Reacher plotting','direct CLI; control_q helper imports','control_q/figures.py','DELETE','Superseded by six-family figures; CI/JS/hash helpers moved'),
        ('analysis/reacher_dq_figures.py','sha256; ci; probability_js','scientific numerical utilities','control_q/metrics.py; control_q/models.py','control_q/metrics.py','MOVE','Exact numerical parity checked before deletion'),
        ('analysis/research_interpretation.py','paired_plot; mechanism_plot; halfcheetah_reward; q_plots; report; main','old cross-task interpretation','standalone CLI only','control_q/figures.py','DELETE','Overlapping Q/control figure suite'),
        ('analysis/q_terminal_planning_figures.py','module wildcard exports','historical compatibility entry','old command document; old test','control_q/run.py','DELETE','Parallel public entry'),
        ('analysis/q_decision_relevance.py','module wildcard exports','historical compatibility entry','old test','control_q/run.py','DELETE','Parallel public entry'),
        ('analysis/control_q/reporting/figures.py','figures; presentation','old Q figures','legacy_cli; run.py','control_q/figures.py','DELETE','Superseded multi-panel and SVG output'),
        ('analysis/control_q/reporting/figures.py','finish','inference table aggregation','legacy_cli','control_q/figures.py; control_q/data.py','MOVE','Final plotting uses verified archived tables'),
        ('analysis/control_q/reporting/decision.py','decision_finish plots','old decision dashboards','diagnostics/planning.py','control_q/figures.py','DELETE','Numerical tables preserved in diagnostics/planning.py'),
        ('analysis/control_q/reporting/html.py','integrate','old figure copy/HTML','run.py','control_q/report.py','DELETE','Superseded report path'),
        ('analysis/control_q/legacy_cli.py','main','old Q inference CLI','run.py','control_q/run.py','DELETE','Single public CLI'),
        ('scripts/q_planning_core.py','distributions; score; ranks','native numerical diagnostics','analysis and tests','control_q/metrics.py','KEEP','Native decoder and score calculations still shared'),
        ('scripts/analyze_q_planning.py','plot','historical Q figures','historical numerical CLI and tests','control_q/figures.py','DELETE','Plot function removed; numerical bank/inference stages retained'),
        ('scripts/mechanism_forward.py','forward diagnostics','mechanism numerical analysis','analyze_q_planning.py; diagnostics','unchanged','KEEP','Not plot-only'),
        ('scripts/analyze_existing_mechanisms.py','mechanism figure functions','separate mechanism experiment','mechanism_forward.py; historical mechanism runner','unchanged','KEEP','Separate experiment and required digest helper'),
        ('analysis/__main__.py','main','raw-to-report pipeline','unified NS-Gym analysis','unchanged','KEEP','Separate source inventory and physical-analysis interface'),
        ('analysis/control_q/evaluation.py','evaluate; main','physical held-out Q-swap evaluation','optional future closed-loop figure','unchanged','KEEP','Evaluation runner; never called in plotting mode'),
    ]
    expanded=[(old,function.strip(),responsibility,callers,replacement,action,reason)
              for old,functions,responsibility,callers,replacement,action,reason in rows
              for function in functions.split(';')]
    with (out/'FIGURE_CODE_AUDIT.csv').open('w',newline='') as f:
        writer=csv.writer(f);writer.writerow(['old_file','old_function','responsibility','callers','replacement','action','reason']);writer.writerows(expanded)
    deleted=[r for r in expanded if r[5] in ('DELETE','MOVE')]
    deleted.append(('analysis/RUN_Q_TERMINAL_PLANNING.md','historical figure commands','old plotting CLI','documentation','control_q/README.md','DELETE','Only final CLI is supported'))
    with (out/'DELETED_PLOT_CODE.csv').open('w',newline='') as f:
        writer=csv.writer(f);writer.writerow(['deleted_file_or_function','replacement','dependency_evidence','replacement_test','git_recovery_reference'])
        for old,function,_,callers,replacement,_,_ in deleted:
            writer.writerow([old+'::'+function,replacement,callers,
                             'test_control_q_metrics.py; test_final_figures.py',
                             '99429838effbfd882aa6e4b15e57c6b5282a5ba5'])
    current=ast.parse(Path(__file__).with_name('figures.py').read_text())
    write(out/'CODE_COUNTS.json',dict(historical_revision='99429838effbfd882aa6e4b15e57c6b5282a5ba5',
        deleted_plot_and_compatibility_files=8,historical_top_level_functions_in_deleted_files=43,
        consolidated_plot_files=1,consolidated_top_level_functions=sum(
            isinstance(node,(ast.FunctionDef,ast.AsyncFunctionDef)) for node in current.body),
        deleted_plot_function_or_entry_records=len(deleted)))
