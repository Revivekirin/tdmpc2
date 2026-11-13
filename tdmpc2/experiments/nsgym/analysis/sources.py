"""Adapters for validated historical CSVs. No simulator or model imports."""

from __future__ import annotations

from pathlib import Path

from .schema import canonical, read_json, rows, sha256, value

SOURCES = read_json(Path(__file__).with_name("config.json"))["sources"]

METHODS = {"frozen_return": "frozen", "ttt_return": "dynamics_only",
           "dq_return": "dynamics_q", "joint_return": "joint"}
REACHER_METHODS = {"Frozen": "frozen", "Dynamics-only": "dynamics_only",
                   "Joint RL": "joint", "Dynamics+Q": "dynamics_q"}


def available(results: Path, task: str) -> dict:
    cfg = SOURCES[task]
    status = read_json(results / cfg["completion"])
    if cfg.get("adapter", task) == "halfcheetah":
        ok = status.get("status") == "PASS" and status.get("paired_rows") == 720
    else:
        ok = status.get("analysis_execution") == "PASS" and status.get("original_stage_audit") == "PASS_SAVED_RECORD_AUDIT"
    if not ok:
        raise ValueError(f"{task} comparison has no verified completed source: {status}")
    return status


def identity(task: str, cfg: dict, row: dict, method: str, source_run: str) -> dict:
    return dict(task=task, environment_id=cfg["environment_id"], method=method,
                checkpoint_id=cfg.get("checkpoint_id", "checkpoint-000670000"),
                training_seed=cfg.get("training_seed", ""), adaptation_seed=value(row, "adaptation_seed"),
                evaluation_seed=value(row, "evaluation_seed", "seed"),
                shift_family=value(row, "family") or ("damping" if "damping" in row.get("parameter", "") else "mass"),
                shift_parameter=value(row, "parameter"),
                severity_multiplier=value(row, "severity"),
                applied_parameter_value=value(row, "actual_parameter", "actual_parameter_min"),
                schedule=value(row, "schedule") or "fixed", shift_onset="",
                adaptation_budget=value(row, "budget") or ("3000" if method in ("dynamics_only", "dynamics_q", "joint") else "0"),
                optimizer_steps=value(row, "gradient_updates") or {
                    "frozen": "0", "dynamics_only": "118" if task == "halfcheetah" else "",
                    "dynamics_q": "236", "joint": "4002" if task == "halfcheetah" else "",
                }.get(method, ""), model_hash=value(row, "snapshot_sha256") or
                (cfg.get("checkpoint_hash", "") if method in ("frozen", "physical_probe") else ""),
                source_run_id=source_run, completion_status="PASS")


def ingest(results: Path, task: str, methods: set[str]):
    cfg = dict(SOURCES[task])
    if cfg.get("adapter", task) == "reacher":
        record = read_json(results / cfg["checkpoint_record"]).get("record", {})
        cfg["checkpoint_hash"] = record.get("model_sha256", "")
    available(results, task)
    evaluation, prediction, physical, q = [], [], [], []
    parameter_info = {}
    path = results / cfg["comparison"]
    file_hash = sha256(path)
    if cfg.get("adapter", task) == "halfcheetah":
        for number, row in enumerate(rows(path), 2):
            parameter_info[(row["parameter"], row["severity"])] = (
                row.get("nominal_parameter", ""), "")
            for column, method in METHODS.items():
                if method not in methods: continue
                key = identity(task, cfg, row, method, "dynamics-q-750k-v1/comparison")
                evaluation.append(canonical(key, kind="shifted", metric="episode_return", value_=row[column],
                                            source=path, source_root=results, row_number=number, source_hash=file_hash))
                mse = value(row, {"frozen": "frozen_latent_mse", "dynamics_only": "ttt_latent_mse",
                                  "dynamics_q": "dq_latent_mse"}.get(method, "unused"))
                if mse:
                    prediction.append(canonical(key, kind="fixed_bank", metric="latent_mse", value_=mse,
                                                source=path, source_root=results, row_number=number, source_hash=file_hash))
    else:
        for number, row in enumerate(rows(path), 2):
            if row["kind"] == "shifted":
                parameter_info[(row["family"], row["severity"])] = (
                    row.get("nominal_parameter", ""), row.get("actual_parameter", ""), row["parameter"])
            for column, method in {k: v for k, v in METHODS.items() if v != "dynamics_q"}.items():
                if method not in methods: continue
                key = identity(task, cfg, row, method, "reacher-analysis-v3")
                evaluation.append(canonical(key, kind=row["kind"], metric="episode_return", value_=row[column],
                                            source=path, source_root=results, row_number=number, source_hash=file_hash))
                if method in ("frozen", "dynamics_only") and row["kind"] == "shifted":
                    mse = value(row, "frozen_mse" if method == "frozen" else "ttt_mse")
                    if mse:
                        prediction.append(canonical(key, kind="fixed_bank", metric="latent_mse", value_=mse,
                                                    source=path, source_root=results, row_number=number, source_hash=file_hash))
        # Task-specific measurements are read from original-record canonical rows.
        # The matched-comparison table remains the sole source for paired returns.
        ep = results / cfg["episodes"]
        eh = sha256(ep)
        episode_metrics = ("distance_initial", "distance_mean", "distance_final", "distance_min",
                           "reward_dist", "reward_ctrl", "action_abs", "action_saturation",
                           "force_rms", "force_saturation", "episode_length")
        for number, row in enumerate(rows(ep), 2):
            method = REACHER_METHODS.get(row["method"])
            if method not in methods or row["completion_status"] != "PASS_original_record":
                continue
            if method == "frozen" and row["budget"] != "0": continue
            if method != "frozen" and row["budget"] != "10000": continue
            key = identity(task, cfg, row, method, "reacher-analysis-v3/original-control")
            key["completion_status"] = row["completion_status"]
            for metric in episode_metrics:
                metric_value = row.get(metric, "")
                if metric_value:
                    evaluation.append(canonical(key, kind=row["kind"], metric=metric, value_=metric_value,
                                                source=ep, source_root=results, row_number=number, source_hash=eh))
    if cfg.get("adapter", task) == "halfcheetah":
        ep = results / "dynamics-q-750k-v1/comparison/per_seed_heldout_control.csv"
        eh = sha256(ep)
        episode_metrics = ("forward_velocity", "control_cost", "action_saturation", "action_abs",
                           "actuator_force_rms", "actuator_force_saturation", "episode_length")
        for number, row in enumerate(rows(ep), 2):
            if "dynamics_q" not in methods: break
            key = identity(task, cfg, row, "dynamics_q", "dynamics-q-750k-v1/comparison")
            for metric in episode_metrics:
                metric_value = row.get(metric, "")
                if metric_value:
                    evaluation.append(canonical(key, kind="shifted", metric=metric, value_=metric_value,
                                                source=ep, source_root=results, row_number=number, source_hash=eh))
    pp = results / cfg["physical"]
    if pp.is_file():
        ph = sha256(pp)
        for number, row in enumerate(rows(pp), 2):
            key = identity(task, cfg, row, "physical_probe", pp.parent.name)
            if cfg.get("adapter", task) == "halfcheetah":
                metric, metric_value, n = "root_velocity_l2", row.get("root_velocity_l2_mean", ""), row.get("seeds", "")
                nominal = parameter_info.get((row["parameter"], row["severity"]), ("", ""))[0]
                if nominal:
                    key["applied_parameter_value"] = str(float(nominal) * float(row["severity"]))
            else:
                metric, metric_value, n = row["metric"], row["mean"], row.get("n", "")
                if metric == "shifted_latent_mse":
                    # This is a prediction diagnostic, not a physical-state response.
                    continue
                info = parameter_info.get((row["family"], row["severity"]))
                if info:
                    key["shift_parameter"] = info[2]
                    key["applied_parameter_value"] = info[1]
            physical.append(canonical(key, kind="same_state_probe", metric=metric, value_=metric_value,
                                      sample_count=n, source=pp, source_root=results, row_number=number, source_hash=ph))
    if cfg.get("adapter", task) == "halfcheetah" and read_json(results / cfg["q_completion"]).get("status") == "PASS":
        for label, table in (("immediate", cfg["q"]), ("divergence", cfg["q_divergence"]),
                             ("planning", cfg["planning"]), ("ranking", cfg["ranking"])):
            qp = results / table
            if not qp.is_file(): continue
            qh = sha256(qp)
            for number, row in enumerate(rows(qp), 2):
                method = {"F": "frozen", "D": "dynamics_only", "DQ": "dynamics_q", "J": "joint"}.get(row.get("method", ""), row.get("method", "comparison"))
                aliases = {"F":"frozen", "D":"dynamics_only", "DQ":"dynamics_q", "J":"joint"}
                required_methods = {'DQ-full': {'dynamics_q'}, 'DQ-dyn-only': {'dynamics_q'},
                                    'F+DQ-Q': {'frozen','dynamics_q'}, 'TTT+DQ-Q': {'dynamics_only','dynamics_q'}}
                if method != "comparison" and not required_methods.get(method, {method}) <= methods: continue
                if method == "comparison":
                    needed=set()
                    for side in ('left','right'):
                        needed.update(required_methods.get(row[side], {aliases.get(row[side],row[side])}))
                    if not needed <= methods: continue
                key = identity(task, cfg, row, method, "q-planning-mechanism-750k-v1")
                nominal = parameter_info.get((row["parameter"], row["severity"]), ("", ""))[0]
                if nominal:
                    key["applied_parameter_value"] = str(float(nominal) * float(row["severity"]))
                if label == "immediate":
                    columns = ("q", "entropy", "ensemble_std", "boundary_mass", "interhead_js", "bellman_residual", "bellman_soft_ce")
                elif label == "divergence":
                    columns = ("js", "q_difference")
                    key["method"] = f"{row['left']}→{row['right']}"
                elif label == "planning":
                    columns = ("reward_sum", "terminal_contribution", "total_score", "ess")
                else:
                    columns = ("top_k_overlap", "argmax_disagreement")
                    key["method"] = f"{row['left']}→{row['right']}"
                for metric in columns:
                    q.append(canonical(key, kind=label, metric=metric, value_=row.get(metric, ""),
                                       sample_count=row.get("evaluation_seeds", ""), source=qp,
                                       source_root=results, row_number=number, source_hash=qh))
    return evaluation, prediction, physical, q


def ingest_completed_reacher_dq(results: Path, aggregate_csv: Path):
    """Read the final four-way table only after the separate completion gate passes."""
    cfg = dict(SOURCES["reacher"])
    cfg["checkpoint_hash"] = read_json(results / cfg["checkpoint_record"]).get("record", {}).get("model_sha256", "")
    file_hash = sha256(aggregate_csv)
    records, predictions = [], []
    for number, row in enumerate(rows(aggregate_csv), 2):
        key = identity("reacher", cfg, row, "dynamics_q", aggregate_csv.parent.name)
        for metric in ("episode_return", "distance_mean", "distance_final", "reward_dist", "reward_ctrl",
                       "action_abs", "action_saturation"):
            raw = row.get("Dynamics+Q_" + metric, "")
            if raw:
                records.append(canonical(key, kind=row["kind"], metric=metric, value_=raw,
                                         source=aggregate_csv, source_root=results,
                                         row_number=number, source_hash=file_hash))
        if row["kind"] == "shifted" and row.get("dq_mse"):
            predictions.append(canonical(key, kind="fixed_bank", metric="latent_mse", value_=row["dq_mse"],
                                         source=aggregate_csv, source_root=results,
                                         row_number=number, source_hash=file_hash))
    return records, predictions


def completed_reacher_dq(results, paths):
    """Fail closed: PASS markers alone cannot turn a header-only export into evidence."""
    try:
        required=('training_completion','prediction_completion','control_completion','aggregate_completion','aggregate_per_seed_csv')
        if not all(paths.get(k) and (results/paths[k]).is_file() for k in required):return False
        if not all(read_json(results/paths[k]).get('status')=='PASS' for k in required[:-1]):return False
        exported=list(rows(results/paths['aggregate_per_seed_csv']))
        shifted=[r for r in exported if r['kind']=='shifted']
        expected={(f,k,a,e) for f in ('D0','D1','M0','M1') for k in (.5,.75,1.,1.25,1.5,2.)
                  for a in (53000001,53000002,53000003) for e in range(54000001,54000021)}
        actual={(r['family'],float(r['severity']),int(r['adaptation_seed']),int(r['evaluation_seed'])) for r in shifted}
        if actual!=expected or len(shifted)!=len(expected):return False
        if any(not r.get('dq_mse') or not r.get('Dynamics+Q_episode_return') for r in shifted):return False
        train=results/paths['training_completion']
        training={p.parent.name:read_json(p) for p in train.parent.glob('*/completion.json')}
        if len(training)!=72 or any(d.get('status')!='PASS' or not d.get('snapshot_sha256') for d in training.values()):return False
        for mode,field in [('prediction','prediction_completion'),('control','control_completion')]:
            folder=(results/paths[field]).parent
            for name,d in training.items():
                done=read_json(folder/name/'completion.json')
                csv_path=folder/name/(mode+'.csv')
                if done.get('status')!='PASS' or done.get('snapshot_sha256')!=d['snapshot_sha256']:return False
                if not csv_path.is_file() or sha256(csv_path)!=done.get('csv_sha256'):return False
        return True
    except (KeyError,ValueError,TypeError,OSError):
        return False
