"""Stage 3B-1 Frozen-only evaluator; never enters the adaptation runner."""
import os
os.environ.setdefault('MUJOCO_GL','egl')
from copy import deepcopy
import importlib.metadata
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import hydra
from omegaconf import OmegaConf
import torch
from common.benchmark_config import cli_aliases,normalize,validate_checkpoint_episodic
from common.parser import parse_cfg
from common.severity_protocol import PROTOCOL,severity,run_path,digest_file,protocol_identity,source_digest
from common.severity_evaluation import evaluate_frozen,serial


@hydra.main(config_path='.',config_name='severity',version_base=None)
def main(cfg):
    from hydra.core.hydra_config import HydraConfig
    cfg=normalize(cfg,HydraConfig.get().overrides.task)
    from common.benchmark_validation import validate
    validate(OmegaConf.to_container(cfg,resolve=True))
    kind=cfg.dynamics_shift.parameter.type
    value=severity(kind,cfg.dynamics_shift.parameter.shifted)
    seeds=list(cfg.stage3a.evaluation_seeds)
    if (cfg.stage3a.mode!='frozen' or cfg.adaptation.mode!='frozen' or cfg.compile
            or cfg.analysis.enabled or cfg.schedule.type!='fixed' or cfg.stage3a.native_budgets
            or cfg.stage3b.resume or cfg.stage3a.fixed_data_collect):
        raise ValueError('Severity protocol requires fixed Frozen evaluation without adaptation/resume/analysis.')
    if (not seeds or len(set(seeds))!=len(seeds) or any(type(s)!=int for s in seeds)
            or seeds != list(range(4000+10000*cfg.stage3a.seed_set,4000+10000*cfg.stage3a.seed_set+len(seeds)))
            or len(seeds)>10000):
        raise ValueError('Use the Stage 3B evaluation seeds 4000 + set-id*10000, contiguous, at most 10000.')
    output=Path(hydra.utils.to_absolute_path(cfg.severity.output))
    expected=run_path(output.parents[6],kind,value,cfg.stage3a.seed_set,cfg.severity.run_id)
    if output!=expected:
        raise ValueError(f'Output path must encode shift/set/method/severity/run-id: {expected}')
    if output.exists(): raise FileExistsError(output)
    checkpoint=Path(hydra.utils.to_absolute_path(cfg.checkpoint))
    if not checkpoint.is_file(): raise FileNotFoundError(checkpoint)
    if not torch.cuda.is_available():
        raise RuntimeError('Real TD-MPC2 MPC requires CUDA; run CPU protocol tests locally.')
    validate_checkpoint_episodic(checkpoint,cfg.episodic,torch)
    cfg.checkpoint=str(checkpoint)
    resolved=OmegaConf.to_container(cfg,resolve=True)
    definition=deepcopy(resolved['dynamics_shift'])
    definition['parameter'].pop('shifted')
    revision=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
    dirty=subprocess.check_output(['git','diff','HEAD'],text=True)
    versions={}
    for package in ('torch','numpy','mani-skill2','sapien','gym'):
        try: versions[package]=importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError: versions[package]='unavailable'
    meta=dict(protocol=PROTOCOL,run_id=cfg.severity.run_id,shift_type=kind,shift_value=value,
              nominal_parameter=cfg.dynamics_shift.parameter.nominal,
              severity_units='native environment steps' if kind=='action_delay' else 'multiplicative factor',
              set_id=cfg.stage3a.seed_set,evaluation_seeds=seeds,task=cfg.task,model_size=cfg.model_size,
              episodic=cfg.episodic,checkpoint_sha256=digest_file(checkpoint),code_revision=revision,
              working_tree_diff_sha256=hashlib.sha256(dirty.encode()).hexdigest(),
              shift_definition_sha256=hashlib.sha256(json.dumps(definition,sort_keys=True).encode()).hexdigest(),
              environment_versions=versions,protocol_controls_sha256=protocol_identity(resolved),
              source_sha256=source_digest(Path(__file__).resolve().parents[1]),
              evaluation_protocol='Stage 3B episode(): eval_mode=True, t0 reset, seed global RNG and env.reset; 2 native repeats; shifted + independent nominal env',
              config=resolved,log_eef=cfg.severity.log_eef)
    cfg=parse_cfg(cfg)
    if cfg.multitask or cfg.obs!='state' or not cfg.mpc:
        raise ValueError('Single-task state-observation MPC required.')
    from envs import make_env
    from envs.wrappers.dynamics_shift.logging import find_shift
    from tdmpc2 import TDMPC2
    from common.seed import set_seed
    from adapt_shift import episode
    set_seed(cfg.seed)
    setup=make_env(cfg)
    try:
        agent=TDMPC2(cfg);agent.load(str(checkpoint))
    finally:
        setup.close()
    output.mkdir(parents=True,exist_ok=False)
    (output/'severity-metadata.json').write_text(json.dumps(meta,indent=2))
    results=evaluate_frozen(agent,cfg,make_env,episode,output,meta,find_shift)
    meta['effective_physical_values']=results['effective_physical_values']
    # Publish measured metadata atomically inside this exclusively allocated run.
    temporary=output/'severity-metadata.tmp'
    temporary.write_text(json.dumps(meta,default=serial,indent=2))
    temporary.replace(output/'severity-metadata.json')
    (output/'severity-effective.json').write_text(json.dumps(meta,default=serial,indent=2))
    (output/'severity-complete.json').write_text(json.dumps(results,default=serial,indent=2))
    print(json.dumps(dict(path=str(output),complete=True),indent=2))


if __name__=='__main__':
    sys.argv[1:]=cli_aliases(sys.argv[1:])
    main()
