"""Read-only artifact audit for the fixed Stage 3B-2 matrix. CPU only."""
from copy import deepcopy
import math
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tdmpc2'))
from common.severity_protocol import value_label,digest_file

MATRIX={'action_delay':[1,2],'action_gain':[.7,.3],
        'object_mass':[1.5,3.0],'contact_friction':[.5,.3]}
BUDGETS=[0,800,1600,3200,6400,12800]
PLANNER=('horizon','iterations','num_samples','num_elites','num_pi_trajs','temperature','min_std','max_std')
ROOT=Path(__file__).resolve().parents[1]


def read(path):return json.loads(Path(path).read_text())


def plain(value):
    from omegaconf import OmegaConf
    return OmegaConf.to_container(value,resolve=True) if OmegaConf.is_config(value) else value


def controls(cfg):
    cfg=deepcopy(cfg)
    for key in ('work_dir','stage3b','checkpoint'):cfg.pop(key,None)
    cfg['stage3a']={k:v for k,v in cfg['stage3a'].items() if k not in ('output','fixed_data_path','fixed_data_collect')}
    return cfg


def expected_config(kind,value,checkpoint):
    from hydra import compose,initialize_config_dir
    from omegaconf import OmegaConf
    from common.benchmark_config import normalize,cli_aliases
    overrides=cli_aliases(['task=pick-cube','model_size=5','episodic=false',
                          f'checkpoint={json.dumps(str(checkpoint))}',f'dynamics_shift={kind}',
                          'schedule=fixed',f'dynamics_shift.parameter.shifted={value}',
                          'stage3a.mode=dynamics_only','stage3a.seed_set=0'])
    with initialize_config_dir(config_dir=str(ROOT/'tdmpc2'),version_base=None):
        cfg=normalize(compose(config_name='stage3b',overrides=overrides),overrides)
    data=OmegaConf.to_container(cfg,resolve=True)
    s=data['stage3a'];s['adaptation_seeds']=list(range(s['adaptation_seed_start'],s['adaptation_seed_start']+s['episodes']))
    if [0]+s['native_budgets']!=BUDGETS:raise ValueError('Stage 3B default budget grid changed.')
    return data


def fixed_identity(cfg,checkpoint_hash):
    return dict(checkpoint_sha256=checkpoint_hash,task=cfg['task'],model_size=cfg['model_size'],
                episodic=cfg['episodic'],obs=cfg['obs'],dynamics_shift=cfg['dynamics_shift'],
                schedule=cfg['schedule'],seeds=cfg['stage3a']['fixed_data_seeds'],
                collector='pretrained Frozen MPC',policy_eval_mode=True,
                planner={k:cfg[k] for k in PLANNER})


def load_pt(path):
    import torch
    return torch.load(path,map_location='cpu',weights_only=False)


def check_dataset(path,cfg,checkpoint_hash,expected_hash=None):
    path=Path(path)
    checksum=digest_file(path)
    if expected_hash and checksum!=expected_hash:raise ValueError('Fixed-data SHA256 differs from metadata.')
    payload=load_pt(path)
    if payload['version']!=1 or plain(payload['identity'])!=fixed_identity(cfg,checkpoint_hash):
        raise ValueError('Fixed-data checkpoint/shift/seeds/planner identity differs.')
    n=len(payload['action'])
    if not n or any(len(payload[k])!=n for k in ('observation','next_observation','episode_ids')):
        raise ValueError('Invalid fixed-data transition arrays.')
    if [r['seed'] for r in payload['episodes']]!=cfg['stage3a']['fixed_data_seeds']:
        raise ValueError('Fixed-data episode seeds differ.')
    import torch
    if set(payload['episode_ids'].tolist())!=set(range(len(payload['episodes']))):
        raise ValueError('Fixed-data episode assignment is incomplete or out of range.')
    if any(not torch.isfinite(payload[k]).all().item() for k in ('observation','action','next_observation')):
        raise ValueError('Non-finite fixed data.')
    return checksum


def rows_valid(rows,cfg,dataset_hash,full=False):
    grid=[r['native_budget'] for r in rows]
    if not grid or grid!=BUDGETS[:len(grid)] or (full and grid!=BUDGETS):
        raise ValueError(f'Invalid milestone prefix: {grid}')
    for r in rows:
        for condition in ('shifted','nominal'):
            for ep in r['evaluation'][condition]['episodes']:
                if any(not math.isfinite(float(ep[k])) for k in ('success','episode_return','episode_length')):
                    raise ValueError('Non-finite evaluation.')
        for condition in ('initial','current'):
            if any(not math.isfinite(float(x)) for x in r['fixed_data_prediction'][condition]['per_episode_means']):
                raise ValueError('Non-finite fixed-data MSE.')
        if (r['adaptation_native_steps']!=r['native_budget'] or r['fixed_data_sha256']!=dataset_hash
                or not r['evaluation_state_invariance'] or not r['reward_q_parameter_invariance']):
            raise ValueError('Budget/data/evaluation/frozen-component invariance mismatch.')
        for condition in ('shifted','nominal'):
            if [ep['seed'] for ep in r['evaluation'][condition]['episodes']]!=cfg['stage3a']['evaluation_seeds']:
                raise ValueError('Held-out evaluation seeds differ.')
        for condition in ('initial','current'):
            if len(r['fixed_data_prediction'][condition]['per_episode_means'])!=len(cfg['stage3a']['fixed_data_seeds']):
                raise ValueError('Fixed-data episode count differs.')


def optimizer_valid(state,cfg):
    if not state or not state.get('param_groups'):raise ValueError('Missing Adam optimizer state.')
    for group in state['param_groups']:
        if (group['lr']!=cfg['stage3a']['lr'] or tuple(group['betas'])!=(.9,.999)
                or group['eps']!=1e-8 or group['weight_decay']!=0 or group['amsgrad'] or group.get('maximize',False)):
            raise ValueError('Optimizer is incompatible with existing episode-wise Adam.')


def checkpoint_valid(path,meta,cfg,checkpoint_hash,dataset_hash):
    state=load_pt(path)
    identity=dict(checkpoint_sha256=checkpoint_hash,fixed_data_sha256=dataset_hash,
                  controls={k:v for k,v in meta['config'].items() if k not in ('stage3a','stage3b','work_dir')})
    if state['version']!=1 or state['identity']!=identity:
        raise ValueError('Resume checkpoint identity differs from metadata.')
    rows_valid(state['rows'],cfg,dataset_hash)
    budgets=[r['native_budget'] for r in state['rows']]
    if state['budgets']!=budgets or state['rollout']['native_steps']!=budgets[-1]:
        raise ValueError('Checkpoint rollout budget differs from saved evaluations.')
    settings={k:v for k,v in state['adapter']['settings'].items() if k not in ('output','fixed_data_path','fixed_data_collect')}
    expected={k:v for k,v in cfg['stage3a'].items() if k not in ('output','fixed_data_path','fixed_data_collect')}
    if settings!=expected:raise ValueError('Adapter update/replay/seed protocol differs.')
    optimizer_valid(state['adapter']['optimizer'],cfg)
    if state['adapter']['in_episode'] and state['environment'] is None:
        raise ValueError('Mid-episode resume lacks simulator/controller state.')
    if len(state['adapter']['replay'])>cfg['stage3a']['capacity']:raise ValueError('Replay exceeds capacity.')
    import torch
    if not state['adapter']['predictor'] or any(not torch.isfinite(v).all().item()
            for v in state['adapter']['predictor'].values() if isinstance(v,torch.Tensor)):
        raise ValueError('Missing/non-finite predictor.')
    for key in ('predictor','rng','counters','in_episode'):
        if key not in state['adapter']:raise ValueError(f'Missing adapter {key}.')
    adapter=state['adapter'];last=state['rows'][-1]
    if adapter.get('version')!=1 or len(adapter['counters'])!=4:
        raise ValueError('Invalid adapter snapshot schema.')
    episodes,transitions,native,updates=adapter['counters']
    if (episodes!=last['adaptation_episodes'] or updates!=last['gradient_update_steps']
            or len(adapter['replay'])!=last['replay_size'] or native>budgets[-1]
            or transitions<0 or any(x<0 for x in adapter['counters'])):
        raise ValueError('Adapter counters/replay disagree with milestone.')
    if len(state['episodes'])!=episodes:
        raise ValueError('Completed episode history disagrees with adapter.')
    if updates and not adapter['optimizer']['state']:
        raise ValueError('Updated predictor lacks Adam moment state.')
    for moment in adapter['optimizer']['state'].values():
        if int(moment['step'])!=updates or any(not torch.isfinite(moment[k]).all().item() for k in ('exp_avg','exp_avg_sq')):
            raise ValueError('Adam step/moment state disagrees with gradient count.')
    if any(len(tr)!=3 or any(not isinstance(t,torch.Tensor) or not torch.isfinite(t).all().item() for t in tr)
           for tr in adapter['replay']):
        raise ValueError('Invalid replay transitions.')
    for key in ('rng','planner','rollout','episodes'):
        if key not in state:raise ValueError(f'Missing checkpoint {key}.')
    return budgets[-1]


def checkpoint_paths(directory,seen=None):
    directory=Path(directory);seen=set() if seen is None else seen
    if directory in seen:raise ValueError('Cycle in resume provenance.')
    seen.add(directory)
    paths={}
    metadata=read(directory/'metadata.json')
    parent=metadata.get('resumed_from') or metadata.get('config',{}).get('stage3b',{}).get('resume')
    if parent:
        paths.update(checkpoint_paths(Path(parent).parent,seen))
    for path in directory.glob('resume-budget-*.pt'):
        try:budget=int(path.stem.split('-')[-1])
        except ValueError:continue
        paths[budget]=path
    return paths


def audit_candidate(path,expected,checkpoint_hash):
    """Never trust the last directory component (legacy runs use cfg.seed)."""
    path=Path(path);answer=dict(path=str(path.parent),status='INCOMPATIBLE',reason='')
    try:
        meta=read(path);cfg=meta['config']
        if meta['active_method']!='dynamics_only' or controls(cfg)!=controls(expected):
            raise ValueError('Resolved experiment controls differ from requested protocol.')
        if meta['checkpoint_sha256']!=checkpoint_hash:raise ValueError('Pretrained checkpoint hash differs.')
        if meta.get('policy_eval_mode', True) is not True or meta.get('evaluation_budget_excluded', True) is not True:
            raise ValueError('Evaluation/adaptation budget protocol differs.')
        if meta.get('optimizer_class') not in (None,'torch.optim.Adam'):raise ValueError('Optimizer class differs.')
        dataset=Path(meta['fixed_data_path'])
        data_hash=check_dataset(dataset,cfg,checkpoint_hash,meta['fixed_data_sha256'])
        paths=checkpoint_paths(path.parent)
        valid={};errors=[]
        for budget,cp in sorted(paths.items()):
            try:
                owner=read(cp.parent/'metadata.json')
                if controls(owner['config'])!=controls(cfg):raise ValueError('Resume ancestor controls differ.')
                found=checkpoint_valid(cp,owner,cfg,checkpoint_hash,data_hash)
                if found!=budget:raise ValueError('Checkpoint filename disagrees with payload.')
                valid[budget]=str(cp)
            except (Exception) as exc:
                errors.append(f'{cp}: {type(exc).__name__}: {exc}')
        answer.update(dataset=str(dataset),dataset_sha256=data_hash,checkpoint_paths=valid,
                      artifact_errors=errors,optimizer='torch.optim.Adam')
        complete_error=None
        if (path.parent/'complete.json').exists():
            try:
                rows=[json.loads(line) for line in (path.parent/'evaluations.jsonl').read_text().splitlines()]
                rows_valid(rows,cfg,data_hash,full=True)
                if set(valid)!=set(BUDGETS):raise ValueError('Missing/corrupt milestone snapshots (including resume ancestors).')
                marker=read(path.parent/'complete.json')
                if marker['adaptation_native_steps']!=12800:raise ValueError('Wrong completion budget.')
                if any(rows[-1].get(k)!=v for k,v in marker.items()):
                    raise ValueError('Completion metrics disagree with final evaluation.')
                final=load_pt(path.parent/'adapted.pt')
                if (final['checkpoint_sha256']!=checkpoint_hash or final['fixed_data_sha256']!=data_hash
                        or final['metrics']!=marker or not final['predictor']):
                    raise ValueError('Final predictor identity/metrics differ.')
                optimizer_valid(final['optimizer'],cfg)
                from common.stage3a import equal_state
                snapshot=load_pt(valid[12800])
                if (not equal_state(final['predictor'],snapshot['adapter']['predictor'])
                        or not equal_state(final['optimizer'],snapshot['adapter']['optimizer'])
                        or snapshot['rows']!=rows):
                    raise ValueError('Final predictor/optimizer/evaluations differ from final milestone.')
                for name in ('diagnostics.jsonl','adaptation-episodes.jsonl'):
                    with (path.parent/name).open() as f:
                        for line in f:json.loads(line)
                answer.update(status='REUSE_COMPLETE',reason='Resolved config, dataset, all milestones and final predictor verified.')
                return answer
            except Exception as exc:complete_error=f'{type(exc).__name__}: {exc}'
        # Requiring a contiguous snapshot prefix prevents endlessly resuming a final
        # checkpoint whose earlier milestone artifacts were deleted.
        prefix=[]
        for budget in BUDGETS:
            if budget not in valid:break
            prefix.append(budget)
        if prefix:
            original=Path(cfg['checkpoint'])
            if not original.is_file() or digest_file(original)!=checkpoint_hash:
                raise ValueError('Exact resume requires the original nominal checkpoint path/hash.')
            budget=max(prefix)
            answer.update(status='RESUME_PARTIAL',resume=valid[budget],budget=budget,
                          reason='Resume latest verified snapshot; preserve prior directory.'+
                          (f' Completion artifact issue: {complete_error}' if complete_error else ''))
        else:raise ValueError('No valid continuation checkpoint. '+str(complete_error or errors))
    except Exception as exc:answer['reason']=f'{type(exc).__name__}: {exc}'
    return answer


def discover(root,unclassified=None):
    by_condition={}
    for path in sorted((Path(root)/'runs').rglob('metadata.json')):
        try:
            meta=read(path);cfg=meta['config'];p=cfg['dynamics_shift']['parameter']
            if meta['active_method']=='dynamics_only' and cfg['stage3a']['seed_set']==0:
                by_condition.setdefault((p['type'],float(p['shifted'])),[]).append(path)
        except (ValueError,KeyError,TypeError,OSError) as exc:
            if unclassified is not None:unclassified.append(dict(path=str(path),reason=f'{type(exc).__name__}: {exc}'))
            continue
    return by_condition
