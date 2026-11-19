"""Validated, read-only inputs for the final research figures."""
from __future__ import annotations

from pathlib import Path
import json
import numpy as np
import pandas as pd

from .config import GRID, TASKS
from .metrics import require
from .provenance import verified_artifact


ARTIFACTS = {
    'terminal': 'q-terminal-planning-v1',
    'decision': 'q-decision-relevance-v1',
    'returns': 'q-terminal-planning-presentation-v2',
}


def load_inputs(root: Path):
    """Verify historical hashes and complete six-by-six coverage before plotting."""
    root = root.resolve()
    verified={key: verified_artifact(root, name) for key, name in ARTIFACTS.items()}
    paths={key:value[0] for key,value in verified.items()}
    tables = {
        'control': pd.read_csv(paths['decision']/'tables/control_prediction_summary.csv'),
        'evidence': pd.read_csv(paths['decision']/'tables/actual_control_prediction_evidence.csv'),
        'ranking': pd.read_csv(paths['decision']/'tables/fixed_summary.csv'),
        'native': pd.read_csv(paths['decision']/'tables/native_summary.csv'),
        'effects': pd.read_csv(paths['terminal']/'tables/terminal_effect_decomposition.csv'),
        'returns': pd.read_csv(paths['returns']/'tables/heldout_return_summary.csv'),
    }
    expected={(task,family,float(k)) for task,t in TASKS.items() for family in t['families'] for k in GRID}
    for name in ('control','ranking','native','effects','returns'):
        frame=tables[name]
        actual=set(zip(frame.task,frame.family,frame.severity.astype(float)))
        require(actual==expected, f'{name}: missing or extra conditions: {expected ^ actual}')
    for task,family,k in expected:
        c=tables['control'];c=c[(c.task==task)&(c.family==family)&(c.severity==k)]
        require(set(c[c.metric=='return'].method)=={'F','D','DQ','J'},
                f'Missing measured control method: {task}/{family}/{k}')
        require(set(c[c.metric=='fixed_encoder_latent_mse'].method)=={'F','D','DQ'},
                f'Missing comparable prediction method: {task}/{family}/{k}')
        rank=tables['ranking'];rank=rank[(rank.task==task)&(rank.family==family)&
                                          (rank.severity==k)&(rank.phase=='fixed_total_score')]
        for metric in ('argmax_disagreement','spearman'):
            require({'D → DQ','DQ_F → DQ'}<=set(rank[rank.metric==metric].comparison),
                    f'Missing ranking metric: {task}/{family}/{k}/{metric}')
        n=tables['native'];n=n[(n.task==task)&(n.family==family)&
                              (n.severity==k)&(n.comparison=='DQ_F → DQ')&
                              (n.metric=='final_action_l2')]
        require(len(n)>0,f'Missing native MPPI metric: {task}/{family}/{k}')
    for task,t in TASKS.items():
        e=tables['evidence']; e=e[e.task==task]
        for family in t['families']:
            for k in GRID:
                v=e[(e.family==family)&(e.severity==k)]
                require(set(v.adaptation_seed)==set(t['adaptation']),f'Adaptation seeds: {task}/{family}/{k}')
                require(set(v.evaluation_seed)==set(t['evaluation']),f'Evaluation seeds: {task}/{family}/{k}')
                for method in ('F','D','DQ','J'):
                    w=v[v.method==method]
                    require(len(w)==3*20 and not w.duplicated(['adaptation_seed','evaluation_seed']).any(),
                            f'Paired evidence: {task}/{family}/{k}/{method}')
    arrays={}
    for task,t in TASKS.items():
        for family in t['families']:
            for k in GRID:
                for seed in t['adaptation']:
                    stem=f'{task}-{family}-k{k:g}-seed{seed}.npz'
                    path=paths['terminal']/'q_arrays'/stem
                    require(path.is_file(),f'Missing saved array: {path}')
                    arrays[(task,family,float(k),seed)]=path
    require(len(arrays)==108,'Expected 108 validated inference arrays')
    return paths,tables,arrays,{key:value[1] for key,value in verified.items()}


def paired_terminal_arrays(path: Path):
    """Fixed heads 0,1; same tape, terminal action and candidate IDs."""
    with np.load(path, allow_pickle=False) as z:
        seeds=z['evaluation_seeds']
        require(len(seeds)==20 and len(set(map(int,seeds)))==20,'Terminal evaluation seeds')
        require(np.array_equal(z['candidate_ids'],np.arange(32)),'Candidate IDs')
        require(np.array_equal(z['F_immediate_scalar'],z['D_immediate_scalar']),
                'Frozen/D immediate Q invariant')
        q={m:np.asarray(z[f'{m}_terminal_scalar'][:2],dtype=np.float64).mean(axis=0)
           for m in ('F','D','DQ_F','DQ','J')}
        score={m:np.asarray(z[f'{m}_scores'],dtype=np.float64)
               for m in ('DQ_F','DQ')}
        terminal={m:np.asarray(z[f'{m}_terminal_scores'],dtype=np.float64)
                  for m in ('DQ_F','DQ')}
        rewards={m:np.asarray(z[f'{m}_rewards'],dtype=np.float64)
                 for m in ('DQ_F','DQ')}
        critic_effect=np.asarray(z['effect_critic_DQ'],dtype=np.float64)
        effects={key:np.asarray(z[f'effect_{key}'],dtype=np.float64)
                 for key in ('predictor_D','predictor_D_to_DQ','total_DQ_minus_F')}
        require(np.allclose(rewards['DQ'],rewards['DQ_F'],atol=1e-5,rtol=1e-5),
                'Q swap changed predicted rewards')
        require(np.allclose(score['DQ']-score['DQ_F'],terminal['DQ']-terminal['DQ_F'],
                            atol=5e-4,rtol=2e-5),'Candidate Q-swap score identity')
        require(all(np.isfinite(v).all() for v in (*q.values(),*score.values(),*terminal.values())),
                'Nonfinite terminal/candidate values')
        return dict(seeds=seeds,q=q,score=score,terminal=terminal,rewards=rewards,
                    critic_effect=critic_effect,effects=effects)


def load_swap(root: Path, source: Path | None):
    """Accept only a complete, provenance-checked physical Q-swap batch."""
    if source is None:
        candidates=list(root.glob('control-q-run-*/q-swap/completion.json'))
        candidates=[p.parent for p in candidates if json.loads((p/'completion.json').read_text()).get('status')=='PASS']
        require(len(candidates)<=1,'Multiple completed Q-swap batches; specify --swap-source')
        if not candidates:return None
        source=candidates[0]
    from .metrics import read,sha256
    source=source.resolve();done=read(source/'completion.json')
    require(done['status']=='PASS' and not done['smoke'] and done['streams']==108,
            'Full 108-stream Q-swap PASS required')
    provenance=read(source/'provenance.json')
    for name,record in provenance['sources'].items():
        path=Path(name)
        require(path.is_file() and sha256(path)==record['sha256'],f'Changed Q-swap source: {path}')
    files=sorted(source.glob('*/episodes.csv'))
    require(len(files)==108,'Q-swap episode stream coverage')
    frames=[]
    for file in files:
        child=read(file.parent/'completion.json')
        require(child['status']=='PASS' and child['episodes']==40,'Incomplete Q-swap child')
        for name,digest in child['files'].items():
            path=file.parent/name
            require(path.is_file() and sha256(path)==digest,f'Changed Q-swap child: {path}')
        frames.append(pd.read_csv(file))
    episodes=pd.concat(frames,ignore_index=True)
    require(len(episodes)==4320,'Q-swap episode count')
    return source,episodes
