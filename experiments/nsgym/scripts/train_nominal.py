"""CUDA-only nominal training using the original agent, buffer and replay format."""
import argparse
import json
from pathlib import Path
import time

from research_utils import ROOT, output_dir, metadata, write_json, csv_rows, bootstrap
from training_support import config_from_file, require_cuda, build, evaluate, checkpoint
import numpy as np
import torch
from common.seed import set_seed


def train(config, destination):
    require_cuda(config['device'])
    # Import lazily: configuration and blocked-host diagnostics do not require the CUDA stack.
    from tdmpc2 import TDMPC2
    from common.buffer import Buffer
    from trainer.online_trainer import OnlineTrainer
    out=output_dir(destination); provenance=metadata(config);write_json(out/'metadata.json',provenance)
    set_seed(config['seed']);cfg,env=build(config)
    agent=TDMPC2(cfg);buffer=Buffer(cfg)
    serializer=OnlineTrainer(cfg,env,agent,buffer,None) # Reuse original to_td, not a copied replay schema.
    updates=0;episode=0;done=True;tds=[];all_eval=[];last_checkpoint=None
    started=time.perf_counter();eval_seconds=0;quality_history=[];metrics_path=out/'training.jsonl'
    try:
        for step in range(cfg.steps):
            if done:
                if tds: buffer.add(torch.cat(tds))
                obs=env.reset(seed=cfg.seed+episode);episode+=1
                tds=[serializer.to_td(obs)]
            action=agent.act(obs,t0=len(tds)==1) if step>cfg.seed_steps else env.rand_act()
            obs,reward,done,info=env.step(action)
            if info['terminated'] and not cfg.episodic:raise ValueError('Unexpected true termination in continuing-task configuration')
            tds.append(serializer.to_td(obs,action,reward,info['terminated']))
            if not torch.isfinite(obs).all() or not torch.isfinite(reward):raise ValueError('Non-finite rollout')
            metrics={}
            if step>=cfg.seed_steps:
                if buffer.num_eps==0:raise RuntimeError('No complete seed episode in replay')
                # Match OnlineTrainer: seed_steps pretraining updates at zero-based
                # step==seed_steps, then one update per subsequent transition.
                count=cfg.seed_steps if step==cfg.seed_steps else 1
                for _ in range(count):
                    metrics={k:float(v) for k,v in agent.update(buffer).items()};updates+=1
                    if not all(np.isfinite(v) for v in metrics.values()):raise ValueError('Non-finite optimizer metrics')
                if not all(torch.isfinite(p).all() for p in agent.model.parameters()):raise ValueError('Non-finite model parameters')
            transitions=step+1
            with metrics_path.open('a') as f:
                f.write(json.dumps(dict(environment_transitions=transitions,gradient_updates=updates,
                    training_wall_seconds=time.perf_counter()-started-eval_seconds,**metrics),allow_nan=False)+'\n')
            if transitions%cfg.eval_freq==0 or transitions==cfg.steps:
                policy,elapsed=evaluate(agent,config,cfg.eval_seeds);eval_seconds+=elapsed
                random_rows,elapsed=evaluate(None,config,cfg.eval_seeds,random_actions=True);eval_seconds+=elapsed
                difference=[a['episode_return']-b['episode_return'] for a,b in zip(policy,random_rows)]
                quality_history.append(dict(transitions=transitions,paired_gain_mean=float(np.mean(difference)),paired_gain_ci=bootstrap(difference)))
                for row in policy+random_rows:
                    row.update(environment_transitions=transitions,gradient_updates=updates,evaluation_wall_seconds=eval_seconds,
                               training_wall_seconds=time.perf_counter()-started-eval_seconds)
                all_eval.extend(policy+random_rows)
            if transitions%cfg.checkpoint_freq==0 or transitions==cfg.steps:
                last_checkpoint=checkpoint(agent,cfg,out,transitions,updates,provenance)
        # Retain the final trajectory even when the explicit transition budget cuts an episode short.
        if len(tds)>cfg.horizon:buffer.add(torch.cat(tds))
        csv_rows(out/'evaluation.csv',all_eval)
        with (out/'evaluation.jsonl').open('x') as f:
            for row in all_eval:f.write(json.dumps(row)+'\n')
        # Predeclared gate: consistent positive paired gain over random at three
        # evaluations, >=10 episodes each. No benchmark-derived threshold invented.
        eligible=not cfg.smoke and cfg.eval_episodes>=10 and len(quality_history)>=3 and all(q['paired_gain_ci'][0]>0 for q in quality_history[-3:])
        quality=dict(status='PRELIMINARY_SUPERIORITY' if eligible else 'INCONCLUSIVE',scientific_sweep_eligible=eligible,
            checkpoint_sha256=__import__('research_utils').digest(last_checkpoint/'model.pt'),
            evaluations=quality_history,criterion='Last 3 evaluations each have positive lower 95% paired bootstrap gain over random, >=10 seeds; smoke is never eligible.',
            limitation='Repeated fixed evaluation seeds; not a benchmark convergence threshold or proof of generalization.')
        write_json(out/'quality.json',quality)
        write_json(out/'completion.json',dict(integration='PASS',environment_transitions=cfg.steps,
            gradient_updates=updates,seed_exploration_transitions=cfg.seed_exploration_transitions,checkpoint=str(last_checkpoint),smoke=cfg.smoke,
            nominal_controller_quality=quality['status']))
        plot_training(out)
        return last_checkpoint
    finally:env.close()


def plot_training(out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import csv
    out=Path(out)
    if (out/'training_curves.png').exists():raise FileExistsError(out/'training_curves.png')
    rows=[json.loads(x) for x in (out/'training.jsonl').read_text().splitlines()]
    ev=list(csv.DictReader((out/'evaluation.csv').open()))
    fig,axes=plt.subplots(2,1,figsize=(8,7))
    loss=[r for r in rows if 'total_loss' in r]
    axes[0].plot([r['environment_transitions'] for r in loss],[r['total_loss'] for r in loss]);axes[0].set_ylabel('Training total loss')
    for random_flag,label in [('False','TD-MPC2'),('True','Random')]:
        steps=sorted({int(r['environment_transitions']) for r in ev})
        axes[1].plot(steps,[np.mean([float(r['episode_return']) for r in ev if int(r['environment_transitions'])==s and r['random_actions']==random_flag]) for s in steps],label=label)
    axes[1].set_ylabel('Evaluation return');axes[1].set_xlabel('Environment transitions');axes[1].legend()
    fig.tight_layout();fig.savefig(out/'training_curves.png');plt.close(fig)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,default=ROOT/'experiments/nsgym/configs/train_nominal.yaml')
    parser.add_argument('--output',type=Path)
    parser.add_argument('--smoke',action='store_true')
    parser.add_argument('--check-config',action='store_true')
    parser.add_argument('--plot',type=Path)
    args=parser.parse_args()
    if args.plot:plot_training(args.plot);return
    config=config_from_file(args.config,args.smoke)
    if args.check_config:print(json.dumps(config,indent=2));return
    print(train(config,args.output or config['output']))

if __name__=='__main__':main()
