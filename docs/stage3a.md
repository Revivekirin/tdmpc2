# Stage 3A: episode 단위 predictor-only TTT

`tdmpc2/adapt_shift.py`가 Frozen MPC와 Dynamics-only TTT + MPC를 실행한다.
기존 Phase A–D / Stage 2 runner와 분석 코드는 수정하지 않았다. `stage3a.mode`
선택지가 이번 runner의 방법을 결정한다. 기존 `adaptation` group은 Stage 2용이다.
`joint_rl`(Standard TD-MPC2 online RL), `dynamics_value`는 명시적으로
NotImplementedError를 발생시키는 후속 확장 지점이다. 새로운 JEPA network,
LoRA, planning-aware loss, value update는 없다.

## 원래 학습 목표와의 관계

저장소의 `TDMPC2._update`는 현재 encoder로 `obs[1:]`를 `no_grad`에서 encode한다.
별도 EMA target encoder는 없다. EMA 업데이트 대상은 Q 함수뿐이다.
현재 관측을 encode한 뒤 predictor를 재귀적으로 unroll하며 다음을 계산한다.

```
L_consistency = (1/H) * sum_t rho**t * MSE(z_pred[t+1], stopgrad(encode(obs[t+1])))
L_total = consistency_coef * L_consistency + reward/value/termination losses
```

state encoder와 dynamics 출력의 기존 SimNorm을 그대로 사용한다. 추가 L2 정규화,
contrastive loss, whitening은 없다. Stage 3A에서는 학습 horizon을 1로 제한한다.
MPC의 planning horizon은 기존 설정을 유지한다.

```
L_TTT = consistency_coef * MSE(
    dynamics(stopgrad(encode(obs)), commanded_action),
    stopgrad(encode(next_obs)))
```

즉 기존 consistency 목표의 H=1 항이다. JEPA-TTT라는 표현은 이 latent prediction
adaptation을 가리킨다. 원래 TD-MPC2와 달리 encoder, reward, termination, policy,
Q 및 target Q를 고정하고 dynamics만 Adam으로 업데이트한다. reward/TD/actor loss,
Q target update는 실행하지 않는다. `consistency_steps`는 확장 지점이며 1 이외는
거부한다. 전체 모델을 eval mode로 유지한다(state encoder/dynamics에는 dropout 없음).

## 시간·replay·평가 규약

- 첫 실험은 single-task state observation, fixed non-identity shift, MPC로 제한한다.
- `begin_episode → observe → end_episode` API. episode 내부 predictor 불변을 검사하고
  종료 후에만 gradient update를 수행한다. 별도 `update` API도 내부/평가 중 호출을 거부한다.
- predictor, Adam state, FIFO replay를 episode 사이 유지한다. pretrained replay는 읽지 않는다.
  replay에는 해당 실행의 shifted 환경에서 얻은 transition만 넣는다. Frozen도 같은
  interaction 수집 규약을 사용하되 optimizer와 update는 없다.
- 기본 capacity=100000 transitions, uniform sampling without replacement,
  batch=256, warmup=1000, episode별 update=50. 이 값은 metadata에 저장된다.
  warmup 및 batch size를 모두 충족해야 업데이트한다. 각 update 사이에는 재샘플링한다.
- commanded action만 학습 입력이다. 실제 executed action과 native-step delay queue는
  `diagnostics.jsonl`에 별도로 기록한다. hidden diagnostic은 encoder/predictor에 넣지 않는다.
- adaptation seed와 evaluation seed의 중복을 거부한다. 각 evaluation checkpoint마다
  별도 shifted/nominal 환경을 생성하고 동일한 held-out seed set을 재사용한다.
  nominal과 shifted evaluation 간 seed 재사용은 의도된 것이며 adaptation과는 분리된다.
- 평가에서 replay addition/update를 금지하고 전체 model state, Adam state, replay,
  학습 counters 및 replay RNG 불변을 검사한다. planner warm-start와 Python/NumPy/
  Torch/CUDA RNG도 복원한다. 평가 환경은 닫는다.
- baseline(episode 0), 설정한 episode 간격, 마지막 episode에서 평가한다.
  모든 방법은 같은 checkpoint를 새로 load하며 SHA-256을 기록한다.
- native steps는 action repeat 상수가 아니라 wrapper의 실제 native record 수로 센다.
  adaptation policy decisions=transitions. 평가의 episodes/native steps/policy decisions는
  별도 누적값으로 기록하고 adaptation budget에서 제외한다. 비교는 동일 episode 수이며
  조기 종료 시 두 방법의 실제 interaction 수가 달라질 수 있다. 결과 비교 시 기록된
  native-step/decision budget을 함께 제시해야 한다.

## 로그 해석

`stage3a.output/<mode>/<seed>/`에 metadata, evaluations.jsonl, diagnostics.jsonl,
최종 adapted.pt와 성공 시 complete.json을 만든다. 기존 결과 디렉터리는 덮어쓰지 않는다.
최종 파일은 predictor/optimizer와 counters를 저장하며 replay까지 복원하는 resume
checkpoint는 아니다.

각 evaluation row에는 adaptation episodes/transitions, native steps, policy decisions,
gradient update 수, replay size, consistency loss, predictor gradient/update norm,
held-out prediction MSE, shifted success/length, nominal success 및 초기 nominal success
대비 retention delta가 있다. loss는 가중치 적용 전 MSE, gradient norm은
`consistency_coef` 적용 후 clipping 전 L2, update norm은 optimizer step 전후 parameter
차이 L2이며 최근 episode의 업데이트 평균이다. 업데이트 전이나 Frozen에서는 null이다.
Frozen의 prediction error는 held-out MSE에서 확인한다.

encoder/reward/Q/target Q/policy/termination의 전체 state 불변을 검사하며 위반 시
실행을 중단한다. `reward_q_parameter_invariance`는 이 검사가 통과했다는 뜻이다.

Q drift는 두 가지로 구분한다.

1. 실제 MPC가 `_Qs`에 전달하는 latent+action input의 mean/std를 누적한다.
   episode 0의 동일 condition 분포와 비교한 L2 drift를 기록한다. 방문 상태와
   후보 action 분포 변화도 포함하며 predictor만의 인과 효과라고 해석하지 않는다.
2. held-out observation과 동일 commanded action을 반복하는 planning-horizon rollout에서
   초기 pretrained predictor와 현재 predictor의 latent를 비교한다. paired MSE 및
   marginal mean/std L2를 기록한다. 이것은 통제된 latent drift diagnostic이며 실제
   planner candidate distribution 자체는 아니다. critic action block은 동일하다.

held-out seed set을 매 checkpoint마다 재사용하므로 이 결과로 hyperparameter를 고르면
별도 최종 test seed set을 추가해야 한다.

## 실행

Linux NVIDIA CUDA + 기존 ManiSkill2 환경과 pretrained checkpoint가 필요하다.
아래 명령은 저장소 root에서 실행한다. checkpoint의 모델 크기/termination head에
맞춰 model_size와 episodic을 지정한다(기존 pick-cube 예시는 model_size=5, episodic=false).

```bash
export TD_MPC2_CHECKPOINT=/absolute/path/to/pick-cube-1.pt

# 짧은 실제 환경 smoke: 각 방법 2 episodes, episode마다 2 updates (TTT만)
python tdmpc2/adapt_shift.py -m \
  hydra/launcher=basic task=pick-cube model_size=5 episodic=false \
  checkpoint="$TD_MPC2_CHECKPOINT" dynamics_shift=action_delay schedule=fixed \
  dynamics_shift.parameter.shifted=1 stage3a.mode=frozen,dynamics_only \
  stage3a.episodes=2 'stage3a.adaptation_seeds=[1000,1001]' \
  'stage3a.evaluation_seeds=[2000,2001]' stage3a.eval_every=1 \
  stage3a.batch_size=16 stage3a.warmup=16 stage3a.updates_per_episode=2 \
  stage3a.output=stage3a-smoke

# 연구용 full evaluation: 방법 2, 각 10 adaptation episodes
# 매 2 episodes마다 shifted 5 + nominal 5 held-out episodes (baseline 포함 총 60)
python tdmpc2/adapt_shift.py -m \
  hydra/launcher=basic task=pick-cube model_size=5 episodic=false \
  checkpoint="$TD_MPC2_CHECKPOINT" dynamics_shift=action_delay schedule=fixed \
  dynamics_shift.parameter.shifted=1 stage3a.mode=frozen,dynamics_only \
  seed=1 stage3a.output=stage3a-full-delay1
```

`seed`는 초기화/replay sampling RNG이며 각 episode의 planner RNG는 explicit episode seed로
설정된다. 독립 반복이 필요하면 adaptation/evaluation seed set을 함께 바꾸고 두 방법에는
동일한 set을 적용한다. root seed만 바꾸면 Frozen 반복은 동일하다.
shift별 결과는 output을 바꿔 보존한다. 예: `dynamics_shift=action_gain
 dynamics_shift.parameter.shifted=0.7 stage3a.output=stage3a-full-gain07`.
성능 결과가 나온 뒤에만 후속 방법 구현 여부를 결정한다.

## 로컬 검증 결과와 한계 (2026-11-03)

CPU protocol/Hydra/평가 통합 smoke test 9개 통과. 합성 모델의 짧은 3-episode 실행:
48 transitions, 96 synthetic native steps, 30 gradient updates. 별도 합성 held-out
데이터의 MSE 0.036998 → 0.007335. 이 수치는 TD-MPC2 checkpoint나 ManiSkill 성능
결과가 아니다. 합성 fixture는 softmax latent를 사용하며 실제 SimNorm network 검증을
대체하지 않는다. 이 호스트의 CUDA 사용 불가, 저장소 내 checkpoint 없음으로 실제
shifted success/nominal retention 실험은 미실행이다. 위 실제 smoke 명령을 CUDA
호스트에서 먼저 수행한 뒤 full evaluation을 실행한다.

```bash
python -m unittest discover -s tests -p 'test_stage3a*.py' -v
```
