# TD-MPC2 Dynamics Shift: 코드·물리 분석과 연구 타당성

작성: 2026-11-08. **상태: 로컬 분석 완료 / 최신 서버 검증 및 신규 simulator response 측정 대기.**

## 1. Executive Summary

현재 네 조건은 서로 다른 문제를 실험한다. Action gain은 **Cartesian 명령 감쇠**, action delay는 **숨겨진 명령 이력에 의존하는 지연 제어**, object mass는 **물체 질량·관성의 동시 변화**, contact friction은 **물체와 지면 재질의 마찰계수 변화**이다. 네 조건을 모두 motor degradation으로 묶을 수 없다.

로컬 pilot에서 gain 0.3은 9/10 성공하지만 평균 33.1 policy decisions를 사용한다. Nominal 6.1보다 길다. Delay 2는 4/10, delay 3은 0/10이다. Mass 3과 friction 0.3은 각각 10/10 성공이다. 이것은 해당 평가 표본의 결과이지, 물리 변화가 없다는 증거나 적응이 불필요하다는 일반적 결론은 아니다.

권고는 gain을 RENAME, delay를 SEPARATE, mass를 물리 조건으로 KEEP하되 성공률 단독 평가를 보완, friction을 표면별로 분리하는 MODIFY이다. 기존 checkpoint를 유지한 PickCube 진단을 먼저 마치고, 접촉 민감도가 부족하면 작업을 확장한다. 성능을 낮추려는 극단적 severity 증가는 권고하지 않는다.

### 증거 구분과 분석 범위

- **[C] 직접 읽은 현재 로컬 코드:** `/Users/jihyekim/Desktop/tdmpc2`, HEAD `32d8730c13f7aa36adeb5ad76781d669c09093b3`. 시작 시 `.DS_Store`, `scripts/run_stage3b.py`가 수정돼 있었으며 그대로 보존했다.
- **[R] 공식 배포본 reference:** PyPI `mani-skill2==0.4.1.post0` wheel에서 추출한 Python 소스. 설치하지 않았다. Wheel SHA256 `04121326858535186db3fcd9598d005f673c43aaab874e61244b8f6e2c7aa1ec`. 이것을 현재 서버 설치본과 동일하다고 인증하지 않는다.
- **[H] 과거 서버 측정의 저장 문서:** `DYNAMICS_SHIFT_AUDIT.md:5–63`. 2026-10-27 수신 결과이며, 이번 작업에서 원본 simulator JSON을 다시 읽거나 simulator를 실행한 것은 아니다.
- **[O] 기존 실험 관측:** 로컬 CSV에 기록된 Frozen pilot와 TTT 집계값. Raw trajectory/완료 metadata가 현재 로컬에 없어 이 작업은 집계표의 수치 해석까지 수행한다.
- **[T] 물리 이론 또는 reference 설정을 대입한 계산:** 측정값과 구분한다.
- **[U] UNKNOWN:** 최신 서버 revision, 현재 설치 binary, 현재 checkpoint의 training configuration과 live response 중 확인되지 않은 부분.

사용자가 지정한 `/nfs4/jhkim/repos/tdmpc2`는 로컬 경로가 아니다. SSH 대상 확인을 요청했으며 현재 대상이 확정되지 않았다. 따라서 이 보고서는 **최신 서버 기준 최종 인증 보고서가 아니다**. 이전 ZIP을 분석 기준으로 사용하지 않았다. 최신 서버 파일·설치 controller·원본 response 로그 확인 전에는 아래 reference 수치를 현재 런타임의 확정값으로 인용하면 안 된다.

## 2. Actual TD-MPC2 → ManiSkill2 execution pipeline

![실행 경로](/Users/jihyekim/Desktop/tdmpc2/analysis/physical_dynamics/action_pipeline.png)

| 순서 | 함수·위치 | 입력 → 출력 / 실제 역할 |
|---|---|---|
| 환경 생성 | `tdmpc2/envs/maniskill.py:88–124`, `envs/__init__.py:56` | PickCube-v0, state observation, `pd_ee_delta_pos`를 명시. 로봇/주파수는 설치 환경 기본값에 의존 |
| 상태 encoding | `tdmpc2/tdmpc2.py:98–120`, `common/world_model.py:105` | CPU observation Tensor `(O,)` → CUDA `(1,O)` → latent `(1,L)` |
| MPC | `tdmpc2/tdmpc2.py:143–219` | `(H,N,A)` 후보 action, predictor rollout, reward 및 terminal Q 평가, elite 선택, 첫 action `(A,)` 반환 |
| model transition | `common/world_model.py:116–123`, `tdmpc2.py:123–140` | latent와 후보 issued action을 연결해 다음 latent 예측. shift wrapper나 delay queue를 직접 호출하지 않음 |
| NumPy 변환 | `envs/wrappers/tensor.py:37` | CPU Tensor `(A,)` → NumPy `(A,)`; 단위 변환 없음 |
| 종료 관리 | `envs/wrappers/timeout.py:21`, `maniskill.py:123` | 최대 100 policy decisions; 내부 native 종료면 조기 중단 |
| 반복 | `ManiSkillWrapper.step`, `maniskill.py:51–76` | 같은 NumPy command를 최대 두 번 전달. reward 합, 마지막 observation/info 반환 |
| 명령 intervention | `dynamics_shift/base.py:83–125` | issued command 검증 → 현재 schedule → transform → native env.step(executed) |
| controller 분할 | [R] `agents/base_controller.py:231–244` | CombinedController가 arm 3차원, gripper 1차원으로 나눔 |
| 목표 계산 | [R] `controllers/pd_ee_pose.py:89–108` | 정규화 action → EEF target pose → IK → 7개 arm joint target |
| 구동 | [R] `controllers/pd_joint_pos.py:27–48` | stiffness/damping/force limit 설정, 각 joint drive target 전달 |
| 물리 적분 | [R] `envs/sapien_env.py:542–558` | set_action 1회 뒤 before_simulation_step와 scene.step 25회 |
| 결과 | [R] `sapien_env.py:531–540`, `pick_cube.py:54–105` | obs, success, reward, done 산출; 외부 tensor wrapper가 float32 reward Tensor로 변환 |

여기서 `H,N,L`은 실행 config의 horizon, sample 수, latent dimension이다. 현재 checkpoint 자체가 로컬에 없으므로 특정 숫자를 추측하지 않는다. [R] 표준 Panda state는 qpos 9 + qvel 9 + base pose 7 + TCP pose 7 + goal 3 + TCP→goal 3 + object pose 7 + TCP→object 3 + object→goal 3 = **51차원**이다. 이 계산은 reference의 observation 구성에 근거하며 현재 서버의 `observation_space.shape` readback은 대기다. 물체 선속도·각속도, delay queue는 이 state vector에 명시적으로 포함되지 않는다.

현재 runner는 `adapt_shift.py:48`에서 `eval_mode=True`로 호출한다. 그러나 MPC elite 선택에 `gumbel_softmax_sample`이 남아 있다(`tdmpc2.py:210`). 따라서 Frozen이나 eval_mode를 모든 경우 결정론적 action과 동의어로 쓰면 안 된다. 본 작업에서는 이를 이유로 pairing 실험을 다시 실행하지 않았다.

### Checkpoint와 controller의 연결

환경 생성 코드의 PickCube controller 선택은 직접 확인됐다. 집계 CSV는 checkpoint SHA256 `1781f0fa959ee51fe96450e5bf5bf24f1d87ce515a89730e6b964c560144a258`를 기록한다. 과거 문서는 `checkpoints/maniskill2/pick-cube-1.pt`를 사용한다. 다만 `TDMPC2.load`는 가중치를 로드할 뿐 controller 설정을 checkpoint에서 복원하지 않는다(`tdmpc2.py:81–95`). **실행 환경 설정 확인과 pretrained checkpoint의 학습 당시 설정 증명은 별개**다. 최신 checkpoint 파일 hash와 학습 provenance는 [U]이다.

## 3. Policy action dimensions and controller semantics

[C] PickCube 설정은 `pd_ee_delta_pos`이다. [R] Panda는 arm 이후 gripper 순서로 controller를 구성한다(`defaults.py:152–162`). 따라서 이 구성에서 `target=arm`의 `slice(None,-1)`은 정확히 0,1,2를 선택한다. wrapper 자체는 controller 이름을 검사하지 않으므로 다른 로봇의 마지막 성분이 항상 gripper라는 일반 보장은 없다.

| index | Controller parameter | 물리 의미 | policy 단위·범위 | [R] native target 단위·범위 |
|---|---|---|---|---|
| 0 | arm Δx | 현재 TCP/EEF 좌표계 x 방향 위치 증분 | 무차원 [-1,1] | [-0.1,0.1] m |
| 1 | arm Δy | 현재 TCP/EEF 좌표계 y 방향 위치 증분 | 무차원 [-1,1] | [-0.1,0.1] m |
| 2 | arm Δz | 현재 TCP/EEF 좌표계 z 방향 위치 증분 | 무차원 [-1,1] | [-0.1,0.1] m |
| 3 | mimic gripper position | 두 finger prismatic joint의 같은 절대 위치 목표 | 무차원 [-1,1] | 각각 [-0.01,0.04] m |

Arm은 **Cartesian EEF position delta control**이며 orientation action이 없다. 내부적으로는 IK로 joint position target을 계산해 drive를 제어한다. Joint velocity나 torque를 policy가 직접 출력하지 않는다. Gripper는 delta가 아닌 absolute position이다. -0.01 m 목표는 실제 finger가 음의 위치까지 움직인다는 뜻이 아니라 닫힘 접촉력을 만들기 위한 목표 설정이다.

[R] `utils/common.py:124–129`의 affine 변환:

\[
b(a;l,h)=\frac{h+l}{2}+\frac{h-l}{2}\operatorname{clip}(a,-1,1).
\]

따라서 arm은 `0.1*a` m, finger 한 관절은 `0.015+0.025*a[3]` m이다. Gripper a=-1,0,+1은 각각 -0.01,0.015,0.04 m 목표다. 두 finger의 간격을 이 숫자 자체와 혼동하지 않는다.

[R] `frame='ee', use_delta=True, use_target=False, interpolate=False`:

\[
p^*_{B}=p_{B,ee}+R_{B,ee}\,(0.1a_{0:3}),\qquad R^*_{B,ee}=R_{B,ee}.
\]

이전 commanded target이 아니라 **현재 측정 EE pose**를 기준으로 매 native call마다 다시 목표를 만든다. IK는 현재 full qpos를 초기값으로 쓰고 arm 7개 joint만 활성화한다. 실패하면 `_start_qpos`로 fallback한다. `_target_pose`, `_target_qpos`, `_start_qpos`, `_step`은 내부 state지만 `use_target=False`에서 `get_state()`는 target을 observation에 추가하지 않는다. [H] 서버 감사 기록도 이 설정을 보고했다.

## 4. Timing and action repeat / 기본 물리 모델

| 시간 구분 | [R] 값 및 [H] 기존 서버 보고 | 의미 |
|---|---|---|
| Physics integration | 500 Hz = 2 ms | scene.step 한 번 |
| Drive solver / passive compensation | physics substep마다 | 고정 target에 대한 drive와 물리 반응은 매 substep 계산 |
| Controller target update | 20 Hz = 50 ms | 새 native action으로 target·IK 재계산 |
| Environment native step | 50 ms, 25 physics substeps | wrapper delay queue의 한 칸 |
| TD-MPC2 policy decision | 정상 100 ms = 10 Hz | 같은 issued action으로 native 두 번 |
| MPC replanning | policy decision당 한 번 | 마지막 native observation으로 다시 계획 |
| 조기 종료 | 50 ms일 수 있음 | 첫 native step에서 done이면 repeat 중단 |

이는 simulation time이다. MPC 계산 wall-clock이 100 ms 이내라는 실시간 보장은 아니다. Action repeat 2는 **IK/target 계산도 두 번** 수행한다. 두 번째는 첫 50 ms 후 실제 EE 위치를 새 기준으로 삼는다. 같은 target을 100 ms 유지하는 것과 다르며, 실제 변위가 `2*0.1*a`라는 보장도 없다.

로봇을 이해하는 일반식은 다음과 같다. 접촉항 부호는 여기서 환경 반력을 좌변에 놓는 관례다.

\[
M(q)\ddot q+C(q,\dot q)\dot q+g(q)+J(q)^Tf_c=\tau_{drive}+\tau_{passive}.
\]

[R] reference는 joint drive에 arm stiffness=1000, damping=100, force_limit=100을 전달한다(`defaults.py:33–35`). Revolute joint의 통상 단위는 Nm/rad, Nm·s/rad, Nm이다. Gripper도 숫자는 같지만 prismatic joint이므로 N/m, N·s/m, N이다. 이들은 **시뮬레이션 drive 설정**이지 실제 Franka 제조사 사양이 아니다.

\[
\tau_{drive}\approx\operatorname{sat}_{\tau_{max}}
[K_p(q^*-q)+K_d(\dot q^*-\dot q)]
\]

위 식은 drive의 물리적 의미를 설명하는 근사식이다. Python에서 이 식을 직접 계산하는 것은 아니다. PhysX implicit drive, force/acceleration mode, constraint solver, contact coupling까지 같은 식으로 완전히 재현했다고 주장하지 않는다. 이 controller는 drive velocity target을 매번 설정하지 않으며 그 런타임 값은 추가 확인 대상이다.

또한 [R] `base_controller.py:204–213`은 `compute_passive_force(external=False)`를 매 physics step에 계산하여 `set_qf`로 전달한다. 따라서 시스템을 보상 없는 순수 PD만으로 설명하면 불완전하다. 로봇 자체의 passive force 보상은 grasp한 외부 물체 하중을 완전히 보상한다는 뜻도 아니다. Gain shift는 이 보상이나 drive gain/force limit을 변경하지 않는다.

## 5. Action Gain code/physics walkthrough

### Nominal, 적용 위치와 수치 예시

[C] `action_shift.py:19,57–73`: nominal은 1이며 arm만 곱한 뒤 전체 vector를 native bounds로 clip한다. `ManiSkillWrapper`는 추가 정규화를 하지 않는다. Planner는 먼저 [-1,1]로 제한하고, shift 다음에 설치 controller가 다시 clip/scale한다.

```python
# 실제 production transform의 핵심
executed = action.astype(np.result_type(action.dtype, self.low.dtype), copy=True)
executed[self.indices] *= value
return np.clip(executed, self.low, self.high)
```

아래는 정책에서 추출한 action이 아니라 **설명용 prescribed input을 실제 transform 함수에 실행한 결과**다. `scripts/analyze_physical_dynamics.py`가 production AST를 추출해 실행했으며 simulator는 사용하지 않았다.

```text
입력                 [ 0.50, -0.20, 0.10, -1.0 ]
gain 0.7 실행 명령    [ 0.35, -0.14, 0.07, -1.0 ]
gain 0.3 실행 명령    [ 0.15, -0.06, 0.03, -1.0 ]
[R] nominal EE delta [ 0.050,-0.020,0.010 ] m
[R] gain .7 delta   [ 0.035,-0.014,0.007 ] m
[R] gain .3 delta   [ 0.015,-0.006,0.003 ] m
7D joint target      UNKNOWN (현재 qpos와 IK 필요)
실측 EE/joint response UNKNOWN (신규 simulator 실행 없음)
```

| gain | 명령/target delta 잔존 비율 | 감소 | 위 예시 x target delta | 판정 |
|---|---:|---:|---:|---|
| 0.9 | 90% | 10% | 45 mm | 가벼운 command perturbation |
| 0.7 | 70% | 30% | 35 mm | 기존 reference 유지 가능 |
| 0.5 | 50% | 50% | 25 mm | command adaptation 중간 조건 |
| 0.3 | 30% | 70% | 15 mm | 감소가 크지만 torque 70% 손실이 아님 |

이 예시에서 saturation은 없다. 정상 범위 command에 0≤gain≤1을 곱하면 새 saturation을 만들지 않는다. 범위 밖 입력의 preclip을 wrapper가 수행하는 것은 아니므로 일반적인 외부 command에 대해서는 곱셈 후 clip 순서가 중요하다.

### 물리적 결과와 연구 타당성

[T] 가까운 EEF target은 일반적으로 joint target 오차를 줄일 수 있다. 그러나 IK의 비선형성, singularity, joint limit, contact, PD damping, passive compensation 때문에 속도나 torque가 정확히 gain배가 되지는 않는다. 방향·좌표계가 유지되는 command attenuation이며 actuator torque effectiveness `τ_actual=ατ_requested`와 다르다.

[O] gain .3에서도 9/10 성공하고 episode length가 33.1로 늘었다. 작은 증분을 여러 번 반복해 목표를 향할 수 있다는 해석과 부합한다. 하지만 성공한 policy의 EEF 속도, torque margin, clipping 비율을 실제로 측정하지 않아 원인을 확정할 수 없다. Frozen은 가중치를 고정할 뿐 observation feedback과 MPC replanning을 중단하지 않는다.

권고 이름은 **EEF delta-command gain**이다. 모터 고장을 연구하려면 drive force limit 감소, drive stiffness/damping 변화, 혹은 drive output torque에 effectiveness를 적용하는 별도 intervention을 설계해야 한다. 세 가지도 각각 포화 능력, servo 특성, torque 전달률이라는 다른 문제다. 현재 action/observation interface와 nominal drive를 유지하면 checkpoint 재사용 가능성은 있으나 nominal-equivalence와 접촉 하중 응답을 먼저 확인해야 한다. 이번에는 그러한 intervention을 구현하지 않았다.

![Gain](/Users/jihyekim/Desktop/tdmpc2/analysis/physical_dynamics/gain_command_vs_response.png)

## 6. Action Delay queue and timing walkthrough

[C] `action_shift.py:34–55,66–73`: queue는 매 native call에 clip 전 issued action의 복사본을 append한다. 길이는 schedule에서 가능한 최대 delay+1로 제한한다. 그 뒤 `queue[-d-1]`을 읽으므로 d=0은 방금 넣은 현재 command다. 처음 d개 call에는 초기 action(default 전부 0)의 arm 성분을 사용한다. Gripper는 현재 action 그대로다.

기본 `reset_queue_on_episode=True`이면 episode reset 후 queue가 비워진다. False라면 episode를 넘어 남을 수 있으므로 별도 실험 정의다. 현재 YAML은 기본 동작을 사용한다. Snapshot은 queue도 저장한다. Schedule 중 delay가 바뀌어도 queue를 비우지 않는다. Delay 증가 때 오래된 명령이 재등장할 수 있고, 감소 때 일부 과거 명령을 건너뛸 수 있다. Maximum delay는 schedule의 possible values로 초기화한다.

### 실제 함수로 확인한 8 native-step timeline

A/B/C/D는 각각 arm x=.2/.4/.6/.8의 prescribed command이고, I는 initial arm=0이다. Gripper는 A=+1,B=-1,C=+1,D=-1로 번갈아 입력했다. 각 행의 gripper는 delay와 무관하게 현재 열의 값이다.

| Native step | Policy decision | Commanded | d=0 arm | d=1 arm | d=2 arm | d=3 arm | gripper |
|---:|---:|---|---|---|---|---|---:|
| 0 | 0 | A | A | I | I | I | +1 |
| 1 | 0 | A | A | A | I | I | +1 |
| 2 | 1 | B | B | A | A | I | -1 |
| 3 | 1 | B | B | B | A | A | -1 |
| 4 | 2 | C | C | B | B | A | +1 |
| 5 | 2 | C | C | C | B | B | +1 |
| 6 | 3 | D | D | C | C | B | -1 |
| 7 | 3 | D | D | D | C | C | -1 |

처음 d개 행만 I를 사용하고, 이후에는 보관된 issued command를 선택한다.이 표는 command-level 실행 검증이며 controller 또는 robot motion 측정은 아니다. Zero arm command는 현재 pose를 목표로 삼는 hold 명령이다. 무토크·모터 정지가 아니다. Gripper는 그대로 움직일 수 있다.

[R,H] 주파수가 유지된다면 `T_delay=d/20`초이므로 0/1/2/3은 0/50/100/150 ms다. Delay1은 정상 policy period의 절반이고 한 decision 동안 이전 명령과 현재 명령이 혼합된다. Delay2는 정상 두-native decision에서 직전 decision의 arm command가 두 번 실행된다. 조기 종료가 있으면 decision↔native 변환을 단순히 2배로 계산하지 않는다.

이 구현은 **controller에 도착하기 전 command transport/execution scheduling delay**이다. 관측을 늦추지 않으므로 sensor latency가 아니며 motor의 1차 lag `τ_m dx/dt+x=u` 같은 response filter도 아니다. 지연되는 것은 과거 target pose가 아니라 정규화 action이다. 그것을 현재 EE pose에 적용해 새 target을 계산한다.

### Predictor 입력과 Markov property

[C] `adapt_shift.py:49–56`에서 `adapter.observe(obs, action, next_obs, len(records))`의 action은 issued command다. `common/world_model.py:116–123`의 predictor는 `(z,a)`를 사용한다. `factory.py:11–13`은 shift parameter/executed action을 agent에게 노출하는 설정을 거부한다. Queue는 log/snapshot에 있으나 observation에는 추가되지 않는다.

\[
P(s_{t+1}\mid s_t,a_t,H_t),\qquad H_t=(a_{t-d},\ldots,a_{t-1})
\]

가 필요한 상황에서 현재 모델은 이력을 명시적으로 받지 않는 `P(s_{t+1}|s_t,a_t)`를 근사한다. 동일한 physical state와 현재 issued command도 pending history에 따라 다른 transition을 만들 수 있다. 특히 delay2에서 현재 issued arm action은 그 decision의 두 native transition에 즉시 사용되지 않는다. 한편 기본 state observation도 물체 속도 전체를 담지 않으므로 delay 이전부터 엄밀한 full-state Markov 관측이라고 단정하지 않는다.

따라서 delay2 성능 저하에는 제어 지연, arm–gripper 시간 불일치, history 누락, predictor/가치함수 mismatch가 모두 가능한 원인이다. 현재 집계표로 원인 비율을 분리할 수 없으며 **UNKNOWN**이다. 구분하려면 짧은 고정 trajectory에서 동일 물리 상태·상이한 pending history의 transition 차이를 기록하고, 별도 향후 연구에서 history-aware oracle의 예측 상한을 비교한다. 이 보고서는 새 adaptation 방법을 구현하지 않는다.

![Delay](/Users/jihyekim/Desktop/tdmpc2/analysis/physical_dynamics/delay_native_step_timeline.png)

## 7. Object Mass code/physics walkthrough

### 적용과 실제 수치의 근거

[C] `physics_shift.py:24–37`에서 actor의 mass, principal inertia, COM pose를 읽어 baseline으로 저장한다. `apply:58–88`은 builder로 새 actor를 만든다.

```python
builder.set_mass_and_inertia(baseline_mass * factor,
                            baseline_cmass,
                            baseline_inertia * factor)
```

[H] 과거 서버의 nominal readback은 **0.0640000030 kg**이다(`DYNAMICS_SHIFT_AUDIT.md:20`). 아래 kg 값은 그 readback에 이번 코드의 배율을 적용한 계산이며 현재 서버에서 다시 읽은 수치는 아니다.

| factor | mass (kg, 역사적 baseline 기반) | principal inertia | 중력 하중 비율 | 물리적 의미 |
|---:|---:|---|---:|---|
| 1.0 | 0.0640000030 | I₀ | 1 | nominal |
| 1.5 | 0.0960000045 | 1.5 I₀ | 1.5 | 같은 geometry에서 질량·회전 저항 증가 |
| 2.0 | 0.1280000060 | 2 I₀ | 2 | 동일 net force에 translational acceleration 절반 |
| 3.0 | 0.1920000090 | 3 I₀ | 3 | 192 g 물체, 동일 torque에 angular acceleration 감소 |

[R] `pick_cube.py:21–26`, `base_env.py:29–47`는 half-size .02 m의 box collision/visual을 만든다. 변 길이 .04 m cube이다. [T] 균질 cube라면 `Ixx=Iyy=Izz=m*(.04²+.04²)/12`이므로 nominal **1.7066667e-5 kg·m²**, factor1.5/2/3은 각각 2.5600001e-5 / 3.4133335e-5 / 5.1200002e-5다. 이것은 균질 box 공식에 대입한 값이며 inertia getter의 신규 측정값이 아니다. Current nominal inertia vector와 COM pose는 [U]다. 코드가 기존 COM를 보존하므로 임의의 COM=0 측정을 만들어 쓰지 않는다.

### 질량이 움직임에 주는 영향

\[
m\ddot x=F_{contact}+mg+F_{other},\quad
I\dot\omega+\omega\times I\omega=\tau_{contact}.
\]

같은 net force일 때 a는 1/m로 줄지만 자유낙하의 `F=mg`도 같이 커지므로 자유낙하 가속도는 mass-independent다. 무조건 mass3이면 모든 동작의 가속도가 1/3이라는 해석은 잘못이다. Grasp 중에는 하중 mg, 필요한 접선 접촉력, 로봇에 전달되는 `J^T f`가 증가한다. 동일 자세·중력 방향이면 payload가 만드는 quasi-static joint load 성분은 mass에 선형적이지만 robot 자체 하중, Jacobian, contact geometry가 있어 전체 joint torque가 3배는 아니다.

[U] 실제 접촉 normal force와 joint drive torque/saturation 로그가 없으므로 192 g이 구동 한계에 얼마나 가까운지 계산할 수 없다. Reference force_limit=100을 물체 질량과 곧바로 비교하거나 두 손가락의 최대 법선력=200 N으로 단정해서는 안 된다. 목표 위치 오차, 관절 constraint, 접촉 patch가 실제 힘을 결정한다.

### Actor replacement의 보존 범위

[C] `physics_shift.py:64–88`은 pose/linear velocity/angular velocity 저장 → shape collision groups 저장 → builder.build → shape 수 검사 → groups/pose/vel 복원 → `env.obj`와 `_actors` cache의 동일 객체 slot 교체 → old actor 제거 순서다. Reset은 nominal restore → native reset → schedule sample → factor apply → 새 observation 생성이다(`140–146`). Step 중 mass를 바꾸지 않으며 `mid_episode_supported=False`다.

Geometry와 material은 retained builder에서 재사용하지만 production 코드는 전체 geometry/material equality를 assert하지 않는다. COM translation/orientation와 mass/inertia readback은 검증한다. Actor ID, 외부 stale reference, nested cache, contact warm-start, sleeping state, solver 내부 상태까지 동일하다는 보장은 없다. nominal factor=1은 rebuild 없이 지나가지만 shifted factor는 새 actor 생성 경로를 탄다는 점도 구분해야 한다.

[H] 이전 감사는 shallow reference와 shape 비교, 3개 seed의 보존/복원, seed17 contact response 차이를 보고했다. 신규 모든 trajectory의 완전 인증은 아니다. [O] factor3 성공 100%, length6.1은 충분한 grasp/torque margin, feedback 보정, task의 짧은 하중 구간과 양립한다. 이 후보들 중 무엇이 주원인인지 현재 수치로 확정하지 않는다. Readback만으로 구현 문제를 완전히 배제할 수도 없다.

**판정:** mass+inertia scaling은 같은 형상의 밀도/하중 변화를 나타내는 유효한 물리 intervention이다. KEEP하되 이름/metadata에 mass와 inertia 동시 scaling을 적고, task success 외에 lift acceleration·hold slip·하중 margin을 평가한다.

![Mass](/Users/jihyekim/Desktop/tdmpc2/analysis/physical_dynamics/mass_physical_response.png)

## 8. Contact Friction code/physics walkthrough

### 변경되는 표면

[C] YAML의 `actor: obj, surfaces: ['@plane']`와 `physics_shift.py:39–54`는 object actor의 모든 collision shape와 **PlaneGeometry를 가진 유일한 actor**의 모든 collision shape를 선택한다. plane을 외형상 테이블 mesh와 혼동하면 안 된다. 명시적인 robot finger/link 선택은 없다. plane이 0개 또는 여러 개면 실패한다.

각 shape의 원래 material과 static/dynamic/restitution 값을 저장한다. 새 material을 shape마다 만들어 배정하므로 기존 shared material 객체를 직접 수정하지 않는다. 물체/지면/robot이 원래 material을 공유했는지는 현재 live identity 조사 전 [U]지만, 원래 material을 in-place 변경해서 finger까지 함께 바꾸는 구현은 아니다.

```python
material = scene.create_physical_material(mu_s0 * f, mu_d0 * f, restitution0)
shape.set_physical_material(material)
```

[R] `sapien_env.py:584–585`의 scene default는 static=dynamic=1.0이고 cube/ground 생성은 별도 physical material을 지정하지 않는다. Panda finger는 `defaults.py:12–19`의 static=dynamic=2.0이다. 아래는 **공식 reference 기본값 기준**이며 최신 서버 shape getter 측정은 [U]다.

| factor | object μs / μd | plane μs / μd | finger μs / μd | 변경 표면 |
|---:|---|---|---|---|
| 1.0 | 1.0 / 1.0 | 1.0 / 1.0 | 2.0 / 2.0 | nominal |
| 0.7 | 0.7 / 0.7 | 0.7 / 0.7 | 2.0 / 2.0 | object+plane |
| 0.5 | 0.5 / 0.5 | 0.5 / 0.5 | 2.0 / 2.0 | object+plane |
| 0.3 | 0.3 / 0.3 | 0.3 / 0.3 | 2.0 / 2.0 | object+plane |

### Coulomb friction와 effective coefficient

\[
|F_t|\leq\mu_s N,\qquad |F_{slide}|\approx\mu_d N.
\]

Static friction은 미끄러지지 않는 접촉에서 허용되는 접선력 한계, dynamic friction은 미끄러질 때 저항을 나타낸다. Gripper가 수직 cube 양쪽을 잡고 외력이 작다고 근사하면 `μ_eff(N_left+N_right)≥mg`가 정적 유지의 필요 조건이다. 실제 다점 접촉·torsional friction·가속도까지 충분히 설명하는 완전한 grasp 모델은 아니다. Table 위에서는 법선력이 주로 하중과 연결되고, finger에서는 gripper 압착과 연결된다.

PhysX는 두 재질을 결합한다. 공식 [PxCombineMode 설명](https://nvidia-omniverse.github.io/PhysX/physx/5.4.1/_api_build/structPxCombineMode.html)은 average/min/multiply/max를 구분한다. 이 문서는 결합 원리의 자료이며 SAPIEN2 binary의 combine mode 인증 자료는 아니다. Wrapper readback의 `combine_rule='PhysX default average; ...'`는 **하드코딩된 설명 문자열**이다(`physics_shift.py:114–118`). getter 측정으로 해석하지 않는다.

[T] Average이고 reference 값이 맞다는 조건에서:

\[
\mu_{obj-plane}(f)=(f+f)/2=f,
\quad\mu_{obj-finger}(f)=(f+2)/2.
\]

| factor | object-plane μeff | object-finger μeff | finger 접촉의 nominal 대비 잔존 비율 |
|---:|---:|---:|---:|
| 1 | 1 | 1.5 | 100% |
| .7 | .7 | 1.35 | 90% |
| .5 | .5 | 1.25 | 83.3% |
| .3 | .3 | 1.15 | 76.7% |

따라서 material factor .3을 finger–object 마찰 70% 감소로 보고하면 안 된다. 이 계산에서는 감소가 23.3%다. 실제 combine mode나 material이 다르면 표도 달라진다. 양쪽 재질을 동시에 같은 factor로 바꿀 경우 average에서도 pair coefficient가 같은 factor로 줄지만, 현재 구현은 finger를 바꾸지 않는다.

### Frozen 100%와 task 평가 한계

[O] .3/.5 모두 10/10 성공이다. 평균 length는 모두6.1이다. [R] PickCube success는 물체-목표 거리≤.025 m와 arm qvel 최대 절댓값≤.2 rad/s이며, success 즉시 done이다(`pick_cube.py:68–83`, `sapien_env.py:564–566`). 명시적인 지속 hold 시간, object slip rate, grasp force margin 조건이 아니다. 잠깐 목표에 도달한 뒤 미끄러지는지 오래 관찰하지 않는다.

낮은 material factor에서도 pair friction 감소가 제한적이거나 normal force 여유가 큰 경우 성공률이 유지될 수 있다. 그러나 현재 policy의 slip/contact trajectory는 없어서 **가설**이다. [H] 이전 seed17 controlled-contact probe의 차이는 wrapper가 접촉 응답을 바꿀 수 있다는 제한적 증거이며, 현재 pilot grasp의 slip 정도를 알려주지는 않는다.

권고는 object-only / plane-only / finger-only를 구분해 결과를 기록하는 MODIFY이다. 표면 분리는 원인 식별을 위한 것이며 숫자를 더 극단적으로 만들기 위한 것이 아니다. Pair coefficient와 slip onset을 확인한 뒤 유효 범위를 선택한다.

![Friction](/Users/jihyekim/Desktop/tdmpc2/analysis/physical_dynamics/friction_contact_response.png)

## 9. Severity values, units and literature evidence

### 현재 수치의 출처

`common/severity_protocol.py:8–10`이 pilot 배열을 정의한다. Git commit `196cadb`(2026-11-05, ADD:compare shift severity on frozen)에 추가됐다. 해당 commit의 코드에는 논문에 근거한 숫자 산정 설명이 없다. 문서 `docs/stage3b_severity.md:49–64`는 탐색 pilot로 설명한다.

| 구분 | 분류 | 근거와 한계 |
|---|---|---|
| gain [.9,.7,.5,.3], delay [0,1,2,3], mass [1,1.5,2,3], friction [1,.7,.5,.3] | Engineering heuristic | 구현상 pilot grid. 정확한 간격 선택 이유는 Unsupported/UNKNOWN |
| 기존 reference .7 / 1 / 1.5 / .5 | Empirical calibration을 통한 유지 근거 존재 | `docs/stage3b_shift_selection.md`가 과거 감사 결과를 인용해 유지. 최초 선택의 문헌 근거는 확인되지 않음 |
| 최신 severe 조건 .3 / 2 / 3 / .3 | 기존 pilot 관측으로 사후 특성화 가능 | 특정 성능을 얻으려고 원래 선택했다는 의도는 문서만으로 입증되지 않음 |
| 관련 논문 범위 | Source-backed 비교 자료 | 아래 범위가 현재 grid의 역사적 출처라는 증거는 없음 |

`docs/stage3b_shift_selection.md:14–30`에는 표 중간에 shell output/traceback이 섞인 문서 손상이 있다. 이 부분의 수치를 근거로 재인용하지 않고 원 CSV를 우선했다. 기존 파일은 변경하지 않았다.

### 1차 자료 비교

| Source | Environment | Shift variable / 종류 | 확인한 range | Time scale | Evaluation purpose |
|---|---|---|---|---|---|
| [CARL 공식 문서](https://automl.github.io/CARL/main/source/environments/environment_families/classic_control.html) | CartPole 등 contextual RL | 실제 mass, gravity, force magnitude | CartPole masscart bounds .1–10, masspole .01–1; **허용 bounds이지 논문 pilot distribution이 아님** | context별 episode 구성 | context generalization/adaptation |
| [RWRL 공식 repo](https://github.com/google-research/realworldrl_suite), [example](https://github.com/google-research/realworldrl_suite/blob/master/examples/run_ppo.py) | dm_control cartpole/walker 등 | action/observation/reward delay를 구분; physical perturbation도 지원 | example `actions=20`; 모든 task의 권장 범위가 아님 | delay는 env step; physical schedule 별도 | 현실적 제약의 영향 평가 |
| [NS-Gym 공식 문서](https://nsgym.io/), [CartPole parameters](https://nsgym.io/env_pages/classic_control/cartpole.html) | Gymnasium classic/MuJoCo 등 | mass/gravity/force 등과 update scheduler | CartPole default masspole=.1, gravity=9.8, force=10, dt=.02; 보편적인 severity range는 없음 | 설정한 schedule에 따라 episode 내 변경 가능 | 비정상 MDP의 탐지·적응·계획 |
| [RoboTwin-Phys Appendix A](https://arxiv.org/html/2609.26292v1#A1) | 50-task bimanual manipulation | mass+inertia, friction, damping 등 실제 물리 속성 | mass ×.125–4; friction .01–1.25; arm/gripper damping ×.5–3 | episode 시작 샘플, rollout 중 고정 | 물리 조건 다양성과 정책 robustness |
| [RMA Table I](https://roboticsproceedings.org/rss17/p011.pdf) | Unitree A1, RaiSim locomotion | payload/friction/motor strength, servo gains | train/test friction .05–4.5 / .04–6; payload0–6 /0–7 kg; motor strength .90–1.10 /.88–1.22 | resample probability train .004/test .01 per step; history50 steps | 빠른 motor/environment adaptation |
| [JEPA-TTT §8](https://arxiv.org/html/2610.00722v1) | PushT, Two-Room, Reacher, OGBench-Cube | action rotation, contact-motion mapping, state-dependent field, coupling, delay+axis routing | PushT90°; Reacher 최대155°/165°; Cube coupling 계수3, delay1 raw step+cyclic routing | shift를 500 test episodes 동안 고정 | frozen latent predictor의 persistent TTT |

RoboTwin-Phys는 task별 feasible domain과 expert feasibility 검사를 둔다. 이 설계 원칙은 유용하지만 .125–4를 PickCube에 그대로 이식할 근거는 아니다. RMA action은 12개 joint position target이고 TD-MPC2 PickCube는 3D Cartesian delta다. 두 환경의 motor strength 숫자를 command gain과 동치로 비교할 수 없다. CARL/MuJoCo friction의 파라미터 구성도 PhysX static/dynamic pair와 동일하지 않다.

JEPA-TTT Cube는 hover/reach task이며 yaw/gripper를 고정한다. Delay와 cyclic routing이 결합돼 있고 predictor는 latent/action context window를 사용한다. 현재 PickCube grasp 및 단일 `(z,a)` predictor와 직접적인 성능 비교를 하면 안 된다. 해당 논문의 synthetic action/contact mapping 변화는 실제 재질 변화와 구분해서 참고한다.

### 적절성 판단

현재 gain 범위는 command sensitivity 진단에 유지할 수 있고, delay0–3은 지연·부분관측 진단으로 유지할 수 있다. Mass1–3, friction1–.3은 readback·접촉 응답 검증을 위한 범위로 유지한다. 다만 mass/friction pilot가 천장 성능이므로 이 네 숫자만으로 TTT 개선 능력을 식별하기 어렵다. 새 숫자 범위는 제시하지 않는다. 먼저 하중/미끄러짐 응답을 측정하고, feasible task 조건에서 측정 가능한 차이가 나는지 결정한다.

![Severity](/Users/jihyekim/Desktop/tdmpc2/analysis/physical_dynamics/shift_severity_physical_comparison.png)

## 10. Physical response validation

> 후속 서버 audit 준비에서 probe를 강화했다. 모든 physics substep 접촉 수집, 숫자 벡터 검증, native tape provenance 검증, world-frame target 저장이 추가됐다. 아래 최초 collector의 마지막 substep 한계 설명은 이전 버전 기록이다. 현재 실행 상태와 검증 범위는 [PHYSICAL_RESPONSE_VALIDATION.md](/Users/jihyekim/Desktop/tdmpc2/PHYSICAL_RESPONSE_VALIDATION.md)를 우선한다. 신규 서버 측정은 아직 없다.

### 이번에 실행한 것과 실행하지 않은 것

| Probe | 실제 수행 | 확보 결과 | 남은 항목 |
|---|---|---|---|
| A gain | production `transform` AST에 5개 gain 실행 | 명령 값·gripper 불변·clip 후 결과 확인 | live IK target, joint response, EEF displacement |
| B delay | production `transform`에 delay0–3 ×8 native commands | 32개 history 선택과 현재 gripper 확인 | live target·물리 response |
| C mass | 코드 및 과거 감사 기록 검토 | 과거 nominal mass readback; 보존/복원 및 contact response 보고 | 현재 mass/inertia live 값, acceleration/lift/torque 측정 |
| D friction | 코드·reference 재질·과거 감사 검토 | material 적용 대상과 conditional combine 계산 | 현재 pair friction, slip, normal/tangential force |

`analysis/physical_dynamics/command_probe.json`은 이번 command 실행 결과다. **실제 simulator probe 결과 파일이 아니다.** `mass_physical_response`는 역사적 mass readback 기반 계산과 동일 net force 이론, `friction_contact_response`는 average combine 가정의 계산이다. 그래프 제목에 명시했다. 새로운 실측 EEF·접촉력 curve를 생성하지 않았다. 기존 CSV success plot도 물리 response 측정 그래프와 구분한다.

### 서버에서 실행할 최소 response probe

`scripts/probe_physical_response.py`를 추가했다. 기존 Linux `tdmpc2-ms2` 환경에서만 실행한다. 기존 severity NPZ의 native `commanded_action` 또는 연구자가 고정한 `(T,4)` NPY를 읽는다. **policy decision action을 그대로 native tape라고 넣으면 repeat 의미가 바뀌므로 금지**한다. 한 조건 최대80 native steps, 기본40이며 최대5조건이다. 새 output 디렉터리만 허용하고 기존 실험 파일을 수정하지 않는다. Controller target/qpos/qvel/EEF/object pose·velocity, contact point impulse, nominal 초기 kinematics, 설치 Python source와 version을 기록한다. 학습·checkpoint 로딩은 없다. 현재 로컬에서 syntax 검증만 했고 실제 SAPIEN 실행은 [U]다.

```bash
# 기존 서버 환경에서, 실제 native action tape 경로를 지정한다.
python scripts/probe_physical_response.py \
  --actions /ABSOLUTE/PATH/shifted-seed-4000.npz \
  --shift action_gain --values 1 .7 .3 --seed 4000 --max-native-steps 40 \
  --output /ABSOLUTE/NEW/physical-probe-gain
# delay: --shift action_delay --values 0 1 2 3
# mass: --shift object_mass --values 1 1.5 2 3
# friction: --shift contact_friction --values 1 .7 .5 .3
```

위 명령은 실행 예시이며 아직 실행하지 않았다.소스가 서버에 존재하는지 먼저 확인해야 한다. Gain에는 free-space reach가 포함된 tape, mass/friction에는 grasp→lift→hold가 포함된 짧은 nominal tape를 사용한다. 각 조건은 동일 seed와 고정 tape이며 policy feedback이 개입하지 않아 물리 변화와 policy 보정을 분리한다. Grasp/contact가 발생하지 않은 조건은 friction/grasp 검증 성공으로 판정하지 않는다. 배포된 scripted collector는 last physics substep의 contact만 보므로 native 구간 전체 impulse나 지속 slip을 완전히 관측하지 못한다.

Object acceleration은 두 native 경계 velocity 차분/50 ms로 얻는 **평균 가속도 추정**이다. Contact `impulse`는 N·s이며 force가 아니다. 마지막 substep impulse를 native 50 ms로 나눠 전체 구간 force라고 보고하면 안 된다. Slip은 TCP–object 상대 pose 변화만으로 단정하지 말고 실제 finger contact 유지와 rotation을 함께 확인해야 한다.

Probe를 평가하는 순서는 (1) actual setter/readback, (2) target 차이, (3) 실제 q/EE/object response, (4) contact/성공 지표다. `mass readback PASS`를 (4)의 인증으로 승격하지 않는다. 전체 paired-determinism 재감사는 범위에 넣지 않았다.

## 11. Frozen/TTT results reinterpretation

원본 입력은 `results/stage3b-generalization/severity_sweep/combined-set0-v1/summary.csv`와 `results/stage3c-report-20261108-170105/tables/stage3b_aggregated_results.csv`다. 모든 아래 TTT 행은 CSV의 **decision_ttt**, 1 seed set, milestone12800 native adaptation steps를 선택했다. 0-budget 행은 동일 TTT 평가 경로의 unadapted 기준이며 모든 값이 pilot와 일치했다. 파일 이름에 stage3c가 있어도 별도 native-step adaptation 결과로 재해석하지 않는다.

| Shift | Severity | Frozen pilot success | Frozen length (policy decisions) | TTT success @12800 | Actual physical effect / 해석 한계 |
|---|---:|---:|---:|---:|---|
| gain | .9 | 100% | 6.5 | 미수록 | command10% 감소, live 운동 미측정 |
| gain | .7 | 100% | 7.9 | 100% | command30% 감소, 성공률 천장 |
| gain | .5 | 90% | 20.1 | 미수록 | command50% 감소 |
| gain | .3 | 90% | 33.1 | 100% | command70% 감소, 시간 비용 증가 |
| delay | 0 | 100% | 6.1 | 미수록 | nominal |
| delay | 1 | 80% | 28.9 | 90% | native1 지연; arm/gripper 비동기 |
| delay | 2 | 40% | 77.2 | 10% | native2 지연; hidden history |
| delay | 3 | 0% | 100.0 | 미수록 | 모두 horizon 도달; 물리 failure 원인 미분리 |
| mass | 1 | 100% | 6.1 | 미수록 | nominal |
| mass | 1.5 | 100% | 6.1 | 100% | mass/inertia1.5배; current response 미측정 |
| mass | 2 | 100% | 6.6 | 미수록 | mass/inertia2배 |
| mass | 3 | 100% | 6.1 | 100% | mass/inertia3배; 성공률상 식별 여지 부족 |
| friction | 1 | 100% | 6.1 | 미수록 | nominal |
| friction | .7 | 100% | 6.1 | 미수록 | object/plane material70% |
| friction | .5 | 100% | 6.1 | 100% | pair 감소율은50%로 단정 불가 |
| friction | .3 | 100% | 6.1 | 100% | pair 감소율은70%로 단정 불가 |

모든 pilot rate의 분모는10이다. 90→100%는 성공1개 차이다. 독립 training/seed-set 반복이1개라 유의성이나 일반화 향상을 선언하지 않는다. Pilot와 TTT의 checkpoint/프로토콜 전체 provenance 일치도 raw metadata까지 재검증한 것은 아니다.

Delay2의 TTT success trajectory는 budgets0/800/1600/3200/6400/12800에서 **40/20/0/0/10/10%**이다. 개선이 아니라 악화가 관측된다. 이는 latent MSE 감소만으로 제어 개선을 보장하지 않는 사례일 수 있지만, 원인을 hidden history로 확정할 수는 없다. Frozen Q/reward/policy prior와 바뀐 predictor의 조합, rollout distribution도 대안이다.

Gain.3은 같은 budgets에서 **90/100/100/90/100/100%**이다. Mass3은 모두100%, friction.3은 **100/90/100/100/90/100%**다. 중간 checkpoint를 임의로 골라 최고 성공률만 보고하지 않는다. 누적 dense reward는 긴 episode에서 커질 수 있다. Delay3의 return179.22가 nominal20.31보다 높지만 success는0%이므로 return 증가를 성능 향상으로 해석하면 오류다.

## 12. Benchmark design problems

1. **Intervention 계층 혼합:** command mapping, temporal history, rigid-body parameter가 서로 다른데 같은 dynamics adaptation 점수로 평균하면 실패 원인과 개선 기여를 잃는다.
2. **Observation sufficiency:** delay pending state가 predictor에 없고 nominal state에도 object velocity가 생략된다. history 없는 predictor의 한계를 파라미터 업데이트만의 한계로 일반화할 수 없다.
3. **Severity의 물리 불일치:** gain.3, mass3, friction.3은 동일 난도가 아니며 effective contact coefficient도 material factor와 다를 수 있다.
4. **성공률 천장과 종결 규칙:** PickCube는 단기 목표 도달+arm 정지 조건이다. 지속 grasp 안정성을 따로 요구하지 않는다. Mass/friction의 적응 필요성이 잘 드러나지 않을 수 있다.
5. **Actor rebuild의 보장 범위:** setter readback과 geometry/reference 검증, 실제 contact response를 분리해야 한다.
6. **증거 provenance 간격:** 현재 로컬 코드, 과거 설치 기록, pilot source hashes, 현재 서버가 하나의 revision이라고 증명되지 않았다. Pilot CSV에서 delay의 source hash와 나머지 shifts의 source hash도 서로 다르므로 통합 실험 한 번으로 부르면 안 된다.
7. **소표본 재사용:** 같은10개 평가 seed를 severity 선택·method 평가에 반복 사용하면 선택 편향을 배제하기 어렵다. 기존 수치를 바꾸지 말고 향후 사전 고정한 final test seeds를 별도로 둔다.

## 13. Redesign recommendations

### A. 현재 ManiSkill2 benchmark의 최소 수정

| 항목 | 권고 | 검증 목표 | 비용/제약 |
|---|---|---|---|
| Gain | RENAME → EEF delta-command attenuation | 현재 bounds/target/응답을 분리 기록 | 낮음; checkpoint와 interface 유지 |
| Delay | SEPARATE → hidden command-delay benchmark | history 누락과 제어 지연을 구분한 해석 | 문서·로그 낮음; history-aware 모델은 별도 연구로 이번 범위 밖 |
| Mass | KEEP mass+inertia reset shift, response 평가 보강 | nominal kg/I/COM, lift/hold response와 하중 margin | 계측 중간; actor rebuild 검증 필요 |
| Friction | MODIFY → object/plane/finger별 intervention 명시 | pair coefficient와 contact slip onset | 중간; 새로운 결과는 별도 protocol version |
| Evaluation | 성공률+시간+최종 오차+hold/slip 진단 병기 | 천장에서도 물리/제어 차이를 식별 | 기존 success protocol은 보존, 진단 평가를 별도 보고 |
| Task | PickCube에서 우선 검증, 필요시 StackCube 별도 | 지속 접촉·정렬·release 안정성 | StackCube 전용 pretrained checkpoint 필요 여부 확인 |

`factory.py:15–18`은 mass/friction을 pick-cube에서만 허용한다. 따라서 task 이름만 StackCube로 바꾸는 것은 지원되지 않는다. StackCube의 object 두 개, actor 참조, 대상 표면, 보상/종료 조건을 별도로 audit해야 한다. PickCube checkpoint를 그대로 StackCube에서 정상 정책이라고 가정하지 않는다. 동일 task에서 wrapper/계측만 보강하는 경로는 기존 checkpoint를 가장 많이 재사용한다.

모터 성능 연구가 필수라면 nominal PD controller를 유지하고 drive force limit 또는 drive gain 변화를 별도 조건으로 설계한다. Force limit 감소를 torque effectiveness loss와 동일시하지 않는다. 실제 정상 trajectory의 saturation/하중을 먼저 읽고 그 의미에 따라 범위를 정한다. 현재 보고서는 새로운 수치 grid나 장기 학습 실행을 제안·수행하지 않는다.

### B. 다른 benchmark로 이전

| 방향 | 기술적 타당성 | 재현성 | 연구 적합성 | 구현 비용·nominal 학습 |
|---|---|---|---|---|
| 현 PickCube 개선 | 현재 코드/runner 연결 유지 가능 | 설치 source·asset·checkpoint hash 보강 필요 | manipulation 물리 연구에 적합하나 task 민감도 확인 필요 | 낮음~중간; 동일 interface면 기존 checkpoint 유지 |
| StackCube | task adapter/physics target 확장 필요 | 두 actor reset/cache 관리 필요 | grasp, stacking, release 안정성 조사에 유용 | 중간; compatible StackCube checkpoint 없으면 nominal 재학습 |
| NS-Gym continuous MuJoCo | Gymnasium API와 scheduler 연결 가능 | env/version/update function/clock 고정 필요 | episode 내 non-stationarity를 명시적으로 연구하기 좋음 | 중간~높음; PickCube checkpoint 재사용 불가 |
| NS-Gym discrete CartPole 등 | 현 continuous MPC action과 직접 호환 안 됨 | 단순 환경이라 원인 계측 쉬움 | 식별성 sanity check에는 유용 | action/planner 변경이 필요할 수 있어 이번 제약 밖 |
| RoboTwin-Phys 등 manipulation | action/obs/robot/simulator adapter 필요 | task별 feasible set와 asset pinning 필요 | 물리 diversity 연구에 강점 | 높음; 기존 PickCube 가중치의 즉시 재사용 근거 없음 |

TTT runner 이식에서는 reset/step 반환형, observation flatten, action normalization, terminated/truncated, native clock/repeat, success 정의, replay의 issued/executed 의미를 모두 맞춰야 한다. 단순 환경 wrapper 교체만으로 기존 reward/Q/encoder가 유효해지지는 않는다. **권고 우선순위는 A의 계측·분류 개선 → 민감도 판단 → 필요할 때만 B**다.

## 14. Unknowns and limitations

- 최신 서버 SSH 대상과 현재 revision/dirty diff, 실제 설치 controller source 및 SAPIEN binary 동일성: UNKNOWN.
- 현재 checkpoint file, 학습 당시 controller·action scaling provenance: UNKNOWN.
- 현재 nominal mass/inertia/COM/material getter 값: 신규 readback 없음. mass는 역사적 측정, friction은 공식 reference 값으로 구분.
- 현재 drive target velocity, mode, 실제 force limit 포화율, object contact force: UNKNOWN.
- 현재 effective friction combine mode 및 finger–object slip: UNKNOWN. 하드코딩 문자열을 readback으로 취급하지 않음.
- 기존 mass/friction 성공100%의 인과적 원인: UNKNOWN. 충분한 margin/feedback/종료 기준/구현 문제 중 일부를 집계표만으로 확정하지 않음.
- 신규 simulator response probe는 미실행. 제공 collector도 real simulator 검증 대기이며 syntax 검증만 수행.
- 시각화6종은 구조도, 실제 command 함수 실행, 명시적 조건부 이론, 기존 CSV 관측으로만 구성. 추측한 실험 trajectory 없음.

최종 서버 기준 보고서를 완성하려면 (1) 서버 HEAD와 해당 파일 hash, (2) 설치 controller config/source 및 action/obs space readback, (3) 기존 severity NPZ/JSON, (4) 필요한 경우 위 짧은 prescribed-action response 결과가 필요하다. 이 자료를 확보하기 전에는 본 문서의 [R]/[H]를 [C]/신규 실측으로 바꾸지 않는다.

## 15. References and exact code locations

로컬 소스는 아래 자동 생성 위치 목록을 참고한다. 보고서 본문의 [R] 경로는 모두 `analysis/physical_dynamics/reference_ms2_0_4_1_post0/mani_skill2/` 아래의 공식 배포본 복사본이다. 현재 서버 site-packages 경로가 아니다. `analysis/physical_dynamics/source_sha256.json`에 분석 파일과 입력 CSV의 hash를 보존한다.

외부 근거는 본문 표의 1차 문서/논문 링크 및 [ManiSkill2 공식 PyPI 배포](https://pypi.org/project/mani-skill2/0.4.1.post0/)다. 문헌의 숫자는 관련 조건 비교에만 쓰며 현재 severity의 원래 출처로 소급 주장하지 않는다.

### 코드 위치 인덱스 (현재 파일에서 AST로 추출)


**tdmpc2/tdmpc2.py**

- [__init__, L17–43](/Users/jihyekim/Desktop/tdmpc2/tdmpc2/tdmpc2.py:17)
- [act, L98–120](/Users/jihyekim/Desktop/tdmpc2/tdmpc2/tdmpc2.py:98)
- [_plan, L143–219](/Users/jihyekim/Desktop/tdmpc2/tdmpc2/tdmpc2.py:143)

**tdmpc2/common/world_model.py**

- [__init__, L17–36](/Users/jihyekim/Desktop/tdmpc2/tdmpc2/common/world_model.py:17)
- [encode, L105–114](/Users/jihyekim/Desktop/tdmpc2/tdmpc2/common/world_model.py:105)
- [next, L116–123](/Users/jihyekim/Desktop/tdmpc2/tdmpc2/common/world_model.py:116)

**tdmpc2/envs/maniskill.py**

- [__init__, L35–44](/Users/jihyekim/Desktop/tdmpc2/tdmpc2/envs/maniskill.py:35)
- [reset, L46–47](/Users/jihyekim/Desktop/tdmpc2/tdmpc2/envs/maniskill.py:46)
- [step, L49–73](/Users/jihyekim/Desktop/tdmpc2/tdmpc2/envs/maniskill.py:49)
- [make_env, L86–117](/Users/jihyekim/Desktop/tdmpc2/tdmpc2/envs/maniskill.py:86)

**tdmpc2/envs/wrappers/tensor.py**

- [__init__, L13–14](/Users/jihyekim/Desktop/tdmpc2/tdmpc2/envs/wrappers/tensor.py:13)
- [reset, L34–35](/Users/jihyekim/Desktop/tdmpc2/tdmpc2/envs/wrappers/tensor.py:34)
- [step, L37–42](/Users/jihyekim/Desktop/tdmpc2/tdmpc2/envs/wrappers/tensor.py:37)

**tdmpc2/envs/wrappers/timeout.py**

- [__init__, L9–11](/Users/jihyekim/Desktop/tdmpc2/tdmpc2/envs/wrappers/timeout.py:9)
- [reset, L17–19](/Users/jihyekim/Desktop/tdmpc2/tdmpc2/envs/wrappers/timeout.py:17)
- [step, L21–25](/Users/jihyekim/Desktop/tdmpc2/tdmpc2/envs/wrappers/timeout.py:21)

**tdmpc2/envs/wrappers/dynamics_shift/action_shift.py**

- [__init__, L10–50](/Users/jihyekim/Desktop/tdmpc2/tdmpc2/envs/wrappers/dynamics_shift/action_shift.py:10)
- [reset, L52–56](/Users/jihyekim/Desktop/tdmpc2/tdmpc2/envs/wrappers/dynamics_shift/action_shift.py:52)
- [transform, L58–73](/Users/jihyekim/Desktop/tdmpc2/tdmpc2/envs/wrappers/dynamics_shift/action_shift.py:58)

**tdmpc2/envs/wrappers/dynamics_shift/physics_shift.py**

- [__init__, L13–21](/Users/jihyekim/Desktop/tdmpc2/tdmpc2/envs/wrappers/dynamics_shift/physics_shift.py:13)
- [bind, L23–54](/Users/jihyekim/Desktop/tdmpc2/tdmpc2/envs/wrappers/dynamics_shift/physics_shift.py:23)
- [apply, L56–106](/Users/jihyekim/Desktop/tdmpc2/tdmpc2/envs/wrappers/dynamics_shift/physics_shift.py:56)
- [readback, L108–118](/Users/jihyekim/Desktop/tdmpc2/tdmpc2/envs/wrappers/dynamics_shift/physics_shift.py:108)
- [__init__, L128–138](/Users/jihyekim/Desktop/tdmpc2/tdmpc2/envs/wrappers/dynamics_shift/physics_shift.py:128)
- [reset, L140–146](/Users/jihyekim/Desktop/tdmpc2/tdmpc2/envs/wrappers/dynamics_shift/physics_shift.py:140)
- [transform, L153–154](/Users/jihyekim/Desktop/tdmpc2/tdmpc2/envs/wrappers/dynamics_shift/physics_shift.py:153)
- [step, L156–160](/Users/jihyekim/Desktop/tdmpc2/tdmpc2/envs/wrappers/dynamics_shift/physics_shift.py:156)

**tdmpc2/envs/wrappers/dynamics_shift/base.py**

- [__init__, L10–23](/Users/jihyekim/Desktop/tdmpc2/tdmpc2/envs/wrappers/dynamics_shift/base.py:10)
- [reset, L49–56](/Users/jihyekim/Desktop/tdmpc2/tdmpc2/envs/wrappers/dynamics_shift/base.py:49)
- [transform, L77–78](/Users/jihyekim/Desktop/tdmpc2/tdmpc2/envs/wrappers/dynamics_shift/base.py:77)
- [step, L83–125](/Users/jihyekim/Desktop/tdmpc2/tdmpc2/envs/wrappers/dynamics_shift/base.py:83)

**tdmpc2/envs/wrappers/dynamics_shift/factory.py**

- [make_shift, L7–21](/Users/jihyekim/Desktop/tdmpc2/tdmpc2/envs/wrappers/dynamics_shift/factory.py:7)

**tdmpc2/adapt_shift.py**

- [episode, L29–62](/Users/jihyekim/Desktop/tdmpc2/tdmpc2/adapt_shift.py:29)
- [evaluate, L146–228](/Users/jihyekim/Desktop/tdmpc2/tdmpc2/adapt_shift.py:146)

**analysis/physical_dynamics/reference_ms2_0_4_1_post0/mani_skill2/agents/configs/panda/defaults.py**

- [__init__, L8–45](/Users/jihyekim/Desktop/tdmpc2/analysis/physical_dynamics/reference_ms2_0_4_1_post0/mani_skill2/agents/configs/panda/defaults.py:8)
- [__init__, L197–199](/Users/jihyekim/Desktop/tdmpc2/analysis/physical_dynamics/reference_ms2_0_4_1_post0/mani_skill2/agents/configs/panda/defaults.py:197)

**analysis/physical_dynamics/reference_ms2_0_4_1_post0/mani_skill2/agents/base_controller.py**

- [__init__, L23–48](/Users/jihyekim/Desktop/tdmpc2/analysis/physical_dynamics/reference_ms2_0_4_1_post0/mani_skill2/agents/base_controller.py:23)
- [set_drive_property, L85–87](/Users/jihyekim/Desktop/tdmpc2/analysis/physical_dynamics/reference_ms2_0_4_1_post0/mani_skill2/agents/base_controller.py:85)
- [reset, L89–91](/Users/jihyekim/Desktop/tdmpc2/analysis/physical_dynamics/reference_ms2_0_4_1_post0/mani_skill2/agents/base_controller.py:89)
- [set_action, L101–105](/Users/jihyekim/Desktop/tdmpc2/analysis/physical_dynamics/reference_ms2_0_4_1_post0/mani_skill2/agents/base_controller.py:101)
- [before_simulation_step, L107–109](/Users/jihyekim/Desktop/tdmpc2/analysis/physical_dynamics/reference_ms2_0_4_1_post0/mani_skill2/agents/base_controller.py:107)
- [__init__, L143–164](/Users/jihyekim/Desktop/tdmpc2/analysis/physical_dynamics/reference_ms2_0_4_1_post0/mani_skill2/agents/base_controller.py:143)
- [set_drive_property, L191–194](/Users/jihyekim/Desktop/tdmpc2/analysis/physical_dynamics/reference_ms2_0_4_1_post0/mani_skill2/agents/base_controller.py:191)
- [reset, L196–198](/Users/jihyekim/Desktop/tdmpc2/analysis/physical_dynamics/reference_ms2_0_4_1_post0/mani_skill2/agents/base_controller.py:196)
- [set_action, L200–202](/Users/jihyekim/Desktop/tdmpc2/analysis/physical_dynamics/reference_ms2_0_4_1_post0/mani_skill2/agents/base_controller.py:200)
- [before_simulation_step, L204–213](/Users/jihyekim/Desktop/tdmpc2/analysis/physical_dynamics/reference_ms2_0_4_1_post0/mani_skill2/agents/base_controller.py:204)
- [set_action, L237–244](/Users/jihyekim/Desktop/tdmpc2/analysis/physical_dynamics/reference_ms2_0_4_1_post0/mani_skill2/agents/base_controller.py:237)

**analysis/physical_dynamics/reference_ms2_0_4_1_post0/mani_skill2/agents/base_agent.py**

- [__init__, L50–79](/Users/jihyekim/Desktop/tdmpc2/analysis/physical_dynamics/reference_ms2_0_4_1_post0/mani_skill2/agents/base_agent.py:50)
- [reset, L153–159](/Users/jihyekim/Desktop/tdmpc2/analysis/physical_dynamics/reference_ms2_0_4_1_post0/mani_skill2/agents/base_agent.py:153)
- [set_action, L161–162](/Users/jihyekim/Desktop/tdmpc2/analysis/physical_dynamics/reference_ms2_0_4_1_post0/mani_skill2/agents/base_agent.py:161)
- [before_simulation_step, L164–165](/Users/jihyekim/Desktop/tdmpc2/analysis/physical_dynamics/reference_ms2_0_4_1_post0/mani_skill2/agents/base_agent.py:164)
- [get_proprioception, L170–175](/Users/jihyekim/Desktop/tdmpc2/analysis/physical_dynamics/reference_ms2_0_4_1_post0/mani_skill2/agents/base_agent.py:170)

**analysis/physical_dynamics/reference_ms2_0_4_1_post0/mani_skill2/agents/controllers/pd_ee_pose.py**

- [reset, L55–57](/Users/jihyekim/Desktop/tdmpc2/analysis/physical_dynamics/reference_ms2_0_4_1_post0/mani_skill2/agents/controllers/pd_ee_pose.py:55)
- [compute_ik, L59–71](/Users/jihyekim/Desktop/tdmpc2/analysis/physical_dynamics/reference_ms2_0_4_1_post0/mani_skill2/agents/controllers/pd_ee_pose.py:59)
- [compute_target_pose, L73–88](/Users/jihyekim/Desktop/tdmpc2/analysis/physical_dynamics/reference_ms2_0_4_1_post0/mani_skill2/agents/controllers/pd_ee_pose.py:73)
- [set_action, L90–109](/Users/jihyekim/Desktop/tdmpc2/analysis/physical_dynamics/reference_ms2_0_4_1_post0/mani_skill2/agents/controllers/pd_ee_pose.py:90)
- [compute_target_pose, L173–195](/Users/jihyekim/Desktop/tdmpc2/analysis/physical_dynamics/reference_ms2_0_4_1_post0/mani_skill2/agents/controllers/pd_ee_pose.py:173)

**analysis/physical_dynamics/reference_ms2_0_4_1_post0/mani_skill2/agents/controllers/pd_joint_pos.py**

- [set_drive_property, L27–38](/Users/jihyekim/Desktop/tdmpc2/analysis/physical_dynamics/reference_ms2_0_4_1_post0/mani_skill2/agents/controllers/pd_joint_pos.py:27)
- [reset, L40–44](/Users/jihyekim/Desktop/tdmpc2/analysis/physical_dynamics/reference_ms2_0_4_1_post0/mani_skill2/agents/controllers/pd_joint_pos.py:40)
- [set_action, L50–68](/Users/jihyekim/Desktop/tdmpc2/analysis/physical_dynamics/reference_ms2_0_4_1_post0/mani_skill2/agents/controllers/pd_joint_pos.py:50)
- [before_simulation_step, L70–76](/Users/jihyekim/Desktop/tdmpc2/analysis/physical_dynamics/reference_ms2_0_4_1_post0/mani_skill2/agents/controllers/pd_joint_pos.py:70)

**analysis/physical_dynamics/reference_ms2_0_4_1_post0/mani_skill2/envs/sapien_env.py**

- [__init__, L76–184](/Users/jihyekim/Desktop/tdmpc2/analysis/physical_dynamics/reference_ms2_0_4_1_post0/mani_skill2/envs/sapien_env.py:76)
- [get_obs, L255–267](/Users/jihyekim/Desktop/tdmpc2/analysis/physical_dynamics/reference_ms2_0_4_1_post0/mani_skill2/envs/sapien_env.py:255)
- [_get_obs_extra, L280–282](/Users/jihyekim/Desktop/tdmpc2/analysis/physical_dynamics/reference_ms2_0_4_1_post0/mani_skill2/envs/sapien_env.py:280)
- [reset, L467–481](/Users/jihyekim/Desktop/tdmpc2/analysis/physical_dynamics/reference_ms2_0_4_1_post0/mani_skill2/envs/sapien_env.py:467)
- [step, L531–540](/Users/jihyekim/Desktop/tdmpc2/analysis/physical_dynamics/reference_ms2_0_4_1_post0/mani_skill2/envs/sapien_env.py:531)
- [step_action, L542–558](/Users/jihyekim/Desktop/tdmpc2/analysis/physical_dynamics/reference_ms2_0_4_1_post0/mani_skill2/envs/sapien_env.py:542)
- [evaluate, L560–562](/Users/jihyekim/Desktop/tdmpc2/analysis/physical_dynamics/reference_ms2_0_4_1_post0/mani_skill2/envs/sapien_env.py:560)
- [get_done, L564–566](/Users/jihyekim/Desktop/tdmpc2/analysis/physical_dynamics/reference_ms2_0_4_1_post0/mani_skill2/envs/sapien_env.py:564)

**analysis/physical_dynamics/reference_ms2_0_4_1_post0/mani_skill2/envs/pick_and_place/pick_cube.py**

- [__init__, L19–22](/Users/jihyekim/Desktop/tdmpc2/analysis/physical_dynamics/reference_ms2_0_4_1_post0/mani_skill2/envs/pick_and_place/pick_cube.py:19)
- [_get_obs_extra, L54–66](/Users/jihyekim/Desktop/tdmpc2/analysis/physical_dynamics/reference_ms2_0_4_1_post0/mani_skill2/envs/pick_and_place/pick_cube.py:54)
- [evaluate, L76–83](/Users/jihyekim/Desktop/tdmpc2/analysis/physical_dynamics/reference_ms2_0_4_1_post0/mani_skill2/envs/pick_and_place/pick_cube.py:76)
- [_get_obs_extra, L135–144](/Users/jihyekim/Desktop/tdmpc2/analysis/physical_dynamics/reference_ms2_0_4_1_post0/mani_skill2/envs/pick_and_place/pick_cube.py:135)

**analysis/physical_dynamics/reference_ms2_0_4_1_post0/mani_skill2/envs/pick_and_place/base_env.py**

- [__init__, L24–27](/Users/jihyekim/Desktop/tdmpc2/analysis/physical_dynamics/reference_ms2_0_4_1_post0/mani_skill2/envs/pick_and_place/base_env.py:24)
- [_build_cube, L29–47](/Users/jihyekim/Desktop/tdmpc2/analysis/physical_dynamics/reference_ms2_0_4_1_post0/mani_skill2/envs/pick_and_place/base_env.py:29)

**analysis/physical_dynamics/reference_ms2_0_4_1_post0/mani_skill2/utils/common.py**

- [__init__, L49–51](/Users/jihyekim/Desktop/tdmpc2/analysis/physical_dynamics/reference_ms2_0_4_1_post0/mani_skill2/utils/common.py:49)
- [clip_and_scale_action, L124–128](/Users/jihyekim/Desktop/tdmpc2/analysis/physical_dynamics/reference_ms2_0_4_1_post0/mani_skill2/utils/common.py:124)
