# Dynamics Shift 재설계 결정 요약

2026-11-08. 기준: 로컬 HEAD `32d8730`, 공식 ManiSkill2 0.4.1.post0 reference, 과거 서버 감사 문서와 기존 결과 CSV. **최신 서버 revision 및 신규 simulator response 검증 전의 잠정 결정**이다. 상세한 근거·숫자·제약은 [분석 보고서](/Users/jihyekim/Desktop/tdmpc2/DYNAMICS_SHIFT_PHYSICAL_ANALYSIS.md)에 있다.

| Shift | Code Validity | Physical Validity | Research Suitability | Decision | Evidence |
|---|---|---|---|---|---|
| Action gain | 실제 transform의 5개 prescribed 입력 검증; arm slice·후단 clipping 확인 | EEF delta-command attenuation으로 타당. Torque effectiveness loss를 구현하지 않음 | 명령 mapping 적응에 적합; motor failure 근거로 부적합 | **RENAME**: EEF delta-command gain | `action_shift.py:19,57–73`; reference `pd_ee_pose.py:89–108`; gain.3 Frozen9/10, length33.1 |
| Action delay | 실제 transform으로 delay0–3 ×8 native steps의 history 선택 확인 | Arm command transport delay. Sensor delay나 motor lag가 아님 | hidden history와 arm/gripper 비동기 문제. 순수 물리 파라미터 적응과 별도 평가 필요 | **SEPARATE**: hidden command-delay benchmark | `action_shift.py:34–55,66–73`; `adapt_shift.py:49–56`; delay2 Frozen40%, decision_ttt@12800=10% |
| Object mass | mass·inertia·COM readback assertion 및 actor/cache 교체 확인. 최신 서버 미인증 | 같은 geometry의 mass+inertia scaling은 타당. 모든 solver state 보존은 보장하지 않음 | 물리 연구에는 적합하나 현재 PickCube 성공률 천장 | **KEEP**, response·hold 평가 보완 | `physics_shift.py:24–37,58–88,140–146`; 과거 nominal .064000003kg; mass3 Frozen10/10 |
| Contact friction | object+plane shape에 새 material 적용, finger 원래 material 불변 | Material perturbation으로 타당. Pair effective coefficient가 factor배인지 미확인 | 표면별 원인 분리와 slip 계측이 필요 | **MODIFY**: object/plane/finger 대상 분리 | `physics_shift.py:39–54,90–95,114–118`; reference finger μ=2; average 가정에서 f=.3의 pair 잔존76.7%; Frozen10/10 |

## 우선 실행 순서

1. 서버 repository HEAD/dirty diff와 설치 controller config/source를 확인한다. 현재 보고서의 reference 값과 대조하고 차이가 있으면 보고서를 수정한다.
2. 기존 severity NPZ/JSON을 먼저 읽는다. 부족한 target/contact 변수만 `scripts/probe_physical_response.py`의 짧은 prescribed-action probe로 기록한다. 장기 학습이나 전체 determinism 감사는 하지 않는다.
3. 기존 결과를 보존한 채 네 intervention의 이름·단위·clock·affected surfaces를 구분한다. 현재 severity 숫자는 진단 grid로 유지한다.
4. Mass/friction에서 readback→contact→response의 연결이 확인되면 hold/slip·완료 시간 지표를 별도 진단으로 추가한다. 기존 success 정의를 소급 변경하지 않는다.
5. 그래도 물리적 구분력이 부족하면 StackCube 등 작업을 확장한다. 현재 physics factory는 PickCube만 지원하므로 actor/reference audit와 compatible checkpoint 확인이 선행돼야 한다.

## 현 환경 수정과 이전의 선택

| 방향 | 기술·재현성 | 연구 적합성 | 비용 및 checkpoint |
|---|---|---|---|
| PickCube 최소 수정 | 기존 runner 유지, source/asset/runtime provenance 보강 | 가장 빨리 현재 benchmark 의미를 검증 가능 | 낮음~중간; 동일 interface면 현재 checkpoint 유지 가능 |
| StackCube 확장 | 두 object·대상 surface·reset cache 추가 검증 | stacking/release/접촉 안정성 연구 | 중간; 전용 checkpoint 없으면 nominal 재학습 |
| NS-Gym continuous 환경 | Gymnasium API·clock·schedule·replay adapter 필요 | episode 내 비정상성 연구에 적합 | 중간~높음; PickCube checkpoint 재사용 불가 |
| 다른 manipulation benchmark | robot/obs/action/reward/asset 차이 큼 | 물리 diversity가 목적이면 검토 가치 있음 | 높음; nominal 학습 및 TTT runner 이식 필요 |

**새 극단적 severity 범위는 권고하지 않는다.** 먼저 실제 하중·contact slip·제어 여유를 읽고, 물리적으로 가능하면서 차이를 측정할 수 있는 조건을 선택해야 한다. 문헌 범위는 비교 근거이며 현재 PickCube 수치의 자동 정당화가 아니다.

## 이번 산출물의 검증 상태

- Command-level 검증: PASS — delay32개, gain5개. 로봇 운동 검증과 구분한다.
- PNG/PDF 각6개: 구조도, command 실행, 조건부 물리 계산, 기존 pilot 관측으로 라벨 구분.
- 서버용 probe: 구문 검증 완료, 실제 SAPIEN 실행 미검증.
- 최신 서버/실측 force·slip/현재 checkpoint 학습 config: UNKNOWN.
- WorldModel/MPC/Q/Actor 및 기존 Stage2/3A/3B 결과: 변경하지 않음.
