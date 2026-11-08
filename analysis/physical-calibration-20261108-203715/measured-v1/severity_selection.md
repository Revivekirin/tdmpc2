# Physical-response severity 선정

기존 PILOT 값은 exploratory engineering heuristic이다. 이 산출물은 기존 config/결과를 변경하지 않는다. 측정 또는 선정 review가 없으면 selected_severities는 비어 있고 benchmark 실행용 확정값이 아니다.

| Shift | Candidate | Selection status |
|---|---:|---|
| action_delay | 0 | NOMINAL_REFERENCE |
| action_delay | 1 | PENDING_REVIEW |
| action_delay | 2 | PENDING_REVIEW |
| action_delay | 3 | PENDING_REVIEW |
| action_gain | 1 | NOMINAL_REFERENCE |
| action_gain | 0.9 | PENDING_REVIEW |
| action_gain | 0.7 | PENDING_REVIEW |
| action_gain | 0.5 | PENDING_REVIEW |
| action_gain | 0.3 | PENDING_REVIEW |
| object_mass | 1 | NOMINAL_REFERENCE |
| object_mass | 1.5 | PENDING_REVIEW |
| object_mass | 2 | PENDING_REVIEW |
| object_mass | 3 | PENDING_REVIEW |
| contact_friction | 1 | NOMINAL_REFERENCE |
| contact_friction | 0.7 | PENDING_REVIEW |
| contact_friction | 0.5 | PENDING_REVIEW |
| contact_friction | 0.3 | PENDING_REVIEW |

## 측정과 선정 기준

Calibration seeds: [704000, 704001, 704002]; final benchmark seeds (미실행): [804000, 804001, 804002, 804003, 804004, 804005, 804006, 804007, 804008, 804009].

- Gain: 접촉 전 후보 구간의 nominal 대비 measured EEF RMSE와 target RMSE를 별도 기록한다. Object 접촉 부재만으로 모든 robot collision 부재를 증명하지는 않는다.
- Delay: 실제 issued/executed history가 요청 native delay와 일치하는지 검사한다. 운동 onset은 측정 변위 threshold 초과 시점이며 motor causal latency 추정값이 아니다.
- Mass: native 경계 velocity 차분의 평균 가속도, lift 변화, 동시 양 finger 접촉 fraction을 기록한다.
- Friction: 지속된 동일 contact pair의 상대 이동 누적량을 finger와 plane으로 분리한다. 이는 deformation/rotation을 포함할 수 있는 relative-motion proxy이며 마찰력 또는 순수 sliding distance로 단정하지 않는다. Contact-free 값은 0이 아닌 missing이며 선정 불가다.
- Frozen success/episode length/goal-distance progress는 open-loop 물리 probe와 별도 CSV 필드에 저장한다. Reward 합을 task progress로 대체하지 않는다.
- Curve는 common native prefix를 사용하고 양쪽 길이를 기록한다. 종료 길이가 다르면 같은 관측 window 내 비교만 유효하다.
- 최소 3개 calibration seed의 완전한 physical+Frozen evidence, shift별 response threshold, 최소 Frozen success, 별도 feasibility 근거와 rationale가 있어야 선정한다. Threshold는 사용자/연구자가 기록하는 연구 기준이며 보편적인 물리 상수가 아니다. 성공률은 소표본 기술통계이며 통계적 유의성을 주장하지 않는다.
- Final seed는 export/selection에 읽지 않는다. 지정한 final seed가 과거 다른 실험에서 사용됐는지 추가 확인해야 한다.
