# Physical-response-calibrated severity 실행 안내

현재 상태는 **도구 준비 완료 / 서버 측정 대기 / 최종 severity 미선정**이다. `pending/`의 CSV는 측정 전 후보 목록이고 PNG는 `NO SIMULATOR MEASUREMENTS` 표시만 있다. 빈 칸을 0으로 해석하지 않는다. 기존 이론 그래프를 측정 곡선으로 복사하지 않았다.

## 변경 파일과 배포

아래 파일을 서버 최신 코드에 반영한다. 기존 파일에 변경사항이 있으면 덮어쓰기 대신 해당 변경만 병합한다.

- `scripts/calibrate_physical_severity.py` — 준비/집계/선정 gate. 서버 실행을 자동 시작하지 않음.
- `scripts/probe_physical_response.py` — 기존 계측기에 repository source digest 추가.
- `tdmpc2/common/severity_evaluation.py` — 기존 `--log-eef`에서 실제 object-goal distance 추가 기록. 모델/행동/보상/종료/seed 설정 변경 없음.
- `tdmpc2/common/severity_protocol.py` — PILOT이 heuristic 후보라는 주석만 추가; 숫자는 그대로.
- `tests/test_physical_severity_calibration.py` — synthetic bookkeeping 검증, 물리 결과 아님.

모든 코드 반영을 마친 뒤 Frozen/probe를 실행한다. 두 실행 사이에 소스가 바뀌면 export가 source mismatch로 거부한다. 기존 Stage3B 디렉터리 대신 새 calibration 디렉터리를 사용한다.

## 서버 명령

```bash
conda activate tdmpc2-ms2
cd /nfs4/jhkim/repos/tdmpc2
export REPO=/nfs4/jhkim/repos/tdmpc2
export CHECKPOINT="$REPO/checkpoints/maniskill2/pick-cube-1.pt"

# 이미 존재하는 디렉터리는 거부한다. 재실행에는 새 경로를 사용한다.
python scripts/calibrate_physical_severity.py prepare \
  --output analysis/calibration-server-v1 \
  --calibration-set 70 --final-set 80 --seeds 3

# 필요 시 먼저 이전에 안내한 nominal 2-step API smoke를 수행한다.
# 아래는 Frozen 평가만 수행하며 TTT/optimizer update는 하지 않는다.
bash analysis/calibration-server-v1/run_frozen.sh

# 같은 seed의 nominal native tape를 모든 shift에 동일하게 replay한다.
bash analysis/calibration-server-v1/run_probes.sh

python scripts/calibrate_physical_severity.py export \
  --manifest analysis/calibration-server-v1/calibration_manifest.json \
  --output analysis/calibration-server-v1/measured-v1
```

기본 calibration seeds는 704000–704002, final seeds는 804000–804009이다. 이는 이번 계획의 예약값이지 과거 모든 서버 실험에서 미사용임을 확인한 값은 아니다. 과거 사용 내역을 확인하고 필요하면 실행 전에 set 번호를 바꾼다. Final seed로 실행하는 명령은 생성하지 않는다. 서로 겹치면 prepare/export가 실패한다.

기본 작업량은 nominal 포함 17개 조건 × 3 seeds의 probe(각 최대40 native steps, 총 최대2040)와, 기존 Frozen runner의 shifted+nominal 평가102 episodes이다. Frozen은 episode당 최대100 policy decisions이므로 최악 상한은20400 native steps다. 실제 조기 종료 시 더 짧다. 이 작업량은 학습 예산이 아니다. `--seeds 1`로 pipeline smoke를 준비할 수 있지만 최종 선정에 필요한 최소3 seeds를 충족하지 못한다.

## Tape의 유효성

생성된 probe script는 action_gain=1 Frozen run의 `nominal-seed-*.npz`를 사용한다. NPZ의 contiguous native timestep과 sidecar의 seed/native count를 검사한다. Policy decision action을 2회 반복 처리 없이 native tape로 넣지 않는다.

Nominal tape가 free-space reach와 grasp–lift를 포함한다고 자동 가정하지 않는다. Export는 접촉 및 lift coverage를 확인한다. Mass/Friction에서 양 finger 접촉과 lift가 부족하거나 friction plane/finger의 지속 접촉 구간이 없으면 PARTIAL로 남고 선정할 수 없다. 이때 사용자가 확인한 다른 짧은 nominal tape로 새로운 probe 디렉터리를 만들고 manifest와 연결 경로를 보존해야 한다. 실패한 조건을 감추거나 접촉 없는 값을 zero sensitivity로 채우지 않는다.

## 산출물

- `physical_severity_calibration.csv`: 후보×seed별 nominal/요청/effective 값, physical response, Frozen success/episode length/goal progress, provenance 및 선정 상태. 구조화된 evidence는 JSON cell이다.
- `physical_response_by_severity.png` / `.pdf`: shift별 서로 다른 단위의 측정 변화량. 동일 seed를 선으로 연결하며 불확실성 CI를 주장하지 않는다.
- `severity_selection.md`: 측정/선정 상태와 해석 제약.
- `calibrated_severity.yaml`: 최종 선정값만 포함. JSON 문법의 유효한 YAML1.2이며 기존 Hydra config에 자동 적용하지 않는다. 네 shift의 선정 근거가 모두 충족되기 전 `ready_for_benchmark=false`다.

## 최종 선정 review

자동으로 가장 실패율이 높은 값을 고르지 않는다. 실제 측정을 검토한 뒤 manifest의 `selection_reviews`에 후보별 결정을 기록하고, **새 output 디렉터리**로 export한다. 아래는 schema 설명이며 숫자 threshold를 추천하는 예가 아니다.

```json
{
  "action_gain:0.7": {
    "response_metric": "free_space_eef_rmse_m",
    "minimum_absolute_response": null,
    "threshold_rationale": "측정 해상도, 관측된 효과 및 연구 목표에 근거한 기준 설명",
    "minimum_frozen_success_rate": null,
    "rationale": "명령 감쇠가 실제 움직임을 바꾸지만 제어 가능성을 유지하는 근거",
    "task_feasibility_rationale": "같은 calibration seeds에서 과제를 실제 수행한 근거",
    "feasibility_seed_set": [704000, 704001, 704002],
    "task_feasibility_evidence": "상대/또는/절대/실제근거파일.json"
  }
}
```

null threshold로는 선정되지 않는다. 연구자가 물리 단위에 맞게 기준을 기록해야 한다. 모든 calibration seed에서 해당 absolute response threshold를 충족하고, Frozen mean success가 명시한 기준을 충족해야 한다. 별도 feasible evidence 파일과 rationale도 필요하다. 파일 hash를 저장한다. 이는 물리적 의미에 대한 연구자 검토를 대체하지 않는다. Threshold 자체를 문헌에서 도출했다거나 통계적 유의성을 확보했다고 주장하지 않는다.

허용 metric:

| Shift | 물리 response 선정 metric |
|---|---|
| Gain | free_space_eef_rmse_m (target_rmse_m은 보조 지표) |
| Delay | eef_rmse_m 또는 eef_motion_onset_delta_s |
| Mass | object_acceleration_rmse_m_s2 또는 max_lift_delta_m |
| Friction | finger_relative_path_delta_m 또는 plane_relative_path_delta_m |

Gain은 target 변화만으로 선정할 수 없도록 검사한다. Measured EEF 변화가 선정 기준이고 target은 보조 지표다. Friction relative-path는 contact-conditioned 상대 운동 proxy다. 진정한 접선 slip, rolling, deformation을 모두 식별한 측정값은 아니다. Normal impulse의 크기를 마찰계수 추정이나 실제 force로 바꿔 쓰지 않는다.

## 검증 상태

로컬 synthetic 단위 검사8개 통과. 기존 severity 회귀 검사17개 중12개 통과,5개는 로컬 Hydra/OmegaConf 의존성 부재로 실행 오류였다. 실제 SAPIEN/CUDA 검증은 서버에서 수행해야 한다. 실험 결과·checkpoint·WorldModel/MPC/Q/Actor는 변경하지 않았다.
