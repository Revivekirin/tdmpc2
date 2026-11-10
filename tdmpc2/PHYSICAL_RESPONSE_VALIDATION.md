# Physical Response Validation — 서버 측정 완료

실제 Linux SAPIEN 측정 run `physical-calibration-20261108-203715`을 검토했다. 원시 probe, Frozen 평가, source provenance, measured CSV/PNG 및 shift별 판정은 [측정 검토 보고서](analysis/physical-calibration-20261108-203715/MEASURED_RESPONSE_REVIEW.md)에 기록했다.

| Shift | 판정 | 핵심 근거 |
|---|---|---|
| Gain | **VERIFIED** | 명령 감쇠와 별도로 measured EEF free-space 응답이 severity에 따라 변화. |
| Delay | **VERIFIED** | Native-step issued/executed 지연 및 measured EEF onset 시차 확인. |
| Mass | **PARTIAL** | Mass/inertia 및 지지 impulse 변화 확인. 2×/3×의 seed 704000에서 실제 grasp–lift 접촉 부족. |
| Friction | **PARTIAL** | Cube/ground material과 두 종류의 접촉 확인. 현재 상대 운동 지표만으로 순수 slip 변화와 benchmark severity를 확정할 수 없음. |

51 candidate–seed comparisons 중 49 `MEASURED`, 2 `PARTIAL`. 기존 [측정 그래프](analysis/physical-calibration-20261108-203715/measured-v1/physical_response_by_severity.png)는 이론 reference 그래프와 별도다. Mass의 두 `PARTIAL` 점도 계산값으로 그려져 있으므로 보고서의 coverage 제한을 함께 읽어야 한다. 최종 [severity YAML](analysis/physical-calibration-20261108-203715/measured-v1/calibrated_severity.yaml)은 아직 `ready_for_benchmark: false`다.
