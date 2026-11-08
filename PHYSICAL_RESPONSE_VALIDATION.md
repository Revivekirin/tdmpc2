# Physical Response Validation — 서버 실행 대기

## 현재 상태

**미완료: 실행할 SSH 호스트가 아직 확정되지 않아 서버 소스 조회와 simulator smoke를 실행하지 않았다.**

지정된 repository는 `/nfs4/jhkim/repos/tdmpc2`다. 로컬 SSH 설정에는 `jhkim@10.20.22.81:53423`, `jhkim@10.20.22.107:55125`, `jhkim@10.20.22.87:4091`이 있으며 어느 호스트가 이번 실행 대상인지 사용자에게 확인 요청했다. 이 목록은 연결 성공이나 repository 존재를 확인한 결과가 아니다.

## 판정

여기서 UNSUPPORTED는 **이번 서버 실행의 측정 증거가 없음**을 의미하며, intervention이 물리적으로 무효이거나 코드가 실패했다는 판정이 아니다. 과거 감사 및 이론 reference는 신규 측정을 대체하지 않는다.

| Shift | 판정 | 확보한 이번 서버 측정 | 연구 목적 적합성 판단 |
|---|---|---|---|
| Gain | UNSUPPORTED | 없음 | Command attenuation의 실제 target/EEF 응답 차이 확인 대기 |
| Delay | UNSUPPORTED | 없음 | Native 지연과 운동 반응 확인 대기; hidden history 문제는 별도로 해석 |
| Mass | UNSUPPORTED | 없음 | Actual mass/inertia, contact 및 lift 응답 확인 대기 |
| Friction | UNSUPPORTED | 없음 | Object–finger/object–plane 접촉과 상대 운동 확인 대기 |

## 준비한 계측 변경

[probe_physical_response.py](/Users/jihyekim/Desktop/tdmpc2/scripts/probe_physical_response.py):

- Native NPZ의 contiguous timestep, JSON sidecar의 evaluation seed와 native-step count를 확인한다. NPY는 명시적 native tape 선언을 요구한다.
- 설치 controller 및 부모 클래스의 Python source/hash, dependency version, repo HEAD/dirty status, wrapper source hash를 저장한다.
- EEF target을 base/world 좌표계로 각각 저장하며 measured EEF pose는 world frame으로 구분한다.
- 모든 physics substep의 object 접촉을 수집하고 object-finger / object-plane / object-other로 분류한다.
- Contact position/normal/impulse를 유한한 3차원 숫자 배열로 변환한다. 지원하지 않는 값은 문자열로 숨기지 않고 실패시킨다.
- 물체 pose/velocity 및 TCP/finger에 대한 상대 pose를 측정한다. 상대 운동 자체를 sustained-contact slip으로 단정하지 않는다.
- 최대 80 native steps/조건, 최대 5조건이며 새 output 디렉터리만 사용한다. 실제 native done이면 멈춘다.

[analyze_physical_response.py](/Users/jihyekim/Desktop/tdmpc2/scripts/analyze_physical_response.py):

- 위 simulator JSON이 있을 때만 measured_* PNG/PDF와 measurement_summary.json을 생성한다.
- 접촉 숫자 배열과 physics substep 개수를 검사한다.
- 기존 reference 그래프와 다른 파일명/출력 디렉터리 및 SIMULATOR MEASUREMENTS 라벨을 사용한다.
- Impulse magnitude 합은 net force가 아니며, native 구간 내 두 finger 접촉은 반드시 동시 접촉인 것은 아니라는 제한을 명시한다.
- 현재 입력 simulator JSON이 없어 실제 측정 그림은 생성하지 않았다.

## 로컬 검증

- 두 스크립트 구문 검사 통과.
- NumPy/list/xyz binding 형태의 contact vector 숫자 변환 및 unsupported object의 strict serialization 검사 통과.
- CLI help 확인. 실제 SAPIEN API 및 hook 호환성은 아직 미검증.
- 모델 학습, checkpoint 변경, 기존 Stage 3B 결과 변경, 새 adaptation method 구현 없음.

## 서버 확정 후 순서

1. 접속 및 최신 repo HEAD/dirty diff 확인. 기존 파일 덮어쓰기 없이 별도 audit 디렉터리에 도구를 배치한다.
2. 설치된 환경의 버전/controller source와 기존 severity native tape/sidecar를 확인한다.
3. Nominal gain 한 조건, 최대 2 native steps로 API/숫자 저장 smoke를 실행한다.
4. Free-space 구간과 실제 grasp/lift contact가 포함되는 tape를 확인한 뒤 필요한 조건만 짧게 실행한다.
5. Gain/Delay는 target·issued/executed·실측 EEF motion, Mass는 contact와 object/lift response, Friction은 접촉 종류 및 contact-conditioned 상대 운동을 확인한다.
6. 이번 측정만으로 보고서·측정 그림·최종 판정을 갱신한다. 접촉 없는 run은 grasp validity 증거로 사용하지 않는다.
