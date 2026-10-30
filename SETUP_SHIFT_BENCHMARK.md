# TD-MPC2 Dynamics Shift v2: 환경 설치와 실행

이 저장소의 `docker/environment.yaml`은 기본 TD-MPC2 의존성만 설치한다.
ManiSkill2는 파일에서 주석 처리되어 있으므로 별도 설치가 필요하다. 아래 전체
평가 절차는 **Linux x86_64 + NVIDIA CUDA** 호스트에서 실행한다. 현재 작업
호스트(macOS arm64)에는 SAPIEN 2.2.x의 호환 wheel과 CUDA가 없어 frozen
checkpoint 평가를 실행할 수 없다.

## 설치 중 `Disk quota exceeded`가 난 경우

사용자가 공유한 로그는 wheel 빌드까지 성공했고, `Installing collected packages`
도중 `[Errno 122] Disk quota exceeded`로 끝났다. `ffmpeg` 빌드 자체의
오류가 아니다. `/home/jhkim/.cache/pip`에 다운로드 캐시가 쌓였고, conda
환경·패키지 캐시·임시 wheel에도 용량이 필요하다. 아래 명령을 **실패한 Linux
호스트**에서 먼저 실행해 큰 파일시스템을 확인한다. `df`의 여유 공간과
`quota`의 사용자 할당량은 별도로 확인한다.

```bash
conda env list
df -h "$HOME" /tmp /scratch 2>/dev/null
quota -s 2>/dev/null || true
du -sh "$HOME/.cache/pip" "$HOME/miniconda3/envs/tdmpc2" \
  "$HOME/.conda/envs/tdmpc2" 2>/dev/null || true
```

아래 `/scratch/jhkim/tdmpc2`는 **예시**다. `df`와 `quota`에서 충분한 공간이
확인된 실제 경로로 바꾼다. 실패한 `tdmpc2` 환경이 `conda env list`에 보이면
`conda env remove -n tdmpc2`로 그 환경만 제거한 뒤 재시도한다. 홈의 pip
캐시를 비워도 괜찮다면 `python -m pip cache purge`로 다운로드 캐시를
정리할 수 있다. 캐시 정리만으로는 새 환경이 들어갈 공간이 생긴다는
보장이 없다.
이후 지정한 prefix에서 재시도가 실패하면 해당 prefix만
`conda env remove -p "$SHIFT_ENV_PREFIX"`로 제거한 뒤 다시 실행한다.

```bash
cd /path/to/tdmpc2
export SHIFT_STORAGE=/scratch/jhkim/tdmpc2
export SHIFT_ENV_PREFIX="$SHIFT_STORAGE/envs/tdmpc2"
mkdir -p "$SHIFT_STORAGE/conda-pkgs" "$SHIFT_STORAGE/tmp" "$SHIFT_STORAGE/assets"
export CONDA_PKGS_DIRS="$SHIFT_STORAGE/conda-pkgs"
export TMPDIR="$SHIFT_STORAGE/tmp"
export PIP_NO_CACHE_DIR=1
conda env create -p "$SHIFT_ENV_PREFIX" -f docker/environment.yaml
conda activate "$SHIFT_ENV_PREFIX"
python -m pip install 'setuptools==65.5.0' 'wheel==0.38.4' 'cython<3'
python -m pip install --no-build-isolation 'gym==0.21.0'
python -m pip install 'mani-skill2==0.4.1.post0'
python -m pip check
export MS2_ASSET_DIR="$SHIFT_STORAGE/assets"
```

이 방식은 conda 환경, conda 패키지 캐시, pip 임시 파일과 ManiSkill 자산을
홈 할당량 밖에 둔다. 새 터미널에서도 `conda activate "$SHIFT_ENV_PREFIX"`
하기 전에 `SHIFT_ENV_PREFIX`의 같은 절대 경로를 설정한다. 저장소와
체크포인트가 홈에 있으면 **평가 로그와 영상은 여전히 홈에 쌓일 수 있다**.
필요하면 저장소와 checkpoint도 여유 있는 파일시스템에 두고 실행한다.
`conda env create`가 다시 실패하면 마지막 오류와 `df -h`, `quota -s`
출력을 함께 확인한다.

## 1. Linux/NVIDIA 호스트에 conda 환경 설치

홈 디렉터리에 충분한 공간이 있는 경우 저장소 루트에서 실행한다. 위 할당량
오류가 났다면 이 일반 명령 대신 앞의 `SHIFT_STORAGE` 절차를 사용한다.
기존 `tdmpc2` 환경을 새로 만들려면 `conda env list`로
상태를 확인한 뒤, 필요할 때만 기존 환경을 제거한다. 기존 환경 제거 시 그
환경에 추가한 패키지도 삭제된다.

```bash
cd /path/to/tdmpc2
conda env list
# 기존 tdmpc2 환경을 의도적으로 재설치할 때만:
# conda env remove -n tdmpc2
conda env create -f docker/environment.yaml
conda activate tdmpc2
python -m pip install 'setuptools==65.5.0' 'wheel==0.38.4' 'cython<3'
python -m pip install --no-build-isolation 'gym==0.21.0'
python -m pip install 'mani-skill2==0.4.1.post0'
python -m pip check
```

`gym==0.21.0`은 오래된 빌드 메타데이터 때문에 최신 wheel/setuptools에서
설치가 실패할 수 있어 위 버전을 먼저 고정한다. ManiSkill2 설치가 요구하는
SAPIEN 버전은 pip resolver에 맡기고 `pip check`와 아래 import 테스트로
실제 설치 결과를 확인한다. `docker/environment.yaml`의 기본 NumPy 버전은
ManiSkill2 설치 과정에서 변경될 수 있으므로 최종 버전을 기록한다.

GPU, simulator, checkpoint 사전 확인:

```bash
python - <<'PY'
import importlib.metadata as m
import torch, gym, gymnasium, sapien.core, mani_skill2, hydra
for name in ('torch', 'gym', 'gymnasium', 'sapien', 'mani-skill2', 'hydra-core', 'numpy'):
    print(name, m.version(name))
print('CUDA available:', torch.cuda.is_available())
assert torch.cuda.is_available(), '이 평가기는 CUDA가 필요합니다'
PY
python -m mani_skill2.utils.download_asset PickCube-v0
```

ManiSkill2는 기본적으로 현재 디렉터리의 `data/`에 asset을 받는다. 별도 경로를
쓰려면 다운로드와 평가 전에 `export MS2_ASSET_DIR=/absolute/asset/dir`를
설정한다. PickCube는 절차적으로 생성되는 자산만 쓸 수 있어 다운로드가
불필요할 수 있지만, 위 명령으로 필요한 자산을 확인한다.

체크포인트는 저장소에 포함되어 있지 않다. 실제 기존 PickCube TD-MPC2
체크포인트의 절대 경로를 지정한다. 모델 크기 `5`는 저장소의 기존 suite
설정이며, 다른 체크포인트를 쓸 때는 해당 아키텍처 크기로 바꾼다.
현재 사용한 `pickcube-nominal-seed0-500k/models/final.pt`에는 termination
head가 있으므로 평가 명령에 `episodic=true`가 필요하다. 이 설정 없이
실행하면 체크포인트 로딩에서 `_termination` 키 오류가 난다.

```bash
export TD_MPC2_CHECKPOINT=/absolute/path/to/pick-cube-1.pt
test -f "$TD_MPC2_CHECKPOINT"
```

## 2. 먼저 검증

저장소 루트에서 실행한다. simulator audit은 새 `audits/` 디렉터리를 만들며
기존 결과 파일을 덮어쓰지 않는다.

```bash
python -m unittest discover -s tests -v
python scripts/audit_shift_simulator.py \
  --out-root audits --seeds 17 18 19 --decisions 50 --atol 1e-6 --contact-probe
python scripts/evaluate_shift_suite.py \
  --suite benchmarks/archive/dynamics_shift_suite_v1.json \
  --checkpoint "$TD_MPC2_CHECKPOINT" --dry-run
```

`audit_shift_simulator.py` 결과의 각 identity/readback/contact 검사가 PASS인지
확인한다. 실패나 UNVERIFIED가 있으면 frozen benchmark 결과를 물리적으로
검증된 것으로 취급하지 않는다.

## 3. Frozen 평가 실행

`evaluate_shift.py`는 `tdmpc2/` 디렉터리에서 실행한다. 아래 세 명령은 각각
고정 shift, episode 중 급변, A→B→A→C 시퀀스 예시다. 실제 checkpoint
경로를 넣어야 한다. `save_video=false`는 영상 렌더링 의존성을 줄인다.

```bash
cd tdmpc2
python evaluate_shift.py task=pick-cube model_size=5 episodic=true \
  checkpoint="$TD_MPC2_CHECKPOINT" eval_episodes=50 seed=1 save_video=false \
  dynamics_shift=action_gain schedule=fixed adaptation=frozen evaluation=standard \
  dynamics_shift.parameter.shifted=0.7

python evaluate_shift.py task=pick-cube model_size=5 episodic=true \
  checkpoint="$TD_MPC2_CHECKPOINT" eval_episodes=50 seed=1 save_video=false \
  dynamics_shift=action_gain schedule=abrupt adaptation=frozen evaluation=paired \
  schedule.switch_step=10 dynamics_shift.parameter.shifted=0.7

python evaluate_shift.py task=pick-cube model_size=5 episodic=true \
  checkpoint="$TD_MPC2_CHECKPOINT" eval_episodes=50 seed=1 save_video=false \
  dynamics_shift=action_gain schedule=sequence adaptation=frozen evaluation=paired
```

`abrupt`에서 `schedule.switch_step=20`으로 바꾸면 두 번째 onset 조건이다.
더 많은 조합과 로그 필드 설명은 `EXPERIMENTS.md`를 참조한다. 각 실행의
고유 출력 디렉터리는 콘솔에 표시되며,
`tdmpc2/logs/pick-cube/1/default/benchmark/eval/` 아래 생성된다.
`summary.json`에서 `status`, `actual_shift_observed`, `exposure_fraction`,
`checkpoint_unchanged`를 확인한다. Paired 실행에서는 shifted run의
`paired.json`도 확인한다. 초기 상태 비교가 PASS여도 full-state pairing은
UNVERIFIED로 기록된다.

## 4. 기존 E0–E7 재현

저장소 루트에서 실행한다. archive는 기존 suite와 byte 동일하며, 아래 명령은
새 결과 디렉터리를 사용한다.

```bash
cd /path/to/tdmpc2
python scripts/evaluate_shift_suite.py \
  --suite benchmarks/archive/dynamics_shift_suite_v1.json \
  --checkpoint "$TD_MPC2_CHECKPOINT"
python scripts/audit_shift_logs.py --logs tdmpc2/logs --out-root audits
```

## 현재 macOS 작업 호스트 상태

이 호스트에서는 repo의 전체 conda 환경 설치가 완료되지 않았다. 앞선
`conda create` 샌드박스 실행은 패키지 조회에 실패했고, 네트워크/환경 생성
권한 요청이 거부되었다. 임시 Python 가상환경에서 Hydra/NumPy 단위 테스트
43개를 실행해 40개 통과, 3개 simulator 통합 테스트 skip을 확인했다.
실제 simulator audit은 `ModuleNotFoundError: gym`으로 UNVERIFIED다.
macOS용 최신 ManiSkill/SAPIEN 3으로 대체 설치하면 이 저장소가 기대하는
ManiSkill2 0.4.1/SAPIEN 2 API와 물리 조건이 달라져 같은 benchmark가 아니다.
