# EvoRL — 경험을 메모리(H)에 둘지 파라미터(P)에 넣을지 RL로 학습

COVE(`../cove`) 위에 얹은 실험 코드. COVE 원본은 수정하지 않고 메모리 매니저·프롬프트·rule 템플릿을 그대로 import해서 씀.

## 설치

COVE 코드는 이 레포에 포함하지 않음 (원 저장소: https://anonymous.4open.science/r/cove-8BCC). 다음 구조로 두면 `evorl/__init__.py`가 `../cove/src`를 자동으로 `sys.path`에 추가함. 다른 위치면 `COVE_SRC=/path/to/cove/src` 지정.

```
<workspace>/
├── cove/      # COVE 체크아웃 (cove/src/...)
└── evorl/     # 이 레포
```

실행은 `<workspace>`에서 `python -m evorl.<module>`.

## 구성

| 파일 | 역할 |
|---|---|
| `tasks/api_world.py` | API형 태스크. COVE rule_templates의 연산 8종 × 가짜 라이브러리 10종. 호출 규약은 프롬프트에 없고 실패 피드백·메모리·가중치로만 알 수 있음. `drift_steps`에서 버전이 바뀌며 규약/동사 이름이 전부 바뀜 (volatile) |
| `tasks/math_tasks.py` | 추론형 태스크. MATH train(Level 1–3) 스트림, MATH-500 검증. math-verify 채점 |
| `tasks/stream.py` | step t → (태스크 타입, 배치 32개, API 버전). 시드 고정이라 모든 분기·정책이 같은 경험을 봄 |
| `agent.py` | 배치 rollout(시도 → 피드백 → 재시도), 검증 V(greedy 1회 시도 정확도) |
| `memory.py` | **H**: COVE `MemorySkillManager`(요약 → add/merge/replace/discard)를 helper LLM으로 구동. HyDE 질의 확장 캐시 |
| `trainer.py` | **P**: 성공 trajectory를 메모리 없는 프롬프트로 LoRA SFT (rejection-sampling fine-tuning). 부모 어댑터에서 이어서 학습 |
| `env.py` | MDP 환경. 노드 = (LoRA, 메모리 뱅크, 히스토리). `expand(node, a)` → 자식 + transition 기록 |
| `features.py` | state 특징. `TASK_FEATURES`(COVE 라우터 입력에 해당) / `AGENT_FEATURES`(학습 상태) |
| `policies.py` | always-H/P/B, random, COVE 재구현(LLM judge + 트리거), Q 라우터 |
| `collect_tree.py` | 루트에서 horizon 3 전체 분기(3+9+27=39 노드) 수집 |
| `qlearn.py` | fitted Q iteration (ExtraTrees), leave-one-root-out 트리 regret |
| `analyze.py` | RQ1 분석: 최적 action 분포, 근시안 vs 장기 불일치, TASK vs ALL 특징 예측력 |
| `run_policy.py` | 정책을 T step 굴려 학습 곡선 기록 |

## 설계 결정 (COVE와 다른 점)

- **policy = Qwen3-0.6B**, **helper = Qwen3-4B-Instruct-2507(고정)**: 메모리 요약/큐레이션과 COVE judge만 담당. 0.6B가 메모리를 직접 쓰면 H가 과소평가되므로 분리. 논문에 "고정 큐레이터" 명시 필요. helper를 0.6B로 바꾸는 ablation 가능 (`helper_url/helper_name`).
- **P = RFT(LoRA SFT on 성공 trajectory)**: COVE는 GRPO. 트리 수집 비용 때문에 교체 (P 한 번 ≈ 수십 초). 같은 환경 안에서 COVE 베이스라인도 같은 P를 쓰므로 비교는 공정.
- **결정 단위 = step(배치 32개)**: action ∈ {H, P, B(=둘 다)}. COVE 재구현은 "항상 H + 트리거 시 pool로 P".
- **보상**: `r = ΔV − λ·C(a)`, C = 액션 wall-clock 초/600. 원시 비용을 저장하므로 λ는 사후에 바꿔서 재계산 가능.

## 서버 실행 (RTX3090 ×2)

```bash
# 컨테이너 재시작 후 /opt가 비었으면
bash /root/cove-rl/evorl/scripts/setup_env.sh
# GPU0: policy(:8001) + helper(:8002)
bash /root/cove-rl/evorl/scripts/start_servers.sh

cd /root/cove-rl
PY=/opt/cove-venv/bin/python
$PY -m evorl.smoke                                              # 소형 E2E 점검
$PY -m evorl.run_policy --exp main --policy random --steps 10   # 행동 정책 (루트 다양화)
$PY -m evorl.collect_tree --exp main --root R0                  # horizon-3 트리
$PY -m evorl.collect_tree --exp main --root R0.H.P.B            # 중간 상태에서 분기
$PY -m evorl.analyze --exp main                                 # RQ1
$PY -m evorl.qlearn --exp main --cv                             # 오프라인 regret
$PY -m evorl.qlearn --exp main --gamma 0.9 --tag rl             # Q 라우터 저장
$PY -m evorl.run_policy --exp main --policy q:/root/cove-rl/results/main/q_rl.pkl --steps 12
```

저장 위치: `/root/cove-rl/results/<exp>/`(노드·transition, 영구), `/opt/cove-work/<exp>/adapters/`(LoRA, 임시), 로그 `/opt/cove-logs/`.
