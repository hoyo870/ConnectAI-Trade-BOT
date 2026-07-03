# 신뢰 재건 로그 (Trust Rebuild Log)

> 시작: 2026-06-24 | 계획: `~/.claude/plans/cosmic-leaping-goblet.md`
> 관련 메모리: `project_pnl_accounting_fix`, `project_singleentry_baseline`, `project_param_sweep`

백테스트가 손실 전략을 큰 수익으로 보이게 만들어 실거래 손실이 발생했다. 이 문서는 원인 규명 →
측정 인프라 재건 → 정직한 재검증 → 파라미터 탐색까지의 진행을 기록한다.

---

## 0. 발단

- 백테스트는 **손실 전략을 +212%/3개월 수익**으로 표시 → 그 숫자만 믿고 실거래 → 큰 손실.
- 사용자가 가동 중 봇을 종료. "실거래뿐 아니라 백테스트도 문제였다."

---

## 1. 근본 원인 — 피라미딩 회계 버그 (수정 완료)

- **버그**: add-to-winner 피라미딩의 수익률을 **최초 진입가** 기준으로 계산 → 추가분 전체에
  "최초가 대비 전체 이동폭"을 적용해 승리 거래를 과대계상.
- **수정**: `strategies/antifragile.py` — 수량가중 평균 진입가로 갱신
  `avg = (rr+add_rr) / (rr/avg + add_rr/price)` (거래소 실현손익 정의와 일치).
- **충격**: 가중평균(정확) 적용 시 BTC 2026 OOS **+수익 → −48.7%**. add-to-winner 피라미딩이
  평단가를 현재가로 끌어당기고 trailing stop이 평단 손실 쪽에서 청산 → **구조적 손실** 판명.
- **검증**: chunk 단위 trace로 전략 공식 `pnl_raw·lev·rr == Σ qty_j·(exit−p_j)` 일치 확인.
- **조치**: 피라미딩 비활성화(`add_levels=0`, 전 프리셋), 단일진입으로 기본 신호부터 재검증.

---

## 2. 인프라 재건 (Phase 1·2)

| ID | 항목 | 파일 | 핵심 |
|----|------|------|------|
| US-101 | 회계 회귀 테스트 | `tests/test_accounting_parity.py` | 가중평균==거래소 청크합산 영구 고정. ALL PASS |
| US-102 | intrabar emergency SL | `strategies/backtest_engine.py` | 거래소 위탁 SL(6×ATR)을 봉 high/low로 체결 모사 → 꼬리손실 반영 |
| US-103 | 피라미딩 비활성 | `config/af_params.py` | `add_levels=0` (DEFAULT+전 프리셋). 단일진입 baseline |
| US-104 | 수치 무효 배너 | README/MODEL_HISTORY/STRATEGY_ANTIFRAGILE | 기존 백테스트 수익 = 과대계상 경고 |
| US-105 | loss_streak skew 점검 | — | train·serve 모두 상수 0 → **skew 없음**, 재학습 불필요 |
| US-106 | 비용 현실화 | `backtest_engine._net_pnl`, `af_params.FUNDING_RATE_8H` | 왕복 수수료+슬리피지+funding, `cost_mult` 민감도 |
| US-107 | walk-forward | `scripts/walkforward.py` | 프리셋을 in-sample에서만 선택, OOS 평가 (튜닝/평가 분리) |
| US-108 | 강건 게이트 | `backtest_engine.robust_metrics` | Sharpe/Sortino/수익집중도/연속손실 → 아웃라이어 의존 탐지 |
| US-109 | parity 하니스 | `scripts/parity_check.py` | 백테스트 trade_log vs paper CSV 거래별 대조 + selftest |

부수: `scripts/backtest_af_exact.py` 에 데이터 자동 갱신(`ensure_data_coverage`) 추가, 4코인 데이터 최신화.

**⏸ 미완(세션 토큰 한도로 차단)**: ralph Step 7 독립 리뷰어(architect) 검증 + Step 7.5 deslop 패스.
저자 self-review로는 6개 고위험 지점 버그 없음 확인했으나 독립 승인 대체 불가.

---

## 3. 정직한 재검증 결과

### 3.1 비용 취약성 (US-106)
BTC 2026 단일진입(10x):

| cost_mult | 수익 | WR | PF |
|---|---|---|---|
| 1× (가정) | +12.55% | 43.6% | 1.139 |
| **2×** | **−34.48%** | 32.7% | 0.676 |
| 3× | −61.89% | 24.3% | 0.424 |

→ 엣지가 비용에 **극도로 취약**. 실측 슬리피지(0.07%)가 모델(0.006%)의 10배라 실제는 2× 영역에 가까울 수 있음.

### 3.2 강건 지표 (US-108)
"+12.6%" 단일진입조차 **Sharpe +0.04, 상위 10% 거래가 순익의 520% 기여**(나머지는 순손실) → 아웃라이어 의존.

### 3.3 hist 검증 (단일진입, 10×91일 랜덤창)
BTC 6/10 · ETH 4/10 · SOL 9/10 · XRP 9/10 통과. **2025 최근 구간이 전부 최약체**(forward에 가장 중요한 신호).

---

## 4. 파라미터 스윕 — 과적합 방지 (`scripts/sweep.py`)

**원칙**: TRAIN(2023–24)에서 후보 선정 → 별개 TEST(2025–현재)에서 평가 + 비용 2배 스트레스 +
robust 지표. in-sample 최댓값을 "최적"이라 부르지 않음.

knob: RSI 선별강도 δ × `trail_atr_init` × ML θ(`ensemble.threshold`). add_levels=0.

### 4코인 독립 합의 결과
- **ML θ=0.45 (기본 0.30보다 엄격)이 robustness의 핵심.** θ=0.30 조합은 비용 2배에서 보편적
  붕괴(top10기여 166~747%). θ=0.45는 비용 2배에도 생존.
- **공통 최선 조합: `δ=10, trail_atr_init=2.0, θ=0.45`** (BTC·SOL·XRP 동일, ETH는 trail=1.0).
- 4코인이 *독립적으로* 같은 영역 수렴 + 과거에서 고른 조합이 미래+비용2배 견딤 → **과적합 아님**.

### 후보 설정 (현재 → 변경)
```
ensemble.threshold : 0.30 → 0.45
RSI δ=10           : dt 22/65→12/75, rg 30/70→20/80, ut 40/85→30/95
trail_atr_init     : 1.0 → 2.0
add_levels         : 0 (단일진입 유지)
```

### 후보 고정 다중 fold 확인 (`scripts/confirm_candidate.py`, 3x vs 10x)
| 코인 | 3x: 양수fold/최악MDD/중앙값 | 10x: 양수/최악MDD/중앙값 |
|---|---|---|
| BTC | 13/13 · 1.3% · +4.1% | 13/13 · 4.2% · +14.0% |
| ETH | 13/13 · 2.8% · +6.1% | 13/13 · 9.1% · +21.5% |
| SOL | 13/13 · 3.4% · +7.5% | 13/13 · 10.8% · +26.9% |
| XRP | 13/13 · 4.7% · +8.2% | 13/13 · 15.1% · +29.5% |

→ **52/52 fold 양수, 3x MDD<5%.** 지금까지 중 가장 강한 후보.

### ⚠️ 결정적 caveat — 이 확인은 완전한 OOS가 아님
1. **부분 순환성**: 후보를 2023–26 데이터로 선택했는데 확인도 2022–26을 봄(겹침).
2. **ML 모델은 2020–2024 학습** → 2022–24 fold는 ML 입장 in-sample. 진짜 ML-OOS(2025+)는 더 약함.
3. 확인은 비용 1배 기준. 3x의 얇은 fold는 비용 2배면 음수 가능.
→ **진짜 시험은 forward(paper)뿐.** 백테스트로는 더 이상 de-risk 불가.

---

## 5. 생성/수정 파일

**신규**: `tests/test_accounting_parity.py`, `scripts/walkforward.py`, `scripts/parity_check.py`,
`scripts/sweep.py`, `scripts/confirm_candidate.py`, `REBUILD_LOG.md`(본 문서).
**수정**: `strategies/antifragile.py`, `strategies/backtest_engine.py`, `config/af_params.py`,
`scripts/backtest_af_exact.py`, README/MODEL_HISTORY/STRATEGY_ANTIFRAGILE(배너).

---

## 6. 다음 단계 (미완)

1. **후보 설정 채택**: candidate 프리셋으로 `af_params` 박제 + ML θ=0.45 반영(meta.json 또는 로드시 오버라이드). prod 보존.
2. **forward 검증 (핵심 게이트)**: 3x · candidate로 **paper 모드 가동** → `parity_check.py`로 백테스트=paper 일치 지속 확인(순환성 없는 유일한 시험).
3. **go-live 게이트**(엄격): 수주 paper에서 ① parity 일치 ② 양의 forward 수익 ③ 감당 가능 MDD 충족 시에만 소액 real.
4. **수익 개선 레버**(전부 OOS+비용+forward 검증 필수):
   - 최우선 **ML 재학습 2024→2026**(2025 약세 주원인 가능성).
   - **실행비용 절감**(taker→maker) — 비용이 binding constraint라 순수익 직접 상승.
   - 출구 개선 / 변동성 사이징 / 진입 필터 추가.
   - ❌ 금지: 레버리지·사이징↑, 같은 데이터 재튜닝, 비용 과소, 피라미딩 복귀(전부 숫자만 부풀림).

**핵심 원칙**: 수익은 "원해서" 오르지 않는다 — **엣지나 실행을 개선**하고, **forward로 검증**해야 오른다.

---

## 7. 후보 백테스트 기록 (paper 검증 전 baseline) — 2026-06-25

설정: `AF_PARAM_PRESET=candidate` (δ=10: RSI 12/75·20/80·30/95, trail_atr_init 2.0, add_levels 0) +
`ML_THRESHOLD=0.45` + **LEVERAGE=3** (paper와 동일) + ML_MODEL_DIR=saved(검증모델). 정정 회계·intrabar·비용 반영.
재현: `.env` 위 설정 후 `python scripts/backtest_af_exact.py --mode {2026|hist|jun} --coin all`.

### (A) 2026 OOS (2026-01-01~05-31, 3x)
| 코인 | 거래 | 수익 | WR | MDD | PF | Sharpe |
|---|---|---|---|---|---|---|
| BTC | 24 | **+5.30%** | 70.8% | 0.6% | 10.40 | +0.55 |
| ETH | 31 | +3.85% | 61.3% | 0.8% | 3.39 | +0.41 |
| SOL | 42 | +3.21% | 54.8% | 1.9% | 1.72 | +0.21 |
| XRP | 41 | +1.77% | 53.7% | 1.3% | 1.54 | +0.17 |
→ 4코인 전부 양수, MDD<2%. (합산 ~+14%)

### (B) HIST (10×91일 랜덤창, seed=42, 3x)
| 코인 | 통과(3/3) | 양수 | 평균수익 |
|---|---|---|---|
| BTC | 0/10 | **10/10** | +3.1% |
| ETH | 0/10 | **10/10** | +6.0% |
| SOL | 0/10 | **10/10** | +5.6% |
| XRP | 0/10 | **10/10** | +8.3% |
→ **40/40 윈도우 전부 양수**, 91일당 +3~8%, MDD 대부분 <5%.

### (C) 2026 6월 (Jun 01~22, 3x)
| 코인 | 거래 | 수익 | WR |
|---|---|---|---|
| BTC | 2 | −0.07% | 0% (소표본 노이즈) |
| ETH | 5 | +0.78% | 60% |
| SOL | 5 | +0.68% | 60% |
| XRP | 3 | +0.23% | 67% |
→ 매우 저빈도(선별적), 대체로 ~breakeven~소폭+. 6월은 표본 적음.

### 해석 (정직)
- **모든 "판정"이 3/3 게이트 탈락 — 단 TPD(≥1.5) 때문**. 후보는 의도적으로 선별적(TPD ~0.1~0.4)이라
  빈도 기준에서만 떨어짐. **수익성은 양호**: 2026 4/4 양수, hist 40/40 양수, 6월 3/4 양수.
  → 저빈도 후보엔 TPD≥1.5 게이트가 부적절. 의미있는 지표는 "양수 일관성 + 낮은 MDD + Sharpe".
- **3x에서 수익은 modest**(안전하지만 작음): 2026 코인당 +2~5%/5개월, hist +3~8%/91일, MDD<5%.
  고변동·고수익은 레버리지로 부풀린 것이었고, 3x는 생존 가능한 현실 수치.
- ⚠️ 이 수치도 in-sample 성격(파라미터를 2026 인접 데이터로 선택) 잔존 → **진짜 시험은 forward(paper)**.
- ⚠️ **[2026-07-04 추가] §7 수치는 look-ahead 버그(§8-2 참조) 포함** — 약간 낙관. 아래 §8 참조.

---

## 8. Forward(Paper) 검증 + Parity 분석 — 2026-06-25~07-03

### 8-1. Paper 실행 결과 (9일, candidate·θ0.45·3x, $10,000 총자본)
설정: AF_PARAM_PRESET=candidate, ML_THRESHOLD=0.45, LEVERAGE=3, TRADE_MODE=paper.
| 코인 | 거래 | 순수익(net) | 비고 |
|---|---|---|---|
| BTC | 1 | +0.094% | 롱, trail_SL |
| ETH | 2 | +0.105% | 롱×2, trail_SL |
| SOL | 2 | +0.408% | 롱×2, trail_SL |
| XRP | 1 | +0.066% | 롱, trail_SL |
| **합산** | **6** | **+0.674%** | |

### 8-2. Parity 개선 3단계 → 완벽 일치 달성
| 수정 | 원인 | 해결 |
|---|---|---|
| ① Bybit 소스 통일 | 백테스트=Binance, paper=Bybit → 가격 basis ~0.07% | `src/data_fetcher.py` bybit 지원 추가, `data/bybit/` 별도 저장 |
| ② paper 수수료 반영 | paper 모드가 gross(수수료 미차감) | `live_tools/live_trader.py` paper 분기에 `2×FEE_TOTAL+funding` 명시 차감 |
| ③ 1h trend look-ahead 제거 | `resample('1h').last()`가 현재 형성 중 봉에 미래(:55) 데이터 → 백테스트 trend에 미래참조 | `strategies/indicators.py` `ema_1h.shift(1)` 완성봉 기준으로 교체 (live와 수학적 동치, 회귀 ALL PASS) |

**최종 결과**: 4코인 전 거래 완전일치, 합산 paper +0.674% vs backtest +0.672% (차이 **0.001%p**).

### 8-3. ⚠️ look-ahead 파급 (수정 전 모든 백테스트에 존재)
- §7 수치 포함 모든 과거 백테스트(2026 OOS, hist, sweep, confirm_candidate)에 look-ahead가 있었음 → 유령 trend-driven 거래로 약간 낙관. §7은 재실행 수치로 갱신 예정(本 섹션 아래).
- ML 라벨도 look-ahead trend로 생성 → saved 모델에 미세 train/serve 불일치. 재학습은 이전 검증(wash)으로 optional.

### 8-4. §7 수치 갱신 (look-ahead 제거 후, 2026-07-04 재실행)

**2026 OOS (2026-01-01~05-31, candidate·θ0.45·3x, causal trend)**
| 코인 | 거래 | 수익 | WR | MDD | Sharpe | §7 대비 |
|---|---|---|---|---|---|---|
| BTC | 6 | +0.04% | 50.0% | 0.5% | +0.08 | 24→6, +5.30→+0.04% |
| ETH | 17 | +1.38% | 41.2% | 1.3% | +0.19 | 31→17, +3.85→+1.38% |
| SOL | 24 | **−0.87%** | 33.3% | 2.6% | −0.10 | 42→24, +3.21→−0.87% |
| XRP | 24 | **−0.65%** | 45.8% | 2.4% | −0.10 | 41→24, +1.77→−0.65% |

**HIST (10×91일, candidate·θ0.45·3x, causal trend)**
| 코인 | 통과(3/3) | 양수 | 평균수익 | §7 대비 |
|---|---|---|---|---|
| BTC | 0/10 | 6/10 | **−0.1%** | 8→6/10, +3.1→−0.1% |
| ETH | 0/10 | 6/10 | +1.0% | 10→6/10, +6.0→+1.0% |
| SOL | 0/10 | 4/10 | +0.7% | 10→4/10, +5.6→+0.7% |
| XRP | 0/10 | 10/10 | +2.9% | 10/10 유지, +8.3→+2.9% |

### 8-5. 정직한 재판정
look-ahead 제거 후 2026 OOS는 BTC/ETH만 양수, SOL/XRP 음수. hist는 XRP만 10/10 양수, 나머지 4~6/10. **§7의 "40/40 양수"와 "4/4 양수"는 대부분 look-ahead에 기인한 거짓 신호였음.** 진짜 엣지는 훨씬 약하고, 특히 SOL/XRP 2026에서 이미 음수.

→ **결론**: 후보 설정의 forward(paper) 결과(+0.674%/9일)는 이 causal 백테스트(+0.672%)와 완벽히 일치 — 백테스트는 이제 신뢰할 수 있다. 그러나 엣지 자체가 약하다는 사실도 백테스트가 정직하게 보여줬다. 실거래 재개는 수주 이상 추가 paper + 종목 다변화·ML 개선 후 재판단 권장.

