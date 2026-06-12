# TODO — Antifragile ML 진입 신호 모델

> 이 파일로 진행 요청 시 순서대로 실행

---

## 사전 준비

- [ ] **temp/scripts/ 기존 파일 삭제**
  - 삭제 대상: 22~32_*.py, backtest_style_a-e.py (총 16개)
  - 결과 기록은 memory/project_style_exploration.md에 보존됨
  - 명령: `rm temp/scripts/*.py`

---

## Phase 1 — BTC 검증 (LightGBM)

- [ ] `temp/scripts/33_af_newmodel.py` 작성
  - Step 1: Antifragile 백테스트로 레이블 생성
  - Step 2: 피처 추출 (23개 기본 + AF-specific 5개)
  - Step 3: LightGBM walk-forward 3-fold 학습
  - Step 4: 결과 출력 + feature importance

- [ ] BTC 실행 및 결과 확인
  ```bash
  /opt/homebrew/Caskroom/miniforge/base/envs/cryptobot/bin/python \
    temp/scripts/33_af_newmodel.py --coin btc
  ```

- [ ] **Phase 1 판정**
  - ✅ PF ≥ 8.45 × 1.10 = 9.30 AND 거래수 ≥ 555건 → Phase 2 진행
  - ❌ 조건 미충족 → 중단 (Antifragile rule-based 유지)

---

## Phase 2 — 4종 전체 + TCN (Phase 1 통과 시)

- [ ] 4종 전체 LightGBM 실행
  ```bash
  /opt/homebrew/Caskroom/miniforge/base/envs/cryptobot/bin/python \
    temp/scripts/33_af_newmodel.py --coin all
  ```

- [ ] TCN 통합 모델 추가 학습 (선택, 시간 여유 있을 때)
  - `temp/models/af_tcn_all.pt`
  - 기존 `src/expert_models.py` 아키텍처 재사용, 레이블만 변경

- [ ] 4종 결과 비교 테이블 확인
  - BTC / ETH / SOL / XRP 각각 PF, WR, 수익률, MDD

---

## Phase 3 — 통합 (7일 paper 완료 후)

- [ ] 7일 paper 거래 결과 확인 (2026-06-10 이후)
- [ ] ML 모델 paper 거래 결과와 비교 분석
- [ ] 조건 충족 시 `live_trader.py`에 `STRATEGY=af_ml` 옵션 추가 검토
  - 진입: ML confidence ≥ threshold
  - exit/sizing: 기존 trailing stop + pyramiding 유지

---

## 참고

- 계획 상세: `temp/PLAN_af_newmodel.md`
- 기존 결과 기록: `memory/project_style_exploration.md`
- 기존 전략 문서: `STRATEGY_ANTIFRAGILE.md`
- backtest 엔진: `scripts/backtest_antifragile.py`
