# ConnectAI Trade Bot — TODO

## Phase 0 — BTC 샌드박스 테스트

- [x] Bybit 테스트넷 API 연결
- [x] live_trader.py 구현 (5분봉 자동 진입/청산)
- [x] 텔레그램 알림 연동 (진입/청산/CB/1시간 보고)
- [ ] 테스트넷 7일 연속 무결함 실행
- [ ] 진입/청산 로직 백테스트 결과와 일치 확인

---

## Phase 1 — BTC Production (완료)

- 모델: `models/production/btc_expert_*.pt`
- 2026 결과: WR=49.6% ✅, TPD=1.84 ✅, daily=+1.30% ✅

## Phase 2 — ETH Production (완료)

- 모델: `models/production/eth_expert_*.pt`
- 2026 결과: WR=48.3% ✅, TPD=1.16, daily=+1.27% ✅ (hold=180, Phase2_gate)

---

## 연구 — ETH Pullback 진입 전략 테스트

> 배경: ETH Pullback v2(이전 봉 피크 + 현재 감소 + 현재 봉 방향 일치) 히스토리 avg WR +2.3%p, daily +0.20%p, MDD -5.5%p 개선 확인.
> 거래수 174→147 (15% 감소, 허용범위). 스크립트: `temp/scripts/22_backtest_pullback.py`

- [ ] ETH Pullback + min_tier_thr 변경 테스트 (0.45, 0.50, 0.55, 0.60)
- [ ] ETH Pullback + hold 변경 테스트 (120, 150, 180, 210)
- [ ] ETH Pullback + sig_upper_thr 조정 (0.50, 0.55, 0.60, 0.65)
- [ ] BTC 즉시 진입 + ETH Pullback 혼합 조합 최종 비교
- [ ] 최적 조합 확정 시 production 파라미터 업데이트

---

## 참고 파일

| 파일 | 내용 |
|------|------|
| `NEXT_GOALS.md` | 단계별 목표 및 판정 기준 |
| `MODEL_HISTORY.md` | 모델/파라미터 이력 |
| `logs/live_state.json` | 실시간 포지션 상태 |
