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

---

## 실거래 전환 — 2026-06-11

- [ ] 소액 실거래 테스트 (2026-06-11 14:00 KST)
  - 전략: Antifragile (ut_rsi_lo=40, ut_rsi_hi=85, leverage=5x)
  - 4종목 (BTC/ETH/SOL/XRP), 자본 25%씩 분할
  - 이상 없으면 본격 운용 전환

---

## live_tools 개선

### 웹 대시보드 디자인
- [ ] 전반적인 UI/UX 개선 (배색, 레이아웃, 가독성)
- [ ] 포지션 카드 디자인 개선 (코인별 수익/손실 시각화)
- [ ] 실시간 차트 또는 수익 히스토리 그래프 추가
- [ ] 모바일 반응형 레이아웃 (NAS 원격 접속 시 폰에서도 확인)

### 프로세스 명령 권한 분리
- [ ] 조회 API (GET `/api/status`, `/api/logs`) — 인증 불필요
- [ ] 제어 API (POST `/api/restart`, `/api/halt`, `/api/stop`) — 토큰/패스워드 인증 필요
- [ ] `.env`에 `DASHBOARD_TOKEN` 추가, Authorization 헤더 검증
- [ ] 대시보드 UI에 제어 버튼 클릭 시 토큰 입력 모달 추가

---

## NAS 배포 — DS920+ (Docker)

> 배경: Intel Celeron J4125 (x86_64), Container Manager 지원. 실거래 테스트 완료 후 진행.

### 경량화 분리 (NAS용)
- [ ] `Dockerfile` 작성 (프로젝트 루트)
- [ ] `requirements-nas.txt` 작성 — torch/SB3/gymnasium 제거 (antifragile 전략만 사용)
  - 필요: `ccxt`, `flask`, `pandas`, `numpy`, `scikit-learn`, `joblib`, `ccxt`, `pyyaml`
  - 제거: `torch`, `stable-baselines3`, `sb3-contrib`, `gymnasium`, `tensorboard`
- [ ] `live_trader.py` — `STRATEGY=antifragile` 시 torch import 스킵 처리
- [ ] `models/` 디렉터리에서 DL 모델 파일 제외 (`.dockerignore`)
- [ ] Docker Compose 파일 작성 (로그 볼륨 마운트, `.env` 바인드)

### NAS 배포 절차 문서화
- [ ] DSM Container Manager GUI 설정 가이드
- [ ] 포트 8765 방화벽 설정
- [ ] 부팅 시 자동 시작 설정 (`--restart unless-stopped`)
- [ ] VPN 접속 후 대시보드 접근 가이드

---

## 참고 파일

| 파일 | 내용 |
|------|------|
| `NEXT_GOALS.md` | 단계별 목표 및 판정 기준 |
| `MODEL_HISTORY.md` | 모델/파라미터 이력 |
| `logs/live_state.json` | 실시간 포지션 상태 |
| `STRATEGY_ANTIFRAGILE.md` | Antifragile 전략 파라미터 및 실험 이력 |
