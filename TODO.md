# ConnectAI Trade Bot — TODO

> 전략: **Antifragile Trailing Stop** (AdaptRSI + ATR trailing, 4종목)
> 실거래 시작: 2026-06-11 | 레버리지: 7x | 프리셋: prod

---

## 실거래 운용

- [ ] Bybit API: `Withdrawal` 권한 OFF 확인 (Bybit 콘솔에서 수동 확인)
- [ ] IP 화이트리스팅: 봇 실행 서버 IP 등록 (보안 강화 시)
- [ ] 레버리지 5x 전환 검토 (XRP MDD 7x=43% → 5x=33%로 개선)

---

## 웹 대시보드 개선

- [ ] 전반적인 UI/UX 개선 (배색, 레이아웃, 가독성)
- [ ] 포지션 카드: 코인별 수익/손실 시각화
- [ ] 실시간 수익 히스토리 그래프 추가
- [ ] 모바일 반응형 레이아웃

### 보안
- [ ] 조회 API (GET) — 인증 불필요, 제어 API (POST) — 토큰 인증 필요
- [ ] `.env`에 `DASHBOARD_TOKEN` 추가, Authorization 헤더 검증

---

## NAS 배포 (DS920+, Docker)

### 경량화 분리
- [ ] `Dockerfile` 작성 (프로젝트 루트)
- [ ] `requirements-nas.txt` — torch/SB3/gymnasium 제거, antifragile 전략 전용
  - 필요: `ccxt`, `flask`, `pandas`, `numpy`, `scikit-learn`, `joblib`, `pyyaml`
- [ ] `live_tools/live_trader.py` — torch import 스킵 처리 (antifragile only 시)
- [ ] Docker Compose 작성 (로그 볼륨 마운트, `.env` 바인드)

### 배포 절차
- [ ] DSM Container Manager GUI 설정
- [ ] 포트 8765 방화벽 설정
- [ ] 부팅 시 자동 시작 (`--restart unless-stopped`)

---

## 참고

| 파일 | 내용 |
|------|------|
| `STRATEGY_ANTIFRAGILE.md` | 전략 파라미터, 스윕 결과, 실험 이력 |
| `SCRIPTS.md` | 스크립트 사용법 |
| `config/af_params.py` | 프리셋/수수료 단일 소스 |
| `logs/live_state*.json` | 실시간 포지션 상태 |
