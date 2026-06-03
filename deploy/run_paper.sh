#!/bin/bash
# Paper 모드 실행 스크립트
# 사용법: bash deploy/run_paper.sh

set -e
cd "$(dirname "$0")/.."

PYTHON="/opt/homebrew/Caskroom/miniforge/base/envs/cryptobot/bin/python"
LOGFILE="logs/paper.log"

mkdir -p logs

# 이미 실행 중이면 종료
if pgrep -f "live_trader.py" > /dev/null 2>&1; then
    echo "[오류] live_trader.py 이미 실행 중. 종료 후 재시작하세요."
    pgrep -fl "live_trader.py"
    exit 1
fi

echo "[시작] TRADE_MODE=paper | 로그: $LOGFILE"
TRADE_MODE=paper nohup "$PYTHON" src/live_trader.py >> "$LOGFILE" 2>&1 &
echo "PID: $!"
echo $! > logs/live_trader.pid
