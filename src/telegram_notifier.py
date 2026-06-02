"""
[Telegram Notifier] 트레이딩 봇 상태를 텔레그램으로 전송.

사용법:
  python src/telegram_notifier.py           # 즉시 1회 전송
  python src/telegram_notifier.py --loop    # 15분 주기 반복
"""

import os
import sys
import time
import argparse
import json
import urllib.request
import urllib.parse
import urllib.error
from datetime import datetime
from pathlib import Path

ENV_PATH = Path(__file__).parent.parent / ".env"


# ── .env 로드 ─────────────────────────────────────────────────────────────────

def load_env(path: Path = ENV_PATH) -> dict:
    env = {}
    if path.exists():
        for line in path.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                env[k.strip()] = v.strip()
    return env


def get_credentials() -> tuple[str, str]:
    env     = load_env()
    token   = env.get("TELEGRAM_BOT_TOKEN") or os.environ.get("TELEGRAM_BOT_TOKEN", "")
    chat_id = env.get("TELEGRAM_CHAT_ID")   or os.environ.get("TELEGRAM_CHAT_ID",   "")
    if not token:
        raise ValueError(".env에 TELEGRAM_BOT_TOKEN이 없습니다.")
    if not chat_id:
        raise ValueError(".env에 TELEGRAM_CHAT_ID가 없습니다. setup_chat_id()를 먼저 실행하세요.")
    return token, chat_id


# ── chat_id 자동 설정 ─────────────────────────────────────────────────────────

def setup_chat_id():
    """봇에 메시지를 보낸 사람의 chat_id를 자동으로 .env에 저장."""
    env   = load_env()
    token = env.get("TELEGRAM_BOT_TOKEN") or os.environ.get("TELEGRAM_BOT_TOKEN", "")
    if not token:
        print("TELEGRAM_BOT_TOKEN이 없습니다.")
        return

    url  = f"https://api.telegram.org/bot{token}/getUpdates"
    req  = urllib.request.Request(url)
    with urllib.request.urlopen(req, timeout=10) as resp:
        data = json.loads(resp.read())

    results = data.get("result", [])
    if not results:
        print("업데이트 없음. 텔레그램에서 봇에게 /start 를 보내고 다시 실행하세요.")
        return

    chat_id = None
    for r in results:
        msg  = r.get("message") or r.get("channel_post") or {}
        chat = msg.get("chat", {})
        if chat.get("id"):
            chat_id = str(chat["id"])
            name    = chat.get("title") or chat.get("username") or chat.get("first_name", "")
            print(f"chat_id 발견: {chat_id} ({name})")
            break

    if chat_id:
        content = ENV_PATH.read_text() if ENV_PATH.exists() else ""
        if "TELEGRAM_CHAT_ID=" in content:
            lines = [
                f"TELEGRAM_CHAT_ID={chat_id}" if l.startswith("TELEGRAM_CHAT_ID=") else l
                for l in content.splitlines()
            ]
            ENV_PATH.write_text("\n".join(lines) + "\n")
        else:
            with open(ENV_PATH, "a") as f:
                f.write(f"TELEGRAM_CHAT_ID={chat_id}\n")
        print(f".env 에 TELEGRAM_CHAT_ID={chat_id} 저장 완료.")
    else:
        print("chat_id를 찾지 못했습니다.")


# ── 메시지 전송 ───────────────────────────────────────────────────────────────

def send_message(text: str, token: str, chat_id: str) -> bool:
    url     = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = json.dumps({
        "chat_id":    chat_id,
        "text":       text,
        "parse_mode": "HTML",
    }).encode()
    req = urllib.request.Request(url, data=payload,
                                  headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            result = json.loads(resp.read())
            return result.get("ok", False)
    except urllib.error.URLError as e:
        print(f"[Telegram] 전송 실패: {e}")
        return False


# ── 상태 보고 수집 ────────────────────────────────────────────────────────────

def collect_status() -> str:
    now    = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines  = [f"<b>🤖 트레이딩 봇 상태 보고</b>", f"⏰ {now}\n"]

    root = Path(__file__).parent.parent

    # 모델 파일 존재 여부
    model_files = {
        "Long Expert":    root / "models/signal_model/expert_long.pt",
        "Short Expert":   root / "models/signal_model/expert_short.pt",
        "Context Expert": root / "models/signal_model/expert_context.pt",
        "RL Best Model":  root / "models/signal_model/rl_best/best_model.zip",
        "Scaler":         root / "models/production/btc_scaler.pkl",
        "Params v17":     root / "models/production/btc_params.json",
    }
    lines.append("<b>📦 모델 파일</b>")
    for name, path in model_files.items():
        icon = "✅" if path.exists() else "❌"
        lines.append(f"  {icon} {name}")

    # 데이터 파일 존재 여부 + 행 수
    data_files = {
        "train_signals": root / "data/train_signals.parquet",
        "val_signals":   root / "data/val_signals.parquet",
        "test_signals":  root / "data/test_signals.parquet",
    }
    lines.append("\n<b>📊 데이터 파일</b>")
    for name, path in data_files.items():
        if path.exists():
            try:
                import pandas as pd
                n = len(pd.read_parquet(path))
                lines.append(f"  ✅ {name}: {n:,} rows")
            except Exception:
                lines.append(f"  ⚠️ {name}: 읽기 오류")
        else:
            lines.append(f"  ❌ {name}: 없음")

    # 백테스트 결과
    summary_path = root / "logs/backtest/backtest_summary.csv"
    lines.append("\n<b>📈 백테스트 결과</b>")
    if summary_path.exists():
        try:
            import pandas as pd
            df = pd.read_csv(summary_path, index_col=0)
            for strategy, row in df.iterrows():
                ret = row.get("total_return", "N/A")
                mdd = row.get("mdd", "N/A")
                lines.append(f"  • {strategy}: 수익률={ret}% / MDD={mdd}%")
        except Exception as e:
            lines.append(f"  ⚠️ 읽기 오류: {e}")
    else:
        lines.append("  ❌ 백테스트 미실행")

    # 최근 에러 로그
    log_dir = root / "logs"
    error_lines = []
    if log_dir.exists():
        for log_file in sorted(log_dir.rglob("*.log"), key=lambda f: f.stat().st_mtime, reverse=True)[:3]:
            try:
                content = log_file.read_text(errors="ignore").splitlines()
                errs = [l for l in content if "ERROR" in l or "Traceback" in l][-3:]
                if errs:
                    error_lines.extend(errs)
            except Exception:
                pass

    lines.append("\n<b>⚠️ 최근 에러 로그</b>")
    if error_lines:
        for e in error_lines[:5]:
            lines.append(f"  {e[:100]}")
    else:
        lines.append("  이상 없음 ✅")

    return "\n".join(lines)


# ── 트레이드 알림 (live_trader에서 호출) ──────────────────────────────────────

def send_trade_alert(text: str) -> bool:
    """진입/청산/CB 등 단발 이벤트 알림. 실패해도 예외를 올리지 않음."""
    try:
        token, chat_id = get_credentials()
        return send_message(text, token, chat_id)
    except Exception as e:
        print(f"[Telegram] 알림 실패 (무시): {e}")
        return False


# ── 메인 ──────────────────────────────────────────────────────────────────────

def notify_once():
    token, chat_id = get_credentials()
    status = collect_status()
    ok = send_message(status, token, chat_id)
    print(f"[Telegram] 전송 {'성공' if ok else '실패'}")
    return ok


def notify_loop(interval_minutes: int = 15):
    print(f"[Telegram] {interval_minutes}분 주기 알림 시작 (Ctrl+C로 중지)")
    while True:
        try:
            notify_once()
        except Exception as e:
            print(f"[Telegram] 오류: {e}")
        time.sleep(interval_minutes * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--loop",    action="store_true", help="15분 주기 반복 실행")
    parser.add_argument("--setup",   action="store_true", help="chat_id 자동 설정")
    parser.add_argument("--interval", type=int, default=15, help="알림 간격 (분, 기본 15)")
    args = parser.parse_args()

    if args.setup:
        setup_chat_id()
    elif args.loop:
        notify_loop(args.interval)
    else:
        notify_once()
