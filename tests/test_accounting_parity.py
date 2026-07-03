"""
회계 회귀 테스트 — 피라미딩 수익률이 거래소(실거래) 실현손익 정의와 일치함을 고정.

배경: 과거 백테스트는 피라미딩 PnL을 "최초 진입가" 기준으로 과대계상해 손실 전략을
+212%/3개월로 보이게 했다 (project_pnl_accounting_fix). 수정 후에는 수량가중 평균
진입가를 사용하며, 이 값이 거래소 청크합산 실현손익과 정확히 일치해야 한다.

이 테스트는 두 가지를 단언한다:
  A) 순수 항등식: 전략의 증분 가중평균 공식이 청크합산 정의와 대수적으로 동일.
  B) 통합: 실제 AntifragileStrategy.process_tick 이 만들어내는 avg_entry / pnl_raw 가
     실제 진입 청크들의 거래소 실현손익과 일치.

실행:  .venv/bin/python tests/test_accounting_parity.py
"""
import os, sys
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
from pathlib import Path
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from config.af_params import DEFAULT_PARAMS
from strategies.antifragile import AntifragileStrategy

TOL = 1e-9


def _exchange_return(chunks, exit_px, pos, lev):
    """거래소 실현손익 정의: Σ qty_j·(exit−p_j)·pos / capital.
    qty_j = capital·rr_j·lev/p_j 이므로 자본 정규화 시 Σ rr_j·lev·pos·(exit−p_j)/p_j."""
    return lev * sum(rr * pos * (exit_px - pj) / pj for rr, pj in chunks)


def _formula_return(rr_total, wavg, exit_px, pos, lev):
    """전략 공식: pnl_raw·lev·rr, pnl_raw = pos·(exit−wavg)/wavg."""
    return lev * rr_total * (pos * (exit_px - wavg) / wavg)


def _weighted_avg(chunks):
    """수량가중 평균: Σrr_j / Σ(rr_j/p_j)."""
    return sum(rr for rr, _ in chunks) / sum(rr / pj for rr, pj in chunks)


def test_identity():
    """A) 가중평균 공식 == 청크합산 (롱/숏, 다양한 청크)."""
    cases = [
        [(0.10, 100.0), (0.15, 110.0)],
        [(0.10, 100.0), (0.15, 101.0), (0.15, 103.0), (0.15, 106.0)],
        [(0.10, 62732.42), (0.15, 62587.27), (0.15, 62354.33), (0.15, 62249.99)],  # 실데이터 숏
        [(0.10, 1.0965), (0.15, 1.0967), (0.15, 1.0968)],
    ]
    for pos in (1, -1):
        for lev in (1, 5, 10):
            for chunks in cases:
                rr_total = sum(rr for rr, _ in chunks)
                wavg = _weighted_avg(chunks)
                exit_px = chunks[-1][1] * 1.01  # 임의 청산가
                a = _exchange_return(chunks, exit_px, pos, lev)
                b = _formula_return(rr_total, wavg, exit_px, pos, lev)
                assert abs(a - b) < TOL, f"항등식 불일치 pos={pos} lev={lev}: {a} != {b}"
    print("  [A] 항등식 (가중평균 == 청크합산): PASS")


def test_strategy_integration():
    """B) 실제 process_tick 이 만든 avg_entry/pnl_raw 가 청크합산과 일치."""
    p = {**DEFAULT_PARAMS, "add_levels": 3, "ml_threshold": None}
    s = AntifragileStrategy(p, ml_filter=None)
    atr = 2.0
    # 진입: RG 추세, rsi<=rg_rsi_lo(30) → 롱. atr >= price*0.0015 보장(100*0.0015=0.15).
    s.process_tick(price=100.0, atr=atr, rsi=10.0, trend_up=False, trend_down=False)
    assert s.pos == 1 and abs(s.avg_entry - 100.0) < TOL, "진입 실패"
    chunks = [(p["rr_base"], 100.0)]

    # 피라미딩: 가격을 충분히 올려 favorable >= next_lvl 달성(경계 1e-9 여유 확보).
    # rsi 중립(재진입 없음).
    pyr_prices = [102.0, 106.0, 112.0]
    for px in pyr_prices:
        res = s.process_tick(price=px, atr=atr, rsi=50.0, trend_up=False, trend_down=False)
        pyr = [e for e in res["events"] if e["type"] == "pyramid"]
        if pyr:
            chunks.append((p["rr_add"], px))
            wavg = _weighted_avg(chunks)
            assert abs(pyr[0]["new_avg"] - wavg) < 1e-6, \
                f"피라미딩 avg 불일치 @ {px}: {pyr[0]['new_avg']} != {wavg}"
            assert abs(s.avg_entry - wavg) < 1e-6, "strategy.avg_entry 불일치"
    assert s.add_cnt == 3, f"피라미딩 3회 기대, 실제 {s.add_cnt}"

    # 청산: trail_sl 하향 관통하도록 낮은 가격 투입 → close 이벤트 pnl_raw 확인.
    exit_px = s.trail_sl - 1.0
    res = s.process_tick(price=exit_px, atr=atr, rsi=50.0, trend_up=False, trend_down=False)
    close = [e for e in res["events"] if e["type"] == "close"]
    assert close, "청산 이벤트 없음"
    pnl_raw = close[0]["pnl_raw"]
    rr_total = sum(rr for rr, _ in chunks)
    # 공식 수익 vs 청크합산 수익 (무수수료, lev=1 정규화)
    formula = _formula_return(rr_total, _weighted_avg(chunks), exit_px, 1, 1)
    chunk = _exchange_return(chunks, exit_px, 1, 1)
    assert abs(formula - chunk) < 1e-9, f"공식≠청크 {formula} {chunk}"
    # 전략이 보고한 pnl_raw·rr 이 청크합산과 동일
    assert abs(pnl_raw * rr_total - chunk) < 1e-9, \
        f"strategy pnl_raw·rr ≠ 청크합산: {pnl_raw*rr_total} != {chunk}"
    print(f"  [B] 통합 (process_tick avg/pnl == 청크합산): PASS  (청크 {len(chunks)}개)")


def test_no_pyramid_equals_single_entry():
    """add_levels=0 이면 단일 진입과 동일 (avg_entry 고정, rr=rr_base)."""
    p = {**DEFAULT_PARAMS, "add_levels": 0, "ml_threshold": None}
    s = AntifragileStrategy(p, ml_filter=None)
    s.process_tick(price=100.0, atr=2.0, rsi=10.0, trend_up=False, trend_down=False)
    for px in [101.0, 104.0, 108.0]:
        res = s.process_tick(price=px, atr=2.0, rsi=50.0, trend_up=False, trend_down=False)
        assert not [e for e in res["events"] if e["type"] == "pyramid"], "피라미딩이 발동됨"
    assert s.add_cnt == 0 and abs(s.avg_entry - 100.0) < TOL and abs(s.rr - p["rr_base"]) < TOL
    print("  [C] add_levels=0 단일진입: PASS")


if __name__ == "__main__":
    print("회계 회귀 테스트")
    test_identity()
    test_strategy_integration()
    test_no_pyramid_equals_single_entry()
    print("ALL PASS ✅")
