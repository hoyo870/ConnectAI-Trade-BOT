---
strategy_id: "STRAT_06_KELTNER_SUPER_BANDS"
strategy_name: "Keltner Channels with Super Bands & Zero-Lag MA"
target_regimes: ["RANGE_BOUND", "TRENDING_PULLBACK", "MEAN_REVERSION"]
timeframes: ["5m", "15m", "1h"]
indicators:
  - id: "kc_basis"
    type: "Custom_MA"
    period: 20
    options: ["SMA", "EMA", "DEMA", "TEMA", "FRAMA", "VWMA"]
  - id: "kc_normal"
    type: "Keltner Channel"
    multiplier: 2.0
    range_type: "ATR (10)"
  - id: "kc_super"
    type: "Keltner Channel Super"
    multiplier: 3.0 # (normal_multiplier * 1.5)
parameters:
  pyramiding_max: 2
  entry_qty_pct: 50.0
  hard_stop_loss_pct: 1.0
---

# STRATEGY SPECIFICATION: STRAT_06_KELTNER_SUPER_BANDS

## 📌 1. OVERVIEW & THEORETICAL EDGE
본 전략은 보편적인 켈트너 채널 역추세(Mean-Reversion) 매매가 겪는 최대 취약점인 **'강한 원웨이 투매/급등 장세에서의 연속 청산'**을 방지하기 위해 설계된 하이브리드 알고리즘입니다. 

일반 밴드(2.0x) 바깥으로 주가가 이탈하면 회귀를 노리고 진입하되, 그 이탈의 에너지가 너무 강력하여 **슈퍼 밴드(3.0x)마저 찢고 나가는 캔들일 경우 '패닉 셀/바이' 장세로 규정하여 진입을 기각(Filter-out)**합니다. 아울러 DEMA, TEMA 등의 Zero-lag 이동평균선을 Basis로 채택해 지표의 선행성을 확보했습니다.

---

## ⚙️ 2. ENTRY LOGIC (LONG / BUY)

### 2.1 State Machine & Conditions
- **전제 필터**: 반드시 `barstate.isconfirmed == true` (확정 종가) 상태에서만 연산하여 리페인팅을 차단합니다.
- **진입 조건 (Normal Break & Super Safe)**:
  1. `Condition_A`: 직전 봉 종가가 일반 하단 밴드보다 낮을 것 (`close[1] < lower[1]`)
  2. `Condition_B`: 직전 봉 종가가 슈퍼 하단 밴드보다는 높을 것 (`close[1] > super_lower[1]`)
  - **[ Action ]**: `Condition_A == True AND Condition_B == True` 일 때 트리거 발동.

### 2.2 Pyramiding (분할 투입) 룰
- **1차 진입 (`Long_A`)**: 포지션이 `0`일 때 자본의 **50%** 투입. (만약 숏을 보유 중이었다면 숏 전량 청산 후 스위칭 진입)
- **2차 진입 (`Long_B`)**: `Long_A`를 보유한 상태에서 주가가 더 하락하여 **위의 진입 조건이 다시 한번 출현할 경우** 남은 자본 **50%** 추가 투입 (최대 2회 매집 완료).

---

## ⚙️ 3. ENTRY LOGIC (SHORT / SELL)

### 3.1 State Machine & Conditions
- **진입 조건**:
  1. `Condition_A`: 직전 봉 종가가 일반 상단 밴드보다 높을 것 (`close[1] > upper[1]`)
  2. `Condition_B`: 직전 봉 종가가 슈퍼 상단 밴드보다는 낮을 것 (`close[1] < super_upper[1]`)
  - **[ Action ]**: 두 조건 동시 충족 시 트리거 발동.

### 3.2 Pyramiding 룰
- **1차 진입 (`Short_A`)**: 포지션 `0`일 때 **50%** 투입. (롱 보유 중이었으면 전량 스위칭 청산)
- **2차 진입 (`Short_B`)**: 1차 보유 중 조건 재충족 시 남은 **50%** 투입.

---

## 🛡️ 4. POSITION MANAGEMENT & EXIT RULES

### 4.1 하드 스탑로스 (Hard Stop Loss) : 버그 보정 완료
- **기준 가격**: 물타기로 인해 왜곡되는 `position_avg_price`를 배제하고, **무조건 '최초 체결된 1차 물량의 진입 단가(`entry_price(0)`)'를 기준 닻(Anchor)으로 설정**합니다.
  - `Long_SL_Price` = `1차 진입가 * (1.0 - 0.01)` (-1% 손절)
  - `Short_SL_Price` = `1차 진입가 * (1.0 + 0.01)` (+1% 손절)
- **집행**: 주가가 해당 가격에 터치하는 즉시 1차, 2차 물량 **동시 전량 시장가 손절**.

### 4.2 중심선/반대밴드 회귀 익절 (Take Profit) : 좀비 버그 보정 완료
- **체크 방식**: 확정봉 종가뿐만 아니라, 봉 진행 중의 꼬리 터치(`OHLC`)까지 밴드 인식 범위로 확장합니다.
- **롱 익절**: 포지션 보유 중 주가의 `Open, High, Low, Close` 중 하나라도 **일반 상단 밴드(`upper`)를 터치하는 순간, 보유 중인 `Long_A`와 `Long_B`를 전량 일괄 청산**(`close_all`)하여 잔여 물량이 방치되는 좀비 포지션을 차단합니다.
- **숏 익절**: 주가가 **일반 하단 밴드(`lower`)** 터치 시 전량 일괄 청산.

---

## 💻 5. COMPLETE PINE SCRIPT v5 CODE
```pinescript
//@version=5
strategy("Keltner Channels SuperBands (STRAT_06)", overlay=true, initial_capital=1000, currency=currency.USDT, default_qty_type=strategy.percent_of_equity, default_qty_value=50, pyramiding=2)

import TradingView/ta/7 as ta7

// --- Inputs ---
len = input.int(20, minval=1, title="Channel Length")
mult = input.float(2.0, title="Normal Multiplier", step=0.1)
src = input(close, title="Source")
BandsStyle = input.string("Average True Range", options=["Average True Range", "True Range", "Range"], title="Range Type")
atrlength = input.int(10, title="ATR Length")
maType = input.string("SMA", title="MA Type", options=["SMA", "EMA", "DEMA", "TEMA", "FRAMA", "VWMA"])
slPctInput = input.float(1.0, title="Stop Loss % (1차 진입가 기반)", minval=0.05, step=0.05) / 100.0

// --- Helper: Zero-lag MAs ---
ma(source, length, _type) =>
    switch _type
        "SMA"   => ta.sma(source, length)
        "EMA"   => ta.ema(source, length)
        "DEMA"  => ta7.dema(source, length)
        "TEMA"  => ta7.tema(source, length)
        "FRAMA" => ta7.frama(source, length)
        "VWMA"  => ta.vwma(source, length)

// --- Bands Calculation ---
basis   = ma(src, len, maType)
rangema = BandsStyle == "True Range" ? ta.tr(true) : BandsStyle == "Average True Range" ? ta.atr(atrlength) : ta.rma(high - low, len)

upper   = basis + rangema * mult
lower   = basis - rangema * mult
super_upper = basis + rangema * (mult * 1.5)
super_lower = basis - rangema * (mult * 1.5)

// --- Plotting ---
u = plot(upper, color=#2962FF, title="Upper")
plot(basis, color=#2962FF, title="Basis")
l = plot(lower, color=#2962FF, title="Lower")
fill(u, l, color=color.rgb(33, 150, 243, 95))

super_u = plot(super_upper, color=#ff2929, title="Super Upper")
super_l = plot(super_lower, color=#ff2929, title="Super Lower")

// --- Signals & Logic ---
aboveUpper = close > upper
belowLower = close < lower
aboveSuper = close > super_upper
belowSuper = close < super_lower

awaitBar = barstate.isconfirmed

enterShortSig = awaitBar and aboveUpper[1] and not aboveSuper[1]
enterLongSig  = awaitBar and belowLower[1] and not belowSuper[1]

var bool longAdded  = false
var bool shortAdded = false

if strategy.position_size == 0
    longAdded  := false
    shortAdded := false

// Entries
if enterLongSig
    if strategy.position_size < 0
        strategy.close_all(comment="Switch to Long")
        shortAdded := false
    if strategy.position_size <= 0
        longAdded := false
        strategy.entry("Long_A", strategy.long)
    else if strategy.position_size > 0 and not longAdded
        strategy.entry("Long_B", strategy.long)
        longAdded := true

if enterShortSig
    if strategy.position_size > 0
        strategy.close_all(comment="Switch to Short")
        longAdded := false
    if strategy.position_size >= 0
        shortAdded := false
        strategy.entry("Short_A", strategy.short)
    else if strategy.position_size < 0 and not shortAdded
        strategy.entry("Short_B", strategy.short)
        shortAdded := true

// Fixed SL (Anchor to First Entry)
first_entry_px = strategy.opentrades.entry_price(0)
longStop  = strategy.position_size > 0 ? first_entry_px * (1.0 - slPctInput) : na
shortStop = strategy.position_size < 0 ? first_entry_px * (1.0 + slPctInput) : na

strategy.exit("SL_Long_A",  from_entry="Long_A",  stop=longStop)
strategy.exit("SL_Long_B",  from_entry="Long_B",  stop=longStop)
strategy.exit("SL_Short_A", from_entry="Short_A", stop=shortStop)
strategy.exit("SL_Short_B", from_entry="Short_B", stop=shortStop)

// Fixed Clean TP
aboveUpperOHLC = open > upper or high > upper or low > upper or close > upper
belowLowerOHLC = open < lower or high < lower or low < lower or close < lower

if awaitBar
    if strategy.position_size > 0 and aboveUpperOHLC
        strategy.close_all(comment="TP @ Upper")
        longAdded := false
    if strategy.position_size < 0 and belowLowerOHLC
        strategy.close_all(comment="TP @ Lower")
        shortAdded := false

plot(longStop,  title="SL_Line_Long",  style=plot.style_linebr, color=color.red, linewidth=2)
plot(shortStop, title="SL_Line_Short", style=plot.style_linebr, color=color.orange, linewidth=2)