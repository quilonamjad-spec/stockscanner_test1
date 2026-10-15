"""Stretch Monitor - Streamlit + yfinance version (quick-look, delayed data).
Run locally:  streamlit run streamlit_app.py
"""
import time
import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf

# yfinance interval -> how much history to ask for
YF = {"ONE_DAY": ("1d", "2y"), "ONE_HOUR": ("1h", "6mo"), "FIFTEEN_MINUTE": ("15m", "55d"), "FIVE_MINUTE": ("5m", "55d")}
TF_LABEL = {"Daily": "ONE_DAY", "1 hour": "ONE_HOUR", "15 min": "FIFTEEN_MINUTE", "5 min": "FIVE_MINUTE"}

@st.cache_data(ttl=300, show_spinner=False)
def _download(sym, tf):
    interval, period = YF[tf]
    raw = yf.Ticker(sym.upper() + ".NS").history(period=period, interval=interval, auto_adjust=False)
    if raw is None or raw.empty:
        raise ValueError("No data from Yahoo for %s.NS (check the NSE symbol)" % sym.upper())
    raw = raw.reset_index()
    tcol = raw.columns[0]
    df = pd.DataFrame({"ts": raw[tcol].astype(str).str[:16], "open": raw["Open"], "high": raw["High"],
                       "low": raw["Low"], "close": raw["Close"], "volume": raw["Volume"]})
    return df.dropna(subset=["close"]).reset_index(drop=True)

def candles(sym, tf):
    return _download(sym, tf).copy()

# ---------------------------------------------------------------- analysis (same logic as the Angel One version)
def build(df):
    c, h, l, o, v = df.close, df.high, df.low, df.open, df.volume
    pc = c.shift()
    tr = pd.concat([h - l, (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1)
    df["atr"] = tr.ewm(alpha=1/14, adjust=False).mean()
    d = c.diff()
    up = d.clip(lower=0).ewm(alpha=1/14, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1/14, adjust=False).mean()
    df["rsi"] = 100 - 100 / (1 + up / dn.replace(0, np.nan))
    df["ema20"] = c.ewm(span=20, adjust=False).mean()
    df["ema50"] = c.ewm(span=50, adjust=False).mean()
    df["st20"] = (c - df.ema20) / df.atr
    df["st50"] = (c - df.ema50) / df.atr
    m, s = c.rolling(20).mean(), c.rolling(20).std()
    df["pb"] = (c - (m - 2 * s)) / (4 * s)
    macd = c.ewm(span=12, adjust=False).mean() - c.ewm(span=26, adjust=False).mean()
    df["mh"] = macd - macd.ewm(span=9, adjust=False).mean()
    lo, hi = l.rolling(14).min(), h.rolling(14).max()
    df["stoch"] = 100 * (c - lo) / (hi - lo)
    df["volr"] = v / v.rolling(20).mean()
    cl = lambda x, a, b: ((x - a) / (b - a)).clip(0, 1)
    # --- extension components (0..1): RSI and ATR carry the most weight
    df["s_rsi"] = cl(df.rsi, 60, 80)
    df["s_atr"] = cl(df.st20, 1, 3)
    df["s_bb"] = cl(df.pb, 0.8, 1.2)
    df["s_50"] = cl(df.st50, 2, 5)
    df["ext"] = 100 * (0.35 * df.s_rsi + 0.35 * df.s_atr + 0.15 * df.s_bb + 0.15 * df.s_50)
    # --- downside stretch (mirror image) -> buy-area / bounce score
    df["dext"] = 100 * (0.35 * cl(40 - df.rsi, 0, 20) + 0.35 * cl(-df.st20, 1, 3)
                        + 0.15 * cl(0.2 - df.pb, 0, 0.4) + 0.15 * cl(-df.st50, 2, 5))
    # --- pullback triggers (0..1)
    df["s_div"] = ((c >= c.rolling(15).max() * 0.995) & (df.rsi < df.rsi.rolling(15).max() - 3)).astype(float)
    fall1 = (df.mh > 0) & (df.mh < df.mh.shift())
    fall2 = fall1 & (df.mh.shift() < df.mh.shift(2))
    df["s_macd"] = fall1 * 0.5 + fall2 * 0.5
    wick = (h - np.maximum(o, c)) / (h - l).replace(0, np.nan)
    df["s_wick"] = cl(wick.fillna(0), 0.3, 0.7)
    climax = (df.volr > 1.8) & (c > o) & (df.st20 > 1.5)
    dry = (df.volr < 0.7) & (c >= c.rolling(15).max() * 0.995)
    df["s_vol"] = climax * 1.0 + (~climax & dry) * 0.6
    df["s_sto"] = cl(df.stoch.fillna(50), 80, 100)
    run = (c > pc).astype(int)
    streak = run.groupby((run == 0).cumsum()).cumsum()
    df["s_run"] = (streak / 6).clip(0, 1)
    df["prob"] = (0.40 * df.ext / 100 + 0.15 * df.s_div + 0.15 * df.s_macd + 0.10 * df.s_wick
                  + 0.10 * df.s_vol + 0.05 * df.s_sto + 0.05 * df.s_run)
    # --- breakout setup (0..1): near highs, squeeze, trend, volume, momentum, room to run
    df["res"] = h.shift(1).rolling(20).max()
    dist = (df.res - c) / df.atr
    df["b_prox"] = pd.Series(np.where(c > df.res, 1.0, cl(1 - dist / 1.5, 0, 1)), index=df.index)
    bbw = 4 * s / m
    df["bbr"] = bbw / bbw.rolling(100).mean()
    df["b_sq"] = cl(df.bbr, 1, 0.5)
    df["b_trend"] = (c > df.ema50) * 0.5 + ((df.ema20 > df.ema50) & (c > df.ema20)) * 0.5
    df["b_vol"] = cl(df.volr, 1, 2)
    df["b_mom"] = cl(df.rsi, 50, 58) * cl(df.rsi, 78, 68)
    df["b_room"] = 1 - df.ext / 100
    df["bk"] = (0.25 * df.b_prox + 0.15 * df.b_sq + 0.20 * df.b_trend + 0.15 * df.b_vol
                + 0.10 * df.b_mom + 0.15 * df.b_room)
    # --- history: after similar stretch (ext>=60), did price drop >= 1 ATR within 5 bars?
    fwd_low = l.rolling(5).min().shift(-5)
    df["hit"] = (fwd_low <= c - df.atr).astype(float).where(fwd_low.notna())
    fwd_hi = h.rolling(5).max().shift(-5)
    df["bhit"] = (fwd_hi >= c + df.atr).astype(float).where(fwd_hi.notna())
    return df

def analyze(sym, tf):
    df = build(candles(sym, tf))
    if len(df) < 60:
        raise ValueError("Not enough history (%d bars)" % len(df))
    x = df.iloc[-1]
    past = df.iloc[:-6]
    sim = past[past.ext >= 60].dropna(subset=["hit"])
    n = len(sim)
    hit = float(sim.hit.mean()) if n else None
    model = float(x.prob)
    blended = n >= 8 and float(x.ext) >= 60   # history only counts when stretched today
    prob = 0.6 * model + 0.4 * hit if blended else model
    sig = [
        ("RSI (14)", f"{x.rsi:.1f}", x.s_rsi, "RSI above 70 = overbought"),
        ("ATR stretch vs EMA20", f"{x.st20:+.2f} ATR", x.s_atr, "Distance from 20 EMA in ATR units; >2 is stretched"),
        ("Bollinger %B", f"{x.pb:.2f}", x.s_bb, "Above 1 = closing outside upper band"),
        ("ATR stretch vs EMA50", f"{x.st50:+.2f} ATR", x.s_50, "Distance from 50 EMA in ATR units"),
        ("RSI bearish divergence", "yes" if x.s_div else "no", x.s_div, "Price at highs while RSI makes a lower high"),
        ("MACD momentum fade", f"{x.mh:+.2f}", x.s_macd, "Histogram positive but shrinking"),
        ("Upper-wick rejection", f"{x.s_wick*100:.0f}%", x.s_wick, "Long upper shadow on latest candle"),
        ("Volume exhaustion", f"{x.volr:.1f}x avg", x.s_vol, "Climax volume or dry-up at highs"),
        ("Stochastic (14)", f"{x.stoch:.0f}", x.s_sto, "Above 80 = overbought zone"),
        ("Up-bar streak", f"{x.s_run*6:.0f} bars", x.s_run, "Consecutive higher closes"),
    ]
    sc = [float(s[2]) for s in sig]
    yes = [s for s in sc if s >= 0.6]; no = [s for s in sc if s <= 0.3]
    voters = len(yes) + len(no)
    agree = max(len(yes), len(no)) / voters if voters else 0
    strength = (np.mean(yes) if len(yes) >= len(no) and yes else np.mean([1 - s for s in no]) if no else 0)
    data_f = min(1, len(df) / 200) * (0.7 + 0.3 * min(1, n / 20))
    conf = 100 * (0.5 * agree + 0.5 * strength) * data_f
    ext = float(x.ext)
    dext = float(x.dext)
    bounce = 0.8 * dext / 100 + 0.2 * float(np.clip((20 - x.stoch) / 20, 0, 1))
    uptrend = x.ema20 > x.ema50
    if ext >= 40 and ext >= dext:
        area, direction, dprob = "Sell area", "Down", prob
        why = "Price is stretched above its averages; a pullback lower is the likelier path."
    elif dext >= 40:
        area, direction, dprob = "Buy area", "Up", bounce
        why = "Price is stretched below its averages; a bounce higher is the likelier path."
    elif uptrend and abs(x.st20) <= 0.7 and x.rsi < 55:
        area, direction, dprob = "Buy area", "Up", max(bounce, 0.45)
        why = "Uptrend (20 EMA above 50 EMA) with price back near the 20 EMA: a dip-buy zone."
    else:
        area, direction, dprob = "Neutral", "Sideways", prob
        why = "No stretch in either direction; no edge from mean reversion."
    pb_warn, pb_calm = len(yes), len(no)
    # --- breakout view
    bsim = past[past.bk >= 0.6].dropna(subset=["bhit"])
    bn = len(bsim)
    bk = float(x.bk)
    bk_prob = 0.6 * bk + 0.4 * float(bsim.bhit.mean()) if (bn >= 8 and bk >= 0.6) else bk
    bsig = [
        ("Near / above 20-bar high", f"{(x.res - x.close) / x.atr:+.2f} ATR to high", x.b_prox, "Negative means already above the prior 20-bar high"),
        ("Volatility squeeze", f"{x.bbr:.2f}x normal", x.b_sq, "Tight Bollinger width often precedes expansion"),
        ("Trend alignment", "up" if x.b_trend >= 1 else "mixed" if x.b_trend > 0 else "down", x.b_trend, "Price above 20 EMA, 20 EMA above 50 EMA"),
        ("Volume expansion", f"{x.volr:.1f}x avg", x.b_vol, "Breakouts need participation"),
        ("Momentum zone", f"RSI {x.rsi:.0f}", x.b_mom, "RSI 58-68 = strong but not exhausted"),
        ("Room to run", f"{100 - ext:.0f}", x.b_room, "Low overextension leaves space to move up"),
    ]
    why2 = None
    if area == "Sell area":
        if prob >= 0.5 and bk_prob >= 0.55:
            outlook, area, direction = "Dip, then continuation", "Wait for dip", "Down"
            why2 = ("Strong trend and a fresh breakout, but price is stretched. A dip toward the breakout level / 20 EMA "
                    "is likely first; that dip is the better entry before the trend resumes.")
        elif bk_prob >= 0.55:
            outlook = "Continuation (chase risk)"
            why2 = "Trend and breakout are strong and pullback signals are weak, but price is stretched, so buying here carries risk."
        else:
            outlook = "Pullback likely" if prob >= 0.5 else "Pullback possible"
    elif area == "Buy area":
        outlook = "Bounce likely"
    else:
        outlook = "Breakout likely" if bk_prob >= 0.55 else "Breakout watch" if bk_prob >= 0.4 else "Range / no edge"
    if why2:
        why = why2
    conf_basis = "breakout signals" if (outlook.startswith("Breakout") or "ontinuation" in outlook) else "pullback signals"
    if conf_basis == "breakout signals":   # confidence should judge the call being made
        sc = [float(b[2]) for b in bsig]
        yes = [v for v in sc if v >= 0.6]; no = [v for v in sc if v <= 0.3]
        voters = len(yes) + len(no)
        agree = max(len(yes), len(no)) / voters if voters else 0
        strength = (np.mean(yes) if len(yes) >= len(no) and yes else np.mean([1 - v for v in no]) if no else 0)
        conf = 100 * (0.5 * agree + 0.5 * strength) * min(1, len(df) / 200) * (0.7 + 0.3 * min(1, bn / 20))
    oversold = x.rsi < 30 and x.st20 < -2
    verdict = "Oversold" if oversold else "Overextended" if ext >= 70 else "Stretched" if ext >= 45 else "Normal"
    return {
        "symbol": sym.upper(), "bars": len(df), "last_bar": str(x.ts)[:16].replace("T", " "), "price": round(float(x.close), 2), "atr": round(float(x.atr), 2),
        "rsi": round(float(x.rsi), 1), "stretch": round(float(x.st20), 2), "ext": round(ext),
        "prob": round(prob * 100), "model_prob": round(model * 100),
        "hist_hit": None if hit is None else round(hit * 100), "hist_n": n,
        "consensus_yes": len(yes), "consensus_total": len(sig), "consensus_no": len(no),
        "confidence": round(conf), "conf_basis": conf_basis,
        "pb_warn": pb_warn, "pb_calm": pb_calm, "pb_total": len(sig),
        "bk_support": int(sum(b[2] >= 0.6 for b in bsig)), "bk_against": int(sum(b[2] <= 0.3 for b in bsig)), "bk_total": len(bsig), "outlook": outlook, "bk_prob": round(bk_prob * 100),
        "bk_hist_n": bn, "resistance": round(float(x.res), 2),
        "bk_levels": {"trigger": round(float(x.res), 2), "t1": round(float(x.res + x.atr), 2),
                      "t2": round(float(x.res + 2 * x.atr), 2), "fail": round(float(x.ema20), 2)},
        "bk_signals": [{"name": b[0], "value": b[1], "score": round(float(b[2]), 2), "note": b[3]} for b in bsig], "area": area, "direction": direction,
        "dir_prob": round(dprob * 100), "why": why, "dext": round(dext), "bounce": round(bounce * 100),
        "up_zones": {"ema20": round(float(x.ema20), 2), "plus1atr": round(float(x.close + x.atr), 2),
                     "plus2atr": round(float(x.close + 2 * x.atr), 2)}, "verdict": verdict,
        "bias": "Pullback likely" if prob >= 0.6 else "Pullback possible" if prob >= 0.4 else "Low pullback risk",
        "zones": {"ema20": round(float(x.ema20), 2), "minus1atr": round(float(x.close - x.atr), 2),
                  "minus2atr": round(float(x.close - 2 * x.atr), 2)},
        "signals": [{"name": s[0], "value": s[1], "score": round(float(s[2]), 2), "note": s[3]} for s in sig],
    }


# ---------------------------------------------------------------- UI

GREEN, RED, AMBER, NAVY, GREY, TRACK = "#2F8F6F", "#CF3F2A", "#D99A00", "#4F7CAC", "#6B7889", "#3A4658"

def chip(text, color):
    fg = "#1b1500" if color == AMBER else "#fff"
    return (f"<span style='display:inline-block;background:{color};color:{fg};font-weight:600;font-size:13px;"
            f"padding:3px 11px;border-radius:99px;margin:0 6px 6px 0'>{text}</span>")

def chips(r):
    v = {"Overextended": RED, "Stretched": AMBER, "Oversold": NAVY}.get(r["verdict"], GREEN)
    a = {"Buy area": GREEN, "Sell area": RED, "Wait for dip": AMBER}.get(r["area"], GREY)
    arrow = {"Up": "↑", "Down": "↓"}.get(r["direction"], "↔")
    o = r["outlook"]
    oc = (NAVY if ("Breakout likely" in o or "ontinuation" in o) else AMBER if "watch" in o or "possible" in o
          else GREY if "Range" in o else GREEN if "Bounce" in o else RED)
    return chip(r["verdict"], v) + chip(f"{r['area']} {arrow}", a) + chip(o, oc)

def dots(signals, good_high):
    """One bar segment per signal. good_high=True: high score = green (breakout). False: high score = red (pullback)."""
    hi, lo = (GREEN, RED) if good_high else (RED, GREEN)
    segs = "".join(f"<div style='flex:1;height:8px;border-radius:2px;background:"
                   f"{hi if x['score'] >= .6 else lo if x['score'] <= .3 else TRACK}'></div>" for x in signals)
    return f"<div style='display:flex;gap:4px;margin:4px 0 12px'>{segs}</div>"

def sig_table(signals, good_high):
    hi, lo = (GREEN, RED) if good_high else (RED, GREEN)
    rows = "".join(
        f"<tr><td style='padding:5px 4px;border-bottom:1px solid #3A4658'>"
        f"<span style='display:inline-block;width:9px;height:9px;border-radius:50%;margin-right:7px;background:"
        f"{hi if x['score'] >= .6 else lo if x['score'] <= .3 else AMBER}'></span>{x['name']}<br>"
        f"<small style='opacity:.65'>{x['note']}</small></td>"
        f"<td style='padding:5px 4px;border-bottom:1px solid #3A4658;text-align:right;white-space:nowrap'>{x['value']}</td></tr>"
        for x in signals)
    return f"<table style='width:100%;border-collapse:collapse;font-size:13px'>{rows}</table>"

def card(r):
    if "error" in r:
        st.error(f"**{r['symbol']}** - {r['error']}")
        return
    with st.container(border=True):
        a, b = st.columns([3, 2])
        a.subheader(r["symbol"])
        b.markdown(f"<div style='text-align:right'>₹{r['price']} · ATR {r['atr']}</div>", unsafe_allow_html=True)
        st.markdown(chips(r), unsafe_allow_html=True)
        st.caption(r["why"])
        pos = max(0, min(100, 50 + r["stretch"] / 3 * 50))
        st.markdown(
            "<div style='position:relative;height:26px;margin:6px 0'>"
            "<div style='position:absolute;top:11px;left:0;right:0;height:6px;border-radius:3px;"
            "background:linear-gradient(90deg,#2F8F6F 0,#2F8F6F 33%,#D99A00 58%,#CF3F2A 85%)'></div>"
            "<div style='position:absolute;top:7px;left:50%;width:2px;height:14px;background:#888'></div>"
            f"<div style='position:absolute;top:0;left:{pos}%;transform:translateX(-50%);font-size:16px'>▼</div></div>"
            f"<div style='display:flex;justify-content:space-between;font-size:12px;opacity:.7'>"
            f"<span>-3 ATR</span><span>20 EMA · now {r['stretch']:+.2f} ATR</span><span>+3 ATR</span></div>",
            unsafe_allow_html=True)
        c = st.columns(4)
        c[0].metric("Overextension", r["ext"])
        c[1].metric("Bounce odds" if r["direction"] == "Up" else "Pullback odds", f"{r['dir_prob']}%")
        c[2].metric("Breakout odds", f"{r['bk_prob']}%")
        c[3].metric("Confidence", r["confidence"])
        st.write(f"Pullback signals: **{r['pb_warn']}** of {r['pb_total']} warn, {r['pb_calm']} calm")
        st.markdown(dots(r["signals"], False), unsafe_allow_html=True)
        st.write(f"Breakout signals: **{r['bk_support']}** of {r['bk_total']} support, {r['bk_against']} against")
        st.markdown(dots(r["bk_signals"], True), unsafe_allow_html=True)
        st.caption(f"Confidence is based on the {r['conf_basis']}.")
        if r["direction"] == "Up":
            z = r["up_zones"]; st.write(f"RSI **{r['rsi']}** · If it bounces: 20 EMA **{z['ema20']}**, +1 ATR **{z['plus1atr']}**, +2 ATR **{z['plus2atr']}**")
        else:
            z = r["zones"]; st.write(f"RSI **{r['rsi']}** · If it pulls back: 20 EMA **{z['ema20']}**, -1 ATR **{z['minus1atr']}**, -2 ATR **{z['minus2atr']}**")
        k = r["bk_levels"]
        if any(w in r["outlook"] for w in ("Breakout", "ontinuation")):
            lead = (f"Already closed above the prior 20-bar high **{k['trigger']}** (retest level)" if r["price"] > k["trigger"]
                    else f"Breakout trigger: close above **{k['trigger']}**")
            st.write(f"{lead} · targets **{k['t1']}**, **{k['t2']}** · fails under 20 EMA **{k['fail']}**")
        hist = f"{r['hist_hit']}% ({r['hist_n']} past cases)" if r["hist_n"] >= 8 else f"too few past cases ({r['hist_n']})"
        st.caption(f"History check: {hist}. Rule-based odds {r['model_prob']}%. Data: {r['bars']} candles, latest {r['last_bar']}.")
        with st.expander("All signals"):
            st.markdown("**Pullback signals** (red = warns, green = calm)")
            st.markdown(sig_table(r["signals"], False), unsafe_allow_html=True)
            st.markdown("**Breakout signals** (green = supports, red = against)")
            st.markdown(sig_table(r["bk_signals"], True), unsafe_allow_html=True)

def main():
    st.set_page_config(page_title="Stretch Monitor", layout="wide")
    st.title("Stretch Monitor")
    st.caption("Overextension (RSI + ATR), pullback and breakout odds, consensus and confidence. "
               "Data from Yahoo Finance: delayed and less exact than Angel One. A screening aid, not advice.")
    c1, c2, c3 = st.columns([4, 1, 1])
    syms = c1.text_input("NSE symbols, comma separated", "RELIANCE, TCS, INFY, HDFCBANK")
    tf = TF_LABEL[c2.selectbox("Timeframe", list(TF_LABEL), index=0)]
    c3.write(""); c3.write("")
    go = c3.button("Analyze", type="primary", use_container_width=True)
    if go:
        out = []
        for sym in [x.strip() for x in syms.split(",") if x.strip()][:20]:
            try:
                out.append(analyze(sym, tf))
            except Exception as e:
                out.append({"symbol": sym.upper(), "error": str(e)})
        out.sort(key=lambda r: r.get("dir_prob", -1), reverse=True)
        st.session_state["out"] = out
    out = st.session_state.get("out")
    if out:
        st.caption("Sorted by odds of the expected move, highest first.")
        cols = st.columns(2)
        for i, r in enumerate(out):
            with cols[i % 2]:
                card(r)

if __name__ == "__main__":
    main()
