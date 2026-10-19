"""Stretch Monitor: Streamlit + Yahoo Finance, one file, no database.
Run locally:   streamlit run yahooscreener.py
Deploy:        push this file and requirements.txt to GitHub, then create the app at share.streamlit.io

Type a list of symbols, press Analyze. Every stock gets a card with its buy / sell outlook:
  breakout / breakdown, bounce / rejection, pullback / rally-then-continuation, neutral watch (both triggers),
  plus candlestick confirmation, overextension, odds and confidence.
The analysis section below is the same code as the Angel One screener's engine.py, so both give identical calls.
A screening aid, not advice.
"""
import datetime as dt
import time
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf

# ================================================================== analysis (shared logic, do not edit by hand)
def _cl(x, a, b):
    """clip((x - a) / (b - a), 0, 1) on a Series, done in numpy (much faster than Series.clip)."""
    return pd.Series(np.clip((x.values - a) / (b - a), 0, 1), index=x.index)

def build(df):
    """All indicators and scores, same formulas as before. Kept as local variables and attached in one go (about 3x faster
    than assigning ~70 DataFrame columns one at a time)."""
    c, h, l, o, v = df.close, df.high, df.low, df.open, df.volume
    cl = _cl
    pc = c.shift()
    tr = pd.concat([h - l, (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1/14, adjust=False).mean()
    d = c.diff()
    up = d.clip(lower=0).ewm(alpha=1/14, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1/14, adjust=False).mean()
    rsi = 100 - 100 / (1 + up / dn.replace(0, np.nan))
    ema20 = c.ewm(span=20, adjust=False).mean()
    ema50 = c.ewm(span=50, adjust=False).mean()
    st20 = (c - ema20) / atr
    st50 = (c - ema50) / atr
    m, s = c.rolling(20).mean(), c.rolling(20).std()
    pb = (c - (m - 2 * s)) / (4 * s)
    macd = c.ewm(span=12, adjust=False).mean() - c.ewm(span=26, adjust=False).mean()
    mh = macd - macd.ewm(span=9, adjust=False).mean()
    lo, hi = l.rolling(14).min(), h.rolling(14).max()
    stoch = 100 * (c - lo) / (hi - lo)
    volr = v / v.rolling(20).mean()
    # --- extension components (0..1): RSI and ATR carry the most weight
    s_rsi, s_atr, s_bb, s_50 = cl(rsi, 60, 80), cl(st20, 1, 3), cl(pb, 0.8, 1.2), cl(st50, 2, 5)
    ext = 100 * (0.35 * s_rsi + 0.35 * s_atr + 0.15 * s_bb + 0.15 * s_50)
    # --- downside stretch (mirror image) -> buy-area / bounce score
    dext = 100 * (0.35 * cl(40 - rsi, 0, 20) + 0.35 * cl(-st20, 1, 3) + 0.15 * cl(0.2 - pb, 0, 0.4) + 0.15 * cl(-st50, 2, 5))
    # --- pullback triggers (0..1)
    s_div = ((c >= c.rolling(15).max() * 0.995) & (rsi < rsi.rolling(15).max() - 3)).astype(float)
    fall1 = (mh > 0) & (mh < mh.shift())
    fall2 = fall1 & (mh.shift() < mh.shift(2))
    s_macd = fall1 * 0.5 + fall2 * 0.5
    wick = (h - np.maximum(o, c)) / (h - l).replace(0, np.nan)
    s_wick = cl(wick.fillna(0), 0.3, 0.7)
    climax = (volr > 1.8) & (c > o) & (st20 > 1.5)
    dry = (volr < 0.7) & (c >= c.rolling(15).max() * 0.995)
    s_vol = climax * 1.0 + (~climax & dry) * 0.6
    s_sto = cl(stoch.fillna(50), 80, 100)
    run = (c > pc).astype(int)
    streak = run.groupby((run == 0).cumsum()).cumsum()
    s_run = (streak / 6).clip(0, 1)
    prob = (0.40 * ext / 100 + 0.15 * s_div + 0.15 * s_macd + 0.10 * s_wick + 0.10 * s_vol + 0.05 * s_sto + 0.05 * s_run)
    # --- breakout setup (0..1): near highs, squeeze, trend, volume, momentum, room to run
    res = h.shift(1).rolling(20).max()
    dist = (res - c) / atr
    b_prox = pd.Series(np.where(c > res, 1.0, cl(1 - dist / 1.5, 0, 1)), index=df.index)
    bbw = 4 * s / m
    bbr = bbw / bbw.rolling(100, min_periods=40).mean()      # min_periods: stocks with 60-118 candles used to fail here (NaN)
    b_sq = cl(bbr, 1, 0.5)
    b_trend = (c > ema50) * 0.5 + ((ema20 > ema50) & (c > ema20)) * 0.5
    b_vol = cl(volr, 1, 2)
    b_mom = cl(rsi, 50, 58) * cl(rsi, 78, 68)
    b_room = 1 - ext / 100
    bk = 0.25 * b_prox + 0.15 * b_sq + 0.20 * b_trend + 0.15 * b_vol + 0.10 * b_mom + 0.15 * b_room
    # --- history: after similar stretch (ext>=60), did price drop >= 1 ATR within 5 bars?
    fwd_low = l.rolling(5).min().shift(-5)
    hit = (fwd_low <= c - atr).astype(float).where(fwd_low.notna())
    fwd_hi = h.rolling(5).max().shift(-5)
    bhit = (fwd_hi >= c + atr).astype(float).where(fwd_hi.notna())
    # --- breakdown setup (0..1): mirror of the breakout setup - near lows, squeeze, downtrend, volume, momentum, room to fall
    sup = l.shift(1).rolling(20).min()
    ddist = (c - sup) / atr
    d_prox = pd.Series(np.where(c < sup, 1.0, cl(1 - ddist / 1.5, 0, 1)), index=df.index)
    d_trend = (c < ema50) * 0.5 + ((ema20 < ema50) & (c < ema20)) * 0.5
    r2 = 100 - rsi
    d_mom = cl(r2, 50, 58) * cl(r2, 78, 68)
    d_room = 1 - dext / 100
    dk = 0.25 * d_prox + 0.15 * b_sq + 0.20 * d_trend + 0.15 * b_vol + 0.10 * d_mom + 0.15 * d_room
    cols = dict(sup=sup, d_prox=d_prox, d_trend=d_trend, d_mom=d_mom, d_room=d_room, dk=dk, atr=atr, rsi=rsi, ema20=ema20, ema50=ema50, st20=st20, st50=st50, pb=pb, mh=mh, stoch=stoch, volr=volr,
                s_rsi=s_rsi, s_atr=s_atr, s_bb=s_bb, s_50=s_50, ext=ext, dext=dext, s_div=s_div, s_macd=s_macd,
                s_wick=s_wick, s_vol=s_vol, s_sto=s_sto, s_run=s_run, prob=prob, res=res, b_prox=b_prox, bbr=bbr,
                b_sq=b_sq, b_trend=b_trend, b_vol=b_vol, b_mom=b_mom, b_room=b_room, bk=bk, hit=hit, bhit=bhit)
    return pd.concat([df, pd.DataFrame(cols, index=df.index)], axis=1)

# ------------------------------------------------------------------ candlestick patterns
CANDLE_WEIGHT = 10                 # confidence points a confirming (or contradicting) candle pattern adds (or removes)
BULL_REV = {"Hammer", "Bullish engulfing", "Piercing line", "Morning star"}
BEAR_REV = {"Shooting star", "Bearish engulfing", "Dark cloud cover", "Evening star"}

def candle_patterns(df, atr):
    """Patterns that end at the latest candle (2- and 3-candle patterns use the candles before it).
    -> list of (name, 'bull' | 'bear' | 'neutral'). Shape only: whether a pattern MATTERS depends on where it forms (see analyze)."""
    if len(df) < 3 or not atr or not np.isfinite(atr):
        return []
    o, h, l, c = (df[k].values[-3:].astype(float) for k in ("open", "high", "low", "close"))
    rng, body, out = h - l, np.abs(c - o), []
    if rng[2] <= 0:
        return out
    up2, lo2 = h[2] - max(o[2], c[2]), min(o[2], c[2]) - l[2]
    bull2, bear2 = c[2] > o[2], c[2] < o[2]
    if lo2 >= 2 * body[2] and lo2 >= 0.55 * rng[2] and up2 <= 0.2 * rng[2]:
        out.append(("Hammer", "bull"))
    if up2 >= 2 * body[2] and up2 >= 0.55 * rng[2] and lo2 <= 0.2 * rng[2]:
        out.append(("Shooting star", "bear"))
    if rng[1] > 0:
        mid1 = (o[1] + c[1]) / 2
        if c[1] < o[1] and bull2 and o[2] <= c[1] and c[2] >= o[1] and body[2] > body[1]:
            out.append(("Bullish engulfing", "bull"))
        elif c[1] > o[1] and bear2 and o[2] >= c[1] and c[2] <= o[1] and body[2] > body[1]:
            out.append(("Bearish engulfing", "bear"))
        elif c[1] < o[1] and body[1] >= 0.5 * rng[1] and bull2 and o[2] < c[1] and mid1 < c[2] < o[1]:
            out.append(("Piercing line", "bull"))
        elif c[1] > o[1] and body[1] >= 0.5 * rng[1] and bear2 and o[2] > c[1] and o[1] < c[2] < mid1:
            out.append(("Dark cloud cover", "bear"))
        if h[2] <= h[1] and l[2] >= l[1]:
            out.append(("Inside bar", "neutral"))
    if rng[0] > 0 and rng[1] > 0:
        mid0 = (o[0] + c[0]) / 2
        if c[0] < o[0] and body[0] >= 0.5 * rng[0] and body[1] <= 0.35 * body[0] and bull2 and c[2] > mid0:
            out.append(("Morning star", "bull"))
        if c[0] > o[0] and body[0] >= 0.5 * rng[0] and body[1] <= 0.35 * body[0] and bear2 and c[2] < mid0:
            out.append(("Evening star", "bear"))
    if body[2] >= 0.85 * rng[2] and body[2] >= 0.8 * atr:
        out.append(("Bullish marubozu" if bull2 else "Bearish marubozu", "bull" if bull2 else "bear"))
    return out

def strong_candle(df, atr):
    """'bull' / 'bear' when the latest candle has a big body (60%+ of its range, 0.5+ ATR) closing near its high / low."""
    o, h, l, c = (float(df[k].iloc[-1]) for k in ("open", "high", "low", "close"))
    rng = h - l
    if rng <= 0 or abs(c - o) < 0.6 * rng or abs(c - o) < 0.5 * atr:
        return None
    return "bull" if (c > o and c >= l + 0.75 * rng) else "bear" if (c < o and c <= l + 0.25 * rng) else None

def call_dir(outlook):
    """Direction a call is betting on, for candle confirmation. None for neutral calls and 'wait for a dip/rally' calls."""
    if outlook.startswith(("Breakout likely", "Continuation", "Bounce")):
        return "bull"
    if outlook.startswith(("Breakdown likely", "Breakdown continuation", "Rejection", "Pullback")):
        return "bear"
    return None

def _consensus(sc, ndata, nsim):
    """Same agreement x strength confidence the breakout view has always used. -> (confidence, supporting, against)."""
    yes = [v for v in sc if v >= 0.6]; no = [v for v in sc if v <= 0.3]
    voters = len(yes) + len(no)
    agree = max(len(yes), len(no)) / voters if voters else 0
    strength = (np.mean(yes) if len(yes) >= len(no) and yes else np.mean([1 - v for v in no]) if no else 0)
    return 100 * (0.5 * agree + 0.5 * strength) * min(1, ndata / 200) * (0.7 + 0.3 * min(1, nsim / 20)), yes, no

def analyze(sym, df):
    df = build(df)
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
    uptrend, downtrend = x.ema20 > x.ema50, x.ema20 < x.ema50
    atr = float(x.atr)
    rej = 0.8 * ext / 100 + 0.2 * float(np.clip((x.stoch - 80) / 20, 0, 1))     # mirror of `bounce`
    pats = candle_patterns(df, atr)
    pnames = [pt[0] for pt in pats]
    bull_rev, bear_rev = any(n in BULL_REV for n in pnames), any(n in BEAR_REV for n in pnames)
    strong = strong_candle(df, atr)
    lo_c, hi_c = float(x.low), float(x.high)
    # a level is "touched" when the candle's range reached it (a wick may pierce it) AND the candle closed back on the right side of it
    touch_sup = lambda lvl: bool(pd.notna(lvl) and lo_c <= lvl + 0.75 * atr and x.close >= lvl - 0.75 * atr)
    touch_res = lambda lvl: bool(pd.notna(lvl) and hi_c >= lvl - 0.75 * atr and x.close <= lvl + 0.75 * atr)
    at_sup = touch_sup(x.sup) or touch_sup(x.ema20) or touch_sup(x.ema50) or dext >= 40     # at a support / EMA, or price is stretched down
    at_res = touch_res(x.res) or touch_res(x.ema20) or touch_res(x.ema50) or ext >= 40      # at a resistance / EMA, or price is stretched up
    brk_dn = ("Bearish marubozu" in pnames or strong == "bear") and x.close < x.sup         # a real breakdown candle
    brk_up = ("Bullish marubozu" in pnames or strong == "bull") and x.close > x.res         # a real breakout candle
    if brk_up and "Bullish marubozu" not in pnames:                                          # name the candle so the output can show why it counted
        pnames.append("Strong bull candle")
    if brk_dn and "Bearish marubozu" not in pnames:
        pnames.append("Strong bear candle")
    if ext >= 40 and ext >= dext:
        area, direction, dprob = "Sell area", "Down", prob
        why = "Price is stretched above its averages; a pullback lower is the likelier path."
    elif dext >= 40:
        area, direction, dprob = "Buy area", "Up", bounce
        why = "Price is stretched below its averages; a bounce higher is the likelier path."
    elif uptrend and abs(x.st20) <= 0.7 and x.rsi < 55:
        area, direction, dprob = "Buy area", "Up", max(bounce, 0.45)
        why = "Uptrend (20 EMA above 50 EMA) with price back near the 20 EMA: a dip-buy zone."
    elif downtrend and abs(x.st20) <= 0.7 and x.rsi > 45:             # mirror of the dip-buy zone: rally into the 20 EMA in a downtrend
        area, direction, dprob = "Rally zone", "Down", max(rej, 0.45)
        why = "Downtrend (20 EMA below 50 EMA) with price rallied back near the 20 EMA: a rally-and-reject sell zone."
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
    # --- breakdown view (mirror of the breakout view; history = how often price fell 1 ATR within 5 bars after similar setups)
    dsim = past[past.dk >= 0.6].dropna(subset=["hit"])
    dn_n = len(dsim)
    dk = float(x.dk)
    dk_prob = 0.6 * dk + 0.4 * float(dsim.hit.mean()) if (dn_n >= 8 and dk >= 0.6) else dk
    dsig = [
        ("Near / below 20-bar low", f"{(x.close - x.sup) / x.atr:+.2f} ATR above low", x.d_prox, "Negative means already below the prior 20-bar low"),
        ("Volatility squeeze", f"{x.bbr:.2f}x normal", x.b_sq, "Tight Bollinger width often precedes expansion"),
        ("Trend alignment", "down" if x.d_trend >= 1 else "mixed" if x.d_trend > 0 else "up", x.d_trend, "Price below 20 EMA, 20 EMA below 50 EMA"),
        ("Volume expansion", f"{x.volr:.1f}x avg", x.b_vol, "Breakdowns need participation"),
        ("Momentum zone", f"RSI {x.rsi:.0f}", x.d_mom, "RSI 32-42 = weak but not washed out"),
        ("Room to fall", f"{100 - dext:.0f}", x.d_room, "Low downside stretch leaves space to fall"),
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
        if dext >= 40 and dk_prob >= 0.55:        # stretched down but breaking down hard: not a bounce call
            if bounce >= 0.5:
                outlook, area, direction = "Rally, then continuation down", "Wait for rally", "Up"
                why2 = ("Strong downtrend and a fresh breakdown, but price is stretched below its averages. A bounce toward the broken level / "
                        "20 EMA is likely first; that rally is the better entry to sell before the decline resumes.")
            else:
                outlook = "Breakdown continuation (chase risk)"
                why2 = ("Trend and breakdown are strong and bounce signals are weak, but price is already stretched below its averages, "
                        "so selling here carries risk.")
        else:
            outlook = "Bounce likely"
    elif area == "Rally zone":
        outlook, area = "Rejection likely", "Sell area"
    else:
        top = max(bk_prob, dk_prob)
        up_first = bk_prob >= dk_prob
        outlook = (("Breakout likely" if up_first else "Breakdown likely") if top >= 0.55 else
                   ("Breakout watch" if up_first else "Breakdown watch") if top >= 0.4 else "Range / no edge")
        if outlook in ("Range / no edge", "Breakout watch", "Breakdown watch"):      # range edge + reversal candle
            if bull_rev and touch_sup(x.sup):
                outlook, area, direction, dprob = "Bounce at support", "Buy area", "Up", max(bounce, 0.45)
                why2 = "Price is at the bottom of its 20-bar range and printed a bullish reversal candle (" + ", ".join(n for n in pnames if n in BULL_REV) + ")."
            elif bear_rev and touch_res(x.res):
                outlook, area, direction, dprob = "Rejection at resistance", "Sell area", "Down", max(rej, 0.45)
                why2 = "Price is at the top of its 20-bar range and printed a bearish reversal candle (" + ", ".join(n for n in pnames if n in BEAR_REV) + ")."
    if why2:
        why = why2
    conf_basis = "breakout signals" if (outlook.startswith("Breakout") or "ontinuation" in outlook) else "pullback signals"
    if outlook.startswith(("Breakdown", "Rally")):            # mirror of the breakout / dip-then-continuation basis
        conf_basis = "breakdown signals"
        conf, yes, no = _consensus([float(d[2]) for d in dsig], len(df), dn_n)
    elif outlook in ("Rejection likely", "Rejection at resistance", "Bounce at support"):
        conf_basis = "reversal signals"
        sell = outlook.startswith("Rejection")
        rsig = ([("Downtrend", "down" if x.d_trend >= 1 else "mixed" if x.d_trend > 0 else "up", x.d_trend),
                 ("At resistance / EMA", "yes" if at_res else "no", float(at_res)),
                 ("Bearish candle pattern", ", ".join(n for n in pnames if n in BEAR_REV) or "none", float(bear_rev)),
                 ("Volume", f"{x.volr:.1f}x avg", x.b_vol),
                 ("Room to fall (RSI not oversold)", f"RSI {x.rsi:.0f}", float(np.clip((x.rsi - 30) / 15, 0, 1)))] if sell else
                [("Uptrend", "up" if x.b_trend >= 1 else "mixed" if x.b_trend > 0 else "down", x.b_trend),
                 ("At support / EMA", "yes" if at_sup else "no", float(at_sup)),
                 ("Bullish candle pattern", ", ".join(n for n in pnames if n in BULL_REV) or "none", float(bull_rev)),
                 ("Volume", f"{x.volr:.1f}x avg", x.b_vol),
                 ("Room to rise (RSI not overbought)", f"RSI {x.rsi:.0f}", float(np.clip((70 - x.rsi) / 15, 0, 1)))])
        rs = [float(v[2]) for v in rsig]
        yes = [v for v in rs if v >= 0.6]; no = [v for v in rs if v <= 0.3]
        conf = 100 * (0.5 * (len(yes) / (len(yes) + len(no)) if (yes or no) else 0) + 0.5 * (np.mean(yes) if yes else 0)) * min(1, len(df) / 200) * 0.7     # 0.7: no past-case history behind reversal calls, same as the other views with no history
    elif conf_basis == "breakout signals":   # confidence should judge the call being made
        sc = [float(b[2]) for b in bsig]
        yes = [v for v in sc if v >= 0.6]; no = [v for v in sc if v <= 0.3]
        voters = len(yes) + len(no)
        agree = max(len(yes), len(no)) / voters if voters else 0
        strength = (np.mean(yes) if len(yes) >= len(no) and yes else np.mean([1 - v for v in no]) if no else 0)
        conf = 100 * (0.5 * agree + 0.5 * strength) * min(1, len(df) / 200) * (0.7 + 0.3 * min(1, bn / 20))
    # --- candlestick confirmation: a pattern counts only where it forms (at a support / resistance / EMA, or when price is stretched)
    d = call_dir(outlook)
    vol_ok = bool(np.isfinite(x.volr) and x.volr >= 1.2)
    f = a = False
    if d == "bull":
        breakout_call = outlook.startswith(("Breakout", "Continuation"))
        f = brk_up if breakout_call else (bull_rev and at_sup)
        a = (bear_rev and at_res) or brk_dn            # a reversal candle at resistance, or a breakdown candle, contradicts a buy call
    elif d == "bear":
        breakdown_call = outlook.startswith("Breakdown")
        f = brk_dn if breakdown_call else (bear_rev and at_res)
        a = (bull_rev and at_sup) or brk_up
    cs_for, cs_against = (0.6 + 0.4 * vol_ok) if f else 0.0, (0.6 + 0.4 * vol_ok) if a else 0.0
    candle = "confirms" if cs_for > cs_against else "against" if cs_against > cs_for else "none"
    adj = round(CANDLE_WEIGHT * (cs_for - cs_against))
    conf = float(np.clip(conf + adj, 0, 100))
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
        "bk_hist_n": bn, "resistance": round(float(x.res), 2), "support": round(float(x.sup), 2),
        "dk_prob": round(dk_prob * 100), "dk_hist_n": dn_n, "bd_support": int(sum(v[2] >= 0.6 for v in dsig)),
        "bd_against": int(sum(v[2] <= 0.3 for v in dsig)), "bd_total": len(dsig),
        "bd_levels": {"trigger": round(float(x.sup), 2), "t1": round(float(x.sup - x.atr), 2),
                      "t2": round(float(x.sup - 2 * x.atr), 2), "fail": round(float(x.ema20), 2)},
        "bd_signals": [{"name": v[0], "value": v[1], "score": round(float(v[2]), 2), "note": v[3]} for v in dsig],
        "patterns": pnames, "candle": candle, "candle_adj": adj,
        "rev_signals": ([{"name": v[0], "value": v[1], "score": round(float(v[2]), 2)} for v in rsig] if conf_basis == "reversal signals" else []),
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

def side_of(outlook):
    """Which way the call leans: buy (breakout, continuation, bounce, dip-then-continuation), sell (pullback, breakdown, rejection,
    rally-then-continuation-down), None (neutral / watch)."""
    return "buy" if outlook.startswith(("Breakout likely", "Continuation", "Bounce", "Dip")) else \
           "sell" if outlook.startswith(("Pullback", "Breakdown likely", "Breakdown continuation", "Rejection", "Rally")) else None

# ================================================================== data: Yahoo Finance, no database
# Every symbol is downloaded once and kept in memory for CACHE_TTL seconds (shared by everyone using the app),
# and symbols are fetched together in ONE request per batch, which keeps Yahoo's rate limiter happy.
IST = ZoneInfo("Asia/Kolkata")
YF = {"ONE_DAY": ("1d", "2y"), "ONE_HOUR": ("1h", "6mo"), "FIFTEEN_MINUTE": ("15m", "55d"), "FIVE_MINUTE": ("5m", "55d")}
TF_LABEL = {"Daily": "ONE_DAY", "1 hour": "ONE_HOUR", "15 min": "FIFTEEN_MINUTE", "5 min": "FIVE_MINUTE"}
TF_MIN = {"ONE_HOUR": 60, "FIFTEEN_MINUTE": 15, "FIVE_MINUTE": 5}
CACHE_TTL = 300            # seconds before a symbol is downloaded again
FAIL_TTL = 60              # seconds before a symbol that returned nothing is asked for again (do not hammer Yahoo)
MAX_SYMBOLS = 25           # per run; keeps the phone UI quick and Yahoo calm
MAX_BARS = 1500            # newest candles used per stock (same cap as the Angel One screener)
BATCH = 20                 # symbols per Yahoo request


def yf_ticker(s):
    """RELIANCE -> RELIANCE.NS. Already-qualified tickers are left alone: ^NSEI, CL=F, GC=F, TCS.BO, RELIANCE.NS."""
    s = s.strip().upper()
    return s if (s.startswith("^") or "=" in s or "." in s) else s + ".NS"


def label_of(s):
    s = s.strip().upper()
    return s[:-3] if s.endswith(".NS") else s


def parse_symbols(text):
    out, seen = [], set()
    for t in text.replace(",", " ").replace(";", " ").split():
        k = label_of(t)
        if k and k not in seen:
            seen.add(k); out.append(k)
    return out


@st.cache_resource
def _store():
    return {}


class YahooError(Exception):
    pass


def _tidy(d, tf):
    """Yahoo OHLCV frame -> ts/open/high/low/close/volume with IST timestamps as text."""
    d = d.dropna(subset=["Close"])
    if d.empty:
        return None
    idx = pd.to_datetime(d.index)
    if getattr(idx, "tz", None) is not None:
        idx = idx.tz_convert(IST).tz_localize(None)
    fmt = "%Y-%m-%d" if tf == "ONE_DAY" else "%Y-%m-%d %H:%M"
    vol = d["Volume"].fillna(0.0).astype(float).values
    df = pd.DataFrame({"ts": [i.strftime(fmt) for i in idx], "open": d["Open"].astype(float).values,
                       "high": d["High"].astype(float).values, "low": d["Low"].astype(float).values,
                       "close": d["Close"].astype(float).values, "volume": vol})
    return df.drop_duplicates("ts", keep="last").reset_index(drop=True)


def _split(data, tickers):
    """Pick each ticker's frame out of a yf.download() result (single- and multi-ticker layouts)."""
    out = {}
    if data is None or data.empty:
        return out
    if isinstance(data.columns, pd.MultiIndex):
        l0, l1 = set(data.columns.get_level_values(0)), set(data.columns.get_level_values(1))
        for t in tickers:
            d = data[t] if t in l0 else data.xs(t, axis=1, level=1) if t in l1 else None
            if d is not None and not d.dropna(subset=["Close"]).empty:
                out[t] = d
    elif len(tickers) == 1 and not data.dropna(subset=["Close"]).empty:
        out[tickers[0]] = data
    return out


def fetch(tickers, tf):
    """-> ({ticker: DataFrame}, last_error_text). Uses the shared in-memory cache; downloads only what is missing or stale."""
    store, now = _store(), time.time()
    need = [t for t in tickers if (t, tf) not in store
            or now - store[(t, tf)][0] > (CACHE_TTL if store[(t, tf)][1] is not None else FAIL_TTL)]
    err = ""
    interval, period = YF[tf]
    for i in range(0, len(need), BATCH):
        chunk = need[i:i + BATCH]
        got, data = {}, None
        for attempt in range(2):
            try:
                data = yf.download(chunk, interval=interval, period=period, group_by="ticker", auto_adjust=False,
                                   progress=False, threads=True)
                break
            except Exception as e:
                err = f"{type(e).__name__}: {e}"; time.sleep(3)
        got = _split(data, chunk)
        for t in chunk:                                   # whatever the batch missed gets one individual try
            if t not in got:
                try:
                    h = yf.Ticker(t).history(period=period, interval=interval, auto_adjust=False)
                    if h is not None and not h.empty:
                        got[t] = h
                except Exception as e:
                    err = f"{type(e).__name__}: {e}"
        for t in chunk:
            store[(t, tf)] = (time.time(), _tidy(got[t], tf) if t in got else None)      # None = nothing came back (remembered briefly)
        if i + BATCH < len(need):
            time.sleep(1)
    if len(store) > 400:                                   # keep memory small: drop the oldest entries
        for k in sorted(store, key=lambda k: store[k][0])[:100]:
            store.pop(k, None)
    return {t: store[(t, tf)][1] for t in tickers if store.get((t, tf), (0, None))[1] is not None}, err


def drop_forming(df, tf):
    """Remove the newest candle if it is still forming (so signals are judged on closed candles only)."""
    if len(df) < 62:
        return df
    now = dt.datetime.now(IST).replace(tzinfo=None)
    try:
        t = dt.datetime.strptime(df.ts.iloc[-1][:16], "%Y-%m-%d %H:%M" if tf != "ONE_DAY" else "%Y-%m-%d")
    except ValueError:
        return df
    if tf == "ONE_DAY":
        forming = t.date() == now.date() and now.weekday() < 5 and now.time() < dt.time(15, 45)
    else:
        forming = t + dt.timedelta(minutes=TF_MIN[tf]) > now
    return df.iloc[:-1].reset_index(drop=True) if forming else df


def run(sym, df, tf, closed_only):
    if df is None:
        raise ValueError("No data from Yahoo for %s (check the symbol)" % sym)
    df = df.copy()
    if closed_only:
        df = drop_forming(df, tf)
    df = df.tail(MAX_BARS).reset_index(drop=True)
    if len(df) < 60:
        raise ValueError("Not enough history (%d candles, need 60)" % len(df))
    no_volume = float(df.volume.sum()) == 0                # indices and some futures have no volume
    if no_volume:
        df["volume"] = 1.0
    r = analyze(sym, df)
    r["side"] = side_of(r["outlook"])
    r["no_volume"] = no_volume
    return r


# ================================================================== UI (built for phones: stacked cards, wrapping tiles)
GREEN, RED, AMBER, NAVY, GREY, TRACK = "#2F8F6F", "#CF3F2A", "#D99A00", "#4F7CAC", "#6B7889", "#3A4658"

CSS = """<style>
.block-container{padding-top:1.2rem;padding-left:.8rem;padding-right:.8rem;max-width:1200px}
.tiles{display:grid;grid-template-columns:repeat(auto-fit,minmax(84px,1fr));gap:8px;margin:10px 0}
.tile{border:1px solid rgba(128,128,128,.35);border-radius:10px;padding:8px 4px;text-align:center}
.tile b{display:block;font-size:19px;line-height:1.15}
.tile span{font-size:11px;opacity:.7}
.line{font-size:14px;margin:4px 0}
.sumbar{display:flex;gap:8px;flex-wrap:wrap;margin:6px 0 10px}
</style>"""


def wide(fn, *a, **kw):
    """Full-width widget on old and new Streamlit (use_container_width is being replaced by width='stretch')."""
    try:
        return fn(*a, width="stretch", **kw)
    except TypeError:
        return fn(*a, use_container_width=True, **kw)


def chip(text, color):
    fg = "#1b1500" if color == AMBER else "#fff"
    return (f"<span style='display:inline-block;background:{color};color:{fg};font-weight:600;font-size:13px;"
            f"padding:3px 11px;border-radius:99px;margin:0 6px 6px 0'>{text}</span>")


def outlook_color(o):
    if o.startswith(("Breakdown likely", "Breakdown continuation", "Rejection", "Pullback likely")):
        return RED
    if o.startswith("Rally") or "watch" in o or "possible" in o:
        return AMBER
    if o.startswith("Bounce"):
        return GREEN
    if "Range" in o:
        return GREY
    return NAVY if o.startswith(("Breakout likely", "Continuation", "Dip")) else RED


def chips(r):
    v = {"Overextended": RED, "Stretched": AMBER, "Oversold": NAVY}.get(r["verdict"], GREEN)
    a = {"Buy area": GREEN, "Sell area": RED, "Wait for dip": AMBER, "Wait for rally": AMBER}.get(r["area"], GREY)
    arrow = {"Up": "↑", "Down": "↓"}.get(r["direction"], "↔")
    out = chip(r["verdict"], v) + chip(f"{r['area']} {arrow}", a) + chip(r["outlook"], outlook_color(r["outlook"]))
    if r.get("patterns") and r["candle"] != "none":
        out += chip(f"{r['patterns'][0]} {'✓' if r['candle'] == 'confirms' else '✗'}", GREEN if r["candle"] == "confirms" else RED)
    return out


def dots(signals, good_high):
    """One bar segment per signal. good_high=True: high score = green (breakout, breakdown). False: high score = red (pullback)."""
    hi, lo = (GREEN, RED) if good_high else (RED, GREEN)
    segs = "".join(f"<div style='flex:1;height:8px;border-radius:2px;background:"
                   f"{hi if x['score'] >= .6 else lo if x['score'] <= .3 else TRACK}'></div>" for x in signals)
    return f"<div style='display:flex;gap:4px;margin:4px 0 12px'>{segs}</div>"


def sig_table(signals, good_high):
    hi, lo = (GREEN, RED) if good_high else (RED, GREEN)
    rows = "".join(
        f"<tr><td style='padding:5px 4px;border-bottom:1px solid rgba(128,128,128,.3)'>"
        f"<span style='display:inline-block;width:9px;height:9px;border-radius:50%;margin-right:7px;background:"
        f"{hi if x['score'] >= .6 else lo if x['score'] <= .3 else AMBER}'></span>{x['name']}"
        f"{('<br><small style=opacity:.65>' + x['note'] + '</small>') if x.get('note') else ''}</td>"
        f"<td style='padding:5px 4px;border-bottom:1px solid rgba(128,128,128,.3);text-align:right;white-space:nowrap'>{x['value']}</td></tr>"
        for x in signals)
    return f"<table style='width:100%;border-collapse:collapse;font-size:13px'>{rows}</table>"


def is_brk(o):
    return o.startswith(("Breakout likely", "Continuation", "Dip, then continuation", "Breakout watch"))


def is_brd(o):
    return o.startswith(("Breakdown likely", "Breakdown continuation", "Rally, then continuation", "Breakdown watch"))


def card(r):
    if "error" in r:
        st.error(f"**{r['symbol']}**: {r['error']}")
        return
    o = r["outlook"]
    with st.container(border=True):
        st.markdown(f"<div style='display:flex;justify-content:space-between;align-items:baseline'>"
                    f"<span style='font-size:22px;font-weight:700'>{r['symbol']}</span>"
                    f"<span>{r['price']} · ATR {r['atr']}</span></div>", unsafe_allow_html=True)
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
            f"<span>-3 ATR</span><span>20 EMA · now {r['stretch']:+.2f} ATR</span><span>+3 ATR</span></div>", unsafe_allow_html=True)
        tiles = [("Overextension", r["ext"]), ("Bounce odds" if r["direction"] == "Up" else "Pullback odds", f"{r['dir_prob']}%"),
                 ("Breakout odds", f"{r['bk_prob']}%"), ("Breakdown odds", f"{r['dk_prob']}%"), ("Confidence", r["confidence"])]
        st.markdown("<div class='tiles'>" + "".join(f"<div class='tile'><b>{v}</b><span>{k}</span></div>" for k, v in tiles) + "</div>",
                    unsafe_allow_html=True)
        st.markdown(f"<div class='line'>Pullback signals: <b>{r['pb_warn']}</b> of {r['pb_total']} warn, {r['pb_calm']} calm</div>"
                    + dots(r["signals"], False)
                    + f"<div class='line'>Breakout signals: <b>{r['bk_support']}</b> of {r['bk_total']} support, {r['bk_against']} against</div>"
                    + dots(r["bk_signals"], True)
                    + f"<div class='line'>Breakdown signals: <b>{r['bd_support']}</b> of {r['bd_total']} support a sell, {r['bd_against']} against</div>"
                    + dots(r["bd_signals"], True), unsafe_allow_html=True)
        st.caption(f"Confidence is based on the {r['conf_basis']}.")
        if r["patterns"]:
            effect = (f":green[✓ confirms the call ({r['candle_adj']:+d} confidence)]" if r["candle"] == "confirms" else
                      f":red[✗ goes against the call ({r['candle_adj']:+d} confidence)]" if r["candle"] == "against" else
                      "(no effect on the call: not at a key level)")
            st.write(f"Candle: **{', '.join(r['patterns'])}** {effect}")
        if r["direction"] == "Up":
            z = r["up_zones"]; st.write(f"RSI **{r['rsi']}** · If it bounces: 20 EMA **{z['ema20']}**, +1 ATR **{z['plus1atr']}**, +2 ATR **{z['plus2atr']}**")
        else:
            z = r["zones"]; st.write(f"RSI **{r['rsi']}** · If it pulls back: 20 EMA **{z['ema20']}**, -1 ATR **{z['minus1atr']}**, -2 ATR **{z['minus2atr']}**")
        if o == "Range / no edge" or o.endswith("watch"):
            st.write(f"Watch both sides: buy on a close above **{r['resistance']}**, sell on a close below **{r['support']}**")
        if is_brk(o):
            k = r["bk_levels"]
            lead = (f"Already closed above the prior 20-bar high **{k['trigger']}** (retest level)" if r["price"] > k["trigger"]
                    else f"Breakout trigger: close above **{k['trigger']}**")
            st.write(f"{lead} · targets **{k['t1']}**, **{k['t2']}** · fails under 20 EMA **{k['fail']}**")
        if is_brd(o) and not o.endswith("watch"):
            k = r["bd_levels"]
            lead = (f"Already closed below the prior 20-bar low **{k['trigger']}** (retest level)" if r["price"] < k["trigger"]
                    else f"Breakdown trigger: close below **{k['trigger']}**")
            st.write(f"{lead} · targets **{k['t1']}**, **{k['t2']}** · fails above 20 EMA **{k['fail']}**")
        hist = f"{r['hist_hit']}% ({r['hist_n']} past cases)" if r["hist_n"] >= 8 else f"too few past cases ({r['hist_n']})"
        st.caption(f"History check: {hist}. Rule-based odds {r['model_prob']}%. Data: {r['bars']} candles, latest {r['last_bar']}."
                   + (" No volume data for this symbol, so volume signals are neutral." if r["no_volume"] else ""))
        with st.expander("All signals"):
            st.markdown("**Pullback signals** (red = warns, green = calm)")
            st.markdown(sig_table(r["signals"], False), unsafe_allow_html=True)
            st.markdown("**Breakout signals** (green = supports, red = against)")
            st.markdown(sig_table(r["bk_signals"], True), unsafe_allow_html=True)
            st.markdown("**Breakdown signals** (green = supports a sell, red = against)")
            st.markdown(sig_table(r["bd_signals"], True), unsafe_allow_html=True)
            if r.get("rev_signals"):
                st.markdown("**Reversal signals** (green = supports the call)")
                st.markdown(sig_table(r["rev_signals"], True), unsafe_allow_html=True)


SIDES = {"All": None, "Buy side": "buy", "Sell side": "sell", "Neutral / watch": "none"}


def main():
    st.set_page_config(page_title="Stretch Monitor", layout="wide", initial_sidebar_state="collapsed")
    st.markdown(CSS, unsafe_allow_html=True)
    st.title("Stretch Monitor")
    st.caption("Buy and sell setups (breakout / breakdown, bounce / rejection, pullback / rally) with candlestick confirmation, "
               "overextension, odds and confidence. Yahoo Finance data: delayed and less exact than Angel One. A screening aid, not advice.")
    syms = st.text_area("Symbols (comma or space separated). NSE names need no suffix; also ^NSEI, CL=F, GC=F, TCS.BO",
                        "RELIANCE, TCS, INFY, HDFCBANK", height=80)
    c1, c2 = st.columns([1, 1])
    tf = TF_LABEL[c1.selectbox("Timeframe", list(TF_LABEL), index=0)]
    closed = c2.checkbox("Closed candles only", value=False,
                         help="Ignore the newest candle while it is still forming, so a signal cannot disappear when the candle closes.")
    go = wide(st.button, "Analyze", type="primary")
    if go:
        names = parse_symbols(syms)[:MAX_SYMBOLS]
        if not names:
            st.warning("Enter at least one symbol.")
        else:
            with st.spinner(f"Fetching {len(names)} symbols from Yahoo..."):
                frames, err = fetch([yf_ticker(s) for s in names], tf)
            out = []
            for s in names:
                try:
                    out.append(run(s, frames.get(yf_ticker(s)), tf, closed))
                except Exception as e:
                    out.append({"symbol": s, "error": str(e)})
            out.sort(key=lambda r: r.get("dir_prob", -1), reverse=True)
            st.session_state["out"] = out
            st.session_state["nodata"] = not frames
            st.session_state["err"] = err
            st.session_state["when"] = dt.datetime.now(IST).strftime("%d %b %H:%M IST")
    out = st.session_state.get("out")
    if not out:
        return
    if st.session_state.get("nodata"):
        st.warning("Yahoo returned no data for any symbol. Check the symbols; if they are right, Yahoo is probably rate-limiting this server "
                   "(common on shared hosting). Wait a few minutes and press Analyze again."
                   + (" Last error: " + st.session_state["err"][:200] if st.session_state.get("err") else ""))
    ok = [r for r in out if "error" not in r]
    nb, ns = sum(r["side"] == "buy" for r in ok), sum(r["side"] == "sell" for r in ok)
    st.markdown(f"<div class='sumbar'>{chip(f'Buy side {nb}', GREEN)}{chip(f'Sell side {ns}', RED)}"
                f"{chip(f'Neutral {len(ok) - nb - ns}', GREY)}{chip(f'Errors {len(out) - len(ok)}', AMBER) if len(out) > len(ok) else ''}</div>",
                unsafe_allow_html=True)
    show = SIDES[st.radio("Show", list(SIDES), horizontal=True)]
    rows = [r for r in out if show is None or "error" in r or (r["side"] or "none") == show]
    st.caption(f"Analysed {st.session_state.get('when', '')}. Sorted by odds of the expected move, highest first. "
               "Data is cached for 5 minutes.")
    cols = st.columns(2)
    for i, r in enumerate(rows):
        with cols[i % 2]:
            card(r)


if __name__ == "__main__":
    main()
