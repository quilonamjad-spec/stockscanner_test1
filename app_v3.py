import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="Warrior Fine-Tuned Engine", layout="wide")
st.title("🛡️ Advanced Fine-Tuned Pattern Scanner")
st.markdown("Features Multi-Timeframe Verification, Dynamic 9 EMA Support Tracks, and Volume Shock filters.")

# --- SIDEBAR INTERFACE CONTROLS ---
st.sidebar.header("🕹️ Core Execution Controls")
exchange = st.sidebar.selectbox("Exchange Registry:", ["NSE (.NS)", "BSE (.BO)"])
raw_ticker = st.sidebar.text_input("Asset Ticker Symbol:", "TCS")
trading_tf = st.sidebar.selectbox("Trading Timeframe (Execution):", ["5m", "15m", "30m"])
trend_tf = st.sidebar.selectbox("Trend Anchor Timeframe (Filter):", ["1h", "2h", "1d"])

# --- NEW EXTENSION: BOUNDARY SELECTION FILTER ---
st.sidebar.markdown("---")
st.sidebar.subheader("📅 Level Boundaries Strategy")
boundary_mode = st.sidebar.radio(
    "Select Structural Range Source:",
    ["Purely Today's Session", "Multi-Day Rolling Window (Last 30 Candles)"]
)

st.sidebar.markdown("---")
st.sidebar.subheader("🎛️ Algorithmic Sensitivity Sliders")
vol_sigma = st.sidebar.slider("Required Volume Shock (Z-Score Sigma):", 1.0, 3.0, 1.5, step=0.1)
flag_retracement_limit = st.sidebar.slider("Max Flag Pullback Depth (% of Pole):", 30, 60, 50, step=5) / 100
max_ema_extension = st.sidebar.slider("Max Allowed Extension from 9 EMA (%):", 0.5, 3.0, 1.5, step=0.1) / 100

ticker = f"{raw_ticker.strip().upper()}.NS" if exchange == "NSE (.NS)" else f"{raw_ticker.strip().upper()}.BO"

# --- MULTI-TIMEFRAME DATA ENGINE ---
@st.cache_data(ttl=15)
def pull_clean_feed(symbol, interval, period="1mo"):
    try:
        data = yf.download(symbol, period=period, interval=interval, group_by='ticker')
        if data.empty: return None
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(-1)
        return data.loc[:, ~data.columns.duplicated()]
    except:
        return None

# Fetch both datasets
df_trade = pull_clean_feed(ticker, trading_tf)
df_trend = pull_clean_feed(ticker, trend_tf, period="3mo")

if df_trade is None or df_trade.empty or df_trend is None or df_trend.empty:
    st.error("❌ Analytical data connection broken. Verify your ticker handle syntax or market connectivity.")
else:
    # --- HIGHER TIMEFRAME TREND FILTER ---
    df_trend['EMA_20'] = df_trend['Close'].ewm(span=20, adjust=False).mean()
    higher_tf_is_bullish = df_trend['Close'].iloc[-1] > df_trend['EMA_20'].iloc[-1]
    trend_status_text = "🟢 Bullish (Price above 20 EMA)" if higher_tf_is_bullish else "🔴 Bearish (Price below 20 EMA)"

    # --- MAIN TRADING TIMEFRAME CALCULATIONS ---
    # Fetch a wider buffer from history to ensure we have enough candles for either rule
    trade_df = df_trade.copy()
    trade_df['EMA_9'] = trade_df['Close'].ewm(span=9, adjust=False).mean()
    
    # --- APPLY USER CHOSEN DATE FILTERING MODE ---
    if boundary_mode == "Purely Today's Session":
        today_date = trade_df.index[-1].date()
        working_df = trade_df[trade_df.index.date == today_date].copy()
        
        # Guard clause: If the market just opened and there aren't enough candles yet for today
        if len(working_df) < 3:
            st.warning("⚠️ Not enough candles generated yet in today's active live session. Temporarily showing rolling view instead.")
            working_df = trade_df.tail(40).copy()
    else:
        # Multi-Day standard 40 candle window
        working_df = trade_df.tail(40).copy()

    # Isolate history window vs current active session row
    history = working_df.iloc[:-1]
    live_candle = working_df.iloc[-1]
    
    resistance_ceiling = history['High'].max() if not history.empty else live_candle['High']
    support_floor = history['Low'].min() if not history.empty else live_candle['Low']
    
    live_close = live_candle['Close']
    live_high = live_candle['High']
    live_low = live_candle['Low']
    live_ema9 = live_candle['EMA_9']
    
    # Advanced Volume Shock Calculation (Z-Score Method)
    vol_mean = history['Volume'].mean() if not history.empty else 1
    vol_std = history['Volume'].std() if not history.empty else 1
    volume_z_score = (live_candle['Volume'] - vol_mean) / vol_std if vol_std > 0 else 0
    volume_shock_confirmed = volume_z_score >= vol_sigma

    # --- FINE-TUNED PATTERN SCANNING MATRICES ---
    is_bull_flag = False
    flagpole_peak = 0
    ema_hold_confirmed = False
    
    if len(history) > 5:
        for i in range(len(history) - 12, len(history) - 3):
            if i < 0: continue
            for j in range(i + 1, i + 6):
                if j >= len(history): break
                base_open = history['Open'].iloc[i]
                peak_high = history['High'].iloc[j]
                
                if (peak_high - base_open) / base_open >= 0.008: 
                    flag_period = history.iloc[j+1:]
                    if len(flag_period) >= 2:
                        lowest_flag_dip = flag_period['Low'].min()
                        highest_flag_peak = flag_period['High'].max()
                        max_allowed_dip = peak_high - ((peak_high - base_open) * flag_retracement_limit)
                        
                        if lowest_flag_dip >= max_allowed_dip and highest_flag_peak <= peak_high * 1.003:
                            is_bull_flag = True
                            flagpole_peak = peak_high
                            matching_ema_bars = flag_period[flag_period['Low'] >= flag_period['EMA_9'] * 0.998]
                            if len(matching_ema_bars) >= (len(flag_period) * 0.7):
                                ema_hold_confirmed = True
                            break
            if is_bull_flag: break

    ema_extension_pct = (live_close - live_ema9) / live_ema9 if live_ema9 > 0 else 0
    is_overextended = ema_extension_pct > max_ema_extension

    # --- PATTERN DECISION MATRIX ---
    status_msg = "Range Rotation Profile"
    analysis_narrative = "Price action is rotating smoothly within horizontal structural boundaries. No breakout parameters met."
    theme_color = "#2b303a"

    trigger_level = flagpole_peak if is_bull_flag else resistance_ceiling

    if live_close > trigger_level:
        if volume_shock_confirmed:
            if is_overextended:
                status_msg = "⚠️ OVEREXTENDED MOMENTUM (High Chasing Risk)"
                analysis_narrative = f"CAUTION: Price cleared the breakout line at ₹{trigger_level:.2f} on high volume, but is extended {ema_extension_pct*100:.2f}% away from the 9 EMA. High mean-reversion trap danger."
                theme_color = "#744210"
            elif not higher_tf_is_bullish:
                status_msg = "⚠️ COUNTER-TREND BREAKOUT (Higher Timeframe Resistance)"
                analysis_narrative = f"CAUTION: Lower timeframe breakout is triggering, but the anchor macro trend ({trend_tf}) is Bearish. Heavy overhead distribution expected."
                theme_color = "#744210"
            else:
                if is_bull_flag:
                    status_msg = "🚀 PREMIUM BULL FLAG ENTRY CONFIRMED"
                    analysis_narrative = f"HIGH ACCURACY MATCH: Validated Flagpole identified with a 9 EMA support hold during consolidation. Volume shock score is {volume_z_score:.1f}x with multi-timeframe trend alignment."
                    theme_color = "#1b4332"
                else:
                    status_msg = "🚀 STANDARD BREAKOUT ENTRY CONFIRMED"
                    analysis_narrative = f"VALID SIGNAL: Resistance ceiling level at ₹{resistance_ceiling:.2f} has been breached cleanly on volume. Higher timeframe trend matches trade direction."
                    theme_color = "#1b4332"
        else:
            status_msg = "⚠️ FALSE BREAKOUT TRAP WARNING (Low Volume Trap)"
            analysis_narrative = f"VOLUME TRAP: Price crossed resistance, but volume shock is flat ({volume_z_score:.2f} sigma). Move lacks institutional backing."
            theme_color = "#744210"

# 🚨 REVERSAL STRATEGY A: RESISTANCE REJECTION (The Short/Sell Fade)
    elif live_high >= (trigger_level * 0.998) and live_close <= trigger_level:
        status_msg = "📉 RESISTANCE REJECTION: POTENTIAL SHORT/SELL ENTRY"
        analysis_narrative = f"""
        <b>Strategy: Range Fade (Short)</b><br>
        The asset pushed up to touch your resistance ceiling at ₹{live_high:.2f} but got rejected. 
        Because it failed to confirm a close above the line, this highlights an opportunity to 
        <b>Short/Sell near the top</b> and take profits near the support floor baseline at ₹{support_floor:.2f}.
        """
        theme_color = "#641e1e"  # Crimson Red

    # 🍏 REVERSAL STRATEGY B: SUPPORT BOUNCE (The Buy/Long Fade)
    elif live_low <= (support_floor * 1.002) and live_close >= support_floor:
        status_msg = "🟢 SUPPORT BOUNCE: POTENTIAL BUY/LONG ENTRY"
        analysis_narrative = f"""
        <b>Strategy: Range Bounce (Buy)</b><br>
        The asset dipped down to test your support floor baseline at ₹{live_low:.2f} but found buyers and bounced. 
        Because it held above the baseline, this highlights an excellent opportunity to 
        <b>Buy near the bottom floor</b> and take profits near the resistance ceiling line at ₹{trigger_level:.2f}.
        """
        theme_color = "#0a5c36"  # Deep Forest Green

    # Standard breakdown fallback
    elif live_close < support_floor:
        status_msg = "📉 STRUCTURAL BREAKDOWN PROFILE"
        analysis_narrative = f"The price has breached the structural support baseline floor of ₹{support_floor:.2f} with strong downward expansion momentum."
        layout_theme_color = "#4a154b"

    # --- UI RENDERING ENGINE ---
    st.subheader("📊 High-Fidelity Evaluation Dashboard")
    st.markdown(f"""
    <div style="background-color:{theme_color}; padding:24px; border-radius:10px; border: 1px solid rgba(255,255,255,0.12); margin-bottom:25px;">
        <h3 style="margin-top:0; color:#ffffff; font-family:sans-serif; font-weight:700;">{status_msg}</h3>
        <p style="font-size:16px; color:#f5f5f5; margin-bottom:0; font-family:sans-serif; line-height:1.6;"><b>Range Filter Applied:</b> {boundary_mode}<br>{analysis_narrative}</p>
    </div>
    """, unsafe_allow_html=True)
    
    d1, d2, d3, d4 = st.columns(4)
    d1.metric(f"Anchor Trend ({trend_tf})", trend_status_text)
    d2.metric("Volume Z-Score Shock", f"{volume_z_score:.2f} σ")
    d3.metric("9 EMA Support Hold?", "EXCELLENT" if ema_hold_confirmed else "WEAK/NO")
    d4.metric("EMA Extension Distance", f"{ema_extension_pct*100:.2f}%")

    # --- RENDER CHART ---
    fig = go.Figure()
    fig.add_trace(go.Candlestick(x=working_df.index, open=working_df['Open'], high=working_df['High'], low=working_df['Low'], close=working_df['Close'], name="Price Bars"))
    fig.add_trace(go.Scatter(x=working_df.index, y=working_df['EMA_9'], line=dict(color="#3a86ff", width=2), name="9 EMA Line"))
    
    fig.add_shape(type="line", x0=working_df.index[0], y0=trigger_level, x1=working_df.index[-1], y1=trigger_level, line=dict(color="#FF4B4B", width=2, dash="dash"), name="Trigger Barrier")
    fig.add_shape(type="line", x0=working_df.index[0], y0=support_floor, x1=working_df.index[-1], y1=support_floor, line=dict(color="#2EB82E", width=2, dash="dash"), name="Support Baseline")
    
    fig.update_layout(xaxis_rangeslider_visible=False, template="plotly_dark", height=550)
    st.plotly_chart(fig, use_container_width=True)