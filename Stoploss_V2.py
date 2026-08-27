import streamlit as st
import yfinance as yf

st.set_page_config(
    page_title="NSE Trailing SL Calculator", page_icon="📈", layout="centered"
)


def fetch_nse_price(symbol):
    formatted_symbol = symbol.upper().strip()
    if not (
        formatted_symbol.endswith(".NS") or formatted_symbol.endswith(".BO")
    ):
        formatted_symbol += ".NS"

    ticker = yf.Ticker(formatted_symbol)
    df = ticker.history(period="1d", interval="1m")

    if df.empty:
        return None, None
    return formatted_symbol, float(df["Close"].iloc[-1])


st.title("📈 Trailing Stop Loss Calculator")
st.caption("NSE Market Protection Tool")

col1, col2 = st.columns(2)
with col1:
    symbol_input = st.text_input("Stock Symbol", value="SARDAEN")
    action = st.selectbox("Action", ["BUY", "SELL"])
with col2:
    entry_price = st.number_input(
        "Entry Price (₹)", min_value=0.0, value=500.0, step=0.05
    )
    is_whipsaw_safe = st.checkbox(
        "Whipsaw Safe Mode (0.40% Gap)",
        value=True,
        help="Milestone protection is active. After 1% profit, the SL uses a fixed 0.30% ratchet gap.",
    )

trailing_gap = 0.0030  # Fixed 0.30% ratchet gap once 1% target is reached

# Persistent trade state: the stop loss can ratchet in the profitable
# direction but must never move backward during the same trade.
if "ratchet_key" not in st.session_state:
    st.session_state.ratchet_key = None
    st.session_state.stop_loss = None
    st.session_state.highest_profit_pct = None
    st.session_state.highest_price = None

if st.button("🔄 Fetch Live Price & Calculate", use_container_width=True):
    with st.spinner("Fetching live NSE price via yfinance..."):
        formatted_symbol, live_price = fetch_nse_price(symbol_input)

    if live_price is None:
        st.error(
            f"Could not fetch data for '{symbol_input}'. Check the symbol or enter market price manually."
        )
    else:
        is_buy = action == "BUY"
        profit_pct = (
            (live_price - entry_price) / entry_price
            if is_buy
            else (entry_price - live_price) / entry_price
        )

        initial_sl = (
            entry_price * (1 - 0.005) if is_buy else entry_price * (1 + 0.005)
        )
        target_tp = (
            entry_price * (1 + 0.010) if is_buy else entry_price * (1 - 0.010)
        )

        # Reset the ratchet when the trade details change.
        trade_key = (formatted_symbol, action, round(entry_price, 4))
        if st.session_state.ratchet_key != trade_key:
            st.session_state.ratchet_key = trade_key
            st.session_state.stop_loss = initial_sl
            st.session_state.highest_profit_pct = profit_pct
            st.session_state.highest_price = live_price

        # Keep the best profit/price seen during this trade.
        st.session_state.highest_profit_pct = max(
            st.session_state.highest_profit_pct, profit_pct
        )

        if is_buy:
            st.session_state.highest_price = max(
                st.session_state.highest_price, live_price
            )
        else:
            st.session_state.highest_price = min(
                st.session_state.highest_price, live_price
            )

        peak_profit = st.session_state.highest_profit_pct
        peak_price = st.session_state.highest_price

        # Calculate the candidate SL from the HIGHEST profit milestone reached,
        # not from the current profit. This creates the ratchet effect.
        if peak_profit >= 0.010:
            # At 1.00% profit, immediately lock +0.70%.
            # Beyond that, trail from the best price reached.
            locked_sl = (
                entry_price * 1.007 if is_buy else entry_price * 0.993
            )
            trailing_sl = (
                peak_price * (1 - trailing_gap)
                if is_buy
                else peak_price * (1 + trailing_gap)
            )
            candidate_sl = (
                max(locked_sl, trailing_sl)
                if is_buy
                else min(locked_sl, trailing_sl)
            )
            status_msg = (
                f"🎯 1.00% Profit Reached! SL locked at minimum +0.70%; "
                f"then ratcheting with a fixed {(trailing_gap*100):.2f}% gap from the peak price"
            )
            badge_type = "success"

        elif peak_profit >= 0.0075:
            # From +0.75% up to (but not including) +1.00%,
            # the SL steps up with every favorable 0.05% move while
            # maintaining a 0.50% profit buffer.
            # Examples: +0.75% profit -> +0.25% SL,
            #           +0.80% profit -> +0.30% SL.
            stepped_profit = (
                int((peak_profit * 100 + 1e-9) / 0.05) * 0.05
            ) / 100
            sl_profit_pct = max(0.0025, stepped_profit - 0.0050)
            candidate_sl = (
                entry_price * (1 + sl_profit_pct)
                if is_buy
                else entry_price * (1 - sl_profit_pct)
            )
            status_msg = (
                f"📈 Profit Ratchet Active — Peak {peak_profit * 100:.2f}%, "
                f"SL locked at +{sl_profit_pct * 100:.2f}%"
            )
            badge_type = "info"

        elif peak_profit >= 0.005:
            candidate_sl = entry_price
            status_msg = "⚡ 0.50% Milestone Reached — SL moved to Break-Even"
            badge_type = "warning"

        else:
            candidate_sl = initial_sl
            status_msg = "⏳ Price below trailing triggers (Original -0.50% SL)"
            badge_type = "secondary"

        # THE RATCHET: SL is never allowed to move backward.
        if is_buy:
            st.session_state.stop_loss = max(
                st.session_state.stop_loss, candidate_sl
            )
        else:
            st.session_state.stop_loss = min(
                st.session_state.stop_loss, candidate_sl
            )

        new_sl = st.session_state.stop_loss

        st.divider()

        # Display Metrics
        m1, m2, m3 = st.columns(3)
        m1.metric("Live Market Price", f"₹{live_price:.2f}")
        m2.metric("P&L %", f"{profit_pct * 100:+.2f}%")
        m3.metric("Target (1.0%)", f"₹{target_tp:.2f}")

        if badge_type == "success":
            st.success(status_msg)
        elif badge_type == "info":
            st.info(status_msg)
        else:
            st.warning(status_msg)

        st.metric(label="UPDATED STOP LOSS LEVEL", value=f"₹{new_sl:.2f}")