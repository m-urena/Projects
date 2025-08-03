import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from alpha_vantage.timeseries import TimeSeries
from alpha_vantage.fundamentaldata import FundamentalData
import statsmodels.api as sm
import altair as alt
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)
st.set_page_config(page_title="ETF Dashboard", layout="wide")

ALPHA_KEY = "ZULKAGMB68HF6I9V"
ts = TimeSeries(key=ALPHA_KEY, output_format='pandas')
fd = FundamentalData(key=ALPHA_KEY)

def format_aum(value):
    if value >= 1e12:
        return f"${value / 1e12:.2f}T"
    elif value >= 1e9:
        return f"${value / 1e9:.2f}B"
    elif value >= 1e6:
        return f"${value / 1e6:.2f}M"
    else:
        return f"${value:,.0f}"

def fetch_close_prices(ticker, start, end):
    try:
        data, _ = ts.get_daily_adjusted(ticker, outputsize='full')
        data = data.rename(columns={"5. adjusted close": "Close"})
        data.index = pd.to_datetime(data.index)
        filtered = data.loc[(data.index >= pd.to_datetime(start)) & (data.index <= pd.to_datetime(end))]
        return filtered.sort_index()["Close"]
    except Exception:
        return pd.Series()

def calculate_stats(prices, benchmark_prices, risk_free_rate=0.01):
    returns = prices.pct_change().dropna()
    benchmark_returns = benchmark_prices.pct_change().dropna()
    aligned = pd.concat([returns, benchmark_returns], axis=1).dropna()
    if aligned.empty:
        return np.nan, np.nan, np.nan, np.nan, np.nan
    returns = aligned.iloc[:, 0]
    benchmark_returns = aligned.iloc[:, 1]
    std_dev = float(returns.std() * np.sqrt(252))
    sharpe = float((returns.mean() * 252 - risk_free_rate) / std_dev)
    downside = returns[returns < 0]
    sortino = float((returns.mean() * 252 - risk_free_rate) / (downside.std() * np.sqrt(252))) if not downside.empty else np.nan
    cumulative = (1 + returns).cumprod()
    drawdown = cumulative / cumulative.cummax() - 1
    max_drawdown = float(drawdown.min())
    beta = float(np.cov(returns, benchmark_returns)[0, 1] / np.var(benchmark_returns))
    return std_dev, sharpe, sortino, max_drawdown, beta

def diligence(tickers, start):
    end = datetime.today()
    prices = {}
    for t in tickers:
        series = fetch_close_prices(t, start, end)
        if series.empty:
            st.error(f"Data not available for {t}")
            return pd.DataFrame(), None, None
        prices[t] = series
    adjusted_start = max(s.index[0] for s in prices.values())
    for t in tickers:
        prices[t] = prices[t][adjusted_start:]
    benchmark = fetch_close_prices("SPY", adjusted_start, end)

    records = []
    for t in tickers:
        try:
            overview, _ = fd.get_company_overview(t)
        except Exception:
            overview = {}

        nav = np.nan
        aum = float(overview.get("MarketCapitalization", "nan"))
        expense = float(overview.get("ExpenseRatio", "nan"))
        yield_ = float(overview.get("DividendYield", "nan"))

        stats = calculate_stats(prices[t], benchmark)

        records.append({
            "Ticker": t,
            "NAV ($)": round(nav, 4) if pd.notna(nav) else np.nan,
            "AUM ($)": format_aum(aum) if pd.notna(aum) else np.nan,
            "Expense Ratio": expense,
            "Dividend Yield": yield_,
            "Volatility": stats[0],
            "Sharpe": stats[1],
            "Sortino": stats[2],
            "Max Drawdown": stats[3],
            "Beta": stats[4]
        })
    return pd.DataFrame(records), adjusted_start, end

def plot_normalized_chart(tickers, start, end):
    df = pd.DataFrame()
    for t in tickers:
        prices = fetch_close_prices(t, start, end)
        if prices.empty:
            continue
        df[t] = prices
    df = df / df.iloc[0]
    df = df.reset_index().melt(id_vars='index', var_name='Ticker', value_name='Normalized Price')
    df = df.rename(columns={"index": "Date"})

    chart = alt.Chart(df).mark_line().encode(
        x='Date:T',
        y=alt.Y('Normalized Price:Q', scale=alt.Scale(zero=False)),
        color='Ticker:N'
    ).properties(title="Normalized Price Chart")

    st.altair_chart(chart, use_container_width=True)

def main():
    st.title("ETF Due Diligence Dashboard")
    tickers_input = st.text_input("Enter ETF tickers (comma-separated):", "SPY,GLD,OVLH")
    start_input = st.text_input("Enter start date (YYYY-MM-DD):", "2020-01-01")

    if st.button("Run Analysis"):
        tickers = [t.strip().upper() for t in tickers_input.split(",")]
        with st.spinner("Gathering data..."):
            df, start, end = diligence(tickers, start_input)
            if df.empty:
                return
            st.subheader("Performance & Risk Metrics")
            st.dataframe(df)

            st.subheader("Normalized Price Chart")
            plot_normalized_chart(tickers, start, end)

main()
