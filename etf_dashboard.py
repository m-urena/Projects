#!/usr/bin/env python
# coding: utf-8

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from alpha_vantage.timeseries import TimeSeries
from alpha_vantage.fundamentaldata import FundamentalData
from alpha_vantage.alpha_vantage_api import AlphaVantage
import statsmodels.api as sm
import warnings
import altair as alt

API_KEY = "ZULKAGMB68HF6I9V"
ts = TimeSeries(key=API_KEY, output_format='pandas')
fd = FundamentalData(key=API_KEY, output_format='pandas')
warnings.filterwarnings("ignore", category=FutureWarning)
st.set_page_config(page_title="ETF Dashboard", layout="wide")

def format_aum(value):
    if value >= 1e12:
        return f"${value / 1e12:.2f}T"
    elif value >= 1e9:
        return f"${value / 1e9:.2f}B"
    elif value >= 1e6:
        return f"${value / 1e6:.2f}M"
    else:
        return f"${value:,.0f}"

def get_alpha_prices(ticker, start, end):
    try:
        df, _ = ts.get_daily_adjusted(symbol=ticker, outputsize='full')
        df = df.sort_index()
        df = df[['5. adjusted close']].rename(columns={'5. adjusted close': 'Close'})
        df = df[(df.index >= start) & (df.index <= end)]
        df.index = pd.to_datetime(df.index)
        return df
    except:
        return pd.DataFrame()

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

def plot_normalized_chart(tickers, start, end):
    norm_data = pd.DataFrame()
    for t in tickers:
        data = get_alpha_prices(t, start, end)
        if not data.empty:
            norm_data[t] = data['Close']
    norm_data = norm_data / norm_data.iloc[0]
    norm_data = norm_data.reset_index().melt(id_vars='date', var_name='Ticker', value_name='Normalized Price')

    chart = alt.Chart(norm_data).mark_line().encode(
        x='date:T',
        y=alt.Y('Normalized Price:Q', scale=alt.Scale(zero=False)),
        color='Ticker:N'
    ).properties(title="Normalized Price Chart")

    st.altair_chart(chart, use_container_width=True)

def diligence(tickers, start_date_str):
    try:
        start_date = datetime.strptime(start_date_str, "%Y-%m-%d")
    except ValueError:
        st.error("Start date must be in YYYY-MM-DD format.")
        return pd.DataFrame(), None, None

    end_date = datetime.today()
    price_data = {}

    for t in tickers:
        data = get_alpha_prices(t, start_date, end_date)
        if data.empty:
            st.error(f"No data available for {t}.")
            return pd.DataFrame(), None, None
        price_data[t] = data

    adjusted_start = max(df.index[0] for df in price_data.values())
    benchmark = get_alpha_prices("SPY", adjusted_start, end_date)
    benchmark = benchmark["Close"]

    records = []

    for t in tickers:
        try:
            overview, _ = fd.get_company_overview(t)
            nav = np.nan
            aum = float(overview.get("MarketCapitalization", np.nan))
            expense = float(overview.get("ExpenseRatio", np.nan))
            div_yield = float(overview.get("DividendYield", np.nan))
        except:
            nav, aum, expense, div_yield = np.nan, np.nan, np.nan, np.nan

        data = price_data[t].loc[adjusted_start:]
        std_dev, sharpe, sortino, max_drawdown, beta = calculate_stats(data["Close"], benchmark)

        records.append({
            "Ticker": t,
            "NAV ($)": round(nav, 4) if pd.notna(nav) else np.nan,
            "AUM ($)": format_aum(aum) if pd.notna(aum) else np.nan,
            "Expense Ratio": expense if pd.notna(expense) else np.nan,
            "Dividend Yield": div_yield if pd.notna(div_yield) else np.nan,
            "Volatility": std_dev,
            "Sharpe": sharpe,
            "Sortino": sortino,
            "Max Drawdown": max_drawdown,
            "Beta": beta
        })

    return pd.DataFrame(records), adjusted_start, end_date

def main():
    st.title("ETF Due Diligence Dashboard")
    tickers_input = st.text_input("Enter ETF tickers (comma-separated):", "SPY, GLD, OVLH")
    start_date_input = st.text_input("Enter start date (YYYY-MM-DD):", "2020-01-01")

    if st.button("Run Analysis"):
        tickers = [t.strip().upper() for t in tickers_input.split(",")]
        with st.spinner("Gathering data..."):
            df, start, end = diligence(tickers, start_date_input)
            if df.empty:
                return
            st.subheader("Performance & Risk Metrics")
            st.dataframe(df)

            st.subheader("Normalized Price Chart")
            plot_normalized_chart(tickers, start, end)

main()
