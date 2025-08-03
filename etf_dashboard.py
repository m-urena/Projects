#!/usr/bin/env python
# coding: utf-8

# In[139]:


import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from pandas.tseries.offsets import DateOffset
from pandas_datareader import data as web
import yfinance as yf
import statsmodels.api as sm
import warnings
import altair as alt
import webbrowser
import yfinance.shared
from yfinance import pdr_override

pdr_override()
yf.enable_debug_mode()  
yf.set_tz_cache_location(".")  
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
    norm_data = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False)["Close"]
    norm_data = norm_data / norm_data.iloc[0]
    norm_data = norm_data.reset_index().melt(id_vars='Date', var_name='Ticker', value_name='Normalized Price')

    chart = alt.Chart(norm_data).mark_line().encode(
        x='Date:T',
        y=alt.Y('Normalized Price:Q', scale=alt.Scale(zero=False)),
        color='Ticker:N'
    ).properties(title="Normalized Price Chart")

    st.altair_chart(chart, use_container_width=True)

def factor_diligence(tickers, start_date, end_date):
    ff = web.DataReader("F-F_Research_Data_Factors_Daily", "famafrench", start=start_date, end=end_date)[0] / 100
    ff.index = pd.to_datetime(ff.index)

    factor_data = []

    for t in tickers:
        prices = yf.download(t, start=start_date, end=end_date, auto_adjust=True, progress=False)["Close"]
        if prices.empty:
            factor_data.append({
                "Ticker": t,
                "Alpha (Ann.)": None,
                "Beta (MKT)": None,
                "Beta (SMB)": None,
                "Beta (HML)": None,
                "R-Squared": None
            })
            continue

        returns = prices.pct_change().dropna()
        df = pd.concat([returns, ff], axis=1).dropna()
        df.columns = ["Return"] + list(ff.columns)  # <- Fix column name after concat

        if df.empty or "Return" not in df.columns:
            factor_data.append({
                "Ticker": t,
                "Alpha (Ann.)": None,
                "Beta (MKT)": None,
                "Beta (SMB)": None,
                "Beta (HML)": None,
                "R-Squared": None
            })
            continue

        X = sm.add_constant(df[["Mkt-RF", "SMB", "HML"]])
        y = df["Return"] - df["RF"]
        model = sm.OLS(y, X).fit()

        factor_data.append({
            "Ticker": t,
            "Alpha (Ann.)": round(model.params["const"] * 252, 4),
            "Beta (MKT)": round(model.params["Mkt-RF"], 4),
            "Beta (SMB)": round(model.params["SMB"], 4),
            "Beta (HML)": round(model.params["HML"], 4),
            "R-Squared": round(model.rsquared, 4)
        })

    return pd.DataFrame(factor_data)


def diligence(tickers, start_date_str):
    try:
        start_date = datetime.strptime(start_date_str, "%Y-%m-%d")
    except ValueError:
        st.error("Start date must be in YYYY-MM-DD format.")
        return pd.DataFrame(), None, None

    end_date = datetime.today()
    price_data = {}

    # Get earliest valid start date across all tickers
    for t in tickers:
        data = yf.download(t, start=start_date, end=end_date, auto_adjust=True, progress=False)["Close"]
        if data.empty:
            st.error(f"No data available for {t}.")
            return pd.DataFrame(), None, None
        price_data[t] = data
    adjusted_start = max(df.index[0] for df in price_data.values())

    benchmark = yf.download("SPY", start=adjusted_start, end=end_date, auto_adjust=True, progress=False)["Close"]
    records = []

    for t in tickers:
        info = yf.Ticker(t).info
        nav = info.get("navPrice", np.nan)
        aum = info.get("totalAssets", np.nan)
        expense = info.get("netExpenseRatio", np.nan)
        div_yield = info.get("yield", np.nan)

        data = price_data[t].loc[adjusted_start:]
        std_dev, sharpe, sortino, max_drawdown, beta = calculate_stats(data, benchmark)

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
            factor_df = factor_diligence(tickers, start, end)

            st.subheader("Performance & Risk Metrics")
            st.dataframe(df)

            st.subheader("Fama-French Factor Loadings")
            st.dataframe(factor_df)

            st.subheader("Normalized Price Chart")
            norm_data = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False)["Close"]
            norm_data = norm_data / norm_data.iloc[0]
            plot_normalized_chart(tickers, start, end)

main()

webbrowser.open("http://localhost:8501")


# In[113]:




