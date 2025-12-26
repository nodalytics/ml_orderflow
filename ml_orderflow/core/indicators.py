import numpy as np
import pandas as pd
import pandas_ta_classic as ta
from typing import Literal

def linear_regression_channel(
    df: pd.DataFrame,
    src: str = "close",
    period: int = 100,
    devlen: float = 2.0,
    trend_mode: Literal["avg", "vote", "extreme"] = "avg",
    prev_slopes_len: int = 3,
) -> pd.DataFrame:
    """
    Linear Regression Channels calculation.
    """
    if src == "hlc3":
        prices = (df["high"] + df["low"] + df["close"]) / 3.0
    elif src == "ohlc4":
        prices = (df["open"] + df["high"] + df["low"] + df["close"]) / 4.0
    else:
        prices = df[src]

    values, uppers, lowers = [], [], []
    slopes, intercepts, degrees = [], [], []
    stdevs, r2s, cfos, trends = [], [], [], []

    # slope_history for trend detection
    slope_history = []

    for i in range(len(prices)):
        if i < period:
            values.append(np.nan)
            uppers.append(np.nan)
            lowers.append(np.nan)
            slopes.append(np.nan)
            intercepts.append(np.nan)
            degrees.append(np.nan)
            stdevs.append(np.nan)
            r2s.append(np.nan)
            cfos.append(np.nan)
            trends.append("⇒")
            continue

        y = prices.iloc[i - period : i].values
        x_arr = np.arange(1, period + 1, dtype=np.float64)
        y_arr = y.astype(float)

        x_sum = 0.5 * period * (period + 1)
        x2_sum = x_sum * (2 * period + 1) / 3.0
        divisor = period * x2_sum - x_sum * x_sum
        y_sum = np.sum(y_arr)
        xy_sum = np.sum(x_arr * y_arr)

        slope = (period * xy_sum - x_sum * y_sum) / divisor
        intercept = (y_sum * x2_sum - x_sum * xy_sum) / divisor

        residuals = (slope * x_arr + intercept) - y_arr
        res_sum_sq = np.sum(residuals**2)

        value = residuals[-1] + y_arr[-1]
        degree = 180.0 / np.pi * np.arctan(slope)
        cfo = 100.0 * residuals[-1] / y_arr[-1] if y_arr[-1] != 0 else 0.0
        stdev = np.sqrt(res_sum_sq / period)

        mean_y = np.mean(y_arr)
        sst = np.sum((y_arr - mean_y) ** 2)
        r2 = 1.0 - res_sum_sq / sst if sst != 0 else -np.inf

        upper = value + stdev * devlen
        lower = value - stdev * devlen

        slope_history.append(slope)
        if len(slope_history) > prev_slopes_len:
            slope_history.pop(0)

        if len(slope_history) < prev_slopes_len:
            trend = "⇒"
        else:
            prev_slopes = slope_history[:-1]
            if trend_mode == "avg":
                avg_prev = np.mean(prev_slopes)
                if slope > 0:
                    trend = "⇑" if slope > avg_prev else "⇗"
                elif slope < 0:
                    trend = "⇓" if slope < avg_prev else "⇘"
                else:
                    trend = "⇒"
            elif trend_mode == "vote":
                greater = sum(slope > ps for ps in prev_slopes)
                smaller = sum(slope < ps for ps in prev_slopes)
                if slope > 0:
                    trend = "⇑" if greater > smaller else "⇗"
                elif slope < 0:
                    trend = "⇓" if smaller > greater else "⇘"
                else:
                    trend = "⇒"
            elif trend_mode == "extreme":
                min_slope, max_slope = min(prev_slopes), max(prev_slopes)
                if slope > 0:
                    trend = "⇑" if slope > max_slope else "⇗"
                elif slope < 0:
                    trend = "⇓" if slope < min_slope else "⇘"
                else:
                    trend = "⇒"
            else:
                trend = "⇒"

        values.append(value)
        uppers.append(upper)
        lowers.append(lower)
        slopes.append(slope)
        intercepts.append(intercept)
        degrees.append(degree)
        stdevs.append(stdev)
        r2s.append(r2)
        cfos.append(cfo)
        trends.append(trend)

    df["reg_value"] = values
    df["reg_upper"] = uppers
    df["reg_lower"] = lowers
    df["reg_slope"] = slopes
    df["reg_intercept"] = intercepts
    df["reg_degree"] = degrees
    df["reg_stdev"] = stdevs
    df["reg_r2"] = r2s
    df["reg_cfo"] = cfos
    df["reg_trend"] = trends

    return df

def liquidity_sentiment_profile(
    df: pd.DataFrame,
    window: int = 100,
    num_bins: int = 20,
    high_value_area_pct: float = 0.70,
    low_value_area_pct: float = 0.10,
) -> pd.DataFrame:
    """
    Compute liquidity sentiment profile identifying high-traded and low-traded nodes.

    Adds columns to df for:
      - lsp_poc_price: price of highest-volume bin (Point of Control)
      - lsp_high_value_low/high: bounds of high traded node area
      - lsp_low_value_low/high: bounds of low traded node area
      - lsp_bin_width: bin width (price)
    
    Parameters
    ----------
    df : DataFrame with ['high','low','close','volume']
    window : int, rolling lookback window
    num_bins : int, number of price bins
    high_value_area_pct : float, e.g. 0.70 = 70% of volume concentrated in high nodes
    low_value_area_pct : float, e.g. 0.10 = 10% of volume concentrated in low nodes
    
    Returns
    -------
    df : DataFrame with new columns for liquidity zones.
    """
    df["lsp_poc_price"] = np.nan
    df["lsp_high_value_low"] = np.nan
    df["lsp_high_value_high"] = np.nan
    df["lsp_low_value_low"] = np.nan
    df["lsp_low_value_high"] = np.nan
    df["lsp_bin_width"] = np.nan

    for idx in range(window - 1, len(df)):
        sub = df.iloc[idx - window + 1 : idx + 1]

        lo = float(sub["low"].min())
        hi = float(sub["high"].max())
        if hi <= lo:
            continue

        bin_edges = np.linspace(lo, hi, num_bins + 1)
        prices = sub["close"].values
        vols = sub["volume"].values 

        bin_idx = np.digitize(prices, bin_edges) - 1
        vol_by_bin = np.zeros(num_bins)
        for b, v in zip(bin_idx, vols):
            if 0 <= b < num_bins:
                vol_by_bin[b] += v

        total_vol = vol_by_bin.sum()
        if total_vol <= 0:
            continue

        # Point of Control (max volume bin)
        poc_bin = int(np.argmax(vol_by_bin))
        poc_price = (bin_edges[poc_bin] + bin_edges[poc_bin + 1]) / 2.0

        # ---- High Traded Nodes (HTN) ----
        sorted_bins_high = sorted(enumerate(vol_by_bin), key=lambda x: x[1], reverse=True)
        cum, value_bins_high = 0.0, []
        target_high = total_vol * high_value_area_pct
        for b, v in sorted_bins_high:
            value_bins_high.append((b, v))
            cum += v
            if cum >= target_high:
                break

        h_bins = [b for b, _ in value_bins_high]
        h_low, h_high = min(h_bins), max(h_bins)
        high_value_low = bin_edges[h_low]
        high_value_high = bin_edges[h_high + 1]

        # ---- Low Traded Nodes (LTN) ----
        sorted_bins_low = sorted(enumerate(vol_by_bin), key=lambda x: x[1])  # ascending
        cum, value_bins_low = 0.0, []
        target_low = total_vol * low_value_area_pct
        for b, v in sorted_bins_low:
            value_bins_low.append((b, v))
            cum += v
            if cum >= target_low:
                break

        l_bins = [b for b, _ in value_bins_low]
        l_low, l_high = min(l_bins), max(l_bins)
        low_value_low = bin_edges[l_low]
        low_value_high = bin_edges[l_high + 1]

        # assign to df
        df.at[df.index[idx], "lsp_poc_price"] = poc_price
        df.at[df.index[idx], "lsp_high_value_low"] = high_value_low
        df.at[df.index[idx], "lsp_high_value_high"] = high_value_high
        df.at[df.index[idx], "lsp_low_value_low"] = low_value_low
        df.at[df.index[idx], "lsp_low_value_high"] = low_value_high
        df.at[df.index[idx], "lsp_bin_width"] = (hi - lo) / num_bins

    return df

def cdl_patterns(df: pd.DataFrame):
    return df.ta.cdl_pattern(name="all")
