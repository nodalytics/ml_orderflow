import numpy as np
import pandas as pd
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
