import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import grangercausalitytests


def topic_timeseries(df:pd.DataFrame, source_name:str):
    """
        This function creates a time-series dataframe with a 'count' column.
        Daily count is summed up here.
    """
    ts = (df.groupby(["topic_num", "date"])
             .size()
             .reset_index(name="count"))
    ts["source"] = source_name
    return ts

def group_by_date(ts: pd.DataFrame, freq: str):
    """
        This function returns a new dataframe grouped weekly or monthly.
        Inputs:
        - ts: time-series dataframe with daily 'count'.
        - freq: a string which tells what frequency: weekly or monthly to use.
    """
    s = pd.to_datetime(ts["date"], utc= True, errors="coerce").dt.tz_localize(None)
    ts = ts.assign(date=s)
    if (freq == 'W'):
        freq = "W-MON"  # weeks ending on Monday
        
        # label each row by the Monday of its week
        week_mon = ts['date'].dt.to_period(freq).dt.end_time.dt.normalize()
        ts = ts.assign(week_mon=week_mon)
        col_name = 'week_mon'
    elif (freq == 'M'):
        freq = "ME"
        
        # label each row by the end of the month
        month_end = ts['date'].dt.to_period(freq).dt.end_time.dt.normalize()
        ts = ts.assign(month_end=month_end)
        col_name = 'month_end'
        
    # group with named aggregation -> flat columns
    df = (
        ts
        .groupby(['topic_num', 'topic_name', 'source', col_name], as_index=False)
        .agg(count=('count', 'sum'))
        .sort_values(['topic_num', 'source', col_name])
    )
    return df

def create_ts_df(df: pd.DataFrame, sources: list[str], topic_num: int | None):
    """
    Return one row per week with two count columns, e.g. News_count and NewPubMed_count.
    If topic_num is not None, restrict to that topic; otherwise use all topics.
    """
    tmp = df.copy()
    if topic_num is not None and topic_num >= 0:
        tmp = tmp[tmp["topic_num"] == topic_num]

    # 1) aggregate to ONE row per (week, source)
    agg = (
        tmp.groupby(["week_mon", "source"], as_index=False)["count"]
           .sum()
    )

    # 2) pivot to columns (prevents many-to-many joins)
    wide = agg.pivot(index="week_mon", columns="source", values="count").fillna(0)

    # 3) keep/rename just the two sources requested, in a stable order
    s1, s2 = sources
    wide = wide.reindex(columns=[s1, s2])
    # wide = wide.rename(columns={s1: f"{s1}", s2: f"{s2}"})

    return wide.reset_index()

def check_causality(ts: pd.DataFrame, attrs: list[str], maxlag: int = 12):
    """
    ts: DataFrame with columns attrs=[X_col, Y_col] at a fixed frequency (e.g., weekly)
    Prints only the best (lag, p) for each direction.
    """
    # 1) stationarize (log1p + diff)
    Z = np.log1p(ts[attrs]).diff().dropna()

    # safety: keep maxlag reasonable vs length
    maxlag = max(1, min(maxlag, len(Z) // 5))

    def best_result(y, x):
        """Test x -> y; return best lag, p, and F."""
        res = grangercausalitytests(Z[[y, x]], maxlag=maxlag, verbose=False)
        best_lag, best_p = min(
            ((lag, r[0]["ssr_ftest"][1]) for lag, r in res.items()),
            key=lambda t: t[1]
        )
        best_F = res[best_lag][0]["ssr_ftest"][0]
        return best_lag, float(best_p), float(best_F)

    # X -> Y
    x, y = attrs[0], attrs[1]
    lag_xy, p_xy, F_xy = best_result(y, x)
    print(f"{x} → {y}: best lag={lag_xy}, F={F_xy:.4f}, p={p_xy:.4g}")

    # Y -> X
    lag_yx, p_yx, F_yx = best_result(x, y)
    print(f"{y} → {x}: best lag={lag_yx}, F={F_yx:.4f}, p={p_yx:.4g}")
