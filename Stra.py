import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ============================================================
# Simple, followable stratification + diagnostics + merging
# - pandas only
# - ALL bins are distribution-based (quantiles)
# - NULLs become their own bin = "NULL"
# - Stratum ID is "visa|balance|web|mobile|offers"
# - Merging is simple: if a stratum is too small, merge it by
#   coarsening (reducing) quantile bins, then (optionally) merge
#   remaining tiny strata into nearest neighbors by balance quantile.
# ============================================================

NULL = "NULL"


# -------------------------
# 1) Preprocess balance (clip 1% / 99%)
# -------------------------
def clip_balance(df, col="balance", q_lo=0.01, q_hi=0.99):
    x = df[col].astype(float)
    lo = x.quantile(q_lo)
    hi = x.quantile(q_hi)
    out = x.clip(lower=lo, upper=hi)
    return out, lo, hi


# -------------------------
# 2) Quantile binning with NULL bucket
# -------------------------
def qbin_with_null(x: pd.Series, q: int, prefix: str):
    """
    Returns string bins like: f"{prefix}:Q1", ..., f"{prefix}:Q{K}"
    NULLs => "NULL"
    """
    x = x.copy()
    is_null = x.isna()
    out = pd.Series(index=x.index, dtype="object")
    out.loc[is_null] = NULL

    # qcut may drop bins if duplicates; that's OK, we handle it.
    if (~is_null).sum() > 0:
        b = pd.qcut(x[~is_null], q=q, duplicates="drop")
        # map intervals to Q1..Qk based on ordered categories
        cats = list(b.cat.categories)
        m = {cat: f"{prefix}:Q{i+1}" for i, cat in enumerate(cats)}
        out.loc[~is_null] = b.map(m).astype(str).values

    return out


def make_stratum_id(df, cols):
    s = df[cols[0]].astype(str)
    for c in cols[1:]:
        s = s + "|" + df[c].astype(str)
    return s


# -------------------------
# 3) Build strata (easy knobs)
# -------------------------
def build_strata_simple(
    DF: pd.DataFrame,
    *,
    q_balance=5,
    q_web=4,
    q_mobile=4,
    q_offers=4,
    balance_clip=(0.01, 0.99),
):
    df = DF.copy()

    # balance clip
    df["balance_clipped"], lo, hi = clip_balance(df, "balance", balance_clip[0], balance_clip[1])

    # bins (distribution-based)
    df["visa_bin"] = df["visa_ind"].apply(lambda v: NULL if pd.isna(v) else f"visa:{int(v)}")
    df["bal_bin"] = qbin_with_null(df["balance_clipped"], q_balance, "bal")
    df["web_bin"] = qbin_with_null(df["n_web_logins"], q_web, "web")
    df["mob_bin"] = qbin_with_null(df["n_mobile_logins"], q_mobile, "mob")
    df["off_bin"] = qbin_with_null(df["n_offers_year"], q_offers, "off")

    df["stratum_id"] = make_stratum_id(df, ["visa_bin", "bal_bin", "web_bin", "mob_bin", "off_bin"])
    meta = {"balance_clip_lo": float(lo), "balance_clip_hi": float(hi)}
    return df, meta


# -------------------------
# 4) Stratum size report + quick plots
# -------------------------
def stratum_sizes(df, stratum_col="stratum_id", control_frac=0.10, p_global=0.03):
    g = df.groupby(stratum_col).size().rename("N").reset_index()
    g["n_control"] = np.floor(g["N"] * control_frac).astype(int)
    g["exp_conv_control"] = g["n_control"] * p_global
    g = g.sort_values("N").reset_index(drop=True)
    summary = {
        "n_strata": int(g.shape[0]),
        "min_N": int(g["N"].min()),
        "median_N": float(g["N"].median()),
        "p10_N": float(g["N"].quantile(0.10)),
        "p25_N": float(g["N"].quantile(0.25)),
        "p75_N": float(g["N"].quantile(0.75)),
        "max_N": int(g["N"].max()),
    }
    return g, summary


def plot_stratum_sizes(g_sizes):
    plt.figure()
    plt.hist(g_sizes["N"], bins=60)
    plt.xlabel("Stratum size (N)")
    plt.ylabel("# strata")
    plt.title("Stratum size distribution")
    plt.show()

    plt.figure()
    plt.plot(np.sort(g_sizes["N"].values))
    plt.xlabel("Strata (sorted)")
    plt.ylabel("N")
    plt.title("Sorted stratum sizes")
    plt.show()


# -------------------------
# 5) Stratified split + representativeness checks
# -------------------------
def stratified_split(df, stratum_col="stratum_id", id_col="cost_ID", control_frac=0.10, seed=42):
    """
    Deterministic-ish split via hash of ID; then take lowest 10% in each stratum as control.
    """
    out = df.copy()
    h = pd.util.hash_pandas_object(out[id_col].astype(str) + f"__{seed}", index=False).astype(np.uint64)
    u = (h % np.uint64(10_000_000)) / 10_000_000.0
    out["_u"] = u
    out["_rank"] = out.groupby(stratum_col)["_u"].rank(method="first")
    out["_n"] = out.groupby(stratum_col)["_u"].transform("size")
    out["_nc"] = np.floor(out["_n"] * control_frac).astype(int).clip(lower=1)
    out["group"] = np.where(out["_rank"] <= out["_nc"], "control", "test")
    return out.drop(columns=["_u", "_rank", "_n", "_nc"])


def smd(x_c, x_t):
    x_c = pd.Series(x_c).dropna().astype(float)
    x_t = pd.Series(x_t).dropna().astype(float)
    if len(x_c) < 2 or len(x_t) < 2:
        return np.nan
    pooled = np.sqrt((x_c.var(ddof=1) + x_t.var(ddof=1)) / 2.0)
    if pooled == 0:
        return 0.0
    return (x_t.mean() - x_c.mean()) / pooled


def balance_checks(df, numeric_cols=("balance_clipped", "n_web_logins", "n_mobile_logins", "n_offers_year"), group_col="group"):
    """
    Global (not per-stratum) representativeness checks.
    Because you stratify, these should be very small.
    """
    dc = df[df[group_col] == "control"]
    dt = df[df[group_col] == "test"]

    rows = []
    for c in numeric_cols:
        rows.append({"feature": c, "SMD(test-control)": smd(dc[c], dt[c])})

    return pd.DataFrame(rows)


# -------------------------
# 6) SIMPLE merging: coarsen quantile bins until min size is satisfied
# -------------------------
def coarsen_until_ok(
    DF: pd.DataFrame,
    *,
    p_global=0.03,
    control_frac=0.10,
    n_control_min=4000,
    min_exp_conv_control=100,
    # start bins
    q_balance=5, q_web=4, q_mobile=4, q_offers=4,
    # minimum bins allowed (don’t go below these)
    min_q_balance=3, min_q_web=2, min_q_mobile=2, min_q_offers=2,
    seed=42,
):
    """
    Strategy:
      1) Build strata with quantile bins
      2) If there are small strata, reduce (#quantile bins) gradually
      3) Stop when all strata satisfy:
           n_control >= n_control_min AND exp_conv_control >= min_exp_conv_control
         OR you hit minimum bins.
    """
    qb, qw, qm, qo = q_balance, q_web, q_mobile, q_offers

    while True:
        df, meta = build_strata_simple(DF, q_balance=qb, q_web=qw, q_mobile=qm, q_offers=qo)
        g, summary = stratum_sizes(df, control_frac=control_frac, p_global=p_global)

        bad = g[(g["n_control"] < n_control_min) | (g["exp_conv_control"] < min_exp_conv_control)]
        if bad.empty:
            # final split + checks
            df_split = stratified_split(df, control_frac=control_frac, seed=seed)
            checks = balance_checks(df_split)
            return df_split, g, summary, meta, {"q_balance": qb, "q_web": qw, "q_mobile": qm, "q_offers": qo}, checks

        # Coarsening plan (simple + readable):
        # reduce bins in the dimension that currently has the most bins left,
        # prioritizing web/mobile/offers before balance (since balance carries lots of meaning).
        changed = False

        for dim in ["q_web", "q_mobile", "q_offers", "q_balance"]:
            if dim == "q_web" and qw > min_q_web:
                qw -= 1; changed = True; break
            if dim == "q_mobile" and qm > min_q_mobile:
                qm -= 1; changed = True; break
            if dim == "q_offers" and qo > min_q_offers:
                qo -= 1; changed = True; break
            if dim == "q_balance" and qb > min_q_balance:
                qb -= 1; changed = True; break

        if not changed:
            # can't coarsen more. Return current state + list of bad strata.
            df_split = stratified_split(df, control_frac=control_frac, seed=seed)
            checks = balance_checks(df_split)
            return df_split, g, summary, meta, {"q_balance": qb, "q_web": qw, "q_mobile": qm, "q_offers": qo, "note": "Hit minimum bins; some strata still small."}, checks


# -------------------------
# Example usage
# -------------------------
# DF is your dataframe (pandas)
#
# df_final, sizes_tbl, sizes_summary, meta, final_bins, checks = coarsen_until_ok(
#     DF,
#     p_global=0.03,
#     control_frac=0.10,
#     n_control_min=4000,           # for +1pp lift, ~90% power
#     min_exp_conv_control=100,
#     q_balance=5, q_web=4, q_mobile=4, q_offers=4,
#     min_q_balance=3, min_q_web=2, min_q_mobile=2, min_q_offers=2,
#     seed=42,
# )
#
# print("Balance clipped to:", meta)
# print("Final quantile bins:", final_bins)
# print("Strata summary:", sizes_summary)
# display(sizes_tbl.head(20))      # smallest strata
# display(sizes_tbl.tail(20))      # largest strata
# plot_stratum_sizes(sizes_tbl)
# display(checks)                  # global control vs test representativeness
#
# # df_final now includes:
# #   visa_bin, bal_bin, web_bin, mob_bin, off_bin, stratum_id, group
