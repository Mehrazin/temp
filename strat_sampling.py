# strat_sampling.py
# ---------------------------------------------------------
# Simple, explainable stratified sampling for large datasets
# ---------------------------------------------------------

from dataclasses import dataclass
from typing import List, Tuple, Optional

from pyspark.sql import DataFrame
from pyspark.sql import functions as F


# =========================================================
# CONFIG
# =========================================================

@dataclass(frozen=True)
class StratConfig:
    id_col: str = "cost_ID"
    min_stratum_size: int = 280_000
    control_frac: float = 0.10

    # ---- Approach A (tree-based)
    split_order: Tuple[str, ...] = (
        "visa_ind",
        "balance",
        "n_offers_year",
        "n_web_logins",
        "n_mobile_logins",
    )

    balance_bins_try: Tuple[int, ...] = (6, 5, 4, 3, 2)
    count_bins_try: Tuple[int, ...] = (4, 3, 2)

    # ---- Approach B (global coarsening)
    b_balance_start: int = 6
    b_web_start: int = 4
    b_mobile_start: int = 4
    b_offers_start: int = 4

    coarsen_priority: Tuple[str, ...] = (
        "n_offers_year",
        "n_mobile_logins",
        "n_web_logins",
        "balance",
    )

    # ---- Validation
    smd_thresh: float = 0.10
    max_prop_diff_thresh: float = 0.02


# =========================================================
# BINNING HELPERS
# =========================================================

def add_visa_bin(df: DataFrame, out_col: str, mode: str = "3way") -> DataFrame:
    if mode == "merge_null_to_0":
        return df.withColumn(
            out_col,
            F.when(F.col("visa_ind") == 1, F.lit("1")).otherwise(F.lit("0_or_null")),
        )

    return df.withColumn(
        out_col,
        F.when(F.col("visa_ind").isNull(), F.lit("__NULL__"))
         .when(F.col("visa_ind") == 1, F.lit("1"))
         .otherwise(F.lit("0")),
    )


def add_quantile_bin(
    df: DataFrame,
    col: str,
    out_col: str,
    n_bins: int,
    rel_err: float = 0.001,
) -> DataFrame:
    base = df.where(F.col(col).isNotNull())
    if base.rdd.isEmpty():
        return df.withColumn(out_col, F.lit("__NULL__"))

    probs = [i / n_bins for i in range(1, n_bins)]
    cuts = base.approxQuantile(col, probs, rel_err)
    cuts = sorted(set([c for c in cuts if c is not None]))

    expr = F.when(F.col(col).isNull(), F.lit("__NULL__"))
    prev = None
    for c in cuts:
        if prev is None:
            expr = expr.when(F.col(col) <= c, F.lit(f"(-inf,{c}]"))
        else:
            expr = expr.when(
                (F.col(col) > prev) & (F.col(col) <= c),
                F.lit(f"({prev},{c}]"),
            )
        prev = c

    expr = expr.otherwise(F.lit(f"({prev},inf)"))
    return df.withColumn(out_col, expr)


def make_stratum_key(df: DataFrame, cols: List[str], out_col: str) -> DataFrame:
    return df.withColumn(
        out_col,
        F.concat_ws(
            " | ",
            *[F.concat(F.lit(c + "="), F.col(c).cast("string")) for c in cols],
        ),
    )


# =========================================================
# CONTROL / TEST ASSIGNMENT
# =========================================================

def assign_control_test(
    df: DataFrame,
    id_col: str,
    stratum_col: str,
    control_frac: float,
    out_col: str,
) -> DataFrame:
    threshold = int(control_frac * 10_000)
    h = F.pmod(
        F.xxhash64(F.col(id_col).cast("string"), F.col(stratum_col)),
        F.lit(10_000),
    )
    return df.withColumn(out_col, (h < threshold).cast("int"))


# =========================================================
# VALIDATION
# =========================================================

def _smd(mean_t, mean_c, var_t, var_c):
    denom = F.sqrt((var_t + var_c) / 2.0)
    return F.when(denom == 0, 0.0).otherwise((mean_t - mean_c) / denom)


def validate_strata(
    df: DataFrame,
    stratum_col: str,
    treat_col: str,
    numeric_cols: Optional[List[str]] = None,
    categorical_cols: Optional[List[str]] = None,
    smd_thresh: float = 0.10,
    max_prop_diff_thresh: float = 0.02,
):
    numeric_cols = numeric_cols or [
        "balance",
        "n_web_logins",
        "n_mobile_logins",
        "n_offers_year",
    ]
    categorical_cols = categorical_cols or ["visa_ind"]

    base = (
        df.groupBy(stratum_col)
          .agg(
              F.count("*").alias("n"),
              F.sum(F.col(treat_col)).alias("n_control"),
          )
          .withColumn("n_test", F.col("n") - F.col("n_control"))
    )

    metrics = base

    for c in numeric_cols:
        g = (
            df.groupBy(stratum_col)
              .agg(
                  F.avg(F.when(F.col(treat_col) == 1, F.col(c))).alias(f"{c}_mc"),
                  F.avg(F.when(F.col(treat_col) == 0, F.col(c))).alias(f"{c}_mt"),
                  F.var_samp(F.when(F.col(treat_col) == 1, F.col(c))).alias(f"{c}_vc"),
                  F.var_samp(F.when(F.col(treat_col) == 0, F.col(c))).alias(f"{c}_vt"),
              )
              .withColumn(
                  f"{c}_smd",
                  F.abs(_smd(
                      F.col(f"{c}_mt"),
                      F.col(f"{c}_mc"),
                      F.col(f"{c}_vt"),
                      F.col(f"{c}_vc"),
                  )),
              )
        )
        metrics = metrics.join(g, stratum_col)

    for c in categorical_cols:
        tmp = df.withColumn(
            "_cat", F.when(F.col(c).isNull(), "__NULL__").otherwise(F.col(c))
        )
        ct = (
            tmp.groupBy(stratum_col, "_cat", treat_col)
               .count()
               .groupBy(stratum_col, "_cat")
               .pivot(treat_col)
               .sum("count")
               .fillna(0)
        )

        totals = base.select(stratum_col, "n_control", "n_test")
        ct = (
            ct.join(totals, stratum_col)
              .withColumn("p_c", F.col("1") / F.col("n_control"))
              .withColumn("p_t", F.col("0") / F.col("n_test"))
              .withColumn("diff", F.abs(F.col("p_c") - F.col("p_t")))
        )

        maxdiff = ct.groupBy(stratum_col).agg(
            F.max("diff").alias(f"{c}_max_prop_diff")
        )
        metrics = metrics.join(maxdiff, stratum_col)

    pass_cond = F.lit(True)
    for c in numeric_cols:
        pass_cond = pass_cond & (F.col(f"{c}_smd") <= smd_thresh)
    for c in categorical_cols:
        pass_cond = pass_cond & (F.col(f"{c}_max_prop_diff") <= max_prop_diff_thresh)

    metrics = metrics.withColumn("pass_all", pass_cond.cast("int"))

    summary = metrics.agg(
        F.count("*").alias("n_strata"),
        F.sum(1 - F.col("pass_all")).alias("n_failed"),
        F.min("n").alias("min_n"),
    )

    return metrics.orderBy("pass_all", "n"), summary


# =========================================================
# APPROACH A — TREE
# =========================================================

def stratify_tree(df: DataFrame, cfg: StratConfig) -> DataFrame:
    work = df
    nodes = [(None, [])]
    leaves = []

    def colname(depth, f):
        return f"__A_L{depth}_{f}_bin"

    depth = 0
    while nodes:
        next_nodes = []
        progressed = False

        for cond, used_cols in nodes:
            sub = work if cond is None else work.where(cond)
            if sub.count() < cfg.min_stratum_size:
                leaves.append((cond, used_cols))
                continue

            split_done = False
            for f in cfg.split_order:
                if f == "visa_ind":
                    c = colname(depth, f)
                    cand = add_visa_bin(sub, c)
                elif f == "balance":
                    for b in cfg.balance_bins_try:
                        c = colname(depth, f)
                        cand = add_quantile_bin(sub, f, c, b)
                        break
                else:
                    for b in cfg.count_bins_try:
                        c = colname(depth, f)
                        cand = add_quantile_bin(sub, f, c, b)
                        break

                counts = cand.groupBy(c).count().collect()
                if len(counts) > 1 and all(r["count"] >= cfg.min_stratum_size for r in counts):
                    work = cand if cond is None else add_quantile_bin(work, f, c, b)
                    for r in counts:
                        new_cond = (F.col(c) == r[c]) if cond is None else (cond & (F.col(c) == r[c]))
                        next_nodes.append((new_cond, used_cols + [c]))
                    split_done = True
                    progressed = True
                    break

            if not split_done:
                leaves.append((cond, used_cols))

        if not progressed:
            break
        nodes = next_nodes
        depth += 1

    out = work
    label = F.lit(None).cast("string")
    for i, (cond, _) in enumerate(leaves):
        label = F.when(cond if cond is not None else F.lit(True), f"A_{i:03d}").otherwise(label)
    return out.withColumn("stratum_A", label)


def run_approach_A_with_validation(df: DataFrame, cfg: StratConfig):
    out = stratify_tree(df, cfg)
    out = assign_control_test(out, cfg.id_col, "stratum_A", cfg.control_frac, "is_control_A")
    rep, summ = validate_strata(
        out,
        "stratum_A",
        "is_control_A",
        smd_thresh=cfg.smd_thresh,
        max_prop_diff_thresh=cfg.max_prop_diff_thresh,
    )
    return out, rep, summ


# =========================================================
# APPROACH B — GLOBAL COARSENING
# =========================================================

def stratify_coarsen(df: DataFrame, cfg: StratConfig) -> DataFrame:
    bins = {
        "balance": cfg.b_balance_start,
        "n_web_logins": cfg.b_web_start,
        "n_mobile_logins": cfg.b_mobile_start,
        "n_offers_year": cfg.b_offers_start,
    }

    def build():
        x = add_visa_bin(df, "__B_visa")
        for f in bins:
            x = add_quantile_bin(x, f, f"__B_{f}", bins[f])
        return make_stratum_key(
            x,
            ["__B_visa", "__B_balance", "__B_n_web_logins", "__B_n_mobile_logins", "__B_n_offers_year"],
            "stratum_B",
        )

    x = build()
    for _ in range(20):
        small = x.groupBy("stratum_B").count().where(F.col("count") < cfg.min_stratum_size)
        if small.count() == 0:
            break
        for f in cfg.coarsen_priority:
            if bins[f] > 2:
                bins[f] -= 1
                break
        x = build()
    return x


def run_approach_B_with_validation(df: DataFrame, cfg: StratConfig):
    out = stratify_coarsen(df, cfg)
    out = assign_control_test(out, cfg.id_col, "stratum_B", cfg.control_frac, "is_control_B")
    rep, summ = validate_strata(
        out,
        "stratum_B",
        "is_control_B",
        smd_thresh=cfg.smd_thresh,
        max_prop_diff_thresh=cfg.max_prop_diff_thresh,
    )
    return out, rep, summ
