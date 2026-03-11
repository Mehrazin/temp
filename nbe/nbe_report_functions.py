"""
nbe_report_functions.py

Reusable helper functions for NBE legacy-vs-new ranking evaluation in PySpark.
Built to be imported into a Databricks notebook or another Python script.
"""

from typing import Dict, List, Optional, Sequence, Union

import pandas as pd
import matplotlib.pyplot as plt

from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from pyspark.sql import types as T


# ============================================================
# Basic utilities
# ============================================================

def _to_pct(x):
    return F.round(100.0 * x, 4)


def _filled_numeric_col(col_name: str, null_fill_value: float = 0.0):
    return F.coalesce(F.col(col_name).cast("double"), F.lit(float(null_fill_value)))


def _safe_product_cols(
    df: DataFrame,
    product_cols: Optional[List[str]],
    customer_id_col: str,
    control_col: str
) -> List[str]:
    if product_cols is None:
        return [c for c in df.columns if c not in {customer_id_col, control_col}]
    return product_cols


# ============================================================
# Subsetting helpers
# ============================================================

def filter_to_nbe_group(
    df: DataFrame,
    control_col: str = "control"
) -> DataFrame:
    return df.filter(F.col(control_col) == 0)


def subset_by_customer_ids(
    df: DataFrame,
    customer_ids: Optional[Union[DataFrame, Sequence]],
    customer_id_col: str = "customer_id"
) -> DataFrame:
    if customer_ids is None:
        return df

    if isinstance(customer_ids, DataFrame):
        ids_df = customer_ids.select(customer_id_col).distinct()
        return df.join(ids_df, on=customer_id_col, how="inner")

    if isinstance(customer_ids, (list, tuple, set)):
        ids_list = list(customer_ids)
        return df.filter(F.col(customer_id_col).isin(ids_list))

    raise ValueError("customer_ids must be None, a Spark DataFrame, or a Python sequence.")


def subset_by_product_cols(
    df: DataFrame,
    product_cols: Optional[List[str]],
    customer_id_col: str = "customer_id",
    control_col: str = "control"
) -> DataFrame:
    if product_cols is None:
        return df

    keep_cols = [customer_id_col, control_col] + product_cols
    keep_cols = [c for c in keep_cols if c in df.columns]
    return df.select(*keep_cols)


def apply_subsetting(
    df: DataFrame,
    customer_ids: Optional[Union[DataFrame, Sequence]] = None,
    product_cols: Optional[List[str]] = None,
    customer_id_col: str = "customer_id",
    control_col: str = "control",
    filter_control_zero: bool = True
) -> DataFrame:
    out = df

    if filter_control_zero:
        out = filter_to_nbe_group(out, control_col=control_col)

    out = subset_by_customer_ids(
        out,
        customer_ids=customer_ids,
        customer_id_col=customer_id_col
    )

    out = subset_by_product_cols(
        out,
        product_cols=product_cols,
        customer_id_col=customer_id_col,
        control_col=control_col
    )

    return out


def get_product_cols_from_df(
    df: DataFrame,
    customer_id_col: str = "customer_id",
    control_col: str = "control"
) -> List[str]:
    return [c for c in df.columns if c not in {customer_id_col, control_col}]


# ============================================================
# Ranking arrays
# ============================================================

def add_sorted_product_array(
    df: DataFrame,
    product_cols: Optional[List[str]] = None,
    output_col: str = "ranked_products",
    customer_id_col: str = "customer_id",
    control_col: str = "control",
    null_fill_value: float = 0.0
) -> DataFrame:
    product_cols = _safe_product_cols(df, product_cols, customer_id_col, control_col)

    product_structs = [
        F.struct(
            (-_filled_numeric_col(c, null_fill_value)).alias("sort_score"),
            F.lit(c).alias("product")
        )
        for c in product_cols
    ]

    return (
        df
        .select(
            customer_id_col,
            F.transform(
                F.array_sort(F.array(*product_structs)),
                lambda x: x["product"]
            ).alias(output_col)
        )
    )


def add_topk_product_array(
    df: DataFrame,
    k: int,
    product_cols: Optional[List[str]] = None,
    output_col: str = "topk_products",
    customer_id_col: str = "customer_id",
    control_col: str = "control",
    null_fill_value: float = 0.0
) -> DataFrame:
    ranked_df = add_sorted_product_array(
        df=df,
        product_cols=product_cols,
        output_col="_ranked_products",
        customer_id_col=customer_id_col,
        control_col=control_col,
        null_fill_value=null_fill_value
    )

    return ranked_df.select(
        customer_id_col,
        F.slice(F.col("_ranked_products"), 1, k).alias(output_col)
    )


# ============================================================
# Core metric functions
# ============================================================

def compute_total_expected_value(
    df: DataFrame,
    product_cols: Optional[List[str]] = None,
    output_col: str = "EV_total",
    customer_id_col: str = "customer_id",
    control_col: str = "control",
    null_fill_value: float = 0.0
) -> DataFrame:
    product_cols = _safe_product_cols(df, product_cols, customer_id_col, control_col)

    total_expr = None
    for c in product_cols:
        term = _filled_numeric_col(c, null_fill_value)
        total_expr = term if total_expr is None else (total_expr + term)

    return df.select(customer_id_col, total_expr.alias(output_col))


def compute_topk_expected_value(
    df: DataFrame,
    k: int,
    product_cols: Optional[List[str]] = None,
    output_col: str = "EV_topK",
    customer_id_col: str = "customer_id",
    control_col: str = "control",
    null_fill_value: float = 0.0
) -> DataFrame:
    product_cols = _safe_product_cols(df, product_cols, customer_id_col, control_col)

    product_structs = [
        F.struct(
            (-_filled_numeric_col(c, null_fill_value)).alias("sort_score"),
            _filled_numeric_col(c, null_fill_value).alias("ev"),
            F.lit(c).alias("product")
        )
        for c in product_cols
    ]

    sorted_array = F.array_sort(F.array(*product_structs))
    topk_array = F.slice(sorted_array, 1, k)

    return df.select(
        customer_id_col,
        F.aggregate(
            topk_array,
            F.lit(0.0),
            lambda acc, x: acc + x["ev"]
        ).alias(output_col)
    )


@F.udf(T.DoubleType())
def _spearman_from_full_orders(old_order: List[str], new_order: List[str]) -> float:
    if old_order is None or new_order is None:
        return None

    n = len(old_order)
    if n == 0:
        return None
    if n == 1:
        return 1.0

    old_rank = {p: i + 1 for i, p in enumerate(old_order)}
    new_rank = {p: i + 1 for i, p in enumerate(new_order)}

    common = [p for p in old_order if p in new_rank]
    m = len(common)
    if m < 2:
        return None

    d2 = sum((old_rank[p] - new_rank[p]) ** 2 for p in common)
    rho = 1.0 - (6.0 * d2) / (m * (m**2 - 1))
    return float(rho)


@F.udf(T.DoubleType())
def _spearman_from_topn_orders(old_topn: List[str], new_topn: List[str]) -> float:
    if old_topn is None or new_topn is None:
        return None

    n_old = len(old_topn)
    n_new = len(new_topn)
    n = max(n_old, n_new)

    all_products = list(dict.fromkeys((old_topn or []) + (new_topn or [])))
    m = len(all_products)

    if m == 0:
        return None

    old_rank = {p: i + 1 for i, p in enumerate(old_topn)}
    new_rank = {p: i + 1 for i, p in enumerate(new_topn)}

    if m == 1:
        only_product = all_products[0]
        return 1.0 if old_rank.get(only_product, n + 1) == new_rank.get(only_product, n + 1) else 0.0

    d2 = 0.0
    for p in all_products:
        r_old = old_rank.get(p, n + 1)
        r_new = new_rank.get(p, n + 1)
        d2 += (r_old - r_new) ** 2

    rho = 1.0 - (6.0 * d2) / (m * (m**2 - 1))
    return float(rho)


def compute_full_rank_correlation(
    legacy_df: DataFrame,
    new_df: DataFrame,
    product_cols: Optional[List[str]] = None,
    output_col: str = "rank_correlation",
    customer_id_col: str = "customer_id",
    control_col: str = "control",
    null_fill_value: float = 0.0
) -> DataFrame:
    legacy_ranked = add_sorted_product_array(
        df=legacy_df,
        product_cols=product_cols,
        output_col="legacy_ranked_products",
        customer_id_col=customer_id_col,
        control_col=control_col,
        null_fill_value=null_fill_value
    )

    new_ranked = add_sorted_product_array(
        df=new_df,
        product_cols=product_cols,
        output_col="new_ranked_products",
        customer_id_col=customer_id_col,
        control_col=control_col,
        null_fill_value=null_fill_value
    )

    return (
        legacy_ranked
        .join(new_ranked, on=customer_id_col, how="inner")
        .select(
            customer_id_col,
            _spearman_from_full_orders(
                F.col("legacy_ranked_products"),
                F.col("new_ranked_products")
            ).alias(output_col)
        )
    )


def compute_topn_rank_correlation(
    legacy_df: DataFrame,
    new_df: DataFrame,
    n: int,
    product_cols: Optional[List[str]] = None,
    output_col: str = "topN_rank_correlation",
    customer_id_col: str = "customer_id",
    control_col: str = "control",
    null_fill_value: float = 0.0
) -> DataFrame:
    legacy_topn = add_topk_product_array(
        df=legacy_df,
        k=n,
        product_cols=product_cols,
        output_col="legacy_topn_products",
        customer_id_col=customer_id_col,
        control_col=control_col,
        null_fill_value=null_fill_value
    )

    new_topn = add_topk_product_array(
        df=new_df,
        k=n,
        product_cols=product_cols,
        output_col="new_topn_products",
        customer_id_col=customer_id_col,
        control_col=control_col,
        null_fill_value=null_fill_value
    )

    return (
        legacy_topn
        .join(new_topn, on=customer_id_col, how="inner")
        .select(
            customer_id_col,
            _spearman_from_topn_orders(
                F.col("legacy_topn_products"),
                F.col("new_topn_products")
            ).alias(output_col)
        )
    )


def compute_topk_overlap(
    legacy_df: DataFrame,
    new_df: DataFrame,
    k: int,
    product_cols: Optional[List[str]] = None,
    output_col: str = "topK_overlap",
    customer_id_col: str = "customer_id",
    control_col: str = "control",
    null_fill_value: float = 0.0
) -> DataFrame:
    legacy_topk = add_topk_product_array(
        df=legacy_df,
        k=k,
        product_cols=product_cols,
        output_col="legacy_topk_products",
        customer_id_col=customer_id_col,
        control_col=control_col,
        null_fill_value=null_fill_value
    )

    new_topk = add_topk_product_array(
        df=new_df,
        k=k,
        product_cols=product_cols,
        output_col="new_topk_products",
        customer_id_col=customer_id_col,
        control_col=control_col,
        null_fill_value=null_fill_value
    )

    return (
        legacy_topk
        .join(new_topk, on=customer_id_col, how="inner")
        .select(
            customer_id_col,
            (
                F.size(F.array_intersect(F.col("legacy_topk_products"), F.col("new_topk_products"))) / F.lit(float(k))
            ).alias(output_col)
        )
    )


def compute_topk_swap_rate(
    legacy_df: DataFrame,
    new_df: DataFrame,
    k: int,
    product_cols: Optional[List[str]] = None,
    output_col: str = "topK_changed",
    customer_id_col: str = "customer_id",
    control_col: str = "control",
    null_fill_value: float = 0.0
) -> DataFrame:
    legacy_topk = add_topk_product_array(
        df=legacy_df,
        k=k,
        product_cols=product_cols,
        output_col="legacy_topk_products",
        customer_id_col=customer_id_col,
        control_col=control_col,
        null_fill_value=null_fill_value
    )

    new_topk = add_topk_product_array(
        df=new_df,
        k=k,
        product_cols=product_cols,
        output_col="new_topk_products",
        customer_id_col=customer_id_col,
        control_col=control_col,
        null_fill_value=null_fill_value
    )

    return (
        legacy_topk
        .join(new_topk, on=customer_id_col, how="inner")
        .select(
            customer_id_col,
            F.when(F.col("legacy_topk_products") == F.col("new_topk_products"), F.lit(0)).otherwise(F.lit(1)).alias(output_col)
        )
    )


def compute_product_distribution_shift(
    legacy_df: DataFrame,
    new_df: DataFrame,
    k: int,
    product_cols: Optional[List[str]] = None,
    customer_id_col: str = "customer_id",
    control_col: str = "control",
    null_fill_value: float = 0.0
) -> DataFrame:
    legacy_topk = add_topk_product_array(
        df=legacy_df,
        k=k,
        product_cols=product_cols,
        output_col="legacy_topk_products",
        customer_id_col=customer_id_col,
        control_col=control_col,
        null_fill_value=null_fill_value
    )

    new_topk = add_topk_product_array(
        df=new_df,
        k=k,
        product_cols=product_cols,
        output_col="new_topk_products",
        customer_id_col=customer_id_col,
        control_col=control_col,
        null_fill_value=null_fill_value
    )

    legacy_n = legacy_topk.count()
    new_n = new_topk.count()

    legacy_freq = (
        legacy_topk
        .select(F.explode("legacy_topk_products").alias("product"))
        .groupBy("product")
        .count()
        .withColumnRenamed("count", "legacy_topK_frequency")
        .withColumn("legacy_topK_pct", F.col("legacy_topK_frequency") / F.lit(float(legacy_n)))
    )

    new_freq = (
        new_topk
        .select(F.explode("new_topk_products").alias("product"))
        .groupBy("product")
        .count()
        .withColumnRenamed("count", "new_topK_frequency")
        .withColumn("new_topK_pct", F.col("new_topK_frequency") / F.lit(float(new_n)))
    )

    return (
        legacy_freq
        .join(new_freq, on="product", how="full")
        .na.fill(0)
        .orderBy("product")
    )


# ============================================================
# Summary / comparison helpers
# ============================================================

def summarize_metric(df: DataFrame, metric_col: str) -> DataFrame:
    return df.select(metric_col).summary(
        "count", "mean", "stddev", "min", "25%", "50%", "75%", "max"
    )


def metric_difference_table(
    legacy_metric_df: DataFrame,
    new_metric_df: DataFrame,
    metric_col: str,
    customer_id_col: str = "customer_id",
    diff_col: str = "metric_diff"
) -> DataFrame:
    return (
        legacy_metric_df.select(customer_id_col, F.col(metric_col).alias(f"{metric_col}_legacy"))
        .join(
            new_metric_df.select(customer_id_col, F.col(metric_col).alias(f"{metric_col}_new")),
            on=customer_id_col,
            how="inner"
        )
        .withColumn(
            diff_col,
            F.col(f"{metric_col}_new") - F.col(f"{metric_col}_legacy")
        )
    )


def add_overlap_count_from_ratio(
    overlap_df: DataFrame,
    k: int,
    overlap_col: str = "topK_overlap",
    output_col: str = "overlap_count"
) -> DataFrame:
    return overlap_df.withColumn(
        output_col,
        F.round(F.col(overlap_col) * F.lit(float(k)), 0).cast("int")
    )


def compare_same_product_values(
    legacy_df: DataFrame,
    new_df: DataFrame,
    product_cols: List[str],
    customer_id_col: str = "customer_id"
) -> DataFrame:
    joined = legacy_df.alias("l").join(new_df.alias("n"), on=customer_id_col, how="inner")

    diff_exprs = []
    for c in product_cols:
        diff_exprs.append(
            F.abs(
                F.coalesce(F.col(f"n.{c}").cast("double"), F.lit(0.0)) -
                F.coalesce(F.col(f"l.{c}").cast("double"), F.lit(0.0))
            ).alias(f"{c}_abs_diff")
        )

    out = joined.select(F.col(customer_id_col), *diff_exprs)

    sum_expr = None
    for c in product_cols:
        term = F.col(f"{c}_abs_diff")
        sum_expr = term if sum_expr is None else (sum_expr + term)

    return out.withColumn("sum_abs_diff", sum_expr)


def summarize_customer_value_diffs(
    legacy_df: DataFrame,
    new_df: DataFrame,
    product_cols: List[str],
    customer_id_col: str = "customer_id"
) -> DataFrame:
    diff_df = compare_same_product_values(
        legacy_df=legacy_df,
        new_df=new_df,
        product_cols=product_cols,
        customer_id_col=customer_id_col
    )
    return diff_df.select("sum_abs_diff").summary(
        "count", "mean", "stddev", "min", "25%", "50%", "75%", "max"
    )


# ============================================================
# Report table builders
# ============================================================

def build_ev_report_tables(
    legacy_df: DataFrame,
    new_df: DataFrame,
    product_cols: List[str],
    top_k_values: List[int],
    customer_id_col: str = "customer_id",
    control_col: str = "control",
) -> Dict[str, DataFrame]:
    legacy_ev_total = compute_total_expected_value(
        legacy_df, product_cols=product_cols, output_col="EV_total",
        customer_id_col=customer_id_col, control_col=control_col
    )
    new_ev_total = compute_total_expected_value(
        new_df, product_cols=product_cols, output_col="EV_total",
        customer_id_col=customer_id_col, control_col=control_col
    )

    ev_total_diff = metric_difference_table(
        legacy_ev_total, new_ev_total, metric_col="EV_total",
        customer_id_col=customer_id_col, diff_col="EV_total_diff"
    )

    out = {
        "legacy_ev_total": legacy_ev_total,
        "new_ev_total": new_ev_total,
        "ev_total_diff": ev_total_diff,
        "ev_total_summary": ev_total_diff.summary("count", "mean", "stddev", "min", "25%", "50%", "75%", "max"),
    }

    for k in top_k_values:
        legacy_topk_ev = compute_topk_expected_value(
            legacy_df, k=k, product_cols=product_cols, output_col="EV_topK",
            customer_id_col=customer_id_col, control_col=control_col
        )
        new_topk_ev = compute_topk_expected_value(
            new_df, k=k, product_cols=product_cols, output_col="EV_topK",
            customer_id_col=customer_id_col, control_col=control_col
        )

        topk_ev_diff = metric_difference_table(
            legacy_topk_ev, new_topk_ev, metric_col="EV_topK",
            customer_id_col=customer_id_col, diff_col=f"EV_top{k}_diff"
        )

        out[f"legacy_top{k}_ev"] = legacy_topk_ev
        out[f"new_top{k}_ev"] = new_topk_ev
        out[f"top{k}_ev_diff"] = topk_ev_diff
        out[f"top{k}_ev_summary"] = topk_ev_diff.summary("count", "mean", "stddev", "min", "25%", "50%", "75%", "max")

    return out


def build_rank_report_tables(
    legacy_df: DataFrame,
    new_df: DataFrame,
    product_cols: List[str],
    top_k_values: List[int],
    customer_id_col: str = "customer_id",
    control_col: str = "control",
) -> Dict[str, DataFrame]:
    out = {}

    full_rank_corr = compute_full_rank_correlation(
        legacy_df, new_df, product_cols=product_cols,
        output_col="rank_correlation",
        customer_id_col=customer_id_col, control_col=control_col
    )
    out["full_rank_corr"] = full_rank_corr
    out["full_rank_corr_summary"] = summarize_metric(full_rank_corr, "rank_correlation")

    for k in top_k_values:
        topk_corr = compute_topn_rank_correlation(
            legacy_df, new_df, n=k, product_cols=product_cols,
            output_col="topN_rank_correlation",
            customer_id_col=customer_id_col, control_col=control_col
        )

        topk_overlap = compute_topk_overlap(
            legacy_df, new_df, k=k, product_cols=product_cols,
            output_col="topK_overlap",
            customer_id_col=customer_id_col, control_col=control_col
        )
        topk_overlap = add_overlap_count_from_ratio(topk_overlap, k=k)

        topk_swap = compute_topk_swap_rate(
            legacy_df, new_df, k=k, product_cols=product_cols,
            output_col="topK_changed",
            customer_id_col=customer_id_col, control_col=control_col
        )

        total_rows = topk_overlap.count()

        overlap_corr = (
            topk_overlap.alias("o")
            .join(topk_corr.alias("r"), on=customer_id_col, how="inner")
            .groupBy("overlap_count")
            .agg(
                F.count("*").alias("n_customers"),
                _to_pct(F.count("*") / F.lit(float(total_rows))).alias("pct_customers"),
                F.avg("topK_overlap").alias("avg_overlap_ratio"),
                F.avg("topN_rank_correlation").alias("avg_rank_correlation"),
                F.expr("percentile_approx(topN_rank_correlation, 0.5)").alias("median_rank_correlation"),
            )
            .orderBy("overlap_count")
        )

        total_swaps = topk_swap.count()
        swap_summary = (
            topk_swap.groupBy("topK_changed")
            .agg(F.count("*").alias("n_customers"))
            .withColumn("pct_customers", _to_pct(F.col("n_customers") / F.lit(float(total_swaps))))
            .orderBy("topK_changed")
        )

        out[f"top{k}_corr"] = topk_corr
        out[f"top{k}_corr_summary"] = summarize_metric(topk_corr, "topN_rank_correlation")
        out[f"top{k}_overlap"] = topk_overlap
        out[f"top{k}_overlap_summary"] = summarize_metric(topk_overlap, "topK_overlap")
        out[f"top{k}_overlap_vs_corr"] = overlap_corr
        out[f"top{k}_swap"] = topk_swap
        out[f"top{k}_swap_summary"] = swap_summary

    return out


def build_product_shift_report_tables(
    legacy_df: DataFrame,
    new_df: DataFrame,
    product_cols: List[str],
    top_k_values: List[int],
    customer_id_col: str = "customer_id",
    control_col: str = "control",
) -> Dict[str, DataFrame]:
    out = {}

    for k in top_k_values:
        shift_df = compute_product_distribution_shift(
            legacy_df, new_df, k=k, product_cols=product_cols,
            customer_id_col=customer_id_col, control_col=control_col
        )

        shift_df = (
            shift_df
            .withColumn("pct_diff", F.col("new_topK_pct") - F.col("legacy_topK_pct"))
            .withColumn("pct_diff_abs", F.abs(F.col("pct_diff")))
            .withColumn("freq_diff", F.col("new_topK_frequency") - F.col("legacy_topK_frequency"))
            .orderBy(F.desc("pct_diff_abs"))
        )

        out[f"top{k}_product_shift"] = shift_df

    return out


# ============================================================
# Plot helpers
# ============================================================

def plot_ev_diff_histogram(
    ev_diff_df: DataFrame,
    diff_col: str,
    title: str,
    bins: int = 50,
    max_rows: int = 200000,
    figsize=(8, 5)
):
    pdf = ev_diff_df.select(diff_col).limit(max_rows).toPandas()
    plt.figure(figsize=figsize)
    plt.hist(pdf[diff_col].dropna(), bins=bins)
    plt.title(title)
    plt.xlabel(diff_col)
    plt.ylabel("Count")
    plt.tight_layout()
    plt.show()


def plot_overlap_distribution(
    overlap_vs_corr_df: DataFrame,
    k: int,
    figsize=(8, 5)
):
    pdf = overlap_vs_corr_df.toPandas()
    plt.figure(figsize=figsize)
    plt.bar(pdf["overlap_count"].astype(str), pdf["pct_customers"])
    plt.title(f"Top-{k} overlap distribution")
    plt.xlabel("Number of common products in Top-K")
    plt.ylabel("% customers")
    plt.tight_layout()
    plt.show()


def plot_overlap_vs_rankcorr(
    overlap_vs_corr_df: DataFrame,
    k: int,
    figsize=(8, 5)
):
    pdf = overlap_vs_corr_df.toPandas()
    plt.figure(figsize=figsize)
    plt.bar(pdf["overlap_count"].astype(str), pdf["avg_rank_correlation"])
    plt.title(f"Top-{k} average rank correlation by overlap count")
    plt.xlabel("Number of common products in Top-K")
    plt.ylabel("Average rank correlation")
    plt.tight_layout()
    plt.show()


def plot_swap_distribution(
    swap_summary_df: DataFrame,
    k: int,
    figsize=(6, 5)
):
    pdf = swap_summary_df.toPandas()
    pdf["label"] = pdf["topK_changed"].map({0: "No change", 1: "Changed"})
    plt.figure(figsize=figsize)
    plt.bar(pdf["label"], pdf["pct_customers"])
    plt.title(f"Top-{k} changed vs unchanged")
    plt.ylabel("% customers")
    plt.tight_layout()
    plt.show()


def plot_top_products_shift(
    product_shift_df: DataFrame,
    top_n: int = 15,
    figsize=(12, 5)
):
    pdf = product_shift_df.limit(top_n).toPandas()

    x = list(range(len(pdf)))
    width = 0.4

    plt.figure(figsize=figsize)
    plt.bar([i - width / 2 for i in x], pdf["legacy_topK_pct"], width=width, label="legacy")
    plt.bar([i + width / 2 for i in x], pdf["new_topK_pct"], width=width, label="new")
    plt.xticks(x, pdf["product"], rotation=90)
    plt.title("Top products by absolute distribution shift")
    plt.ylabel("Top-K appearance %")
    plt.legend()
    plt.tight_layout()
    plt.show()


# ============================================================
# Scenario runner
# ============================================================

def make_scenario(
    scenario_name: str,
    customer_subset=None,
    product_cols: Optional[List[str]] = None,
    top_k_values: Optional[List[int]] = None,
):
    return {
        "scenario_name": scenario_name,
        "customer_subset": customer_subset,
        "product_cols": product_cols,
        "top_k_values": top_k_values,
    }


def run_nbe_report_scenario(
    spark,
    legacy_df: DataFrame,
    new_df: DataFrame,
    scenario_name: str,
    customer_subset=None,
    product_cols: Optional[List[str]] = None,
    top_k_values: Optional[List[int]] = None,
    customer_id_col: str = "customer_id",
    control_col: str = "control",
    show_displays: bool = True,
    show_plots: bool = True,
) -> Dict[str, DataFrame]:
    if top_k_values is None:
        top_k_values = [1, 3, 5]

    legacy_base = apply_subsetting(
        legacy_df,
        customer_ids=customer_subset,
        product_cols=product_cols,
        customer_id_col=customer_id_col,
        control_col=control_col,
        filter_control_zero=True,
    )

    new_base = apply_subsetting(
        new_df,
        customer_ids=customer_subset,
        product_cols=product_cols,
        customer_id_col=customer_id_col,
        control_col=control_col,
        filter_control_zero=True,
    )

    analysis_product_cols = get_product_cols_from_df(
        legacy_base,
        customer_id_col=customer_id_col,
        control_col=control_col
    )

    report = {
        "scenario_meta": spark.createDataFrame(
            [(
                scenario_name,
                legacy_base.count(),
                len(analysis_product_cols),
                ",".join(str(x) for x in top_k_values)
            )],
            ["scenario_name", "n_customers", "n_products", "top_k_values"]
        )
    }

    report["value_diff_summary"] = summarize_customer_value_diffs(
        legacy_base, new_base, analysis_product_cols, customer_id_col=customer_id_col
    )

    report.update(build_ev_report_tables(
        legacy_base, new_base, analysis_product_cols, top_k_values=top_k_values,
        customer_id_col=customer_id_col, control_col=control_col
    ))

    report.update(build_rank_report_tables(
        legacy_base, new_base, analysis_product_cols, top_k_values=top_k_values,
        customer_id_col=customer_id_col, control_col=control_col
    ))

    report.update(build_product_shift_report_tables(
        legacy_base, new_base, analysis_product_cols, top_k_values=top_k_values,
        customer_id_col=customer_id_col, control_col=control_col
    ))

    if show_displays:
        print(f"\n===== Scenario: {scenario_name} =====")
        display(report["scenario_meta"])
        display(report["value_diff_summary"])
        display(report["ev_total_summary"])
        display(report["full_rank_corr_summary"])

        for k in top_k_values:
            display(report[f"top{k}_ev_summary"])
            display(report[f"top{k}_overlap_vs_corr"])
            display(report[f"top{k}_swap_summary"])
            display(report[f"top{k}_product_shift"])

    if show_plots:
        plot_ev_diff_histogram(
            report["ev_total_diff"],
            diff_col="EV_total_diff",
            title=f"{scenario_name} - EV total difference"
        )

        for k in top_k_values:
            plot_ev_diff_histogram(
                report[f"top{k}_ev_diff"],
                diff_col=f"EV_top{k}_diff",
                title=f"{scenario_name} - Top-{k} EV difference"
            )
            plot_overlap_distribution(report[f"top{k}_overlap_vs_corr"], k=k)
            plot_overlap_vs_rankcorr(report[f"top{k}_overlap_vs_corr"], k=k)
            plot_swap_distribution(report[f"top{k}_swap_summary"], k=k)
            plot_top_products_shift(report[f"top{k}_product_shift"], top_n=15)

    return report


def export_report_to_excel(
    report: Dict[str, DataFrame],
    output_path: str,
    max_product_shift_rows: int = 100
):
    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        for name, df in report.items():
            if not isinstance(df, DataFrame):
                continue

            pdf = df.limit(max_product_shift_rows).toPandas() if "product_shift" in name else df.toPandas()
            pdf.to_excel(writer, sheet_name=name[:31], index=False)
