
"""
nbe_report_functions_final.py

Restored and extended NBE analysis framework.

Design goals
------------
- keep the original 8-scenario workflow
- preserve the rich report-generation style from the earlier version
- keep code modular and readable for ad-hoc analysis
- separate:
    1) report generation
    2) plot generation
    3) Excel export
- support additional analyses:
    - priority alignment
    - response hierarchy
    - decile diagnostics
    - historical month comparison

Conventions
-----------
- Legacy = old model
- PRISM = new model
- decile 1 = highest score
- decile 10 = lowest score
"""

from typing import Dict, List, Optional, Sequence, Union
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

from pyspark.sql import DataFrame, Window
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
    control_col: Optional[str],
    extra_exclude_cols: Optional[Sequence[str]] = None,
) -> List[str]:
    exclude = {customer_id_col}
    if control_col is not None:
        exclude.add(control_col)
    for c in (extra_exclude_cols or []):
        exclude.add(c)

    if product_cols is None:
        return [c for c in df.columns if c not in exclude]
    return product_cols


def ensure_dir(path: Union[str, Path]) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def scenario_output_dir(base_output_path: Union[str, Path], scenario_name: str) -> Path:
    return ensure_dir(Path(base_output_path) / scenario_name)


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
    control_col: Optional[str] = "control",
    extra_id_cols: Optional[List[str]] = None,
) -> DataFrame:
    if product_cols is None:
        return df

    extra_id_cols = extra_id_cols or []
    keep_cols = [customer_id_col] + extra_id_cols
    if control_col is not None and control_col in df.columns:
        keep_cols.append(control_col)

    keep_cols = keep_cols + product_cols
    keep_cols = [c for c in keep_cols if c in df.columns]
    return df.select(*keep_cols)


def apply_subsetting(
    df: DataFrame,
    customer_ids: Optional[Union[DataFrame, Sequence]] = None,
    product_cols: Optional[List[str]] = None,
    customer_id_col: str = "customer_id",
    control_col: Optional[str] = "control",
    extra_id_cols: Optional[List[str]] = None,
    filter_control_zero: bool = True
) -> DataFrame:
    out = df

    if filter_control_zero and control_col is not None and control_col in out.columns:
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
        control_col=control_col,
        extra_id_cols=extra_id_cols,
    )

    return out


def get_product_cols_from_df(
    df: DataFrame,
    customer_id_col: str = "customer_id",
    control_col: Optional[str] = "control",
    extra_exclude_cols: Optional[Sequence[str]] = None,
) -> List[str]:
    return _safe_product_cols(
        df,
        product_cols=None,
        customer_id_col=customer_id_col,
        control_col=control_col,
        extra_exclude_cols=extra_exclude_cols,
    )


# ============================================================
# Ranking arrays
# ============================================================

def add_sorted_product_array(
    df: DataFrame,
    product_cols: Optional[List[str]] = None,
    output_col: str = "ranked_products",
    customer_id_col: str = "customer_id",
    control_col: Optional[str] = "control",
    null_fill_value: float = 0.0,
    extra_exclude_cols: Optional[Sequence[str]] = None,
) -> DataFrame:
    product_cols = _safe_product_cols(
        df, product_cols, customer_id_col, control_col, extra_exclude_cols=extra_exclude_cols
    )

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
    control_col: Optional[str] = "control",
    null_fill_value: float = 0.0,
    extra_exclude_cols: Optional[Sequence[str]] = None,
) -> DataFrame:
    ranked_df = add_sorted_product_array(
        df=df,
        product_cols=product_cols,
        output_col="_ranked_products",
        customer_id_col=customer_id_col,
        control_col=control_col,
        null_fill_value=null_fill_value,
        extra_exclude_cols=extra_exclude_cols,
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
    control_col: Optional[str] = "control",
    null_fill_value: float = 0.0,
    extra_exclude_cols: Optional[Sequence[str]] = None,
) -> DataFrame:
    product_cols = _safe_product_cols(
        df, product_cols, customer_id_col, control_col, extra_exclude_cols=extra_exclude_cols
    )

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
    control_col: Optional[str] = "control",
    null_fill_value: float = 0.0,
    extra_exclude_cols: Optional[Sequence[str]] = None,
) -> DataFrame:
    product_cols = _safe_product_cols(
        df, product_cols, customer_id_col, control_col, extra_exclude_cols=extra_exclude_cols
    )

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
    """
    Penalized Spearman-style top-N correlation:
    - compare the union of products appearing in either top-N list
    - missing products receive rank N+1
    """
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
    control_col: Optional[str] = "control",
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
    control_col: Optional[str] = "control",
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
    control_col: Optional[str] = "control",
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
    control_col: Optional[str] = "control",
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
    control_col: Optional[str] = "control",
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
# Artifact registration
# ============================================================

def make_table_artifact(
    name: str,
    df: DataFrame,
    section: str,
    customer_level: bool,
    include_in_excel: bool
) -> Dict:
    return {
        "name": name,
        "type": "table",
        "section": section,
        "customer_level": customer_level,
        "include_in_excel": include_in_excel,
        "df": df,
    }


def make_metadata_artifact(
    name: str,
    records: List[dict],
    section: str,
    include_in_excel: bool = True
) -> Dict:
    return {
        "name": name,
        "type": "metadata",
        "section": section,
        "customer_level": False,
        "include_in_excel": include_in_excel,
        "records": records,
    }


# ============================================================
# Report generation for original 8-scenario workflow
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


def generate_report(
    spark,
    legacy_df: DataFrame,
    new_df: DataFrame,
    scenario_name: str,
    customer_subset=None,
    product_cols: Optional[List[str]] = None,
    top_k_values: Optional[List[int]] = None,
    customer_id_col: str = "customer_id",
    control_col: str = "control",
) -> Dict:
    """
    Generate the internal report object for one of the original scenarios.
    This is the object later consumed by plotting and Excel export.
    """
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

    artifacts = []

    meta_rows = [{
        "scenario_name": scenario_name,
        "n_customers": legacy_base.count(),
        "n_products": len(analysis_product_cols),
        "top_k_values": ",".join(str(x) for x in top_k_values),
    }]
    artifacts.append(make_metadata_artifact("scenario_meta", meta_rows, section="Scenario"))

    value_diff_summary = summarize_customer_value_diffs(
        legacy_base, new_base, analysis_product_cols, customer_id_col=customer_id_col
    )
    artifacts.append(make_table_artifact(
        "value_diff_summary", value_diff_summary, section="Input Value Comparison",
        customer_level=False, include_in_excel=True
    ))

    legacy_ev_total = compute_total_expected_value(
        legacy_base, product_cols=analysis_product_cols, output_col="EV_total",
        customer_id_col=customer_id_col, control_col=control_col
    )
    new_ev_total = compute_total_expected_value(
        new_base, product_cols=analysis_product_cols, output_col="EV_total",
        customer_id_col=customer_id_col, control_col=control_col
    )
    ev_total_diff = metric_difference_table(
        legacy_ev_total, new_ev_total, metric_col="EV_total",
        customer_id_col=customer_id_col, diff_col="EV_total_diff"
    )
    ev_total_summary = ev_total_diff.summary("count", "mean", "stddev", "min", "25%", "50%", "75%", "max")

    artifacts.append(make_table_artifact("ev_total_diff", ev_total_diff, "Expected Value", True, False))
    artifacts.append(make_table_artifact("ev_total_summary", ev_total_summary, "Expected Value", False, True))

    full_rank_corr = compute_full_rank_correlation(
        legacy_base, new_base, product_cols=analysis_product_cols,
        output_col="rank_correlation",
        customer_id_col=customer_id_col, control_col=control_col
    )
    full_rank_corr_summary = summarize_metric(full_rank_corr, "rank_correlation")

    artifacts.append(make_table_artifact("full_rank_corr", full_rank_corr, "Rank Stability", True, False))
    artifacts.append(make_table_artifact("full_rank_corr_summary", full_rank_corr_summary, "Rank Stability", False, True))

    for k in top_k_values:
        legacy_topk_ev = compute_topk_expected_value(
            legacy_base, k=k, product_cols=analysis_product_cols, output_col="EV_topK",
            customer_id_col=customer_id_col, control_col=control_col
        )
        new_topk_ev = compute_topk_expected_value(
            new_base, k=k, product_cols=analysis_product_cols, output_col="EV_topK",
            customer_id_col=customer_id_col, control_col=control_col
        )
        topk_ev_diff = metric_difference_table(
            legacy_topk_ev, new_topk_ev, metric_col="EV_topK",
            customer_id_col=customer_id_col, diff_col=f"EV_top{k}_diff"
        )
        topk_ev_summary = topk_ev_diff.summary("count", "mean", "stddev", "min", "25%", "50%", "75%", "max")

        artifacts.append(make_table_artifact(f"top{k}_ev_diff", topk_ev_diff, f"Top-{k} Metrics", True, False))
        artifacts.append(make_table_artifact(f"top{k}_ev_summary", topk_ev_summary, f"Top-{k} Metrics", False, True))

        topk_corr = compute_topn_rank_correlation(
            legacy_base, new_base, n=k, product_cols=analysis_product_cols,
            output_col="topN_rank_correlation",
            customer_id_col=customer_id_col, control_col=control_col
        )

        topk_overlap = compute_topk_overlap(
            legacy_base, new_base, k=k, product_cols=analysis_product_cols,
            output_col="topK_overlap",
            customer_id_col=customer_id_col, control_col=control_col
        )
        topk_overlap = add_overlap_count_from_ratio(topk_overlap, k=k)

        topk_swap = compute_topk_swap_rate(
            legacy_base, new_base, k=k, product_cols=analysis_product_cols,
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

        topk_corr_summary = summarize_metric(topk_corr, "topN_rank_correlation")
        topk_overlap_summary = summarize_metric(topk_overlap, "topK_overlap")

        artifacts.append(make_table_artifact(f"top{k}_corr", topk_corr, f"Top-{k} Metrics", True, False))
        artifacts.append(make_table_artifact(f"top{k}_corr_summary", topk_corr_summary, f"Top-{k} Metrics", False, True))
        artifacts.append(make_table_artifact(f"top{k}_overlap", topk_overlap, f"Top-{k} Metrics", True, False))
        artifacts.append(make_table_artifact(f"top{k}_overlap_summary", topk_overlap_summary, f"Top-{k} Metrics", False, True))
        artifacts.append(make_table_artifact(f"top{k}_overlap_vs_corr", overlap_corr, f"Top-{k} Metrics", False, True))
        artifacts.append(make_table_artifact(f"top{k}_swap", topk_swap, f"Top-{k} Metrics", True, False))
        artifacts.append(make_table_artifact(f"top{k}_swap_summary", swap_summary, f"Top-{k} Metrics", False, True))

        shift_df = compute_product_distribution_shift(
            legacy_base, new_base, k=k, product_cols=analysis_product_cols,
            customer_id_col=customer_id_col, control_col=control_col
        )
        shift_df = (
            shift_df
            .withColumn("pct_diff", F.col("new_topK_pct") - F.col("legacy_topK_pct"))
            .withColumn("pct_diff_abs", F.abs(F.col("pct_diff")))
            .withColumn("freq_diff", F.col("new_topK_frequency") - F.col("legacy_topK_frequency"))
            .orderBy(F.desc("pct_diff_abs"))
        )
        artifacts.append(make_table_artifact(f"top{k}_product_shift", shift_df, f"Top-{k} Product Shift", False, True))

    return {
        "scenario_name": scenario_name,
        "customer_id_col": customer_id_col,
        "control_col": control_col,
        "product_cols": analysis_product_cols,
        "artifacts": artifacts,
    }


# ============================================================
# Plot configuration
# ============================================================

DEFAULT_PLOT_CONFIG = {
    "theme": {
        "figsize": (10, 6),
        "title_size": 15,
        "label_size": 11,
        "tick_size": 10,
        "dpi": 150,
        "palette": {
            "legacy": "#1f77b4",
            "prism": "#ff7f0e",
            "changed": "#d62728",
            "unchanged": "#2ca02c",
            "neutral": "#7f7f7f",
            "accent": "#9467bd",
        }
    },

    "ev_total_diff_hist": {
        "enabled": True,
        "title": "EV Total Difference (PRISM - Legacy)",
        "xlabel": "EV Total Difference",
        "ylabel": "Customer Count",
        "bins": 50,
        "max_rows": 200000,
        "filename": "01_ev_total_diff_hist.png",
    },

    "topk_ev_diff_hist": {
        "enabled": True,
        "title_template": "Top-{k} EV Difference (PRISM - Legacy)",
        "xlabel_template": "Top-{k} EV Difference",
        "ylabel": "Customer Count",
        "bins": 50,
        "max_rows": 200000,
        "filename_template": "02_top{k}_ev_diff_hist.png",
    },

    "topk_overlap_distribution": {
        "enabled": True,
        "title_template": "Top-{k} Overlap Distribution",
        "xlabel": "Number of Common Products in Top-K",
        "ylabel": "% of Customers",
        "filename_template": "03_top{k}_overlap_distribution.png",
    },

    "topk_overlap_vs_rankcorr": {
        "enabled": True,
        "title_template": "Top-{k} Avg Rank Correlation by Overlap Count",
        "xlabel": "Number of Common Products in Top-K",
        "ylabel": "Average Rank Correlation",
        "filename_template": "04_top{k}_overlap_vs_rankcorr.png",
    },

    "topk_swap_distribution": {
        "enabled": True,
        "title_template": "Top-{k} Changed vs Unchanged",
        "ylabel": "% of Customers",
        "filename_template": "05_top{k}_swap_distribution.png",
    },

    "topk_product_shift": {
        "enabled": True,
        "title_template": "Top-{k} Product Appearance Shift",
        "ylabel": "Top-K Appearance %",
        "top_n": 15,
        "filename_template": "06_top{k}_product_shift.png",
    },

    "priority_alignment_distribution": {
        "enabled": True,
        "title": "Distribution of Proposed Product Rank in Priority Table",
        "xlabel": "Priority Rank",
        "ylabel": "Offer Count",
        "filename": "07_priority_alignment_distribution.png",
    },

    "response_state_distribution": {
        "enabled": True,
        "title": "Response State Distribution",
        "xlabel": "Response State",
        "ylabel": "Count",
        "filename": "08_response_state_distribution.png",
    },

    "outcome_by_group": {
        "enabled": True,
        "title_template": "{metric} by {group_col}",
        "ylabel_template": "{metric}",
        "filename_template": "09_{group_col}_{metric}.png",
    },

    "decile_distribution": {
        "enabled": True,
        "title_template": "{decile_col} distribution",
        "ylabel": "Count",
        "filename_template": "10_{decile_col}_distribution.png",
    },
}


# ============================================================
# Plot generation
# ============================================================

def _apply_theme(cfg):
    plt.rcParams["font.size"] = cfg["title_size"]
    plt.rcParams["axes.titlesize"] = cfg["title_size"]
    plt.rcParams["axes.labelsize"] = cfg["label_size"]
    plt.rcParams["xtick.labelsize"] = cfg["tick_size"]
    plt.rcParams["ytick.labelsize"] = cfg["tick_size"]


def _find_artifact(report: Dict, name: str) -> Dict:
    for artifact in report["artifacts"]:
        if artifact["name"] == name:
            return artifact
    raise KeyError(f"Artifact not found: {name}")


def _save_histogram(df: DataFrame, col: str, output_path: Path, cfg_plot: Dict, cfg_theme: Dict, title: str, xlabel: str, ylabel: str):
    pdf = df.select(col).limit(cfg_plot["max_rows"]).toPandas()

    plt.figure(figsize=cfg_theme["figsize"], dpi=cfg_theme["dpi"])
    plt.hist(pdf[col].dropna(), bins=cfg_plot["bins"], color=cfg_theme["palette"]["accent"], edgecolor="white")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(axis="y", alpha=0.2)
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()


def _save_bar_plot(x, y, output_path: Path, cfg_theme: Dict, title: str, xlabel: str, ylabel: str, color: str, rotation: int = 0):
    plt.figure(figsize=cfg_theme["figsize"], dpi=cfg_theme["dpi"])
    plt.bar(x, y, color=color, edgecolor="white")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(axis="y", alpha=0.2)
    plt.xticks(rotation=rotation)
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()


def _save_grouped_bar(pdf: pd.DataFrame, output_path: Path, cfg_theme: Dict, title: str, ylabel: str):
    x = list(range(len(pdf)))
    width = 0.4

    plt.figure(figsize=(12, 6), dpi=cfg_theme["dpi"])
    plt.bar(
        [i - width / 2 for i in x], pdf["legacy_topK_pct"], width=width,
        label="Legacy", color=cfg_theme["palette"]["legacy"], edgecolor="white"
    )
    plt.bar(
        [i + width / 2 for i in x], pdf["new_topK_pct"], width=width,
        label="PRISM", color=cfg_theme["palette"]["prism"], edgecolor="white"
    )
    plt.title(title)
    plt.ylabel(ylabel)
    plt.xticks(x, pdf["product"], rotation=90)
    plt.grid(axis="y", alpha=0.2)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()


def generate_plots(
    report: Dict,
    base_output_path: Union[str, Path],
    plot_config: Optional[Dict] = None
) -> Dict[str, str]:
    """
    Generate plots only for keys present in plot_config.
    """
    plot_config = plot_config or DEFAULT_PLOT_CONFIG
    theme_cfg = plot_config["theme"]
    _apply_theme(theme_cfg)

    scenario_name = report["scenario_name"]
    out_dir = scenario_output_dir(base_output_path, scenario_name)
    output_files = {}

    if "ev_total_diff_hist" in plot_config and plot_config["ev_total_diff_hist"].get("enabled", True):
        cfg = plot_config["ev_total_diff_hist"]
        artifact = _find_artifact(report, "ev_total_diff")
        output_path = out_dir / cfg["filename"]
        _save_histogram(
            artifact["df"],
            col="EV_total_diff",
            output_path=output_path,
            cfg_plot=cfg,
            cfg_theme=theme_cfg,
            title=cfg["title"],
            xlabel=cfg["xlabel"],
            ylabel=cfg["ylabel"],
        )
        output_files["ev_total_diff_hist"] = str(output_path)

    topk_values = sorted(
        int(a["name"].split("_")[0].replace("top", ""))
        for a in report["artifacts"]
        if a["name"].startswith("top") and a["name"].endswith("_ev_diff")
    )

    for k in topk_values:
        if "topk_ev_diff_hist" in plot_config and plot_config["topk_ev_diff_hist"].get("enabled", True):
            cfg = plot_config["topk_ev_diff_hist"]
            artifact = _find_artifact(report, f"top{k}_ev_diff")
            output_path = out_dir / cfg["filename_template"].format(k=k)
            _save_histogram(
                artifact["df"],
                col=f"EV_top{k}_diff",
                output_path=output_path,
                cfg_plot=cfg,
                cfg_theme=theme_cfg,
                title=cfg["title_template"].format(k=k),
                xlabel=cfg["xlabel_template"].format(k=k),
                ylabel=cfg["ylabel"],
            )
            output_files[f"top{k}_ev_diff_hist"] = str(output_path)

        if "topk_overlap_distribution" in plot_config and plot_config["topk_overlap_distribution"].get("enabled", True):
            cfg = plot_config["topk_overlap_distribution"]
            pdf = _find_artifact(report, f"top{k}_overlap_vs_corr")["df"].toPandas()
            output_path = out_dir / cfg["filename_template"].format(k=k)
            _save_bar_plot(
                x=pdf["overlap_count"].astype(str),
                y=pdf["pct_customers"],
                output_path=output_path,
                cfg_theme=theme_cfg,
                title=cfg["title_template"].format(k=k),
                xlabel=cfg["xlabel"],
                ylabel=cfg["ylabel"],
                color=theme_cfg["palette"]["legacy"],
            )
            output_files[f"top{k}_overlap_distribution"] = str(output_path)

        if "topk_overlap_vs_rankcorr" in plot_config and plot_config["topk_overlap_vs_rankcorr"].get("enabled", True):
            cfg = plot_config["topk_overlap_vs_rankcorr"]
            pdf = _find_artifact(report, f"top{k}_overlap_vs_corr")["df"].toPandas()
            output_path = out_dir / cfg["filename_template"].format(k=k)
            _save_bar_plot(
                x=pdf["overlap_count"].astype(str),
                y=pdf["avg_rank_correlation"],
                output_path=output_path,
                cfg_theme=theme_cfg,
                title=cfg["title_template"].format(k=k),
                xlabel=cfg["xlabel"],
                ylabel=cfg["ylabel"],
                color=theme_cfg["palette"]["accent"],
            )
            output_files[f"top{k}_overlap_vs_rankcorr"] = str(output_path)

        if "topk_swap_distribution" in plot_config and plot_config["topk_swap_distribution"].get("enabled", True):
            cfg = plot_config["topk_swap_distribution"]
            pdf = _find_artifact(report, f"top{k}_swap_summary")["df"].toPandas()
            pdf["label"] = pdf["topK_changed"].map({0: "Unchanged", 1: "Changed"})
            pdf["plot_color"] = pdf["topK_changed"].map({
                0: theme_cfg["palette"]["unchanged"],
                1: theme_cfg["palette"]["changed"],
            })
            output_path = out_dir / cfg["filename_template"].format(k=k)

            plt.figure(figsize=theme_cfg["figsize"], dpi=theme_cfg["dpi"])
            plt.bar(pdf["label"], pdf["pct_customers"], color=list(pdf["plot_color"]), edgecolor="white")
            plt.title(cfg["title_template"].format(k=k))
            plt.ylabel(cfg["ylabel"])
            plt.grid(axis="y", alpha=0.2)
            plt.tight_layout()
            plt.savefig(output_path, bbox_inches="tight")
            plt.close()

            output_files[f"top{k}_swap_distribution"] = str(output_path)

        if "topk_product_shift" in plot_config and plot_config["topk_product_shift"].get("enabled", True):
            cfg = plot_config["topk_product_shift"]
            pdf = _find_artifact(report, f"top{k}_product_shift")["df"].limit(cfg["top_n"]).toPandas()
            output_path = out_dir / cfg["filename_template"].format(k=k)
            _save_grouped_bar(
                pdf=pdf,
                output_path=output_path,
                cfg_theme=theme_cfg,
                title=cfg["title_template"].format(k=k),
                ylabel=cfg["ylabel"],
            )
            output_files[f"top{k}_product_shift"] = str(output_path)

    return output_files


# ============================================================
# Excel export
# ============================================================

def export_report_to_excel_single_sheet(
    report: Dict,
    base_output_path: Union[str, Path],
    excel_filename: Optional[str] = None,
    max_rows_per_table: int = 5000
) -> str:
    """
    One Excel file per scenario.
    One worksheet per scenario.
    Customer-level tables are excluded by explicit flag.
    """
    scenario_name = report["scenario_name"]
    out_dir = scenario_output_dir(base_output_path, scenario_name)
    excel_filename = excel_filename or f"{scenario_name}_report.xlsx"
    excel_path = out_dir / excel_filename

    exportable = [
        a for a in report["artifacts"]
        if a["type"] in {"table", "metadata"} and a["include_in_excel"] and not a["customer_level"]
    ]

    section_order = []
    for artifact in exportable:
        if artifact["section"] not in section_order:
            section_order.append(artifact["section"])

    with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
        sheet_name = scenario_name[:31]
        startrow = 0

        for section in section_order:
            section_items = [a for a in exportable if a["section"] == section]

            section_header = pd.DataFrame({"Section": [section]})
            section_header.to_excel(writer, sheet_name=sheet_name, startrow=startrow, index=False)
            startrow += len(section_header) + 1

            for artifact in section_items:
                title_df = pd.DataFrame({"Artifact": [artifact["name"]]})
                title_df.to_excel(writer, sheet_name=sheet_name, startrow=startrow, index=False)
                startrow += len(title_df)

                if artifact["type"] == "metadata":
                    pdf = pd.DataFrame(artifact["records"])
                else:
                    pdf = artifact["df"].limit(max_rows_per_table).toPandas()

                pdf.to_excel(writer, sheet_name=sheet_name, startrow=startrow, index=False)
                startrow += len(pdf) + 2

    return str(excel_path)


# ============================================================
# Combined scenario runner
# ============================================================

def run_nbe_scenario_outputs(
    spark,
    legacy_df: DataFrame,
    new_df: DataFrame,
    scenario_name: str,
    base_output_path: Union[str, Path],
    customer_subset=None,
    product_cols: Optional[List[str]] = None,
    top_k_values: Optional[List[int]] = None,
    customer_id_col: str = "customer_id",
    control_col: str = "control",
    plot_config: Optional[Dict] = None,
    export_excel: bool = True
) -> Dict:
    """
    End-to-end runner for the original 8-scenario workflow:
    1) generate internal report object
    2) generate plots into scenario directory
    3) export one Excel file per scenario
    """
    report = generate_report(
        spark=spark,
        legacy_df=legacy_df,
        new_df=new_df,
        scenario_name=scenario_name,
        customer_subset=customer_subset,
        product_cols=product_cols,
        top_k_values=top_k_values,
        customer_id_col=customer_id_col,
        control_col=control_col,
    )

    plot_files = generate_plots(
        report=report,
        base_output_path=base_output_path,
        plot_config=plot_config
    )

    excel_file = None
    if export_excel:
        excel_file = export_report_to_excel_single_sheet(
            report=report,
            base_output_path=base_output_path
        )

    return {
        "scenario_name": scenario_name,
        "scenario_dir": str(scenario_output_dir(base_output_path, scenario_name)),
        "report": report,
        "plot_files": plot_files,
        "excel_file": excel_file,
    }


# ============================================================
# Additional analyses built on top of the original framework
# ============================================================

# -----------------------------
# Priority alignment
# -----------------------------

def compute_priority_alignment_report(
    priority_df: DataFrame,
    proposed_df: DataFrame,
    active_product_cols: List[str],
    customer_id_col: str = "customer_id",
    proposed_product_col: str = "proposed_product",
) -> Dict[str, DataFrame]:
    priority_rank_df = compute_rank_array(
        priority_df,
        product_cols=active_product_cols,
        customer_id_col=customer_id_col,
    )

    priority_alignment_df = (
        proposed_df.join(priority_rank_df, on=customer_id_col, how="inner")
        .select(
            customer_id_col,
            proposed_product_col,
            F.array_position(F.col("ranked_products"), F.col(proposed_product_col)).alias("priority_rank"),
        )
    )

    alignment_summary = priority_alignment_df.agg(
        F.count("*").alias("n_offers"),
        F.avg((F.col("priority_rank") == 1).cast("double")).alias("pct_top1"),
        F.avg((F.col("priority_rank") <= 3).cast("double")).alias("pct_top3"),
        F.avg((F.col("priority_rank") <= 5).cast("double")).alias("pct_top5"),
        F.avg((F.col("priority_rank") <= 10).cast("double")).alias("pct_top10"),
    )

    alignment_distribution = (
        priority_alignment_df.groupBy("priority_rank").count().orderBy("priority_rank")
    )

    return {
        "priority_rank_df": priority_rank_df,
        "priority_alignment_df": priority_alignment_df,
        "priority_alignment_summary": alignment_summary,
        "priority_alignment_distribution": alignment_distribution,
    }


# -----------------------------
# Response hierarchy
# -----------------------------

def add_response_flags_if_needed(
    df: DataFrame,
    response_state_col: str = "response_state",
) -> DataFrame:
    """
    If activated / actioned / interested do not exist,
    create broad flags from response_state.
    """
    cols = set(df.columns)
    out = df

    if {"activated", "actioned", "interested"}.issubset(cols):
        return out

    state = F.upper(F.col(response_state_col))

    if "activated" not in cols:
        out = out.withColumn(
            "activated",
            F.when(
                state.isin(
                    "ACTIVATED", "ACTIONED", "INTERESTED", "NOT INTERESTED"
                ),
                1,
            ).otherwise(0),
        )

    if "actioned" not in cols:
        out = out.withColumn(
            "actioned",
            F.when(
                state.isin(
                    "ACTIONED", "INTERESTED", "NOT INTERESTED"
                ),
                1,
            ).otherwise(0),
        )

    if "interested" not in cols:
        out = out.withColumn(
            "interested",
            F.when(state == "INTERESTED", 1).otherwise(0),
        )

    return out


def compute_response_hierarchy_report(
    measurement_df: DataFrame,
    customer_group_df: Optional[DataFrame] = None,
    customer_id_col: str = "customer_id",
    group_col_name: str = "top1_same",
    response_state_col: str = "response_state",
) -> Dict[str, DataFrame]:
    """
    Example use:
    - attach PRISM/Legacy agreement bucket to measurement data
    - compare Activated / Actioned / Interested by group
    """
    df = add_response_flags_if_needed(measurement_df, response_state_col=response_state_col)

    if customer_group_df is not None:
        df = df.join(customer_group_df, on=customer_id_col, how="left")

    response_state_summary = (
        df.groupBy(response_state_col)
        .agg(
            F.count("*").alias("n"),
            F.avg(F.col("activated").cast("double")).alias("activation_rate"),
            F.avg(F.col("actioned").cast("double")).alias("action_rate"),
            F.avg(F.col("interested").cast("double")).alias("interest_rate"),
        )
        .orderBy(F.desc("n"))
    )

    funnel_summary = df.agg(
        F.count("*").alias("n_proposed"),
        F.avg(F.col("activated").cast("double")).alias("activation_rate"),
        F.avg(F.col("actioned").cast("double")).alias("action_rate"),
        F.avg(F.col("interested").cast("double")).alias("interest_rate"),
    )

    outcome_by_group = None
    if customer_group_df is not None and group_col_name in df.columns:
        outcome_by_group = (
            df.groupBy(group_col_name)
            .agg(
                F.count("*").alias("n"),
                F.avg(F.col("activated").cast("double")).alias("activation_rate"),
                F.avg(F.col("actioned").cast("double")).alias("action_rate"),
                F.avg(F.col("interested").cast("double")).alias("interest_rate"),
            )
            .orderBy(group_col_name)
        )

    return {
        "measurement_enriched": df,
        "response_state_summary": response_state_summary,
        "funnel_summary": funnel_summary,
        "outcome_by_group": outcome_by_group,
    }


# -----------------------------
# Decile diagnostics
# -----------------------------

def assign_score_deciles(
    df: DataFrame,
    score_col: str,
    output_col: str,
    partition_cols: Optional[List[str]] = None,
) -> DataFrame:
    """
    decile 1 = highest
    decile 10 = lowest
    """
    partition_cols = partition_cols or []

    if partition_cols:
        w = Window.partitionBy(*partition_cols).orderBy(F.desc(F.col(score_col)))
    else:
        w = Window.orderBy(F.desc(F.col(score_col)))

    return df.withColumn(output_col, F.ntile(10).over(w))


def compute_decile_diagnostics_report(
    df: DataFrame,
    prism_score_col: str,
    legacy_score_col: str,
    product_col: Optional[str] = None,
) -> Dict[str, DataFrame]:
    out = df

    partition_cols = [product_col] if product_col else []

    if "prism_decile" not in out.columns:
        out = assign_score_deciles(out, prism_score_col, "prism_decile", partition_cols=partition_cols)

    if "legacy_decile" not in out.columns:
        out = assign_score_deciles(out, legacy_score_col, "legacy_decile", partition_cols=partition_cols)

    prism_decile_dist = out.groupBy("prism_decile").count().orderBy("prism_decile")
    legacy_decile_dist = out.groupBy("legacy_decile").count().orderBy("legacy_decile")

    transition = (
        out.groupBy("prism_decile", "legacy_decile")
        .count()
        .orderBy("prism_decile", "legacy_decile")
    )

    return {
        "decile_ready_df": out,
        "prism_decile_distribution": prism_decile_dist,
        "legacy_decile_distribution": legacy_decile_dist,
        "decile_transition_matrix": transition,
    }


# -----------------------------
# Historical month comparison
# -----------------------------

def compute_historical_month_report(
    monthly_df: DataFrame,
    month_col: str = "month",
    product_col: str = "product",
    customer_id_col: str = "customer_id",
    month_a_value: Optional[str] = None,
    month_b_value: Optional[str] = None,
) -> Dict[str, DataFrame]:
    df = add_response_flags_if_needed(monthly_df, response_state_col="response_state") if "response_state" in monthly_df.columns else monthly_df

    month_product_rates = (
        df.groupBy(month_col, product_col)
        .agg(
            F.count("*").alias("n_offers"),
            F.avg(F.col("activated").cast("double")).alias("activation_rate"),
            F.avg(F.col("actioned").cast("double")).alias("action_rate"),
            F.avg(F.col("interested").cast("double")).alias("interest_rate"),
        )
        .orderBy(month_col, product_col)
    )

    out = {
        "month_product_rates": month_product_rates
    }

    if month_a_value is not None and month_b_value is not None:
        month_a_df = df.filter(F.col(month_col) == month_a_value)
        month_b_df = df.filter(F.col(month_col) == month_b_value)

        common_ids = (
            month_a_df.select(customer_id_col).distinct()
            .intersect(month_b_df.select(customer_id_col).distinct())
        )

        common_monthly_df = df.join(common_ids, on=customer_id_col, how="inner")

        common_rates = (
            common_monthly_df.groupBy(month_col, product_col)
            .agg(
                F.count("*").alias("n_offers"),
                F.avg(F.col("activated").cast("double")).alias("activation_rate"),
                F.avg(F.col("actioned").cast("double")).alias("action_rate"),
                F.avg(F.col("interested").cast("double")).alias("interest_rate"),
            )
            .orderBy(month_col, product_col)
        )

        out["common_customer_ids"] = common_ids
        out["common_monthly_df"] = common_monthly_df
        out["common_month_product_rates"] = common_rates

    return out


# ============================================================
# Scenario helper
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
