# Databricks notebook source
# COMMAND ----------
# Cell 1 — Imports

from typing import List, Optional, Sequence, Union

import math
import pandas as pd
import matplotlib.pyplot as plt

from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from pyspark.sql import types as T

# Databricks notebook source
# COMMAND ----------
# Cell 1 — Imports

from typing import List, Optional, Sequence, Union

import math
import pandas as pd
import matplotlib.pyplot as plt

from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from pyspark.sql import types as T

# Databricks notebook source
# COMMAND ----------
# Cell 3 — Subsetting Helpers

def filter_to_nbe_group(
    df: DataFrame,
    control_col: str = CONTROL_COL
) -> DataFrame:
    """
    Keep only actual NBE group rows: control == 0
    """
    return df.filter(F.col(control_col) == 0)


def subset_by_customer_ids(
    df: DataFrame,
    customer_ids: Optional[Union[DataFrame, Sequence]],
    customer_id_col: str = CUSTOMER_ID_COL
) -> DataFrame:
    """
    Subset by customer IDs.

    Supported inputs:
    - Spark DataFrame containing customer_id_col
    - Python list / tuple / set of customer IDs

    If customer_ids is None, returns df unchanged.
    """
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
    customer_id_col: str = CUSTOMER_ID_COL,
    control_col: str = CONTROL_COL
) -> DataFrame:
    """
    Keep only:
    - customer_id
    - control
    - selected product columns

    If product_cols is None, returns df unchanged.
    """
    if product_cols is None:
        return df

    keep_cols = [customer_id_col, control_col] + product_cols
    keep_cols = [c for c in keep_cols if c in df.columns]
    return df.select(*keep_cols)


def apply_subsetting(
    df: DataFrame,
    customer_ids: Optional[Union[DataFrame, Sequence]] = None,
    product_cols: Optional[List[str]] = None,
    customer_id_col: str = CUSTOMER_ID_COL,
    control_col: str = CONTROL_COL,
    filter_control_zero: bool = True
) -> DataFrame:
    """
    Combined helper:
    1) optionally filter to control == 0
    2) optionally subset customers
    3) optionally subset products
    """
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
    customer_id_col: str = CUSTOMER_ID_COL,
    control_col: str = CONTROL_COL
) -> List[str]:
    """
    Infer product columns as all columns except customer_id and control.
    """
    return [c for c in df.columns if c not in {customer_id_col, control_col}]


# Databricks notebook source
# COMMAND ----------
# Cell 4 — Core Metric Functions

def _safe_product_cols(
    df: DataFrame,
    product_cols: Optional[List[str]],
    customer_id_col: str = CUSTOMER_ID_COL,
    control_col: str = CONTROL_COL
) -> List[str]:
    """
    If product_cols is None, infer them from the dataframe.
    """
    if product_cols is None:
        return get_product_cols_from_df(df, customer_id_col, control_col)
    return product_cols


def _filled_numeric_col(col_name: str, null_fill_value: float = NULL_FILL_VALUE):
    """
    Fill nulls and cast to double.
    """
    return F.coalesce(F.col(col_name).cast("double"), F.lit(float(null_fill_value)))


def add_sorted_product_array(
    df: DataFrame,
    product_cols: Optional[List[str]] = None,
    output_col: str = "ranked_products",
    customer_id_col: str = CUSTOMER_ID_COL,
    control_col: str = CONTROL_COL,
    null_fill_value: float = NULL_FILL_VALUE
) -> DataFrame:
    """
    Add a column containing all product names sorted by expected value descending.
    Ties are broken deterministically by product name.

    Output:
    customer_id | ranked_products (array<string>)
    """
    product_cols = _safe_product_cols(df, product_cols, customer_id_col, control_col)

    # Sort ascending by:
    # 1) negative score  -> equivalent to score descending
    # 2) product name    -> deterministic tie break
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
    customer_id_col: str = CUSTOMER_ID_COL,
    control_col: str = CONTROL_COL,
    null_fill_value: float = NULL_FILL_VALUE
) -> DataFrame:
    """
    Add top-K product names per customer.

    Output:
    customer_id | topk_products (array<string>)
    """
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


def compute_total_expected_value(
    df: DataFrame,
    product_cols: Optional[List[str]] = None,
    output_col: str = "EV_total",
    customer_id_col: str = CUSTOMER_ID_COL,
    control_col: str = CONTROL_COL,
    null_fill_value: float = NULL_FILL_VALUE
) -> DataFrame:
    """
    Compute total expected value across all selected products for each customer.

    Output:
    customer_id | EV_total
    """
    product_cols = _safe_product_cols(df, product_cols, customer_id_col, control_col)

    total_expr = None
    for c in product_cols:
        term = _filled_numeric_col(c, null_fill_value)
        total_expr = term if total_expr is None else (total_expr + term)

    return df.select(
        customer_id_col,
        total_expr.alias(output_col)
    )


def compute_topk_expected_value(
    df: DataFrame,
    k: int,
    product_cols: Optional[List[str]] = None,
    output_col: str = "EV_topK",
    customer_id_col: str = CUSTOMER_ID_COL,
    control_col: str = CONTROL_COL,
    null_fill_value: float = NULL_FILL_VALUE
) -> DataFrame:
    """
    Compute sum of the top-K product expected values for each customer.

    Output:
    customer_id | EV_topK
    """
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
    """
    Spearman rank correlation for full rankings.
    Assumes both arrays contain the same product set exactly once each.
    """
    if old_order is None or new_order is None:
        return None

    n = len(old_order)
    if n == 0:
        return None
    if n == 1:
        return 1.0

    old_rank = {p: i + 1 for i, p in enumerate(old_order)}
    new_rank = {p: i + 1 for i, p in enumerate(new_order)}

    # Only compare common products
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
    Spearman-style top-N rank correlation.

    Logic:
    - Take the union of products appearing in either top-N list
    - Assign ranks 1..N for present items
    - Assign rank N+1 if a product is missing from a side
    - Compute Spearman correlation over that union

    Notes:
    - If both top-N lists are identical and N=1, returns 1.0
    - If N=1 and they differ, returns 0.0
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
    customer_id_col: str = CUSTOMER_ID_COL,
    control_col: str = CONTROL_COL,
    null_fill_value: float = NULL_FILL_VALUE
) -> DataFrame:
    """
    Compare full product rankings between legacy and new for each customer.

    Output:
    customer_id | rank_correlation
    """
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
    customer_id_col: str = CUSTOMER_ID_COL,
    control_col: str = CONTROL_COL,
    null_fill_value: float = NULL_FILL_VALUE
) -> DataFrame:
    """
    Compare top-N rankings between legacy and new for each customer.

    Output:
    customer_id | topN_rank_correlation
    """
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
    customer_id_col: str = CUSTOMER_ID_COL,
    control_col: str = CONTROL_COL,
    null_fill_value: float = NULL_FILL_VALUE
) -> DataFrame:
    """
    Overlap = |intersection(topK_old, topK_new)| / K

    Output:
    customer_id | topK_overlap
    """
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
    customer_id_col: str = CUSTOMER_ID_COL,
    control_col: str = CONTROL_COL,
    null_fill_value: float = NULL_FILL_VALUE
) -> DataFrame:
    """
    Whether the top-K set/order changed between legacy and new.

    Output:
    customer_id | topK_changed (0 or 1)
    """
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
    customer_id_col: str = CUSTOMER_ID_COL,
    control_col: str = CONTROL_COL,
    null_fill_value: float = NULL_FILL_VALUE
) -> DataFrame:
    """
    For each product, compute how often it appears in top-K for legacy and new.

    Output:
    product | legacy_topK_frequency | legacy_topK_pct | new_topK_frequency | new_topK_pct
    """
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

# Databricks notebook source
# COMMAND ----------
# Cell 5 — Plot / Summary Helpers

def summarize_metric(
    df: DataFrame,
    metric_col: str
) -> DataFrame:
    """
    Spark summary table for a single metric column.
    """
    return df.select(metric_col).summary(
        "count", "mean", "stddev", "min", "25%", "50%", "75%", "max"
    )


def compare_metric_summary(
    legacy_metric_df: DataFrame,
    new_metric_df: DataFrame,
    metric_col: str,
    customer_id_col: str = CUSTOMER_ID_COL
) -> DataFrame:
    """
    Join legacy and new metric outputs and return summary stats for both.
    """
    joined = (
        legacy_metric_df.select(customer_id_col, F.col(metric_col).alias(f"{metric_col}_legacy"))
        .join(
            new_metric_df.select(customer_id_col, F.col(metric_col).alias(f"{metric_col}_new")),
            on=customer_id_col,
            how="inner"
        )
    )

    return joined.summary(
        "count", "mean", "stddev", "min", "25%", "50%", "75%", "max"
    )


def metric_difference_table(
    legacy_metric_df: DataFrame,
    new_metric_df: DataFrame,
    metric_col: str,
    customer_id_col: str = CUSTOMER_ID_COL,
    diff_col: str = "metric_diff"
) -> DataFrame:
    """
    Return a customer-level joined table with old, new, and difference.
    """
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


def to_pandas_sample(
    df: DataFrame,
    max_rows: int = PANDAS_SAMPLE_MAX_ROWS
) -> pd.DataFrame:
    """
    Convert a Spark DataFrame to pandas, optionally limiting rows first.
    Useful for plotting.
    """
    return df.limit(max_rows).toPandas()


def plot_single_distribution(
    df: DataFrame,
    metric_col: str,
    title: Optional[str] = None,
    bins: int = 50,
    max_rows: int = PANDAS_SAMPLE_MAX_ROWS
):
    """
    Plot one metric distribution.
    """
    pdf = to_pandas_sample(df.select(metric_col), max_rows=max_rows)

    plt.figure(figsize=(8, 5))
    plt.hist(pdf[metric_col].dropna(), bins=bins)
    plt.title(title or metric_col)
    plt.xlabel(metric_col)
    plt.ylabel("Count")
    plt.show()


def plot_metric_comparison(
    legacy_metric_df: DataFrame,
    new_metric_df: DataFrame,
    metric_col: str,
    customer_id_col: str = CUSTOMER_ID_COL,
    bins: int = 50,
    max_rows: int = PANDAS_SAMPLE_MAX_ROWS,
    title: Optional[str] = None
):
    """
    Plot legacy vs new metric distributions on the same chart.
    """
    legacy_pdf = to_pandas_sample(
        legacy_metric_df.select(customer_id_col, metric_col),
        max_rows=max_rows
    )
    new_pdf = to_pandas_sample(
        new_metric_df.select(customer_id_col, metric_col),
        max_rows=max_rows
    )

    plt.figure(figsize=(8, 5))
    plt.hist(legacy_pdf[metric_col].dropna(), bins=bins, alpha=0.5, label="legacy")
    plt.hist(new_pdf[metric_col].dropna(), bins=bins, alpha=0.5, label="new")
    plt.title(title or f"{metric_col}: legacy vs new")
    plt.xlabel(metric_col)
    plt.ylabel("Count")
    plt.legend()
    plt.show()


def plot_product_distribution_shift(
    product_shift_df: DataFrame,
    top_n_products: int = 30,
    order_by_abs_diff: bool = True
):
    """
    Plot product top-K appearance frequencies for legacy vs new.
    """
    df_plot = product_shift_df

    if order_by_abs_diff:
        df_plot = df_plot.withColumn(
            "abs_diff",
            F.abs(F.col("new_topK_pct") - F.col("legacy_topK_pct"))
        ).orderBy(F.desc("abs_diff"))
    else:
        df_plot = df_plot.orderBy("product")

    pdf = df_plot.limit(top_n_products).toPandas()

    x = range(len(pdf))
    width = 0.4

    plt.figure(figsize=(14, 6))
    plt.bar([i - width / 2 for i in x], pdf["legacy_topK_pct"], width=width, label="legacy")
    plt.bar([i + width / 2 for i in x], pdf["new_topK_pct"], width=width, label="new")
    plt.xticks(list(x), pdf["product"], rotation=90)
    plt.ylabel("Top-K frequency %")
    plt.title("Product distribution shift in Top-K")
    plt.legend()
    plt.tight_layout()
    plt.show()

# Databricks notebook source
# COMMAND ----------
# Cell 6 — Example Analysis Flow: Prepare data

# Assumptions:
# - legacy_df and new_df already exist
# - both are built for the SAME month
# - same customer rows
# - same product columns
# - only changed products should differ if reconstruction is correct

# Example optional inputs:
# customer_subset_df = spark.table("my_customer_subset_table").select(CUSTOMER_ID_COL)
# customer_id_list = ["123", "456", "789"]
# selected_product_cols = ["product_1", "product_7", "product_9"]

customer_subset = None          # e.g. customer_subset_df or customer_id_list
selected_product_cols = None    # e.g. ["product_1", "product_7", "product_9"]

legacy_base = apply_subsetting(
    legacy_df,
    customer_ids=customer_subset,
    product_cols=selected_product_cols,
    customer_id_col=CUSTOMER_ID_COL,
    control_col=CONTROL_COL,
    filter_control_zero=True
)

new_base = apply_subsetting(
    new_df,
    customer_ids=customer_subset,
    product_cols=selected_product_cols,
    customer_id_col=CUSTOMER_ID_COL,
    control_col=CONTROL_COL,
    filter_control_zero=True
)

analysis_product_cols = get_product_cols_from_df(
    legacy_base,
    customer_id_col=CUSTOMER_ID_COL,
    control_col=CONTROL_COL
)

print(f"Number of selected product columns: {len(analysis_product_cols)}")
print(f"Legacy rows: {legacy_base.count():,}")
print(f"New rows:    {new_base.count():,}")

display(legacy_base.limit(5))
display(new_base.limit(5))

# Databricks notebook source
# COMMAND ----------
# Cell 7 — Total Expected Value

legacy_ev_total = compute_total_expected_value(
    legacy_base,
    product_cols=analysis_product_cols,
    output_col="EV_total",
    customer_id_col=CUSTOMER_ID_COL,
    control_col=CONTROL_COL
)

new_ev_total = compute_total_expected_value(
    new_base,
    product_cols=analysis_product_cols,
    output_col="EV_total",
    customer_id_col=CUSTOMER_ID_COL,
    control_col=CONTROL_COL
)

display(legacy_ev_total.limit(10))
display(new_ev_total.limit(10))

display(compare_metric_summary(
    legacy_ev_total,
    new_ev_total,
    metric_col="EV_total",
    customer_id_col=CUSTOMER_ID_COL
))

ev_total_diff = metric_difference_table(
    legacy_ev_total,
    new_ev_total,
    metric_col="EV_total",
    customer_id_col=CUSTOMER_ID_COL,
    diff_col="EV_total_diff"
)

display(ev_total_diff.limit(10))

plot_metric_comparison(
    legacy_ev_total,
    new_ev_total,
    metric_col="EV_total",
    customer_id_col=CUSTOMER_ID_COL,
    bins=50,
    title="Total Expected Value: legacy vs new"
)

# Databricks notebook source
# COMMAND ----------
# Cell 8 — Expected Value of Top-K Products

topk_ev_results = {}

for k in TOP_K_VALUES:
    legacy_topk_ev = compute_topk_expected_value(
        legacy_base,
        k=k,
        product_cols=analysis_product_cols,
        output_col="EV_topK",
        customer_id_col=CUSTOMER_ID_COL,
        control_col=CONTROL_COL
    )

    new_topk_ev = compute_topk_expected_value(
        new_base,
        k=k,
        product_cols=analysis_product_cols,
        output_col="EV_topK",
        customer_id_col=CUSTOMER_ID_COL,
        control_col=CONTROL_COL
    )

    topk_ev_results[k] = {
        "legacy": legacy_topk_ev,
        "new": new_topk_ev
    }

    print(f"Top-{k} expected value summary")
    display(compare_metric_summary(
        legacy_topk_ev,
        new_topk_ev,
        metric_col="EV_topK",
        customer_id_col=CUSTOMER_ID_COL
    ))

    plot_metric_comparison(
        legacy_topk_ev,
        new_topk_ev,
        metric_col="EV_topK",
        customer_id_col=CUSTOMER_ID_COL,
        bins=50,
        title=f"Top-{k} Expected Value: legacy vs new"
    )

# Databricks notebook source
# COMMAND ----------
# Cell 9 — Full Rank Correlation

full_rank_corr = compute_full_rank_correlation(
    legacy_base,
    new_base,
    product_cols=analysis_product_cols,
    output_col="rank_correlation",
    customer_id_col=CUSTOMER_ID_COL,
    control_col=CONTROL_COL
)

display(full_rank_corr.limit(10))
display(summarize_metric(full_rank_corr, "rank_correlation"))

plot_single_distribution(
    full_rank_corr,
    metric_col="rank_correlation",
    title="Full Rank Correlation"
)

# Databricks notebook source
# COMMAND ----------
# Cell 10 — Top-N Rank Correlation

topn_rank_corr_results = {}

for n in TOP_N_VALUES:
    topn_rank_corr = compute_topn_rank_correlation(
        legacy_base,
        new_base,
        n=n,
        product_cols=analysis_product_cols,
        output_col="topN_rank_correlation",
        customer_id_col=CUSTOMER_ID_COL,
        control_col=CONTROL_COL
    )

    topn_rank_corr_results[n] = topn_rank_corr

    print(f"Top-{n} rank correlation summary")
    display(summarize_metric(topn_rank_corr, "topN_rank_correlation"))

    plot_single_distribution(
        topn_rank_corr,
        metric_col="topN_rank_correlation",
        title=f"Top-{n} Rank Correlation"
    )

# Databricks notebook source
# COMMAND ----------
# Cell 11 — Top-K Overlap

topk_overlap_results = {}

for k in TOP_K_VALUES:
    topk_overlap = compute_topk_overlap(
        legacy_base,
        new_base,
        k=k,
        product_cols=analysis_product_cols,
        output_col="topK_overlap",
        customer_id_col=CUSTOMER_ID_COL,
        control_col=CONTROL_COL
    )

    topk_overlap_results[k] = topk_overlap

    print(f"Top-{k} overlap summary")
    display(summarize_metric(topk_overlap, "topK_overlap"))

    plot_single_distribution(
        topk_overlap,
        metric_col="topK_overlap",
        title=f"Top-{k} Overlap"
    )

# Databricks notebook source
# COMMAND ----------
# Cell 12 — Top-K Swap Rate

topk_swap_results = {}

for k in TOP_K_VALUES:
    topk_swap = compute_topk_swap_rate(
        legacy_base,
        new_base,
        k=k,
        product_cols=analysis_product_cols,
        output_col="topK_changed",
        customer_id_col=CUSTOMER_ID_COL,
        control_col=CONTROL_COL
    )

    topk_swap_results[k] = topk_swap

    swap_summary = topk_swap.agg(
        F.count("*").alias("n_customers"),
        F.avg(F.col("topK_changed").cast("double")).alias("swap_rate")
    )

    print(f"Top-{k} swap rate")
    display(swap_summary)
    display(topk_swap.limit(10))

# Databricks notebook source
# COMMAND ----------
# Cell 13 — Product Distribution Shift

product_shift_results = {}

for k in TOP_K_VALUES:
    product_shift = compute_product_distribution_shift(
        legacy_base,
        new_base,
        k=k,
        product_cols=analysis_product_cols,
        customer_id_col=CUSTOMER_ID_COL,
        control_col=CONTROL_COL
    )

    product_shift = (
        product_shift
        .withColumn("pct_diff", F.col("new_topK_pct") - F.col("legacy_topK_pct"))
        .withColumn("freq_diff", F.col("new_topK_frequency") - F.col("legacy_topK_frequency"))
    )

    product_shift_results[k] = product_shift

    print(f"Top-{k} product distribution shift")
    display(product_shift.orderBy(F.desc(F.abs(F.col("pct_diff")))))

    plot_product_distribution_shift(
        product_shift,
        top_n_products=30,
        order_by_abs_diff=True
    )

# Databricks notebook source
# COMMAND ----------
# Cell 14 — Optional sanity checks focused on unchanged vs changed products

# Example:
# changed_products = ["product_2", "product_8", "product_14", "product_17", "product_22", "product_25", "product_31", "product_36"]
# unchanged_products = [c for c in analysis_product_cols if c not in changed_products]

changed_products = []
unchanged_products = [c for c in analysis_product_cols if c not in changed_products]

print(f"Changed products:   {len(changed_products)}")
print(f"Unchanged products: {len(unchanged_products)}")

if unchanged_products:
    legacy_unchanged_ev = compute_total_expected_value(
        legacy_base,
        product_cols=unchanged_products,
        output_col="EV_total_unchanged",
        customer_id_col=CUSTOMER_ID_COL,
        control_col=CONTROL_COL
    )

    new_unchanged_ev = compute_total_expected_value(
        new_base,
        product_cols=unchanged_products,
        output_col="EV_total_unchanged",
        customer_id_col=CUSTOMER_ID_COL,
        control_col=CONTROL_COL
    )

    unchanged_diff = metric_difference_table(
        legacy_unchanged_ev,
        new_unchanged_ev,
        metric_col="EV_total_unchanged",
        customer_id_col=CUSTOMER_ID_COL,
        diff_col="EV_total_unchanged_diff"
    )

    print("Unchanged-products EV diff summary")
    display(unchanged_diff.summary("count", "mean", "stddev", "min", "25%", "50%", "75%", "max"))

if changed_products:
    legacy_changed_ev = compute_total_expected_value(
        legacy_base,
        product_cols=changed_products,
        output_col="EV_total_changed",
        customer_id_col=CUSTOMER_ID_COL,
        control_col=CONTROL_COL
    )

    new_changed_ev = compute_total_expected_value(
        new_base,
        product_cols=changed_products,
        output_col="EV_total_changed",
        customer_id_col=CUSTOMER_ID_COL,
        control_col=CONTROL_COL
    )

    changed_diff = metric_difference_table(
        legacy_changed_ev,
        new_changed_ev,
        metric_col="EV_total_changed",
        customer_id_col=CUSTOMER_ID_COL,
        diff_col="EV_total_changed_diff"
    )

    print("Changed-products EV diff summary")
    display(changed_diff.summary("count", "mean", "stddev", "min", "25%", "50%", "75%", "max"))

# Databricks notebook source
# COMMAND ----------
# Cell 15 — Optional quick-access intermediate outputs

# Ranked products
legacy_ranked_products = add_sorted_product_array(
    legacy_base,
    product_cols=analysis_product_cols,
    output_col="ranked_products",
    customer_id_col=CUSTOMER_ID_COL,
    control_col=CONTROL_COL
)

new_ranked_products = add_sorted_product_array(
    new_base,
    product_cols=analysis_product_cols,
    output_col="ranked_products",
    customer_id_col=CUSTOMER_ID_COL,
    control_col=CONTROL_COL
)

display(legacy_ranked_products.limit(10))
display(new_ranked_products.limit(10))

# Example Top-3 product arrays
legacy_top3_products = add_topk_product_array(
    legacy_base,
    k=3,
    product_cols=analysis_product_cols,
    output_col="top3_products",
    customer_id_col=CUSTOMER_ID_COL,
    control_col=CONTROL_COL
)

new_top3_products = add_topk_product_array(
    new_base,
    k=3,
    product_cols=analysis_product_cols,
    output_col="top3_products",
    customer_id_col=CUSTOMER_ID_COL,
    control_col=CONTROL_COL
)

display(legacy_top3_products.limit(10))
display(new_top3_products.limit(10))

