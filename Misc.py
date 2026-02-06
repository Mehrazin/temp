from pyspark.sql import functions as F
from pyspark.sql import Window

def stratify_tree_A_correct(
    df,
    min_stratum_size: int,
    split_order=("visa_ind", "balance", "n_offers_year", "n_web_logins", "n_mobile_logins"),
    balance_bins_try=(6, 5, 4, 3, 2),
    discrete_bins_try=(6, 5, 4, 3, 2),
    max_depth=6,
    rel_err=0.001,
    visa_mode="3way",   # "3way" keeps {0,1,NULL}; "merge_null_to_0" merges null with 0
):
    """
    Correct, readable hierarchical stratification with hard min stratum size.

    Notes on args:
    - balance_bins_try: candidates for number of quantile bins when splitting on balance (continuous).
    - discrete_bins_try: candidates for number of quantile bins for count-valued features (offers/web/mobile).
      These are NOT "categorical"; they are numeric but discrete.
    """

    # ---------- helper: quantile cutpoints ----------
    def _cuts(df_nonnull, col, nb):
        probs = [i / nb for i in range(1, nb)]
        qs = df_nonnull.approxQuantile(col, probs, rel_err)
        return sorted(set([q for q in qs if q is not None]))

    # ---------- helper: bin-id expression from cuts (bin_id=-1 for NULL) ----------
    def _bin_id_expr(col, cuts):
        c = F.col(col).cast("double")
        e = F.when(c.isNull(), F.lit(-1))
        prev = None
        idx = 0
        for cut in cuts:
            if prev is None:
                e = e.when(c <= F.lit(cut), F.lit(idx))
            else:
                e = e.when((c > F.lit(prev)) & (c <= F.lit(cut)), F.lit(idx))
            prev = cut
            idx += 1
        e = e.otherwise(F.lit(idx))
        return e

    # ---------- helper: merge adjacent small bins (greedy, stable) ----------
    def _merge_adjacent(bin_counts, min_size):
        # bin_counts: list of (bin_id, count) where -1 denotes NULL
        null_cnt = 0
        nonnull = []
        for b, n in bin_counts:
            if b == -1:
                null_cnt = n
            else:
                nonnull.append((b, n))
        nonnull.sort()

        if not nonnull:
            return {-1: 0}, [null_cnt]

        groups = [{"bins": [b], "n": n} for b, n in nonnull]

        def merge(i, j):
            groups[i]["bins"] += groups[j]["bins"]
            groups[i]["n"] += groups[j]["n"]
            del groups[j]

        # merge smallest group into a neighbor until all >= min_size or only one group remains
        while len(groups) > 1 and min(g["n"] for g in groups) < min_size:
            i = min(range(len(groups)), key=lambda k: groups[k]["n"])
            if i == 0:
                merge(0, 1)
            elif i == len(groups) - 1:
                merge(i - 1, i)
            else:
                if groups[i - 1]["n"] <= groups[i + 1]["n"]:
                    merge(i - 1, i)
                else:
                    merge(i, i + 1)

        # handle NULL bin: keep separate if big; else merge into first group
        if null_cnt > 0:
            if null_cnt < min_size:
                groups[0]["bins"] = [-1] + groups[0]["bins"]
                groups[0]["n"] += null_cnt
            else:
                groups = [{"bins": [-1], "n": null_cnt}] + groups

        mapping = {}
        sizes = []
        for gid, g in enumerate(groups):
            sizes.append(g["n"])
            for b in g["bins"]:
                mapping[b] = gid
        return mapping, sizes

    # ---------- helper: visa bin expression ----------
    def _visa_bin_expr():
        if visa_mode == "merge_null_to_0":
            return F.when(F.col("visa_ind") == 1, F.lit("1")).otherwise(F.lit("0_or_null"))
        return F.when(F.col("visa_ind").isNull(), F.lit("__NULL__")) \
                .when(F.col("visa_ind") == 1, F.lit("1")).otherwise(F.lit("0"))

    work = df
    created_cols = []          # bin columns that define the leaf key
    nodes = [(None, 0)]        # (condition, depth)

    for depth in range(max_depth):
        next_nodes = []
        did_split_any = False

        for cond, _ in nodes:
            node_df = work if cond is None else work.where(cond)
            n_node = node_df.count()

            # can't split into >=2 children each >= min
            if n_node < 2 * min_stratum_size:
                continue

            best = None  # (out_col, expr, child_values, obj=(n_children, min_child))

            for feat in split_order:
                out_col = f"__A_L{depth}_{feat}_bin"

                # ---- categorical: visa ----
                if feat == "visa_ind":
                    expr = _visa_bin_expr()
                    test = node_df.withColumn(out_col, expr)
                    sizes = [r["count"] for r in test.groupBy(out_col).count().collect()]
                    if len(sizes) < 2 or min(sizes) < min_stratum_size:
                        continue
                    obj = (len(sizes), min(sizes))
                    child_vals = [r[out_col] for r in test.select(out_col).distinct().collect()]
                    if best is None or obj > best[-1]:
                        best = (out_col, expr, child_vals, obj)
                    continue

                # ---- numeric/count: quantiles + merge small bins ----
                bins_try = balance_bins_try if feat == "balance" else discrete_bins_try
                nn = node_df.where(F.col(feat).isNotNull())
                if nn.rdd.isEmpty():
                    continue

                # try nb in order, keep the first nb that yields a feasible split for this feat
                for nb in bins_try:
                    cuts = _cuts(nn, feat, nb)
                    if not cuts:
                        continue
                    bid = _bin_id_expr(feat, cuts)
                    tmp = node_df.withColumn("__bid", bid)
                    bc = [(r["__bid"], r["count"]) for r in tmp.groupBy("__bid").count().collect()]
                    mapping, sizes = _merge_adjacent(bc, min_stratum_size)

                    if len(sizes) < 2 or min(sizes) < min_stratum_size:
                        continue

                    # map bid -> group label G0,G1,...
                    gid = None
                    for old_b, new_g in mapping.items():
                        gid = F.when(bid == F.lit(int(old_b)), F.lit(int(new_g))) if gid is None else gid.when(bid == F.lit(int(old_b)), F.lit(int(new_g)))
                    gid = gid.otherwise(F.lit(-999))
                    expr = F.concat(F.lit("G"), gid.cast("string"))

                    obj = (len(sizes), min(sizes))
                    child_vals = [f"G{i}" for i in range(len(sizes))]
                    if best is None or obj > best[-1]:
                        best = (out_col, expr, child_vals, obj)

                    break  # stability + simplicity

            if best is None:
                continue

            out_col, expr, child_vals, obj = best
            created_cols.append(out_col)
            did_split_any = True

            # apply bin column only inside this node
            inside = cond if cond is not None else F.lit(True)
            if out_col not in work.columns:
                work = work.withColumn(out_col, F.when(inside, expr).otherwise(F.lit(None)))
            else:
                work = work.withColumn(out_col, F.when(inside, expr).otherwise(F.col(out_col)))

            # create children nodes
            for v in child_vals:
                child_cond = (F.col(out_col) == F.lit(v)) if cond is None else (cond & (F.col(out_col) == F.lit(v)))
                next_nodes.append((child_cond, depth + 1))

        if not did_split_any:
            break
        nodes = next_nodes

    # ---- build non-null key (fill __NA__ for levels not used on that branch) ----
    created_cols = sorted(set(created_cols))
    if not created_cols:
        work = work.withColumn("stratum_A_key", F.lit("ROOT"))
    else:
        filled = [F.coalesce(F.col(c), F.lit("__NA__")) for c in created_cols]
        work = work.withColumn(
            "stratum_A_key",
            F.concat_ws(" | ", *[F.concat(F.lit(c + "="), x.cast("string")) for c, x in zip(created_cols, filled)])
        )

    # compact IDs (never null)
    w = Window.orderBy("stratum_A_key")
    work = work.withColumn(
        "stratum_A",
        F.concat(F.lit("A_"), F.lpad(F.dense_rank().over(w).cast("string"), 3, "0"))
    )

    return work


def stratify_C_optbinning(
    df,
    min_stratum_size: int,
    numeric_cols=("balance", "n_offers_year", "n_web_logins", "n_mobile_logins"),
    visa_col="visa_ind",
    max_bins=6,
    seed=7,
    sample_for_fit=2_000_00,  # fit bins on a sample; apply to full DF
):
    """
    Approach C: optbinning for cutpoints + cross-product strata.

    Notes:
    - optbinning runs in Python (driver), so we fit on a sampled pandas df.
    - Then we apply cutpoints back to Spark via SQL CASE expression.
    - This is very practical in Databricks.
    """

    # lazy import so your pipeline still runs without the package unless you call this
    from optbinning import OptimalBinning

    # 1) sample to pandas for fitting cutpoints
    pdf = df.select([visa_col] + list(numeric_cols)).sample(False, min(1.0, sample_for_fit / max(df.count(), 1)), seed).toPandas()

    cutpoints = {}
    for c in numeric_cols:
        x = pdf[c]
        # optbinning expects target y; we can use a dummy y (or a proxy). For unsupervised binning,
        # a practical workaround is to use quantile-ish by passing a random y.
        # If you DO have a target (response), pass it instead for supervised bins.
        y = (pdf[visa_col].fillna(-1).astype("int") == 1).astype("int")  # cheap proxy
        opt = OptimalBinning(name=c, dtype="numerical", max_n_bins=max_bins)
        opt.fit(x, y)
        cps = opt.splits  # cutpoints
        cutpoints[c] = [float(v) for v in cps] if cps is not None else []

    # 2) apply cutpoints to Spark as bin labels (NULL separate)
    out = df

    def apply_cut(col, cuts, out_col):
        expr = F.when(F.col(col).isNull(), F.lit("__NULL__"))
        prev = None
        for i, cut in enumerate(cuts):
            if prev is None:
                expr = expr.when(F.col(col) <= F.lit(cut), F.lit(f"B{i}:(-inf,{cut}]"))
            else:
                expr = expr.when((F.col(col) > F.lit(prev)) & (F.col(col) <= F.lit(cut)), F.lit(f"B{i}:({prev},{cut}]"))
            prev = cut
        expr = expr.otherwise(F.lit(f"B{len(cuts)}:({prev},inf)")) if cuts else expr.otherwise(F.lit("B0:[NON_NULL]"))
        return out.withColumn(out_col, expr)

    for c in numeric_cols:
        out = apply_cut(c, cutpoints[c], f"__C_{c}_bin")

    out = out.withColumn(
        "__C_visa_bin",
        F.when(F.col(visa_col).isNull(), F.lit("__NULL__"))
         .when(F.col(visa_col) == 1, F.lit("1"))
         .otherwise(F.lit("0"))
    )

    key_cols = ["__C_visa_bin"] + [f"__C_{c}_bin" for c in numeric_cols]
    out = out.withColumn(
        "stratum_C_key",
        F.concat_ws(" | ", *[F.concat(F.lit(k + "="), F.col(k)) for k in key_cols])
    )

    # 3) compact IDs
    w = Window.orderBy("stratum_C_key")
    out = out.withColumn("stratum_C", F.concat(F.lit("C_"), F.lpad(F.dense_rank().over(w).cast("string"), 4, "0")))

    # (Optional) You can now check small strata and reduce max_bins if needed, but I’m keeping this function focused.
    return out


def stratify_D_kmeans(
    df,
    min_stratum_size: int,
    k_start=50,
    k_min=2,
    seed=7,
    feature_cols=("visa_ind", "balance", "n_offers_year", "n_web_logins", "n_mobile_logins"),
):
    """
    Approach D: KMeans clusters as strata (Spark-native, scalable).

    Pros: very short, works at 16M scale.
    Cons: less interpretable than bin-based; but you can still summarize each cluster.
    """

    from pyspark.ml.feature import VectorAssembler
    from pyspark.ml.clustering import KMeans
    from pyspark.ml import Pipeline

    # basic null handling: keep null indicators + fill numeric
    out = df
    for c in feature_cols:
        out = out.withColumn(f"__D_{c}_isnull", F.col(c).isNull().cast("int"))

    # fill numeric nulls with 0 (or median if you want), and visa null with -1
    out = out.withColumn("visa_ind_f", F.when(F.col("visa_ind").isNull(), F.lit(-1.0)).otherwise(F.col("visa_ind").cast("double")))
    for c in ("balance", "n_offers_year", "n_web_logins", "n_mobile_logins"):
        if c in feature_cols:
            out = out.withColumn(f"{c}_f", F.coalesce(F.col(c).cast("double"), F.lit(0.0)))

    vec_cols = []
    for c in feature_cols:
        if c == "visa_ind":
            vec_cols.append("visa_ind_f")
        elif c in ("balance", "n_offers_year", "n_web_logins", "n_mobile_logins"):
            vec_cols.append(f"{c}_f")
        vec_cols.append(f"__D_{c}_isnull")

    assembler = VectorAssembler(inputCols=vec_cols, outputCol="__D_features")

    k = k_start
    best = None

    while k >= k_min:
        km = KMeans(k=k, seed=seed, featuresCol="__D_features", predictionCol="__D_cluster")
        pipe = Pipeline(stages=[assembler, km])
        model = pipe.fit(out)
        pred = model.transform(out)

        sizes = pred.groupBy("__D_cluster").count()
        min_n = sizes.agg(F.min("count").alias("min_n")).collect()[0]["min_n"]

        if min_n >= min_stratum_size:
            best = pred
            break

        k = max(k_min, int(k * 0.7))  # shrink k fast

    if best is None:
        # if nothing satisfies min size, fall back to k_min (still non-null strata)
        km = KMeans(k=k_min, seed=seed, featuresCol="__D_features", predictionCol="__D_cluster")
        best = Pipeline(stages=[assembler, km]).fit(out).transform(out)

    best = best.withColumn(
        "stratum_D",
        F.concat(F.lit("D_"), F.lpad(F.col("__D_cluster").cast("string"), 4, "0"))
    )
    return best.drop("__D_features")


from pyspark.sql import functions as F

def describe_strata(
    df,
    stratum_col: str,
    bin_cols: list,
    max_values_per_bin: int = 5,
    sort_by_size: bool = True,
):
    """
    Summarize and explain strata in a generic way.

    Parameters
    ----------
    df : Spark DataFrame
        Final dataframe after stratification.
    stratum_col : str
        Column name of the stratum id (e.g., 'stratum_A').
    bin_cols : list[str]
        Columns that define the stratum (bin / cluster columns).
        Example:
          - Approach A: ['__A_L0_visa_ind_bin', '__A_L1_balance_bin', ...]
          - Approach B: ['visa_bin', 'balance_bin', 'offers_bin', ...]
          - Approach C: ['__C_visa_bin', '__C_balance_bin', ...]
          - Approach D: ['__D_cluster']
    max_values_per_bin : int
        Max number of distinct values to show per bin column.
    sort_by_size : bool
        Whether to sort strata by descending size.

    Returns
    -------
    Spark DataFrame
        One row per stratum with:
        - stratum id
        - count
        - list of values observed for each bin column
    """

    agg_exprs = []

    for c in bin_cols:
        agg_exprs.append(
            F.slice(
                F.sort_array(F.collect_set(F.col(c))),
                1,
                max_values_per_bin
            ).alias(c)
        )

    out = (
        df
        .groupBy(stratum_col)
        .agg(
            F.count("*").alias("n"),
            *agg_exprs
        )
    )

    if sort_by_size:
        out = out.orderBy(F.desc("n"))

    return out


bin_cols_A = [c for c in dfA.columns if c.startswith("__A_L") and c.endswith("_bin")]

describe_strata(
    dfA,
    stratum_col="stratum_A",
    bin_cols=bin_cols_A
)

bin_cols_B = [
    "visa_bin",
    "balance_bin",
    "offers_bin",
    "web_bin",
    "mobile_bin",
]

describe_strata(
    dfB,
    stratum_col="stratum_B",
    bin_cols=bin_cols_B
)

bin_cols_C = [c for c in dfC.columns if c.startswith("__C_") and c.endswith("_bin")]

describe_strata(
    dfC,
    stratum_col="stratum_C",
    bin_cols=bin_cols_C
)

describe_strata(
    dfD,
    stratum_col="stratum_D",
    bin_cols=["__D_cluster"]
)


def add_stratum_description_text(df, stratum_col, bin_cols, out_col="stratum_desc"):
    parts = []
    for c in bin_cols:
        parts.append(
            F.concat(
                F.lit(c.replace("_bin", "").replace("__", "") + "="),
                F.col(c).cast("string")
            )
        )
    return df.withColumn(out_col, F.concat_ws(" | ", *parts))
