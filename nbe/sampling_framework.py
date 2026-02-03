import numpy as np
import pandas as pd
from scipy import linalg, stats
from pyspark.sql import functions as F
from pyspark.sql.window import Window
from pyspark.ml.feature import QuantileDiscretizer, StandardScaler, VectorAssembler

class CubeSampler:
    """Local kernel for Cube Sampling Flight Phase."""
    def __init__(self, pi, X, eps=1e-9):
        self.pi = np.array(pi, dtype=float)
        self.X = np.array(X, dtype=float)
        self.N, self.p = self.X.shape
        # Construct A matrix: a_k = x_k / pi_k
        with np.errstate(divide='ignore', invalid='ignore'):
            self.A = (self.X / self.pi[:, None]).T
        self.eps = eps

    def fast_flight(self):
        """Optimized Fast Flight Algorithm (Chauvet and Tillé, 2006)."""
        undecided = np.where((self.pi > self.eps) & (self.pi < 1 - self.eps)).tolist()
        
        while len(undecided) > self.p:
            # Pick first p+1 undecided units
            active_idx = undecided[:self.p + 1]
            B = self.A[:, active_idx]
            
            # Find a vector in the null space of B
            u = linalg.null_space(B)
            if u.shape[1] == 0: # Numerical stability fallback
                break
            u = u[:, 0]
            
            # Calculate step lengths to hit cube boundaries
            l1_candidates = []
            l2_candidates = []
            for i, idx in enumerate(active_idx):
                if u[i] > self.eps:
                    l1_candidates.append((1 - self.pi[idx]) / u[i])
                    l2_candidates.append(self.pi[idx] / u[i])
                elif u[i] < -self.eps:
                    l1_candidates.append(-self.pi[idx] / u[i])
                    l2_candidates.append((self.pi[idx] - 1) / u[i])
            
            l1 = min(l1_candidates) if l1_candidates else 0
            l2 = min(l2_candidates) if l2_candidates else 0
            
            # Update probabilities (Martingale property)
            q1 = l2 / (l1 + l2) if (l1 + l2) > 0 else 0.5
            if np.random.rand() < q1:
                self.pi[active_idx] += l1 * u
            else:
                self.pi[active_idx] -= l2 * u
                
            # Filter out decided units
            undecided = [idx for idx in undecided if self.pi[idx] > self.eps and self.pi[idx] < 1 - self.eps]
            
        return self.pi



class SamplingFramework:
    def __init__(self, spark_session):
        self.spark = spark_session

    # --- 1. EDA & PREPROCESSING ---
    def get_summary_stats(self, df, numeric_cols):
        """Generates summary statistics for numeric columns including null percentages."""
        total_count = df.count()
        # Use summary() for descriptive statistics
        summary = df.select(numeric_cols).summary("mean", "stddev", "min", "25%", "50%", "75%", "max")
        
        # Calculate null counts efficiently
        null_exprs = [F.count(F.when(F.col(c).isNull(), c)).alias(c) for c in numeric_cols]
        null_counts_row = df.select(null_exprs).collect()[0]
        null_counts = null_counts_row.asDict()
        null_pcts = {k: (v / total_count) if v is not None else 0.0 for k, v in null_counts.items()}
        
        return summary, null_pcts

    def preprocess_data(self, df, config):
        """
        config: dict with 'categorical_cols', 'continuous_cols', 'id_col'
        Treats NULLs as "UNKNOWN" and caps outliers at 99th percentile.
        """
        processed_df = df
        
        # Handle Categorical/Discrete Nulls (Explicit Unknown)
        for col in config['categorical_cols'] + config.get('discrete_numeric_cols', []):
            processed_df = processed_df.withColumn(
                col, F.coalesce(F.col(col).cast("string"), F.lit("UNKNOWN"))
            )
            
        # Handle Continuous Outliers (Winsorization)
        for col in config['continuous_cols']:
            quantiles = df.stat.approxQuantile(col, [0.01, 0.99], 0.001)
            lower, upper = quantiles[0], quantiles[1]
            
            processed_df = processed_df.withColumn(
                col, 
                F.when(F.col(col) < lower, lower)
               .when(F.col(col) > upper, upper)
               .otherwise(F.col(col))
            )
            
        return processed_df

    # --- 2. BINNING & STRATA ---
    def apply_quantile_binning(self, df, input_col, n_buckets):
        """Applies QuantileDiscretizer with handleInvalid='keep' for NULL preservation."""
        if input_col not in df.columns:
            raise ValueError(f"Column {input_col} not found in DataFrame")
        if n_buckets < 2:
            raise ValueError(f"n_buckets must be >= 2, got {n_buckets}")
        
        qd = QuantileDiscretizer(
            numBuckets=n_buckets, 
            inputCol=input_col, 
            outputCol=f"{input_col}_bin",
            handleInvalid="keep"  # Crucial: Puts NULLs/NaNs into a separate bucket
        )
        model = qd.fit(df)
        return model.transform(df)

    def merge_sparse_strata(self, df, strata_col, target_floor=30):
        """Identifies small strata and merges them into an 'OTHER' category."""
        strata_counts = df.groupBy(strata_col).count()
        
        # Identify small strata IDs
        small_strata = strata_counts.filter(F.col("count") < target_floor).select(strata_col)
        small_strata_list = [row[strata_col] for row in small_strata.collect()]
        
        print(f"Merging {len(small_strata_list)} small strata into 'STRATA_MERGED'")
        
        return df.withColumn(
            f"{strata_col}_final",
            F.when(F.col(strata_col).isin(small_strata_list), "STRATA_MERGED")
           .otherwise(F.col(strata_col))
        )

    # --- 3. SAMPLING ---
    def random_sample(self, df, id_col, test_fraction=0.9, seed=42):
        """Simple random 90/10 split."""
        if id_col not in df.columns:
            raise ValueError(f"ID column {id_col} not found in DataFrame")
        if not (0 < test_fraction < 1):
            raise ValueError(f"test_fraction must be between 0 and 1, got {test_fraction}")
        if df.count() == 0:
            raise ValueError("Cannot sample from empty DataFrame")
        
        test_df = df.sample(False, test_fraction, seed=seed)
        control_df = df.join(test_df, on=id_col, how="left_anti")
        return test_df, control_df

    def stratified_sample(self, df, id_col, strata_col, test_fraction=0.9, seed=42):
        """Stratified sampling using sampleBy."""
        if id_col not in df.columns:
            raise ValueError(f"ID column {id_col} not found in DataFrame")
        if strata_col not in df.columns:
            raise ValueError(f"Strata column {strata_col} not found in DataFrame")
        if not (0 < test_fraction < 1):
            raise ValueError(f"test_fraction must be between 0 and 1, got {test_fraction}")
        if df.count() == 0:
            raise ValueError("Cannot sample from empty DataFrame")
        
        unique_strata_rows = df.select(strata_col).distinct().collect()
        unique_strata = [row[strata_col] for row in unique_strata_rows]
        
        if len(unique_strata) == 0:
            raise ValueError(f"No unique strata found in column {strata_col}")
        
        fractions = {s: test_fraction for s in unique_strata}
        
        test_df = df.sampleBy(strata_col, fractions, seed=seed)
        control_df = df.join(test_df, on=id_col, how="left_anti")
        return test_df, control_df

    # --- 4. VALIDATION ---
    def calculate_smd(self, test_df, control_df, numeric_cols):
        """Computes Standardized Mean Difference (Invariant to Sample Size)."""
        res = []
        for col in numeric_cols:
            # Validate column exists
            if col not in test_df.columns or col not in control_df.columns:
                res.append({"feature": col, "smd": None, "test_mean": None, "ctrl_mean": None, "error": f"Column {col} not found"})
                continue
            
            try:
                t_stats_row = test_df.select(F.mean(col).alias("mean"), F.stddev(col).alias("stddev")).collect()
                c_stats_row = control_df.select(F.mean(col).alias("mean"), F.stddev(col).alias("stddev")).collect()
                
                mu_t = t_stats_row[0]["mean"] if t_stats_row and t_stats_row[0]["mean"] is not None else 0
                sd_t = t_stats_row[0]["stddev"] if t_stats_row and t_stats_row[0]["stddev"] is not None else 0
                mu_c = c_stats_row[0]["mean"] if c_stats_row and c_stats_row[0]["mean"] is not None else 0
                sd_c = c_stats_row[0]["stddev"] if c_stats_row and c_stats_row[0]["stddev"] is not None else 0
                
                pooled_sd = np.sqrt((sd_t**2 + sd_c**2) / 2) if (sd_t is not None and sd_c is not None and not np.isnan(sd_t) and not np.isnan(sd_c)) else 0
                smd = abs(mu_t - mu_c) / pooled_sd if pooled_sd != 0 and not np.isnan(pooled_sd) else 0
                res.append({"feature": col, "smd": smd, "test_mean": mu_t, "ctrl_mean": mu_c})
            except Exception as e:
                res.append({"feature": col, "smd": None, "test_mean": None, "ctrl_mean": None, "error": str(e)})
            
        return pd.DataFrame(res)

    def calculate_psi(self, test_df, control_df, binned_col):
        """Calculates Population Stability Index for a binned/categorical column."""
        t_counts = test_df.groupBy(binned_col).count().withColumnRenamed("count", "t_n")
        c_counts = control_df.groupBy(binned_col).count().withColumnRenamed("count", "c_n")
        
        t_total = test_df.count()
        c_total = control_df.count()
        
        joined = t_counts.join(c_counts, on=binned_col, how="full").fillna(0)
        
        # PSI calculation with epsilon to prevent division by zero
        psi_df = joined.withColumn("t_pct", F.col("t_n") / F.lit(t_total) + 1e-6) \
                     .withColumn("c_pct", F.col("c_n") / F.lit(c_total) + 1e-6) \
                     .withColumn("psi", (F.col("t_pct") - F.col("c_pct")) * F.log(F.col("t_pct") / F.col("c_pct")))
        
        psi_result = psi_df.select(F.sum("psi").alias("psi_sum")).collect()[0]["psi_sum"]
        return psi_result if psi_result is not None else 0.0

    def calculate_ks_test(self, test_df, control_df, numeric_col, sample_fraction=0.2, max_sample_size=10000):
        """
        Calculates Kolmogorov-Smirnov test statistic and p-value for a continuous variable.
        Uses sampling for large datasets to avoid memory issues.
        """
        # Validate column exists
        if numeric_col not in test_df.columns or numeric_col not in control_df.columns:
            return {"statistic": None, "pvalue": None, "message": f"Column {numeric_col} not found"}
        
        # Get counts and handle empty DataFrames
        test_count = test_df.count()
        ctrl_count = control_df.count()
        
        if test_count == 0 or ctrl_count == 0:
            return {"statistic": None, "pvalue": None, "message": "Empty DataFrame(s)"}
        
        # Sample data for KS test (works on pandas)
        n_test = min(int(test_count * sample_fraction), max_sample_size)
        n_ctrl = min(int(ctrl_count * sample_fraction), max_sample_size)
        
        # Calculate sample fraction safely
        test_frac = min(n_test / test_count, 1.0) if test_count > 0 else 0.0
        ctrl_frac = min(n_ctrl / ctrl_count, 1.0) if ctrl_count > 0 else 0.0
        
        if test_frac <= 0 or ctrl_frac <= 0:
            return {"statistic": None, "pvalue": None, "message": "Invalid sample fraction"}
        
        pdf_test = test_df.select(numeric_col).sample(False, test_frac, seed=42).toPandas()
        pdf_ctrl = control_df.select(numeric_col).sample(False, ctrl_frac, seed=42).toPandas()
        
        # Remove nulls
        pdf_test = pdf_test[numeric_col].dropna()
        pdf_ctrl = pdf_ctrl[numeric_col].dropna()
        
        if len(pdf_test) == 0 or len(pdf_ctrl) == 0:
            return {"statistic": None, "pvalue": None, "message": "Insufficient data after sampling and null removal"}
        
        try:
            statistic, pvalue = stats.ks_2samp(pdf_test, pdf_ctrl)
            return {"statistic": statistic, "pvalue": pvalue, "test_size": len(pdf_test), "ctrl_size": len(pdf_ctrl)}
        except Exception as e:
            return {"statistic": None, "pvalue": None, "message": f"KS test failed: {str(e)}"}

    def calculate_chi_square(self, test_df, control_df, categorical_col):
        """
        Calculates chi-square test for independence between group assignment and categorical variable.
        Returns test statistic and p-value.
        """
        # Validate column exists
        if categorical_col not in test_df.columns or categorical_col not in control_df.columns:
            return {"statistic": None, "pvalue": None, "message": f"Column {categorical_col} not found"}
        
        # Check for empty DataFrames
        if test_df.count() == 0 or control_df.count() == 0:
            return {"statistic": None, "pvalue": None, "message": "Empty DataFrame(s)"}
        
        # Get counts for each category in test and control
        test_counts = test_df.groupBy(categorical_col).count().withColumnRenamed("count", "test_count")
        ctrl_counts = control_df.groupBy(categorical_col).count().withColumnRenamed("count", "ctrl_count")
        
        # Join and create contingency table
        contingency = test_counts.join(ctrl_counts, on=categorical_col, how="full").fillna(0)
        
        # Convert to pandas for scipy.stats.chi2_contingency
        pdf = contingency.toPandas()
        
        if len(pdf) == 0:
            return {"statistic": None, "pvalue": None, "message": "Empty contingency table"}
        
        pdf = pdf.set_index(categorical_col)
        
        # Create 2xN contingency table (2 groups, N categories)
        if "test_count" not in pdf.columns or "ctrl_count" not in pdf.columns:
            return {"statistic": None, "pvalue": None, "message": "Missing count columns"}
        
        observed = pdf[["test_count", "ctrl_count"]].values.T
        
        if observed.shape[1] < 2:
            return {"statistic": None, "pvalue": None, "message": "Insufficient categories (need at least 2)"}
        
        # Check for zero rows (all zeros in a row)
        row_sums = observed.sum(axis=0)
        if np.any(row_sums == 0):
            # Remove zero-sum rows
            observed = observed[:, row_sums > 0]
            if observed.shape[1] < 2:
                return {"statistic": None, "pvalue": None, "message": "Insufficient non-zero categories"}
        
        try:
            # Perform chi-square test
            statistic, pvalue, dof, expected = stats.chi2_contingency(observed)
            return {"statistic": statistic, "pvalue": pvalue, "dof": dof}
        except Exception as e:
            return {"statistic": None, "pvalue": None, "message": f"Chi-square test failed: {str(e)}"}

    def prepare_for_cube(self, df, balance_cols):
        """Scales continuous variables for balancing stability."""
        assembler = VectorAssembler(inputCols=balance_cols, outputCol="features_unscaled")
        df_vec = assembler.transform(df)
        scaler = StandardScaler(inputCol="features_unscaled", outputCol="features_scaled", withStd=True, withMean=True)
        return scaler.fit(df_vec).transform(df_vec)

    def distributed_cube_sampling(self, df, balance_cols, id_col="cust_ID", test_fraction=0.9):
        """
        Distributed Split-and-Combine execution of the Cube Method.
        
        Args:
            df: Spark DataFrame with features_scaled column
            balance_cols: List of column names used for balancing
            id_col: Name of the ID column (default: "cust_ID")
            test_fraction: Fraction of data to assign to test group (default: 0.9)
        """
        p = len(balance_cols)
        
        # Validate required columns exist
        required_cols = [id_col, "features_scaled"]
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # Step 1: Map Phase - Local Fast Flight on Partitions
        def map_flight(iterator):
            rows = list(iterator)
            if not rows: return
            pi_init = [test_fraction] * len(rows)
            X_init = [r.features_scaled.toArray() for r in rows]
            # Use getattr to handle dynamic column name
            ids = [getattr(r, id_col) for r in rows]
            
            sampler = CubeSampler(pi_init, X_init)
            pi_star = sampler.fast_flight()
            
            for i in range(len(rows)):
                yield (ids[i], float(pi_star[i]), X_init[i].tolist())

        # step 2: Execute Map
        rdd_residuals = df.select(id_col, "features_scaled").rdd.mapPartitions(map_flight)
        
        # Step 3: Combine Phase - Handle residuals on Driver
        # Units are residual if 0 < pi < 1. Expected residuals per partition <= p.
        residuals = rdd_residuals.filter(lambda x: 0.001 < x[1] < 0.999).collect()
        decided_rdd = rdd_residuals.filter(lambda x: x[1] <= 0.001 or x[1] >= 0.999) \
                              .map(lambda x: (x[0], 1 if x[1] > 0.5 else 0))
        decided = self.spark.createDataFrame(decided_rdd, [id_col, "is_test"])

        if residuals:
            res_ids, res_pi, res_X = zip(*residuals)
            # Ensure proper array shape: list of lists -> numpy array
            X_array = np.array(list(res_X))
            sampler = CubeSampler(list(res_pi), X_array)
            final_pi = sampler.fast_flight()
            # Landing: simple Bernoulli for remaining few (usually < p)
            # Ensure probabilities are valid [0, 1]
            final_pi = np.clip(final_pi, 0.0, 1.0)
            final_selection = np.random.binomial(1, final_pi)
            res_decided = self.spark.createDataFrame(
                [(res_ids[i], int(final_selection[i])) for i in range(len(res_ids))],
                [id_col, "is_test"]
            )
            final_assignment = decided.union(res_decided)
        else:
            final_assignment = decided

        return final_assignment