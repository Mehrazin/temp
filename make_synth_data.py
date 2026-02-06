# make_synth_data.py
# ---------------------------------------------------------
# Synthetic dataset for stratification experiments
# ---------------------------------------------------------

from dataclasses import dataclass
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class SynthConfig:
    n: int = 100_000
    seed: int = 7

    p_visa_null: float = 0.30
    p_visa_1_given_nonnull: float = 0.45

    p_null_balance: float = 0.07
    p_null_web: float = 0.40
    p_null_mobile: float = 0.70
    p_null_offers: float = 0.33


def make_synth_pandas(cfg: SynthConfig = SynthConfig()) -> pd.DataFrame:
    rng = np.random.default_rng(cfg.seed)
    n = cfg.n

    cost_ID = np.arange(1, n + 1)

    visa = np.zeros(n, dtype="float64")
    u = rng.random(n)
    visa[u < cfg.p_visa_null] = np.nan
    visa[(u >= cfg.p_visa_null) & (rng.random(n) < cfg.p_visa_1_given_nonnull)] = 1

    seg = rng.choice([0, 1, 2], size=n, p=[0.55, 0.33, 0.12])

    balance = rng.lognormal(7.5 + 0.7 * (seg == 2), 1.0, size=n)
    balance[rng.random(n) < cfg.p_null_balance] = np.nan

    web = rng.poisson(3 + seg)
    mobile = rng.poisson(4 + seg)

    web[rng.random(n) < cfg.p_null_web] = np.nan
    mobile[rng.random(n) < cfg.p_null_mobile] = np.nan

    offers = rng.poisson(1 + seg)
    offers[rng.random(n) < cfg.p_null_offers] = np.nan

    return pd.DataFrame({
        "cost_ID": cost_ID,
        "visa_ind": visa,
        "balance": balance,
        "n_web_logins": web,
        "n_mobile_logins": mobile,
        "n_offers_year": offers,
    })


def make_synth_spark(spark, cfg: SynthConfig = SynthConfig()):
    return spark.createDataFrame(make_synth_pandas(cfg))
