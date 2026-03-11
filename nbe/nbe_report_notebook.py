"""
nbe_report_notebook.py

Notebook-style driver script for running NBE evaluation scenarios.
The actual reusable logic lives in nbe_report_functions.py.

Copy blocks into Databricks cells if you want, or run the script as a single file.
"""

# ============================================================
# CELL 1 START
# Purpose:
# Imports and load reusable functions.
# ============================================================

from nbe_report_functions import (
    make_scenario,
    run_nbe_report_scenario,
    export_report_to_excel,
)

# ============================================================
# CELL 1 END
# ============================================================


# ============================================================
# CELL 2 START
# Purpose:
# Define parameters / config for your notebook run.
# Replace the placeholder product lists with your real product names.
# ============================================================

CUSTOMER_ID_COL = "customer_id"
CONTROL_COL = "control"

TOP_K_VALUES = [1, 3, 5]

ALL_37_PRODUCTS = [
    "product_1", "product_2", "product_3", "product_4", "product_5",
    "product_6", "product_7", "product_8", "product_9", "product_10",
    "product_11", "product_12", "product_13", "product_14", "product_15",
    "product_16", "product_17", "product_18", "product_19", "product_20",
    "product_21", "product_22", "product_23", "product_24", "product_25",
    "product_26", "product_27", "product_28", "product_29", "product_30",
    "product_31", "product_32", "product_33", "product_34", "product_35",
    "product_36", "product_37",
]

CHANGED_8_PRODUCTS = [
    "product_1", "product_2", "product_3", "product_4",
    "product_5", "product_6", "product_7", "product_8"
]

ACTIVE_12_PRODUCTS = [
    "product_1", "product_2", "product_3", "product_4",
    "product_5", "product_6", "product_7", "product_8",
    "product_9", "product_10", "product_11", "product_12"
]

ACTIVE_CHANGED_PRODUCTS = [
    "product_1", "product_2", "product_3", "product_4"
]

OUTPUT_DIR = "/dbfs/FileStore/nbe_reports"

# ============================================================
# CELL 2 END
# ============================================================


# ============================================================
# CELL 3 START
# Purpose:
# Load your dataframes.
# Replace these with your real sources.
# Both legacy_df and new_df must be for the SAME month.
# ============================================================

# Example:
# legacy_df = spark.table("your_catalog.your_schema.legacy_priority_table_dec")
# new_df = spark.table("your_catalog.your_schema.new_priority_table_dec")

# Optional 1M customer subset:
# one_million_customer_df = spark.table("your_catalog.your_schema.customer_subset_1m").select(CUSTOMER_ID_COL).distinct()

legacy_df = None
new_df = None
one_million_customer_df = None

# ============================================================
# CELL 3 END
# ============================================================


# ============================================================
# CELL 4 START
# Purpose:
# Define the 8 scenarios you want to run.
# Dimensions:
# - customer universe: full 8M vs 1M subset
# - product universe: all 37, changed 8, active 12, active changed
# ============================================================

SCENARIOS = [
    make_scenario(
        scenario_name="S1_full_8m_all37",
        customer_subset=None,
        product_cols=ALL_37_PRODUCTS,
        top_k_values=TOP_K_VALUES,
    ),
    make_scenario(
        scenario_name="S2_full_8m_changed8",
        customer_subset=None,
        product_cols=CHANGED_8_PRODUCTS,
        top_k_values=TOP_K_VALUES,
    ),
    make_scenario(
        scenario_name="S3_full_8m_active12",
        customer_subset=None,
        product_cols=ACTIVE_12_PRODUCTS,
        top_k_values=TOP_K_VALUES,
    ),
    make_scenario(
        scenario_name="S4_full_8m_active_changed",
        customer_subset=None,
        product_cols=ACTIVE_CHANGED_PRODUCTS,
        top_k_values=TOP_K_VALUES,
    ),
    make_scenario(
        scenario_name="S5_1m_all37",
        customer_subset=one_million_customer_df,
        product_cols=ALL_37_PRODUCTS,
        top_k_values=TOP_K_VALUES,
    ),
    make_scenario(
        scenario_name="S6_1m_changed8",
        customer_subset=one_million_customer_df,
        product_cols=CHANGED_8_PRODUCTS,
        top_k_values=TOP_K_VALUES,
    ),
    make_scenario(
        scenario_name="S7_1m_active12",
        customer_subset=one_million_customer_df,
        product_cols=ACTIVE_12_PRODUCTS,
        top_k_values=TOP_K_VALUES,
    ),
    make_scenario(
        scenario_name="S8_1m_active_changed",
        customer_subset=one_million_customer_df,
        product_cols=ACTIVE_CHANGED_PRODUCTS,
        top_k_values=TOP_K_VALUES,
    ),
]

# ============================================================
# CELL 4 END
# ============================================================


# ============================================================
# CELL 5 START
# Purpose:
# Run one scenario only.
# Use this first to test everything before running all 8.
# ============================================================

# example_report = run_nbe_report_scenario(
#     spark=spark,
#     legacy_df=legacy_df,
#     new_df=new_df,
#     scenario_name=SCENARIOS[0]["scenario_name"],
#     customer_subset=SCENARIOS[0]["customer_subset"],
#     product_cols=SCENARIOS[0]["product_cols"],
#     top_k_values=SCENARIOS[0]["top_k_values"],
#     customer_id_col=CUSTOMER_ID_COL,
#     control_col=CONTROL_COL,
#     show_displays=True,
#     show_plots=True,
# )

# export_report_to_excel(
#     example_report,
#     output_path=f"{OUTPUT_DIR}/{SCENARIOS[0]['scenario_name']}.xlsx"
# )

# ============================================================
# CELL 5 END
# ============================================================


# ============================================================
# CELL 6 START
# Purpose:
# Run all 8 scenarios and export one Excel file per scenario.
# Each Excel file contains multiple sheets from the report tables.
# ============================================================

# all_reports = {}

# for sc in SCENARIOS:
#     report = run_nbe_report_scenario(
#         spark=spark,
#         legacy_df=legacy_df,
#         new_df=new_df,
#         scenario_name=sc["scenario_name"],
#         customer_subset=sc["customer_subset"],
#         product_cols=sc["product_cols"],
#         top_k_values=sc["top_k_values"],
#         customer_id_col=CUSTOMER_ID_COL,
#         control_col=CONTROL_COL,
#         show_displays=True,
#         show_plots=True,
#     )

#     all_reports[sc["scenario_name"]] = report

#     export_report_to_excel(
#         report,
#         output_path=f"{OUTPUT_DIR}/{sc['scenario_name']}.xlsx"
#     )

# ============================================================
# CELL 6 END
# ============================================================


# ============================================================
# CELL 7 START
# Purpose:
# Example custom single-scenario runs.
# These are useful if you want ad hoc analyses outside the 8 standard scenarios.
# ============================================================

# Example A: full customer universe + only changed products
# custom_report_a = run_nbe_report_scenario(
#     spark=spark,
#     legacy_df=legacy_df,
#     new_df=new_df,
#     scenario_name="custom_full_changed8",
#     customer_subset=None,
#     product_cols=CHANGED_8_PRODUCTS,
#     top_k_values=[1, 3, 5],
#     customer_id_col=CUSTOMER_ID_COL,
#     control_col=CONTROL_COL,
#     show_displays=True,
#     show_plots=True,
# )

# Example B: 1M subset + active 12 products
# custom_report_b = run_nbe_report_scenario(
#     spark=spark,
#     legacy_df=legacy_df,
#     new_df=new_df,
#     scenario_name="custom_1m_active12",
#     customer_subset=one_million_customer_df,
#     product_cols=ACTIVE_12_PRODUCTS,
#     top_k_values=[1, 3, 5],
#     customer_id_col=CUSTOMER_ID_COL,
#     control_col=CONTROL_COL,
#     show_displays=True,
#     show_plots=True,
# )

# ============================================================
# CELL 7 END
# ============================================================
