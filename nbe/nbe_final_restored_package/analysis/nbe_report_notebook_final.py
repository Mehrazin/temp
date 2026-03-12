
"""
nbe_report_notebook_final.py

Notebook-style driver script for the original 8-scenario workflow.
This mirrors the structure the user previously liked:
- rich config section
- 8 scenario definitions
- scenario runner
- Excel export
- plot generation in per-scenario directories
"""

# ============================================================
# CELL 1 START
# Purpose:
# Imports and load reusable functions.
# ============================================================

from copy import deepcopy

from nbe_report_functions_final import (
    DEFAULT_PLOT_CONFIG,
    make_scenario,
    run_nbe_scenario_outputs,
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

BASE_OUTPUT_PATH = "/dbfs/FileStore/nbe_reports_final"

CUSTOMER_ID_COL = "customer_id"
CONTROL_COL = "control"

TOP_K_VALUES = [1, 3, 5]
TOP_N_VALUES = [1, 3, 5, 10]

NULL_FILL_VALUE = 0.0
PANDAS_SAMPLE_MAX_ROWS = 200000

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

plot_config = deepcopy(DEFAULT_PLOT_CONFIG)

# Example customizations:
# plot_config["theme"]["palette"]["legacy"] = "#2F5597"
# plot_config["theme"]["palette"]["prism"] = "#ED7D31"
# plot_config["topk_product_shift"]["top_n"] = 20

# ============================================================
# CELL 2 END
# ============================================================


# ============================================================
# CELL 3 START
# Purpose:
# Load your dataframes.
# Both legacy_df and new_df must be for the SAME month.
# ============================================================

# Replace these with your real sources.
# legacy_df = spark.table("your_catalog.your_schema.legacy_priority_table_nov")
# new_df = spark.table("your_catalog.your_schema.prism_priority_table_nov")
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
# Define the original 8 scenarios.
# ============================================================

SCENARIOS = [
    make_scenario("S1_full_8m_all37", customer_subset=None, product_cols=ALL_37_PRODUCTS, top_k_values=TOP_K_VALUES),
    make_scenario("S2_full_8m_changed8", customer_subset=None, product_cols=CHANGED_8_PRODUCTS, top_k_values=TOP_K_VALUES),
    make_scenario("S3_full_8m_active12", customer_subset=None, product_cols=ACTIVE_12_PRODUCTS, top_k_values=TOP_K_VALUES),
    make_scenario("S4_full_8m_active_changed", customer_subset=None, product_cols=ACTIVE_CHANGED_PRODUCTS, top_k_values=TOP_K_VALUES),
    make_scenario("S5_1m_all37", customer_subset=one_million_customer_df, product_cols=ALL_37_PRODUCTS, top_k_values=TOP_K_VALUES),
    make_scenario("S6_1m_changed8", customer_subset=one_million_customer_df, product_cols=CHANGED_8_PRODUCTS, top_k_values=TOP_K_VALUES),
    make_scenario("S7_1m_active12", customer_subset=one_million_customer_df, product_cols=ACTIVE_12_PRODUCTS, top_k_values=TOP_K_VALUES),
    make_scenario("S8_1m_active_changed", customer_subset=one_million_customer_df, product_cols=ACTIVE_CHANGED_PRODUCTS, top_k_values=TOP_K_VALUES),
]

SCENARIOS

# ============================================================
# CELL 4 END
# ============================================================


# ============================================================
# CELL 5 START
# Purpose:
# Run one scenario only.
# ============================================================

# one_output = run_nbe_scenario_outputs(
#     spark=spark,
#     legacy_df=legacy_df,
#     new_df=new_df,
#     scenario_name=SCENARIOS[0]["scenario_name"],
#     base_output_path=BASE_OUTPUT_PATH,
#     customer_subset=SCENARIOS[0]["customer_subset"],
#     product_cols=SCENARIOS[0]["product_cols"],
#     top_k_values=SCENARIOS[0]["top_k_values"],
#     customer_id_col=CUSTOMER_ID_COL,
#     control_col=CONTROL_COL,
#     plot_config=plot_config,
#     export_excel=True,
# )
#
# print(one_output["scenario_dir"])
# print(one_output["excel_file"])
# print(one_output["plot_files"])

# ============================================================
# CELL 5 END
# ============================================================


# ============================================================
# CELL 6 START
# Purpose:
# Run all 8 scenarios in batch.
# Each scenario gets its own directory automatically.
# ============================================================

# batch_outputs = {}
#
# for sc in SCENARIOS:
#     out = run_nbe_scenario_outputs(
#         spark=spark,
#         legacy_df=legacy_df,
#         new_df=new_df,
#         scenario_name=sc["scenario_name"],
#         base_output_path=BASE_OUTPUT_PATH,
#         customer_subset=sc["customer_subset"],
#         product_cols=sc["product_cols"],
#         top_k_values=sc["top_k_values"],
#         customer_id_col=CUSTOMER_ID_COL,
#         control_col=CONTROL_COL,
#         plot_config=plot_config,
#         export_excel=True,
#     )
#     batch_outputs[sc["scenario_name"]] = out
#     print(sc["scenario_name"], out["scenario_dir"])

# ============================================================
# CELL 6 END
# ============================================================


# ============================================================
# CELL 7 START
# Purpose:
# Example custom runs outside the 8 standard scenarios.
# ============================================================

# Example A: full customer universe + only changed products
# custom_report_a = run_nbe_scenario_outputs(
#     spark=spark,
#     legacy_df=legacy_df,
#     new_df=new_df,
#     scenario_name="custom_full_changed8",
#     base_output_path=BASE_OUTPUT_PATH,
#     customer_subset=None,
#     product_cols=CHANGED_8_PRODUCTS,
#     top_k_values=[1, 3, 5],
#     customer_id_col=CUSTOMER_ID_COL,
#     control_col=CONTROL_COL,
#     plot_config=plot_config,
#     export_excel=True,
# )
#
# Example B: 1M subset + active 12 products
# custom_report_b = run_nbe_scenario_outputs(
#     spark=spark,
#     legacy_df=legacy_df,
#     new_df=new_df,
#     scenario_name="custom_1m_active12",
#     base_output_path=BASE_OUTPUT_PATH,
#     customer_subset=one_million_customer_df,
#     product_cols=ACTIVE_12_PRODUCTS,
#     top_k_values=[1, 3, 5],
#     customer_id_col=CUSTOMER_ID_COL,
#     control_col=CONTROL_COL,
#     plot_config=plot_config,
#     export_excel=True,
# )

# ============================================================
# CELL 7 END
# ============================================================
