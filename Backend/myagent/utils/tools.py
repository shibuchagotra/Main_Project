# my_agent/utils/tools.py
import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")

from dotenv import load_dotenv

load_dotenv(override=True)

def load_env_var(name: str) -> str:
    value = os.getenv(name)
    if not value or value.strip() == "":
        raise ValueError(f"Missing or empty environment variable: {name}")
    return value

def load_data():
    df = pd.read_csv("Data\AQ_met_data.csv")
    df["Timestamp"] = pd.to_datetime(df["Timestamp"])
    states_df = pd.read_csv("Data\states_data.csv")
    ncap_df = pd.read_csv("Data\state_funding_data.csv")
    data=pd.read_csv("Data\Data.csv")
    return df, states_df, ncap_df,data

def run_safe_exec(full_code, df=None, extra_globals=None):
    """Safely execute generated code in sandbox-like globals."""
    import numpy as np, seaborn as sns, streamlit as st, uuid, calendar

    local_vars = {}
    plt.rcParams.update({
        "figure.dpi": 1200,
        "savefig.dpi": 1200,
        "figure.figsize": [9, 6],
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
        "savefig.bbox": "tight",
        "font.size": 11,
        "axes.titlesize": 14,
        "axes.labelsize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
    })

    global_vars = {
        "pd": pd, "plt": plt, "os": os, "sns": sns,
        "uuid": uuid, "calendar": calendar, "np": np,
        "df": df, "st": st
    }
    if extra_globals:
        global_vars.update(extra_globals)

    try:
        exec(full_code, global_vars, local_vars)
        return (
            local_vars.get("answer", "Code executed but no result was saved in 'answer' variable"),
            None
        )
    except Exception as e:
        return None, str(e)
