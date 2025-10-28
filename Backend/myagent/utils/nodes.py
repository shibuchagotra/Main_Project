# my_agent/utils/nodes.py
import os
import pandas as pd

from langchain_groq import ChatGroq
from myagent.utils.state import GraphState
from myagent.utils.tools import load_env_var,run_safe_exec,load_data

models = {
    "gpt-oss-120b": "openai/gpt-oss-120b",
    "qwen3-32b": "qwen/qwen3-32b",
    "gpt-oss-20b": "openai/gpt-oss-20b",
    "llama4 maverik":"meta-llama/llama-4-maverick-17b-128e-instruct",
    "llama3.3": "llama-3.3-70b-versatile",
    "deepseek-R1": "deepseek-r1-distill-llama-70b",
    "gemini-2.5-flash": "gemini-2.5-flash",
    "gemini-2.5-pro": "gemini-2.5-pro",
    "gemini-2.5-flash-lite": "gemini-2.5-flash-lite",
    "gemini-2.0-flash": "gemini-2.0-flash",
    "gemini-2.0-flash-lite": "gemini-2.0-flash-lite",
    # "llama4 scout":"meta-llama/llama-4-scout-17b-16e-instruct"
    # "llama3.1": "llama-3.1-8b-instant"
}

token = load_env_var("GROQ_API_KEY")



def generate_code(state: GraphState) -> GraphState:
    """Generate Python code to answer the user's question."""
    try:
        df, states_df, ncap_df,data_df = load_data()
        new_line = "\n"

        # Template without Markdown backticks
        template = f"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import uuid
import calendar
import numpy as np

plt.style.use('vayuchat.mplstyle')
df = pd.read_csv("Data\AQ_met_data.csv")
df["Timestamp"] = pd.to_datetime(df["Timestamp"])
states_df = pd.read_csv("Data\states_data.csv")
ncap_df = pd.read_csv("Data\state_funding_data.csv")
data_df=pd.read_csv("Data\Data.csv")

# df info:
{new_line.join(['# '+x for x in str(df.dtypes).split(new_line)])}
# states_df info:
{new_line.join(['# '+x for x in str(states_df.dtypes).split(new_line)])}
# ncap_df info:
{new_line.join(['# '+x for x in str(ncap_df.dtypes).split(new_line)])}
#data_info:
{new_line.join(['# '+x for x in str(data_df.dtypes).split(new_line)])}

# Question: {state.question.strip()}
# Generate code to answer the question and save result in 'answer' variable
# If creating a plot, save it with a unique filename and store the filepath in 'answer' and save in folder name  Results
# If returning text/numbers, store the result directly in 'answer'
"""

        # Read system prompt
        with open("prompts\system_prompt.txt", "r", encoding="utf-8") as f:
            system_prompt = f.read().strip()

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Complete the following code while keeping all imports:\n{template}"}
        ]

        # Generate code using LLM
        llm = ChatGroq(model=models[state.model], api_key=token, temperature=0)
        response = llm.invoke(messages)
        answer = response.content.strip()

        # Extract code from response (if wrapped in ```python)
        if "```python" in answer:
            code_part = answer.split("```python")[1].split("```")[0].strip()
        else:
            code_part = answer

        state.generated_code = f"{template}\n{code_part}"

    except Exception as e:
        state.error = str(e)

    return state


# def exec_code(state: GraphState) -> GraphState:
#     """Execute the generated code safely and store the result in state.answer."""
#     if not state.generated_code:
#         state.error = "No code generated"
#         return state

#     try:
#         df, states_df, ncap_df,data_df = load_data()
#         answer_result, code_error = run_safe_exec(
#             state.generated_code,
#             df=df,
#             extra_globals={"states_df": states_df, "ncap_df": ncap_df,"data_df":data_df}
#         )

#         if code_error:
#             state.error = code_error
#         else:
#             state.answer = answer_result
#             state.executed_code = state.generated_code
#             print(state.answer)

#     except Exception as e:
#         state.error = str(e)

#     return state

def exec_code(state: GraphState) -> GraphState:
    if not state.generated_code:
        state.error = "No code generated"
        return state

    df, states_df, ncap_df,data_df= load_data()

    answer_result, code_error = run_safe_exec(
        state.generated_code,
        df=df,
        extra_globals={"states_df": states_df, "ncap_df": ncap_df,"data_df":data_df}
    )

    if code_error:
        state.error = code_error
    else:
        # Convert DataFrame to JSON-serializable format
        if isinstance(answer_result, pd.DataFrame):
            state.answer = answer_result.to_dict(orient="records")
        else:
            state.answer = answer_result

        state.executed_code = state.generated_code

    return state
