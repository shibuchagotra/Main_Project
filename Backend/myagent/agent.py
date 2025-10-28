# my_agent/agent.py
from myagent.utils.state import GraphState
from myagent.utils.nodes import generate_code, exec_code
from langgraph.graph import StateGraph, END, START

# Create the workflow graph with GraphState
workflow = StateGraph(GraphState)

# Add nodes
workflow.add_node("generate_code", generate_code)
workflow.add_node("exec_code", exec_code)

# Define edges
workflow.add_edge(START, "generate_code")   # START node is the entry point
workflow.add_edge("generate_code", "exec_code")
workflow.add_edge("exec_code", END)

# Compile the workflow
graph = workflow.compile()
