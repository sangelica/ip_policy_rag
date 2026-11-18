from langgraph.graph import StateGraph, START, END
from src.schemas import QueryState
from src.nodes import guard, retrieve, grade, rewrite, generate, check, fallback
import sys

def is_relevant(state: QueryState):
    if state["relevant"] or state["retries"] >= 2:
        return "generate"
    else:
        return "rewrite"

def is_grounded(state: QueryState):
    if state["grounded"]:
        return "end"
    else:
        return "fallback"

stateGraph = StateGraph(QueryState)

# Build and compile the graph
# create all graph nodes
for name, func in [("guard", guard), ("retrieve", retrieve), ("grade", grade),
                 ("rewrite", rewrite), ("generate", generate), ("check", check),
                 ("fallback", fallback)]:
    stateGraph.add_node(name, func)

# create all connections
stateGraph.add_edge(START, "guard")
stateGraph.add_edge("guard", "retrieve")
stateGraph.add_edge("retrieve", "grade")
stateGraph.add_conditional_edges("grade",
                                is_relevant,
                                {
                                    "generate" : "generate",
                                    "rewrite" : "rewrite"
                                }
                                )

stateGraph.add_edge("rewrite", "retrieve")

stateGraph.add_edge("generate", "check")
stateGraph.add_conditional_edges("check",
                                 is_grounded,
                                 {
                                     "end" : END,
                                     "fallback" : "fallback"
                                 }
                                )

# small "fallback" node that sets the "Insufficient policy data" answer → END
stateGraph.add_edge("fallback", END)

app = stateGraph.compile()


if __name__ == "__main__":
    res = app.invoke({"query": "What is the 2025 premium rate for direct contributors?"})
    print(res['answer'])