from src.state_machine import app as IP_AGENT
from src.config import PROJECT_ROOT
from pathlib import Path

import streamlit as st

st.title("Philhealth Insurance Assistant")
st.write("Welcome! Enter your question below.")

query = st.text_input("What is your question?")

print(query)
res = IP_AGENT.invoke({"query": query})
st.write(res['answer'])

