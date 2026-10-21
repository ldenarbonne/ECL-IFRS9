"""
Credit Risk Suite – entry point.

Module 1: IFRS 9 expected credit loss engine (staging, ECL, IFRS 7, IRB EL
comparison, PD validation). The router is ready for further credit modules
(IRB capital, credit stress testing).

Author: Laurent Motais de Narbonne
"""
import streamlit as st

st.set_page_config(page_title="Credit Risk Suite", page_icon="🏦", layout="wide")

pages = [
    st.Page("views/ifrs9_page.py", title="IFRS 9 – Expected credit loss", icon="📉", url_path="ifrs9", default=True),
]
st.navigation(pages).run()
