import os 
import numpy as np
import streamlit as st
import pandas as pd
import streamlit.components.v1 as components
from pivottablejs import pivot_ui


st.set_page_config(layout="wide")
css = '''
<style>
    [data-testid="stSidebar"]{
        min-width: 400px;
        max-width: 4800px;
    }
</style>
'''
st.markdown(css, unsafe_allow_html=True)


iris = pd.read_csv(    "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv")
#iris = pd.read_excel("c:/mfa/book11.xlsx")

t = pivot_ui(iris)

with open(t.src) as t:
    components.html(t.read(), width=1900, height=1000, scrolling=True)



#body { font-family: Verdana;}
#.c3-line, .c3-focused {stroke-width: 3px !important;}
#.c3-bar {stroke: white !important; stroke-width: 1;}
#.c3 text { font-size: 12px; color: grey;}
#.tick line {stroke: white;}
#.c3-axis path {stroke: grey;}
#.c3-circle { opacity: 1 !important; }

