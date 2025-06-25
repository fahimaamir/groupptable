import mysql.connector
import datetime
import numpy as np
import os
import streamlit.components.v1 as components
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
import numpy as np
import matplotlib.pyplot as plt
import numpy
import streamlit as st
import matplotlib.pyplot as plt
import streamlit as st
import pandas as pd
from st_aggrid import JsCode, AgGrid, GridOptionsBuilder
from st_aggrid import GridOptionsBuilder, AgGrid, GridUpdateMode, ColumnsAutoSizeMode
from st_aggrid.shared import GridUpdateMode

from st_aggrid import GridOptionsBuilder, AgGrid, GridUpdateMode, DataReturnMode
from st_aggrid.shared import ColumnsAutoSizeMode
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')       #서버에서, 화면에 표시하기 위해서 필요
import seaborn as sns
import altair as alt               ##https://altair-viz.github.io/
import plotly.express as px
import matplotlib.pyplot as plt
import numpy as np

import numpy as np
import streamlit.components.v1 as components
from pivottablejs import pivot_ui
import pandas as pd
import streamlit as st
import pandas as pd
from st_aggrid import AgGrid, GridOptionsBuilder



import numpy
import streamlit as st
import matplotlib.pyplot as plt
import streamlit as st
import pandas as pd
from st_aggrid import GridOptionsBuilder, AgGrid, GridUpdateMode, DataReturnMode
from st_aggrid.shared import ColumnsAutoSizeMode
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')       #서버에서, 화면에 표시하기 위해서 필요
import seaborn as sns
import altair as alt               ##https://altair-viz.github.io/
import plotly.express as px
import matplotlib.pyplot as plt
import numpy as np
import streamlit.components.v1 as components

import os 
import numpy as np
import streamlit as st
import pandas as pd
from st_aggrid import AgGrid, GridOptionsBuilder, GridUpdateMode, DataReturnMode
from st_aggrid import GridOptionsBuilder, AgGrid, GridUpdateMode, DataReturnMode
from st_aggrid import GridOptionsBuilder, AgGrid, GridUpdateMode, DataReturnMode
from st_aggrid.shared import ColumnsAutoSizeMode
from st_aggrid import GridOptionsBuilder, AgGrid, GridUpdateMode
from st_aggrid import AgGrid, GridOptionsBuilder, JsCode
from collections import defaultdict
from st_aggrid import AgGrid, GridOptionsBuilder

st.set_page_config(layout="wide")

def generate_sales_data():
        """Generate dataset simulating sales data."""
        np.random.seed(42)
        rows = 50

        # Create a more complex dataset
        df = pd.DataFrame({
            'Product_ID': range(1, rows + 1),
            'City': np.random.choice(['Karachi', 'Islamabad', 'Quata', 'Pishawar'], rows),
            'Category': np.random.choice(['Electronics', 'Clothing', 'Home', 'Sports'], rows),
            'Item': np.random.choice(['IRON', 'pant', 'Flate', 'football'], rows),
            'Sale_person': np.random.choice(['Fahim', 'Aamir', 'Zahir', 'Asim'], rows),
            'Base_Price': np.random.uniform(10, 500, rows).round(2),
            'Quantity_Sold': np.random.randint(1, 100, rows),
            'commission': np.random.randint(100, 1000, rows),
        })

        return df


def configure_grid_options(df):
        """Configure advanced grid options with multiple features."""
        gb = GridOptionsBuilder.from_dataframe(df)
        # Configure row grouping and aggregation
        #gb.configure_column("allColumns", filter=True)
        for column in df.columns:
            gb.configure_column(column, filter=True)

        gb.configure_default_column(
                    groupable=True,
                    value=True,
                    enableRowGroup=True,
                    aggFunc='sum'
                )

                # Add filter and sort options
        gb.configure_grid_options(
                    enableRangeSelection=True,
                    enableRangeHandle=True,
                    suppressColumnMoveAnimation=False,
                    suppressRowClickSelection=False
                )
        
        return gb.build()
sales_data = generate_sales_data()

grid_options = configure_grid_options(sales_data)
gb = GridOptionsBuilder()
gb.configure_default_column( groupable=True,value=True,enableRowGroup=True,aggFunc='sum'  )

                # Add filter and sort options
gb.configure_grid_options(
                    enableRangeSelection=True,
                    enableRangeHandle=True,
                    #allColumnsfilter=True,
                    #gob.configure_column("allColumns", filter=True)
                    suppressColumnMoveAnimation=False,
                    suppressRowClickSelection=False)
gb.build()

        

    #st.subheader('Interactive Sales Data Grid')
st.markdown("""
    **Features:**
    - Edit Base Price and Quantity Sold
    - Automatic Total Revenue calculation
    """)







    # AgGrid with custom options
mfa = AgGrid(
        sales_data,
        gridOptions=grid_options,
        height=500,
        theme='alpine',
        allow_unsafe_jscode=True,
        #update_mode=GridUpdateMode.MODEL_CHANGED,
        update_mode=GridUpdateMode.GRID_CHANGED,
        data_return_mode=DataReturnMode.FILTERED_AND_SORTED,
        fit_columns_on_grid_load=True,
        width =2900 ,
        reload_data=False
    )


#if st.button('Check availability'):
mfa4 =mfa['data']
forpivot =mfa['data']
st.write(mfa4)


#st.expander("See explanation"):   
colname = list(mfa4)

col1, col2, col3 = st.columns(3)

col1.write('Rows')
col2.write('Columns')
col3.write('Values')

with col1:
    options = st.multiselect(    "What are your favorite colors?",    colname,    default=colname[0], )
with col2:
    options1 = st.multiselect(    "What are your favorite colors?",    colname,    default=colname[1], )
with col3:
    options2 = st.multiselect(    "What are your favorite colors?",    colname,    default=colname[-1], )
    
if st.button('Group by'):
    #st.write("Muhammad")
    
    #df.groupby(['Animal']).mean()
    #aggregated_data = mfa4.groupby(options).agg(options2,'sum')


    a,  *rest = options2
    #aggregated_data = mfa4.groupby(options).agg(a,'sum')
    #aggregated_data = mfa4.groupby('city').agg('commission','sum')
    df_grouped = (mfa4.groupby(options)   [a].sum()).reset_index()
    #df.groupby('A').B.agg(['min', 'max'])
    #df.groupby("A")[["B"]].agg(lambda x: x.astype(float).min())
    #df2.groupby(["X"], sort=False).sum()
    st.write(df_grouped)
    #aggregated_data = mfa4.groupby(options).agg(total_salary=(options2, 'sum'))
    #aggregated_data = mfa4.groupby('city').agg(total_salary=('commission', 'sum'))
    #st.write(aggregated_data)
    
    #total_salary=('Salary', 'sum'),
    #avg_salary=('Salary', 'mean'),
    #player_count=('Name', 'count')
    
    #https://stackoverflow.com/questions/66350904/pandas-subtotal-similar-to-excel
if st.button('Pivot Table'):
    #muhammad = pd.pivot_table(mfa,values="QTY",index=['ITME'], columns='BOOMSIZE',aggfunc='sum')
    muhammad = pd.pivot_table(mfa4,values=options2,index=options, columns=options1,aggfunc='sum')
    #df3 = mfa4.pivot(index=value1, columns=value2 ,values=value3,aggfunc='sum')
    #os.remove('pivott.csv') 
    st.write(muhammad)
    pcsv = muhammad.to_csv(index=True).encode('utf-8')
    #pcsv = muhammad.to_csv().encode('utf-8')
    
    #csv = convert_df(pcsv)

    st.download_button(
        label="Pivot data as CSV",
        data=pcsv,
        file_name='pivottable.csv',
        mime='text/csv',         )
if st.button('Dynamic Pivot'):
    st.write("muhammad")
    t = pivot_ui(sales_data)

    with open(t.src) as t:
        components.html(t.read(), width=4900, height=1000, scrolling=True)
        #components.html(t.read(), width=3900,  scrolling=True)

