from collections import namedtuple
import math
import streamlit as st
import streamlit.components.v1 as stc

# File Processing Pkgs
import altair as alt
import pandas as pd
import docx2txt
from PIL import Image 
from PyPDF2 import PdfFileReader
import pdfplumber
import numpy as np
import os 
import matplotlib.pyplot as plt
import plotly.express as px

#start
"""
#  SIA Data Analysis and Visualization !

It is a simple GUI for data analysis and visualization :heart:

[Introduction](https://docs.streamlit.io) 

[User guid](https://discuss.streamlit.io).

"""

# Functions 
def save_uploadedfile(uploadedfile):
     with open(os.path.join("tempDir",uploadedfile.name),"wb") as f:
         f.write(uploadedfile.getbuffer())
     return st.success("Saved File:{} to tempDir".format(uploadedfile.name))


# Left side - Navigation bar
menu = ["Home","Import Dataset","Data Analysis","Data Visualization"] 
choice = st.sidebar.selectbox("Part1",menu)
# menu = ["Import Dataset","Home","Data Analysis","Data Visualization"] 
# choice = st.sidebar.selectbox("Part2",menu)
# menu = ["Data Analysis","Home","Import Dataset","Data Visualization"] 
# choice = st.sidebar.selectbox("Part3",menu)

# Navigation 
if choice == "Home":
    st.subheader("Home")

elif choice == "Import Dataset":
    st.subheader("Import Dataset")
    # Import Dataset Page 
    data_file = st.file_uploader("Upload CSV",type=['csv'])    
    if st.button("Process"):
        if data_file is not None:
            file_details = {"Filename":data_file.name,"FileType":data_file.type,"FileSize":data_file.size}
            st.write(file_details)
            df = pd.read_csv(data_file)
            st.dataframe(df)

            x_col = st.text_input("Column x : ")
            y_col= st.text_input("Column y: ")

            df_colx = df.iloc[:,14]
            # #st.dataframe(df_colx)
            df_coly1 = df.iloc[:,15]
            df_coly2 = df.iloc[:,13]
            df_coly3 = df.iloc[:,12]
                
            #select data 
            FEL = df['FEL']
            followers = df['followers']
            make_choice = st.selectbox('Select your data:', df)
            year_choice = st.selectbox('', FEL)
            model_choice = st.selectbox('', followers)
            
            #st.dataframe(df_coly)
            save_uploadedfile(data_file)
            chart_data = pd.DataFrame(df_colx,df_coly1)
            st.line_chart(chart_data)
            # st.altair_chart(df_colx,df_coly, use_container_width=True)

            #mul lines in one graph
            chart_data = pd.DataFrame(df_colx, columns=['FEL', 'frends_zTransformation'])
            # Chart Filter1
            chart_menu1 = ["Line chart","Area chart","Bar chart"] 
            chart_choice1 = st.sidebar.selectbox("Chart Types",chart_menu1)
            if chart_choice1 == "Line chart":
                st.subheader("Line chart ")
                st.line_chart(chart_data)
            
            st.area_chart(chart_data)
            st.bar_chart(chart_data)
            

            # Chart Filter2
            chart_menu = ["Line chart","Area chart","Bar chart","Radar chart","Column Chart","Dual-Axis Chart","Mekko Chart","Pie Chart","Scatter Plot"] 
            chart_choice = st.sidebar.selectbox("Chart Types",chart_menu)
            
            #line chart
            if chart_choice == "Line chart":
                st.subheader("Line chart ")
                chart_data = pd.DataFrame(np.random.randn(20, 3),columns=['a', 'b', 'c'])
                st.line_chart(chart_data)
            #area chart
            elif chart_choice == "Area chart":
                st.subheader("Area chart")
                chart_data = pd.DataFrame(np.random.randn(20, 3),columns=['a', 'b', 'c'])
                st.area_chart(chart_data)
            #bar chart
            elif chart_choice == "Bar chart":
                st.subheader("Bar chart")
                chart_data = pd.DataFrame(np.random.randn(50, 3),columns=["a", "b", "c"])
                st.bar_chart(chart_data)

		
elif choice == "Data Analysis":
		st.subheader("Data Analysis")

elif choice == "Data Visualization":
    		st.subheader("Data Visualization")
		
else:
    st.subheader("About")
    st.info("Built with Streamlit")
    st.info("Jesus Saves @JCharisTech")
    st.text("Jesse E.Agbe(JCharis)")




# with st.echo(code_location='below'):
#     total_points = st.slider("Number of points in spiral", 1, 5000, 2000)
#     num_turns = st.slider("Number of turns in spiral", 1, 100, 9)

#     Point = namedtuple('Point', 'x y')
#     data = []

#     points_per_turn = total_points / num_turns

#     for curr_point_num in range(total_points):
#         curr_turn, i = divmod(curr_point_num, points_per_turn)
#         angle = (curr_turn + 1) * 2 * math.pi * i / points_per_turn
#         radius = curr_point_num / total_points
#         x = radius * math.cos(angle)
#         y = radius * math.sin(angle)
#         data.append(Point(x, y))

#     st.altair_chart(alt.Chart(pd.DataFrame(data), height=500, width=500)
#         .mark_circle(color='#0068c9', opacity=0.5)
#         .encode(x='x:Q', y='y:Q'))



