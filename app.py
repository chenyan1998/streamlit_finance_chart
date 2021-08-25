import pandas as pd
import streamlit as st
import yfinance
import os 
import glob
import plotly.express as px
import plotly.figure_factory as ff
from collections import namedtuple
import streamlit.components.v1 as stc

# File Processing Pkgs
import altair as alt
from PIL import Image 
from PyPDF2 import PdfFileReader
import numpy as np
import os 
import matplotlib.pyplot as plt
import plotly.express as px

from os import listdir

def find_csv_filenames( path_to_dir, suffix=".csv" ):
    filenames = listdir(path_to_dir)
    return [ filename for filename in filenames if filename.endswith( suffix ) ]

storefiles = []
@st.cache
def load_data(path):
    # components = pd.read_csv("./orignal_data.csv")
    components = pd.read_csv(path)
    return components.drop('SEC filings', axis=1).set_index('Symbol')

@st.cache()
def load_quotes(asset):
    return yfinance.download(asset)

# Functions 
def save_uploadedfile(uploadedfile):
     with open(os.path.join("tempDir",uploadedfile.name),"wb") as f:
         f.write(uploadedfile.getbuffer())
         storefiles.append(uploadedfile.name)
     return st.success("Saved File:{} to tempDir".format(uploadedfile.name))


#start
"""
#  SIA Data Analysis and Visualization !

It is a simple GUI for data analysis and visualization :heart:

[Introduction](https://docs.streamlit.io) 

[User guide](https://discuss.streamlit.io).

"""

def main():
    # Left side - Navigation bar
    menu = ["Home","Import Dataset","Data Analysis","Data Visualization"] 
    choice = st.sidebar.selectbox("Part1",menu)
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
                save_uploadedfile(data_file)
                st.write(storefiles)

                # x_col = st.text_input("Column x : ")
                # y_col= st.text_input("Column y: ")

                # df_colx = df.iloc[:,14]
                # # #st.dataframe(df_colx)
                # df_coly1 = df.iloc[:,15]
                # df_coly2 = df.iloc[:,13]
                # df_coly3 = df.iloc[:,12]
                    
                # #select data 
                # FEL = df['FEL']
                # followers = df['followers']
                # make_choice = st.selectbox('Select your data:', df)
                # year_choice = st.selectbox('', FEL)
                # model_choice = st.selectbox('', followers)
                
                #st.dataframe(df_coly)
                
                # chart_data = pd.DataFrame(df_colx,df_coly1)
                # st.line_chart(chart_data)
                # # st.altair_chart(df_colx,df_coly, use_container_width=True)

                # #mul lines in one graph
                # chart_data = pd.DataFrame(df_colx, columns=['FEL', 'frends_zTransformation'])

                            
    elif choice == "Data Analysis":
            st.subheader("All import dataset ")
            result = find_csv_filenames("/Users/chenyan/Documents/GitHub/streamlit_finance_chart/tempDir",suffix=".csv")

            all_path = []
            for i in range(len(result)):
                string = str(result[i])
                path = "/Users/chenyan/Documents/GitHub/streamlit_finance_chart/tempDir" + "/" + string
                all_path.append(path) 
            
            all_dataset = st.checkbox('Show All import dataset')
            if all_dataset: 
                st.write(result)

            # Select 
            st.subheader('Select one of the dataset that you want to analysis ')
            options = result
            value = st.selectbox("Click below to select a new dataset", options)
            string = value
            path = "/Users/chenyan/Documents/GitHub/streamlit_finance_chart/tempDir" + "/" + string
            dataset_select = pd.read_csv(path)
            this_dataset = st.checkbox('Show dataset')
            if this_dataset: 
                st.write(dataset_select)
             

            # Select X axis 
            st.subheader("Select axis X : ")
            col_name = list(dataset_select.columns)
            value = st.selectbox("Click below to select axis X ", col_name)
            st.write("You selected the column ",value,"as X axis")
            st.subheader("Show column X : ")
            select_colx = value
            data_x = dataset_select[select_colx]
            dataframe_x = st.checkbox('Show dataframe of column X : ')
            if dataframe_x: 
                st.write(data_x)
            

            # Select Y axis
            st.subheader("Select axis Y : ")
            col_name = list(dataset_select.columns)
            value = st.selectbox("Click below to select axis Y ", col_name)
            st.write("You selected the column ",value,"as Y axis")
            st.subheader("Show column Y : ")
            select_coly = value
            data_y = dataset_select[select_coly]
            dataframe_y = st.checkbox('Show dataframe of column Y : ')
            if dataframe_y: 
                st.write(data_y)

            # Visualize datax, datay seperately on one graph, two col, two lines
            st.subheader("Visualize the column x and column y seperately")
            chart_menu_sep = ["Line Chart","Bar Chart","Scatter Chart"] 
            chart_choice_sep = st.selectbox("Chose the types of chart you want ",chart_menu_sep)
            chart_data = pd.concat([data_x, data_y], axis=1)

            if chart_choice_sep == "Line Chart":
                st.subheader('Line Chart')
                st.line_chart(chart_data)

           
            if chart_choice_sep == "Bar Chart":
                st.subheader('Bar Chart')
                st.bar_chart(chart_data)

            
            if chart_choice_sep == "Area Chart":
                st.subheader('Area Chart')
                st.area_chart(chart_data)

            # Visualize the relationship between X and Y 
            st.subheader("Visualize the relationship between X and Y")
            chart_menu = ["Line chart","Bar chart","Scatter chart"] 
            chart_choice = st.selectbox("Chose the types of chart you want ",chart_menu)
            chart_data = pd.concat([data_x, data_y], axis=1)
            df_xy = st.checkbox('Show dataframe of this chart')
            if df_xy: 
                st.write(chart_data)

            if chart_choice == "Line chart":
                st.subheader("Line chart")
                fig = px.line(dataset_select,x=data_x,y=data_y,title=f'{select_colx} vs. {select_coly}')
                st.plotly_chart(fig)
            
            elif chart_choice == "Bar chart":
                st.subheader("Bar chart")
                fig = px.bar(dataset_select,x=data_x,y=data_y,title=f'{select_colx} vs. {select_coly}')
                st.plotly_chart(fig)

            elif chart_choice == "Scatter chart":
                st.subheader("Scatter chart")
                fig = px.scatter(x=data_x,y=data_y,title=f'{select_colx} vs. {select_coly}')
                st.plotly_chart(fig)
            

            




    elif choice == "Data Visualization":
            # Components 
            components = load_data("./orignal_data.csv")
            title = st.empty()
            st.sidebar.title("Options")

            def label(symbol):
                a = components.loc[symbol]
                return symbol + ' - ' + a.Security

            if st.sidebar.checkbox('View companies list'):
                st.dataframe(components[['Security',
                                        'GICS Sector',
                                        'Date first added',
                                        'Founded']])

            st.sidebar.subheader('Select asset')
            asset = st.sidebar.selectbox('Click below to select a new asset',
                                        components.index.sort_values(), index=3,
                                        format_func=label)
            title.title(components.loc[asset].Security)
            # if st.sidebar.checkbox('View company info', True):
            #     st.table(components.loc[asset])
            # data0 = load_quotes(asset)
            # data = data0.copy().dropna()
            # data0.to_csv('price0.csv')
            # data.to_csv('price.csv')

            data01 = pd.read_csv("/Users/chenyan/Documents/GitHub/streamlit_finance_chart/price0.csv")
            data02 = pd.read_csv("/Users/chenyan/Documents/GitHub/streamlit_finance_chart/price.csv")

            st.subheader('Data0')
            st.write(data01)
            st.subheader('Data')
            st.write(data02)
            data02.index.name = None



            section = st.sidebar.slider('Number of quotes', min_value=30,
                                max_value=min([2000, data02.shape[0]]),
                                value=500,  step=10)

            data2 = data02[-section:]['Adj Close'].to_frame('Adj Close')
            st.subheader('Data2')
            st.write(data2)
            sma = st.sidebar.checkbox('SMA')
            if sma:
                period= st.sidebar.slider('SMA period', min_value=5, max_value=500,
                                    value=20,  step=1)
                data02[f'SMA {period}'] = data02['Adj Close'].rolling(period ).mean()
                data2[f'SMA {period}'] = data02[f'SMA {period}'].reindex(data2.index)

            sma2 = st.sidebar.checkbox('SMA2')
            if sma2:
                period2= st.sidebar.slider('SMA2 period', min_value=5, max_value=500,
                                    value=100,  step=1)
                data02[f'SMA2 {period2}'] = data02['Adj Close'].rolling(period2).mean()
                data2[f'SMA2 {period2}'] = data02[f'SMA2 {period2}'].reindex(data2.index)
                st.subheader('Data2_period')
                st.write(data2)

            line_chart = st.sidebar.checkbox('Line Chart')
            if line_chart:  
                st.subheader('Chart')
                st.line_chart(data2)

            bar_chart = st.sidebar.checkbox('Bar Chart')
            if bar_chart:  
                st.subheader('BarChart')
                st.bar_chart(data2)

            area_chart = st.sidebar.checkbox('Area Chart')
            if area_chart:  
                st.subheader('AreaChart')
                st.area_chart(data2)

            plotly_chart = st.sidebar.checkbox('Plotly Chart')
            if plotly_chart:  
                st.subheader('PlotlyChart')
                x1 = data2.iloc[:,0]
                x2 = data2.iloc[:,1]
                x3 = data2.iloc[:,2]
                # x1 = np.random.randn(200) - 2
                # x2 = np.random.randn(200)
                # x3 = np.random.randn(200) + 2
                # Group data together
                hist_data = [x1, x2, x3]
                group_labels = ['Group 1', 'Group 2', 'Group 3']
                # # Create distplot with custom bin_size
                fig = ff.create_distplot(hist_data, group_labels, bin_size=[.1, .25, .5])
                # # Plot!
                st.plotly_chart(fig, use_container_width=True)

            map_chart = st.sidebar.checkbox('Map Chart')
            if map_chart:
                df = pd.DataFrame(np.random.randn(1000, 2) / [50, 50] + [37.76, -122.4],columns=['lat', 'lon'])
                st.map(df)

            if st.sidebar.checkbox('View stadistic'):
                st.subheader('Stadistic')
                st.table(data2.describe())

            if st.sidebar.checkbox('View quotes'):
                st.subheader(f'{asset} historical data')
                st.write(data2)

            st.sidebar.title("About")
            st.sidebar.info('This app is a simple example of '
                            'using Strealit to create a financial data web app.\n'
                            '\nIt is maintained by [Paduel]('
                            'https://twitter.com/paduel_py).\n\n'
                            'Check the code at https://github.com/paduel/streamlit_finance_chart')

if __name__ == '__main__':
    main()
