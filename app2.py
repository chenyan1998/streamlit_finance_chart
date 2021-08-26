import pandas as pd
import streamlit as st
import yfinance
import docx2txt
import os 
import glob
import plotly.express as px
import plotly.figure_factory as ff
from collections import namedtuple
import streamlit.components.v1 as stc
from streamlit_tags import st_tags, st_tags_sidebar


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

def read_pdf(file):
    pdfReader = PdfFileReader(file)
    count = pdfReader.numPages
    all_page_text = ""
    for i in range(count):
        page = pdfReader.getPage(i)
        all_page_text += page.extractText()
        return all_page_text

def read_pdf_with_pdfplumber(file):
    with pdfplumber.open(file) as pdf:
        page = pdf.pages[0]
        return page.extract_text()

@st.cache
def load_image(image_file):
	img = Image.open(image_file)
	return img 

    

#start
"""
# :airplane: SIA Data Analysis and Visualization 

It is a simple GUI for data analysis and visualization :heart:

:point_up: [Introduction](https://docs.streamlit.io) 

:point_up: [User guide](https://discuss.streamlit.io).

:point_up: [Others](https://discuss.streamlit.io).

"""

def main():
    # Left side - Navigation bar
    st.sidebar.subheader(":blue_book: Whole Information ")
    menu = ["Home","Import Dataset","Data Analysis","Data Visualization","Others"] 
    choice = st.sidebar.selectbox("Part1",menu)

    # Navigation 
    if choice == "Home":
        st.subheader("Home")

    elif choice == "Import Dataset":
        st.subheader("Import Dataset :inbox_tray:")
        # Import Dataset  
        data_file = st.file_uploader("Upload CSV",type=['csv'])    
        if st.button("Process dataset "):
            if data_file is not None:
                file_details = {"Filename":data_file.name,"FileType":data_file.type,"FileSize":data_file.size}
                st.write(file_details)
                df = pd.read_csv(data_file)
                st.dataframe(df)
                save_uploadedfile(data_file)
                st.write(storefiles)

        # Import Image
        st.subheader("Import image :rice_scene: ")
        image_file = st.file_uploader("Upload Image",type=['png','jpeg','jpg'])
        if image_file is not None:
			# To See Details
			# st.write(type(image_file))
			# st.write(dir(image_file))
            file_details = {"Filename":image_file.name,"FileType":image_file.type,"FileSize":image_file.size}
            st.write(file_details)
            img = load_image(image_file)
            st.image(img)

        # Import Document files 
        st.subheader("DocumentFiles :bookmark_tabs: ")
        docx_file = st.file_uploader("Upload File",type=['txt','docx','pdf'])
        if st.button("Process documentfiles"):
            if docx_file is not None:
                file_details = {"Filename":docx_file.name,"FileType":docx_file.type,"FileSize":docx_file.size}
                st.write(file_details)
				# Check File Type
                if docx_file.type == "text/plain":
                    st.text(str(docx_file.read(),"utf-8")) # empty
                    raw_text = str(docx_file.read(),"utf-8") # works with st.text and st.write,used for futher processing
                    st.write(raw_text) # works
                elif docx_file.type == "application/pdf":
                    try:
                        with pdfplumber.open(docx_file) as pdf:
                            page = pdf.pages[0]
                            st.write(page.extract_text())
                    except:
                        st.write("None")
                elif docx_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
				# Use the right file processor ( Docx,Docx2Text,etc)
                    raw_text = docx2txt.process(docx_file) # Parse in the uploadFile Class directory
                    st.write(raw_text)

                
                            
    elif choice == "Data Analysis":
            st.subheader(":open_file_folder: All import dataset ")
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
            st.subheader(':file_folder: Select one of the dataset that you want to analysis ')
            options = result
            value = st.selectbox("Click below to select a new dataset", options)
            string = value
            path = "/Users/chenyan/Documents/GitHub/streamlit_finance_chart/tempDir" + "/" + string
            dataset_select = pd.read_csv(path)
            this_dataset = st.checkbox('Show dataset')
            if this_dataset: 
                st.write(dataset_select) 

            # Select X axis 
            st.subheader(":pushpin: Select axis X : ")
            col_namex = list(dataset_select.columns)
            value = st.selectbox("Click below to select axis X " , col_namex)
            st.write("You selected the column ",value,"as X axis")
            st.subheader("Show column X : ")
            select_colx = value
            data_x = dataset_select[select_colx]
            dataframe_x = st.checkbox('Show dataframe of column X : ')
            if dataframe_x: 
                st.write(data_x)
            

            # Select Y axis
            st.subheader(":pushpin: Select axis Y : ")
            col_namey = list(dataset_select.columns)
            value = st.selectbox("Click below to select axis Y ", col_namey)
            st.write("You selected the column ",value,"as Y axis")
            st.subheader("Show column Y : ")
            select_coly = value
            data_y = dataset_select[select_coly]
            dataframe_y = st.checkbox('Show dataframe of column Y : ')
            if dataframe_y: 
                st.write(data_y)

            # Multi col select 
            st.write("# Visualize Data ")
            maxtags = st.slider('Number of tags allowed?', 1, 10, 3, key='jfnkerrnfvikwqejn')
            col_name = list(dataset_select.columns)
            st.write("You should write column name on the list ",col_name)

            keywords = st_tags(
                label='# Enter Column name:',
                text='Press enter to add more',
                value= col_name,
                suggestions=['five', 'six', 'seven', 'eight', 'nine', 'three', 'eleven', 'ten', 'four'],
                maxtags=maxtags,
                key="aljnf")
            
            # for i in range(len(keywords)):
            #     n = keywords[i]
            #     select_col = n
            #     dataselect = dataset_select[select_col]
                
            #     data = st.checkbox('Show data select : ')
            #     if data: 
            #         st.write(dataselect)
            #     st.write("Keywords Select(one by one)")
            #     st.write(n)

            # Visualize datax, datay seperately on one graph, two col, two lines
            st.subheader(":chart_with_upwards_trend: Visualize the column x and column y seperately")
            chart_menu_sep = ["Line Chart","Bar Chart","Scatter Chart"] 
            chart_choice_sep = st.selectbox("Chose the types of chart you want ",chart_menu_sep)
            col_name = keywords
            chart_data = dataset_select[col_name]

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
            st.subheader(":chart_with_downwards_trend: Visualize the relationship between X and Y")
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
    elif choice == "Others":
        st.subheader(" Generate Report ")
        # Multi col select 
        st.write("# Chose Multiple Dataset ")
        maxtags = st.slider('Number of Dataset allowed?', 1, 10, 3, key='jfnkerrnfvikwqejn')
        result = find_csv_filenames("/Users/chenyan/Documents/GitHub/streamlit_finance_chart/tempDir",suffix=".csv")
        dataset_name = result
        st.write("You should write the name of dataset that on the list ",dataset_name)
        
        keywords1 = st_tags(
            label='# Enter Dataset name:',
            text='Press enter to add more',
            value= dataset_name,
            suggestions=['five', 'six', 'seven', 'eight', 'nine', 'three', 'eleven', 'ten', 'four'],
            maxtags=maxtags,
            key="aljnf")

        for i in range(len(keywords1)+1):
            st.write("keywords length : ", keywords1)
            string = keywords1[i] 
            path = "/Users/chenyan/Documents/GitHub/streamlit_finance_chart/tempDir" + "/" + string
            st.write(path)
            dataset_select = pd.read_csv(path)
            this_dataset = st.checkbox('Show dataset' + str(i))
            if this_dataset: 
                st.write(dataset_select) 

            # Select X axis 
            st.subheader(":pushpin: Select axis X : ")
            col_namex = list(dataset_select.columns)
            value = st.selectbox("Click below to select axis X " + str(i), col_namex)
            st.write("You selected the column ",value,"as X axis")
            st.subheader("Show column X : ")
            select_colx = value
            data_x = dataset_select[select_colx]
            dataframe_x = st.checkbox('Show dataframe of column X : ' + str(i))
            if dataframe_x: 
                st.write(data_x)
            

            # Select Y axis
            st.subheader(":pushpin: Select axis Y : ")
            col_namey = list(dataset_select.columns)
            value = st.selectbox("Click below to select axis Y " + str(i), col_namey)
            st.write("You selected the column ",value,"as Y axis")
            st.subheader("Show column Y : ")
            select_coly = value
            data_y = dataset_select[select_coly]
            dataframe_y = st.checkbox('Show dataframe of column Y : ' + str(i))
            if dataframe_y: 
                st.write(data_y)

            # Multi col select 
            st.write("# Visualize Data ")
            maxtags = st.slider('Number of tags allowed?', 1, 10, 3, key='jfnkerrnfvikwqejn'+ str(i))
            col_name = list(dataset_select.columns)
            st.write("You should write column name on the list ",col_name)

            keywords = st_tags(
                label='# Enter Column name:',
                text='Press enter to add more',
                value= col_name,
                suggestions=['five', 'six', 'seven', 'eight', 'nine', 'three', 'eleven', 'ten', 'four'],
                maxtags=maxtags,
                key="aljnf" + str(i))
            
            # for i in range(len(keywords)):
            #     n = keywords[i]
            #     select_col = n
            #     dataselect = dataset_select[select_col]
                
            #     data = st.checkbox('Show data select : ')
            #     if data: 
            #         st.write(dataselect)
            #     st.write("Keywords Select(one by one)")
            #     st.write(n)

            # Visualize datax, datay seperately on one graph, two col, two lines
            st.subheader(":chart_with_upwards_trend: Visualize the column x and column y seperately")
            chart_menu_sep = ["Line Chart","Bar Chart","Scatter Chart"] 
            chart_choice_sep = st.selectbox("Chose the types of chart you want " + str(i),chart_menu_sep)
            col_name = keywords
            chart_data = dataset_select[col_name]

            if chart_choice_sep == "Line Chart":
                st.subheader('Line Chart')
                st.line_chart(chart_data)


           
            if chart_choice_sep == "Bar Chart":
                st.subheader('Bar Chart')
                st.bar_chart(chart_data)

            
            if chart_choice_sep == "Area Chart":
                st.subheader('Area Chart')
                st.area_chart(chart_data)

            # # Visualize the relationship between X and Y 
            # st.subheader(":chart_with_downwards_trend: Visualize the relationship between X and Y")
            # chart_menu = ["Line chart","Bar chart","Scatter chart"] 
            # chart_choice = st.selectbox("Chose the types of chart you want ",chart_menu)
            # chart_data = pd.concat([data_x, data_y], axis=1)
            # df_xy = st.checkbox('Show dataframe of this chart')
            # if df_xy: 
            #     st.write(chart_data)

            # if chart_choice == "Line chart":
            #     st.subheader("Line chart")
            #     fig = px.line(dataset_select,x=data_x,y=data_y,title=f'{select_colx} vs. {select_coly}')
            #     st.plotly_chart(fig)
            
            # elif chart_choice == "Bar chart":
            #     st.subheader("Bar chart")
            #     fig = px.bar(dataset_select,x=data_x,y=data_y,title=f'{select_colx} vs. {select_coly}')
            #     st.plotly_chart(fig)

            # elif chart_choice == "Scatter chart":
            #     st.subheader("Scatter chart")
            #     fig = px.scatter(x=data_x,y=data_y,title=f'{select_colx} vs. {select_coly}')
            #     st.plotly_chart(fig)
            
            
            
            

        # Select 
        # st.subheader(':file_folder: Select Multiple dataset that you want to analysis ')
        # path = "/Users/chenyan/Documents/GitHub/streamlit_finance_chart/tempDir" + "/" + string
        # dataset_select = pd.read_csv(path)
        # this_dataset = st.checkbox('Show dataset')
        # if this_dataset: 
        #     st.write(dataset_select)

    # Part2 
    # Left side - Navigation bar
    st.sidebar.subheader(":closed_book: Whole Information ")

    p2_menu = ["Whole information Home","Segment","Range of Issue Date","Range of departure date","Total number","Total revenue","Others"] 
    p2_choice = st.sidebar.selectbox("Part2 : Whole information",p2_menu)
    # Navigation 
    if p2_choice == "Whole information Home":
        st.subheader("Whole information")
    elif p2_choice == "Range of Issue Date":
        st.subheader("Range of Issue Date")
    elif p2_choice == "Range of departure date":
        st.subheader("Range of departure date")
    elif p2_choice == "Total number":
        st.subheader("Total number")
    elif p2_choice == "Total revenue":
        st.subheader("Total revenue")
    elif p2_choice == "Others":
        st.subheader("Others")

    # Part3 Segment (for SIN-PER and PER-SIN)
    # Left side - Navigation bar
    st.sidebar.subheader(":green_book: Segment")
    p3_menu = ["Segment","Range of Issue Date","Range of departure date","Total number","Total revenue","Others"] 
    p3_choice = st.sidebar.selectbox("Part3 : Segment (for SIN-PER and PER-SIN)",p3_menu)
    # Navigation 
    if p3_choice == "Whole information Home":
        st.subheader("Whole information")
    elif p3_choice == "Range of Issue Date":
        st.subheader("Range of Issue Date")
    elif p3_choice == "Range of departure date":
        st.subheader("Range of departure date")
    elif p3_choice == "Total number":
        st.subheader("Total number")
    elif p3_choice == "Total revenue":
        st.subheader("Total revenue")
    elif p3_choice == "Others":
        st.subheader("Others")


    # Part4
    # Left side - Navigation bar
    st.sidebar.subheader(":orange_book: Flight")
    p4_menu = ["Whole information Home","Segment","Range of Issue Date","Range of departure date","Total number","Total revenue","Others"] 
    p4_choice = st.sidebar.selectbox("Part4 : Flight ",p4_menu)
    # Navigation 
    if p4_choice == "Whole information Home":
        st.subheader("Whole information")
    elif p4_choice == "Range of Issue Date":
        st.subheader("Range of Issue Date")
    elif p4_choice == "Range of departure date":
        st.subheader("Range of departure date")
    elif p4_choice == "Total number":
        st.subheader("Total number")
    elif p4_choice == "Total revenue":
        st.subheader("Total revenue")
    elif p4_choice == "Others":
        st.subheader("Others")

    #Part5 
    st.sidebar.subheader(":clipboard: Generate Report")
    wholeinformation = st.sidebar.checkbox('Whole information ')
    if wholeinformation: 
        st.write("data_x")
    segment = st.sidebar.checkbox('Segment')
    if segment: 
        st.write("data_x")
    flight = st.sidebar.checkbox('Flight')
    if flight: 
        st.write("data_x")
    flight_given = st.sidebar.checkbox('Flight for a given departure date')
    if flight: 
        st.write("data_x")

if __name__ == '__main__':
    main()
