import streamlit as st
from streamlit_tags import st_tags, st_tags_sidebar

st.write("# Code for streamlit tags")
maxtags = st.slider('Number of tags allowed?', 1, 10, 3, key='jfnkerrnfvikwqejn')

keywords = st_tags(
    label='# Enter Keywords:',
    text='Press enter to add more',
    value=['Zero', 'One', 'Two'],
    suggestions=['five', 'six', 'seven', 'eight', 'nine', 'three', 'eleven', 'ten', 'four'],
    maxtags=maxtags,
    key="aljnf")

for i in range(len(keywords)):
    n = keywords[i]
    st.write("Keywords Select(one by one)")
    st.write(n)

st.write("### Results:")
st.write(type(keywords))



st.sidebar.write("# Code for streamlit tags sidebar")

maxtags_sidebar = st.sidebar.slider('Number of tags allowed?', 1, 10, 3, key='ehikwegrjifbwreuk')


keyword = st_tags_sidebar(label='# Enter Keywords:',
                          text='Press enter to add more',
                          value=['Zero', 'One', 'Two'],
                          suggestions=['five', 'six', 'seven', 'eight', 'nine', 'three', 'eleven', 'ten', 'four'],
                          maxtags=maxtags_sidebar,
                          key="afrfae")

st.sidebar.write("### Results:")
st.sidebar.write((keyword))