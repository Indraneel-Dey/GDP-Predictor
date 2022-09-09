import streamlit as st
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
import pickle

gdp_regressor, year_regressor = pickle.load(open('gdp.pkl', 'rb')), pickle.load(open('year.pkl', 'rb'))

st.write('Indraneel Dey')
st.write('Indian Institute of Technology, Madras')
st.title('GDP Predictor')

with st.sidebar:
    s = st.radio('Select what you want to do', ('Forecast GDP (India) of a year', 'Predict year for target GDP (India)'))

if s == 'Forecast GDP (India) of a year':
    st.write('Input the year to get the forecast of GDP in billion USD')
    year = st.number_input('Year', min_value=2003)
    L, M = [], []
    M.append(year)
    L.append(M)
    year_ = PolynomialFeatures(degree=2).fit_transform(L)
    if st.button('Forecast'):
        st.header('GDP')
        st.text(gdp_regressor.predict(year_)[0])

if s == 'Predict year for target GDP (India)':
    st.write('Input the target GDP in billion USD')
    gdp_input = st.number_input('GDP', min_value=750.0)
    N, X = [], []
    X.append(gdp_input)
    N.append(X)
    if st.button('Predict Year'):
        st.header('Year')
        st.text(int(year_regressor.predict(N)[0]))
