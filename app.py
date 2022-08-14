import streamlit as st
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
import pickle

gdp_regressor, year_regressor = pickle.load(open('gdp.pkl', 'rb')), pickle.load(open('year.pkl', 'rb'))

st.write('Indraneel Dey')
st.write('Indian Institute of Technology, Madras')
st.title('GDP Predictor')
st.text('You can get the GDP forecast (India) of a year')
st.write('Input the year below to get the forecast of GDP in billion USD')
year = int(st.text_input('Year', '2024'))
L, M = [], []
M.append(year)
L.append(M)
year_ = PolynomialFeatures(degree=2).fit_transform(L)
if st.button('Forecast'):
    st.header('GDP')
    st.text(gdp_regressor.predict(year_)[0])

st.text('You can also predict the year a target GDP will be achieved')
st.write('Input the target GDP in billion USD')
gdp_input = float(st.text_input('GDP', '4000'))
N, X = [], []
X.append(gdp_input)
N.append(X)
if st.button('Predict Year'):
    st.header('Year')
    st.text(int(year_regressor.predict(N)[0]))
