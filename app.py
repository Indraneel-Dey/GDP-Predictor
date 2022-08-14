import streamlit as st
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
import pickle

gdp_regressor, sensex_regressor = pickle.load(open('gdp.pkl', 'rb')), pickle.load(open('sensex.pkl', 'rb'))
combined_regressor = pickle.load(open('combined.pkl', 'rb'))

st.write('Indraneel Dey')
st.write('Indian Institute of Technology, Madras')
st.title('GDP-SENSEX Predictor')
st.text('You can get the GDP and SENSEX forecast of a year')
st.write('Input the year below to get the forecast of GDP in billion USD and opening SENSEX of that year')
year = int(st.text_input('Year', '2024'))
L, M = [], []
M.append(year)
L.append(M)
year_ = PolynomialFeatures(degree=2).fit_transform(L)
if st.button('Forecast'):
    col1, col2 = st.columns(2)
    with col1:
        st.header('SENSEX')
        st.text(sensex_regressor.predict(year_)[0])
    with col2:
        st.header('GDP')
        st.text(gdp_regressor.predict(year_)[0])

st.text('You can also predict the GDP of a hypothetical year given the opening SENSEX')
st.write('Input the opening SENSEX of the year to get GDP in billion USD of that year')
sensex_input = float(st.text_input('SENSEX', '50000'))
N, X = [], []
X.append(sensex_input)
N.append(X)
if st.button('Predict'):
    st.header('GDP')
    st.text(combined_regressor.predict(N)[0])
