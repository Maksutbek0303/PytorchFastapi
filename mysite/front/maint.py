import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if ROOT not in sys.path:
    sys.path.insert(0,str(ROOT))

import streamlit as st
from mysite.front.front_numbers import check_number
from mysite.front.clothes_front import check_image_clothes
from mysite.front.CIFAR_front import check_image_ci

with st.sidebar:
    name = st.radio(label='Models : ', options=['Info', 'Checking numbers', 'Guessing clothes', 'CIFAR10-classifier'])
if name == 'Info':
    st.title('Добро пожаловать')
    st.write('Checking numbers - Проверка цифр')
    st.write('Guessing clothes - Угадывание одежды')
    st.write('CIFAR10-classifier')

elif name == 'Checking numbers':
    check_number()

elif name == 'Guessing clothes':
    check_image_clothes()

elif name == 'CIFAR10-classifier':
    check_image_ci()

