import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from PIL import Image


st.write("""
# Сингулярное разложение матрицы на примере изображений
Каждую черно-белую картинку размером M x N можно представить как матрицу размером M x N,
где каждое значение в строке или столбце будет в диапазоне от 0 до 255 и будет обозначать
степень градации серого от 0 - черный, до 255 - белый. 

Что если нам нужно сократить объем хранимой информации, пусть и ценой потери качества? 
При этом изображение должно оставаться узнаваемым.

В этом может помочь [сингулярное разложение (SVD)]
(https://ru.wikipedia.org/wiki/%D0%A1%D0%B8%D0%BD%D0%B3%D1%83%D0%BB%D1%8F%D1%80%D0%BD%D0%BE%D0%B5_%D1%80%D0%B0%D0%B7%D0%BB%D0%BE%D0%B6%D0%B5%D0%BD%D0%B8%D0%B5)
а точнее его главное практическое применение - возможность приблизить исходную матрицу 
матрицей меньшего ранга.
Разложив исходную матрицу изображения на три матрицы - U, Sigma и V мы можем взять 
только первые k диагональных элементов (сингулярных значений) из матрицы Sigma, сохранив 
при этом основную информацию об изображении. 

Размер хранимой информации может сократиться очень существенно.

Давайте попробуем! 
""")
st.caption('''
*Для этого загрузите изображение на панели слева*.
''')

st.sidebar.header('Загрузите картинку')

st.sidebar.markdown("""
Можно драг-н-дроп 
""")


uploaded_file = st.sidebar.file_uploader(
    "Лучше ч/б, но и цветная не проблема - мы её обесцветим", 
    type=["jpg", "jpeg", "png"]
    )


if uploaded_file is not None:

    # Читаем с помощью PIL и сразу переводим в grayscale
    # без усреднения измерений
    img_arr = np.array(Image.open(uploaded_file).convert('L'))

    # делаем разложение сразу для выяснения max_k
    V, sing_values, U = np.linalg.svd(img_arr) 
    max_k = len(sing_values)

    # оформляем по-другому, чтобы не превышало максимальной длины строки по PEP
    k_components = st.sidebar.slider(
        label='Количество сингулярных значений', 
        min_value=1, 
        max_value=len(sing_values), 
        value=50
    )

    # делаем две колонки для наглядности
    col1, col2 = st.columns(2)

    with col1:
        st.write('''
        Исходная картинка:
        ''')
        fig, ax = plt.subplots(1,1)
        ax.imshow(img_arr, cmap='gray')
        ax.axis('off')
        st.pyplot(fig)
        

    with col2:
        square_diagonal_sigma = np.diag(sing_values)
        num_col = U.shape[0] - square_diagonal_sigma.shape[1]
        num_col = int(num_col)
        sigma = np.hstack(
            (square_diagonal_sigma, np.zeros((square_diagonal_sigma.shape[0], num_col)))
        )
        
        st.write(
            k_components, '''сингулярных значений из''', max_k)

        V3,     = V[:, :k_components], 
        sigma3  = sigma[:k_components, :k_components], 
        U3      = U[:k_components, :]
        img_top = V3 @ sigma3 @ U3
        
        fig_result, ax_result = plt.subplots(1,1)
        ax_result.imshow(img_top[0], cmap='gray')
        ax_result.axis('off')
        st.pyplot(fig_result)

    st.write('''#### Размер исходных матриц:''')
    st.write('V = ', V.shape[0], 'x', V.shape[1], '=', V.shape[0]*V.shape[1], 'значений'  )
    st.write('Sigma = ', sigma.shape[0], 'x', sigma.shape[1], '=', sigma.shape[0] * sigma.shape[1], 'значений'  )
    st.write('U = ', U.shape[0], 'x', U.shape[1], '=', U.shape[0] * U.shape[1], 'значений'  )
    st.write('''#### Размер новых матриц:''')
    st.write('V = ', V3.shape[0], 'x', V3.shape[1], '=', V3.shape[0]*V3.shape[1], 'значений'  )
    st.write('Sigma = ', sigma3[0].shape[0], 'x', sigma3[0].shape[1], '=', sigma3[0].shape[0] * sigma3[0].shape[1], 'значений'  )
    st.write('U = ', U3.shape[0], 'x', U3.shape[1], '=', U3.shape[0] * U3.shape[1], 'значений'  )
