import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

st.write("""
# Сингулярное разложение матрицы на примере изображений
Каждую черно-белую картинку размером M x N можно представить как матрицу размером M x N, где каждое значение в строке или столбце будет в диапазоне от 0 до 255 и будет обозначать степень
градации серого от 0 - черный, до 255 - белый. 

Что если нам нужно сократить объем хранимой информации, пусть и ценой потери качества? При этом изображение должно оставаться узнаваемым.

В этом может помочь [сингулярное разложение (SVD)](https://ru.wikipedia.org/wiki/%D0%A1%D0%B8%D0%BD%D0%B3%D1%83%D0%BB%D1%8F%D1%80%D0%BD%D0%BE%D0%B5_%D1%80%D0%B0%D0%B7%D0%BB%D0%BE%D0%B6%D0%B5%D0%BD%D0%B8%D0%B5)
а точнее его главное практическое применение - возможность приблизить исходную матрицу матрицей меньшего ранга.
Разложив исходную матрицу изображения на три матрицы - U, Sigma и V мы можем взять только первые k диагональных элементов (сингулярных значений) из матрицы Sigma, сохранив при этом 
основную информацию об изображении. 

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


uploaded_file = st.sidebar.file_uploader("Лучше ч/б, но и цветная не проблема - мы её обесцветим", type=["jpg"])
if uploaded_file is not None:
    st.write('''
    ### Ваша исходная картинка:
    ''')
    img_arr = plt.imread(uploaded_file)
    
    if len(img_arr.shape) == 3:
        img_arr = img_arr.mean(axis=2)

    fig, ax = plt.subplots(1,1)
    ax.imshow(img_arr, cmap='gray')
    st.pyplot(fig)

    V, sing_values, U = np.linalg.svd(img_arr)
    max_k = len(sing_values)
    square_diagonal_sigma = np.diag(sing_values)
    num_col = U.shape[0] - square_diagonal_sigma.shape[1]
    num_col = int(num_col)
    sigma = np.hstack((square_diagonal_sigma, np.zeros((square_diagonal_sigma.shape[0], num_col))))
    
    
    k_components = st.sidebar.slider('Количество сингулярных значений', 0, max_k ,50)
    
    st.write('''
    ### Картинка, сохраненная с использованием первых''', k_components, '''сингулярных значений из''', max_k)

    V3, sigma3, U3 = V[:, :k_components], sigma[:k_components, :k_components], U[:k_components, :]
    img_top = V3@sigma3@U3
    
    fig1, ax1 = plt.subplots(1,1)
    ax1.imshow(img_top, cmap='gray')
    st.pyplot(fig1)

    st.write('''### Размер исходных матриц:''')
    st.write('V = ', V.shape[0], 'x', V.shape[1], '=', V.shape[0]*V.shape[1], 'значений'  )
    st.write('Sigma = ', sigma.shape[0], 'x', sigma.shape[1], '=', sigma.shape[0] * sigma.shape[1], 'значений'  )
    st.write('U = ', U.shape[0], 'x', U.shape[1], '=', U.shape[0] * U.shape[1], 'значений'  )

    st.write('''### Размер новых матриц:''')
    st.write('V = ', V3.shape[0], 'x', V3.shape[1], '=', V3.shape[0]*V3.shape[1], 'значений'  )
    st.write('Sigma = ', sigma3.shape[0], 'x', sigma3.shape[1], '=', sigma3.shape[0] * sigma3.shape[1], 'значений'  )
    st.write('U = ', U3.shape[0], 'x', U3.shape[1], '=', U3.shape[0] * U3.shape[1], 'значений'  )

else:
    pass
