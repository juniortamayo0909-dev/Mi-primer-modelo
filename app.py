import streamlit as st
import joblib
import numpy as np

# Configuración
st.set_page_config(page_title="Clasificador Iris", page_icon="🌸")

st.title("🌸 Clasificador de Iris")
st.write("Prueba tus modelos entrenados (KNN y SVM)")

# Cargar modelos
@st.cache_resource
def cargar_modelos():
    knn = joblib.load("modelo_iris_knn.pkl")
    svm = joblib.load("modelo_iris_svm.pkl")
    return knn, svm

knn_model, svm_model = cargar_modelos()

# Selector de modelo
modelo_opcion = st.selectbox(
    "Selecciona el modelo:",
    ["KNN", "SVM"]
)

# Inputs de características
st.subheader("Ingresa las características:")

sepal_length = st.slider("Sepal Length", 4.0, 8.0, 5.5)
sepal_width  = st.slider("Sepal Width", 2.0, 4.5, 3.0)
petal_length = st.slider("Petal Length", 1.0, 7.0, 4.0)
petal_width  = st.slider("Petal Width", 0.1, 2.5, 1.0)

# Preparar datos
X = np.array([[sepal_length, sepal_width, petal_length, petal_width]])

# Predicción
if st.button("Predecir"):
    with st.spinner("Analizando..."):
        if modelo_opcion == "KNN":
            pred = knn_model.predict(X)
        else:
            pred = svm_model.predict(X)

    clases = ["Setosa", "Versicolor", "Virginica"]

    resultado = clases[int(pred[0])]
    
    st.success(f"🌼 Predicción: {resultado}")
    st.balloons()
