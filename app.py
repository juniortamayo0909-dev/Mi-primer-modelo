import streamlit as st
import joblib
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt

# Configuración
st.set_page_config(page_title="Iris App", page_icon="🌸", layout="wide")

st.title("🌸 Clasificador Iris (KNN vs SVM)")
st.write("Prueba tus modelos entrenados")

# =========================
# CARGAR MODELOS
# =========================
@st.cache_resource
def cargar_modelos():
    knn = joblib.load("modelo_iris_knn.pkl")
    svm = joblib.load("modelo_iris_svm.pkl")
    return knn, svm

try:
    knn_model, svm_model = cargar_modelos()
except:
    st.error("❌ No se encontraron los modelos .pkl en la carpeta")
    st.stop()

# =========================
# DATASET
# =========================
iris = load_iris()
X = iris.data
y = iris.target
clases = iris.target_names

df = pd.DataFrame(X, columns=iris.feature_names)
df["target"] = y

# =========================
# MENÚ
# =========================
menu = st.sidebar.selectbox(
    "Opciones",
    ["Predicción", "Comparación", "Dataset"]
)

# =========================
# 🔮 PREDICCIÓN
# =========================
if menu == "Predicción":
    st.header("🔮 Predicción individual")

    col1, col2 = st.columns(2)

    with col1:
        sl = st.slider("Sepal Length", 4.0, 8.0, 5.5)
        sw = st.slider("Sepal Width", 2.0, 4.5, 3.0)

    with col2:
        pl = st.slider("Petal Length", 1.0, 7.0, 4.0)
        pw = st.slider("Petal Width", 0.1, 2.5, 1.0)

    entrada = np.array([[sl, sw, pl, pw]])

    if st.button("Predecir"):
        with st.spinner("Procesando..."):
            pred_knn = knn_model.predict(entrada)[0]
            pred_svm = svm_model.predict(entrada)[0]

        st.subheader("Resultados")

        col1, col2 = st.columns(2)

        with col1:
            st.success(f"KNN: {clases[pred_knn]}")
            if hasattr(knn_model, "predict_proba"):
                st.write("Probabilidades:", knn_model.predict_proba(entrada)[0])

        with col2:
            st.success(f"SVM: {clases[pred_svm]}")
            if hasattr(svm_model, "predict_proba"):
                st.write("Probabilidades:", svm_model.predict_proba(entrada)[0])

        st.balloons()

# =========================
# 📊 COMPARACIÓN
# =========================
elif menu == "Comparación":
    st.header("📊 Evaluación de modelos")

    y_pred_knn = knn_model.predict(X)
    y_pred_svm = svm_model.predict(X)

    acc_knn = accuracy_score(y, y_pred_knn)
    acc_svm = accuracy_score(y, y_pred_svm)

    col1, col2 = st.columns(2)

    col1.metric("Accuracy KNN", f"{acc_knn:.2f}")
    col2.metric("Accuracy SVM", f"{acc_svm:.2f}")

    st.subheader("Matriz de confusión")

    modelo = st.selectbox("Modelo", ["KNN", "SVM"])

    if modelo == "KNN":
        cm = confusion_matrix(y, y_pred_knn)
    else:
        cm = confusion_matrix(y, y_pred_svm)

    fig, ax = plt.subplots()
    ax.imshow(cm)
    ax.set_title(f"Matriz - {modelo}")
    st.pyplot(fig)

# =========================
# 📁 DATASET
# =========================
else:
    st.header("📁 Dataset Iris")
    st.dataframe(df)
    st.write(df.describe())
