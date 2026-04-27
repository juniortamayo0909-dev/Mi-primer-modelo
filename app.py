import streamlit as st
import joblib
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt

st.set_page_config(page_title="Dashboard Iris", page_icon="🌸", layout="wide")

st.title("🌸 Dashboard Clasificador Iris")
st.write("Comparación de modelos KNN vs SVM")

# Cargar modelos
@st.cache_resource
def cargar_modelos():
    knn = joblib.load("modelo_iris_knn.pkl")
    svm = joblib.load("modelo_iris_svm.pkl")
    return knn, svm

knn_model, svm_model = cargar_modelos()

# Dataset Iris
iris = load_iris()
X = iris.data
y = iris.target
clases = iris.target_names

df = pd.DataFrame(X, columns=iris.feature_names)
df["target"] = y

# Sidebar navegación
opcion = st.sidebar.selectbox(
    "Menú",
    ["Predicción", "Comparación modelos", "Dataset"]
)

# =========================
# 🔮 PREDICCIÓN
# =========================
if opcion == "Predicción":
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
        with st.spinner("Analizando..."):
            pred_knn = knn_model.predict(entrada)[0]
            pred_svm = svm_model.predict(entrada)[0]

        st.subheader("Resultados:")

        col1, col2 = st.columns(2)

        with col1:
            st.success(f"KNN: {clases[pred_knn]}")
            if hasattr(knn_model, "predict_proba"):
                probs = knn_model.predict_proba(entrada)[0]
                st.write("Probabilidades:", probs)

        with col2:
            st.success(f"SVM: {clases[pred_svm]}")
            if hasattr(svm_model, "predict_proba"):
                probs = svm_model.predict_proba(entrada)[0]
                st.write("Probabilidades:", probs)

        st.balloons()

# =========================
# 📊 COMPARACIÓN
# =========================
elif opcion == "Comparación modelos":
    st.header("📊 Evaluación de modelos")

    # Predicciones
    y_pred_knn = knn_model.predict(X)
    y_pred_svm = svm_model.predict(X)

    acc_knn = accuracy_score(y, y_pred_knn)
    acc_svm = accuracy_score(y, y_pred_svm)

    col1, col2 = st.columns(2)

    with col1:
        st.metric("Accuracy KNN", f"{acc_knn:.2f}")

    with col2:
        st.metric("Accuracy SVM", f"{acc_svm:.2f}")

    # Matriz de confusión
    st.subheader("Matriz de confusión")

    modelo = st.selectbox("Selecciona modelo", ["KNN", "SVM"])

    if modelo == "KNN":
        cm = confusion_matrix(y, y_pred_knn)
    else:
        cm = confusion_matrix(y, y_pred_svm)

    fig, ax = plt.subplots()
    ax.imshow(cm)
    ax.set_title(f"Matriz de Confusión - {modelo}")
    st.pyplot(fig)

# =========================
# 📁 DATASET
# =========================
else:
    st.header("📁 Dataset Iris")
    st.dataframe(df)

    st.subheader("Estadísticas")
    st.write(df.describe())
