import streamlit as st
import random
import time

# Configuración de página
st.set_page_config(page_title="Practica Ecuaciones ", page_icon="🧮")

# Función para generar ecuación
def generar_ecuacion():
    a = random.randint(1, 10)
    b = random.randint(-10, 10)
    x = random.randint(-10, 10)

    resultado = a * x + b
    ecuacion = f"{a}x + ({b}) = {resultado}"

    return ecuacion, x

# Estado
if "ecuacion" not in st.session_state:
    st.session_state.ecuacion, st.session_state.respuesta = generar_ecuacion()

# UI
st.title("🧮 Generador de Ecuaciones")
st.write("Resuelve la siguiente ecuación:")

st.subheader(st.session_state.ecuacion)

# Input
respuesta_usuario = st.number_input("Valor de x:", step=1)

col1, col2 = st.columns(2)

# Verificar respuesta
with col1:
    if st.button("Verificar"):
        with st.spinner("Verificando..."):
            time.sleep(1.2)  # animación de carga

        if respuesta_usuario == st.session_state.respuesta:
            st.balloons()  # 🎈 animación
            st.success("¡Correcto! 🎉")
        else:
            st.error(f"Incorrecto 😢. Era: {st.session_state.respuesta}")

# Nueva ecuación
with col2:
    if st.button("Nueva ecuación"):
        st.session_state.ecuacion, st.session_state.respuesta = generar_ecuacion()
        st.rerun()
