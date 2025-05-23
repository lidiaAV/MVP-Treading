import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(page_title="Simulación", layout="wide")
st.title("📊 Simulación de Estrategia")

# Verificar si hay datos cargados y señales generadas
if 'df' not in st.session_state or st.session_state.df is None:
    st.warning("Por favor, sube un archivo CSV y entrena el modelo en la página principal.")
    st.stop()

df_total = st.session_state.df.copy()
if 'Señal' not in df_total.columns:
    st.error("⚠️ No se han generado señales todavía. Ve a la página principal y entrena el modelo.")
    st.stop()

# Selección de activo si hay múltiples
if 'Ticker' in df_total.columns:
    tickers = df_total['Ticker'].unique()
    ticker_seleccionado = st.selectbox("Selecciona el activo", tickers)
    df = df_total[df_total['Ticker'] == ticker_seleccionado].copy()
else:
    df = df_total.copy()

# Asegurar tipos correctos
df['Fecha'] = pd.to_datetime(df['Fecha'])
df['Señal'] = df['Señal'].astype(str)

# Inputs en la barra lateral
spread = st.sidebar.number_input("Spread (en puntos)", min_value=0.0, value=st.session_state.get("spread", 0.0), step=0.01, key="spread_sim")
comision = st.sidebar.number_input("Comisión fija (en €)", min_value=0.0, value=st.session_state.get("comision", 0.0), step=0.1, key="comision_sim")
capital_inicial = st.sidebar.number_input("Capital inicial (€)", min_value=100.0, value=1000.0, step=100.0, key="capital_inicial_sim")

st.session_state.spread = spread
st.session_state.comision = comision

# Mostrar info
st.sidebar.markdown("---")
st.sidebar.markdown("### ℹ️ Costes aplicados")
st.sidebar.markdown(f"- Spread actual: `{spread}` puntos")
st.sidebar.markdown(f"- Comisión actual: `{comision} €`")
st.sidebar.markdown(f"- Capital inicial: `{capital_inicial} €`")

# Selección de año
años_disponibles = sorted(df['Fecha'].dt.year.dropna().unique(), reverse=True)
año_seleccionado = st.selectbox("Selecciona el año para la simulación", años_disponibles)

df_sim = df[df['Fecha'].dt.year == año_seleccionado].reset_index(drop=True)

# Simulación con interés compuesto
capital = capital_inicial
en_posicion = False
precio_compra = 0
capital_evolucion = []
transacciones = []

for _, fila in df_sim.iterrows():
    señal = fila['Señal']
    precio = fila['Último']

    if not en_posicion and señal == '📈 Comprar':
        precio_compra = precio + spread
        fecha_compra = fila['Fecha']
        capital -= comision
        en_posicion = True

    elif en_posicion and señal == '📉 Vender':
        precio_venta = precio
        fecha_venta = fila['Fecha']
        beneficio = (precio_venta - precio_compra) / precio_compra * capital
        capital += beneficio
        capital -= comision

        transacciones.append({
            'Fecha Compra': fecha_compra.date(),
            'Precio Compra': round(precio_compra, 2),
            'Fecha Venta': fecha_venta.date(),
            'Precio Venta': round(precio_venta, 2),
            'Ganancia (€)': round(beneficio - comision, 2),
            'Capital Tras Venta (€)': round(capital, 2)
        })

        en_posicion = False

    capital_evolucion.append({'Fecha': fila['Fecha'], 'Capital': capital})

capital_final = round(capital, 2)

# Simulación sin interés compuesto
capital_fijo = capital_inicial
capital_actual_fijo = capital_inicial

for i in range(1, len(df_sim)):
    señal_anterior = df_sim.iloc[i - 1]['Señal']
    señal_actual = df_sim.iloc[i]['Señal']
    precio_entrada = df_sim.iloc[i - 1]['Último'] + spread
    precio_salida = df_sim.iloc[i]['Último'] - spread

    if señal_anterior == '📈 Comprar' and señal_actual == '📉 Vender':
        ganancia_pct = (precio_salida - precio_entrada) / precio_entrada
        ganancia_fija = capital_fijo * ganancia_pct - 2 * comision
        capital_actual_fijo += ganancia_fija

capital_final_fijo = round(capital_actual_fijo, 2)

# Buy & Hold y Buy & Max
precio_inicio = df_sim['Último'].iloc[0] + spread
precio_final = df_sim['Último'].iloc[-1]
precio_maximo = df_sim['Último'].max()

ganancia_hold = ((precio_final - precio_inicio) / precio_inicio) * capital_inicial - 2 * comision
capital_hold = capital_inicial + ganancia_hold

ganancia_max = ((precio_maximo - precio_inicio) / precio_inicio) * capital_inicial - 2 * comision
capital_max = capital_inicial + ganancia_max

# Resultados
st.subheader(f"📈 Evolución del Capital en {año_seleccionado}")

if transacciones:
    df_transacciones = pd.DataFrame(transacciones)
    st.dataframe(df_transacciones)

    st.markdown("### 💰 Resultados de la Simulación")
    st.success(f"🔁 Con interés compuesto: **{capital_final:,.2f} €**")
    st.info(f"➖ Sin interés compuesto: **{capital_final_fijo:,.2f} €**")

    st.subheader("📉 Comparativa: Buy & Hold, Buy & Max vs Estrategia")
    st.markdown(f"💼 **Buy & Hold** ({df_sim['Fecha'].iloc[0].date()} → {df_sim['Fecha'].iloc[-1].date()}): **{capital_hold:.2f} €**")
    st.markdown(f"🔝 **Buy & Max**: **{capital_max:.2f} €**")

    df_capital = pd.DataFrame(capital_evolucion)
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(df_capital['Fecha'], df_capital['Capital'], color='green')
    ax.set_title("Crecimiento del Capital (Compuesto)")
    ax.set_xlabel("Fecha")
    ax.set_ylabel("Capital (€)")
    st.pyplot(fig)
else:
    st.warning("⚠️ No se realizaron operaciones en el año seleccionado.")

# Botón para resetear
if st.button("❌ Quitar archivo cargado y reiniciar"):
    st.session_state.df = None
    st.rerun()
