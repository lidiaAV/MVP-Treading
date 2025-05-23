import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier

st.set_page_config(page_title="Resultados", layout="wide")
st.title("📊 Resultados de hoy")

# Inputs en la barra lateral
spread = st.sidebar.number_input("Spread (en puntos)", min_value=0.0, value=0.0, step=0.01, key="spread_input")
comision = st.sidebar.number_input("Comisión fija (en €)", min_value=0.0, value=0.0, step=0.1, key="comision_input")

st.sidebar.markdown("---")
st.sidebar.markdown("### ℹ️ Costes aplicados")
st.sidebar.markdown(f"- Spread actual: `{spread}` puntos")
st.sidebar.markdown(f"- Comisión actual: `{comision} €`")

# Verificar si hay datos
if "df" not in st.session_state:
    st.warning("🔁 Carga los datos desde la página principal primero.")
    st.stop()

df = st.session_state.df.copy()

# Inicializar diccionario para guardar análisis por ticker
if "dfs_por_ticker" not in st.session_state:
    st.session_state.dfs_por_ticker = {}

# === Si hay múltiples activos ===
if "Ticker" in df.columns:
    st.subheader("📋 Resumen de señales de hoy por activo")

    resumen = []
    for ticker in df["Ticker"].unique():
        df_t = df[df["Ticker"] == ticker].copy()

        # Año máximo en datos de este ticker
        ultimo_año = df_t['Fecha'].dt.year.max()

        # Entrenar solo con datos hasta el año anterior
        df_train = df_t[df_t['Fecha'].dt.year < ultimo_año].copy()
        if df_train.empty or len(df_train) < 10:
            continue

        # Calcular indicadores para df_train
        delta = df_train['Último'].diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.rolling(window=5).mean()
        avg_loss = loss.rolling(window=5).mean()
        rs = avg_gain / avg_loss
        df_train['RSI'] = 100 - (100 / (1 + rs))

        df_train['Target'] = ((df_train['Último'].shift(-1) - (df_train['Último'] + spread)) > 0).astype(int)
        df_train.dropna(subset=['RSI', 'Vol.', 'Target'], inplace=True)

        if len(df_train) < 10:
            continue

        X = df_train[['RSI', 'Vol.']]
        y = df_train['Target']
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X, y)

        # Ahora calculamos indicadores y predecimos en todo df_t (incluye último año)
        delta_all = df_t['Último'].diff()
        gain_all = delta_all.clip(lower=0)
        loss_all = -delta_all.clip(upper=0)
        avg_gain_all = gain_all.rolling(window=5).mean()
        avg_loss_all = loss_all.rolling(window=5).mean()
        rs_all = avg_gain_all / avg_loss_all
        df_t['RSI'] = 100 - (100 / (1 + rs_all))

        df_t['Target'] = ((df_t['Último'].shift(-1) - (df_t['Último'] + spread)) > 0).astype(int)
        df_t.dropna(subset=['RSI', 'Vol.', 'Target'], inplace=True)

        df_t['Predicción'] = model.predict(df_t[['RSI', 'Vol.']])
        df_t['Señal'] = df_t['Predicción'].apply(lambda x: '📈 Comprar' if x == 1 else '📉 Vender')

        ultima = df_t.iloc[-1]

        resumen.append({
            "Ticker": ticker,
            "Fecha": ultima["Fecha"].date(),
            "Precio": round(ultima["Último"], 2),
            "RSI": round(ultima["RSI"], 2),
            "Volumen": round(ultima["Vol."], 0),
            "Señal": ultima["Señal"]
        })

        # Guardar dataframe procesado en session_state
        st.session_state.dfs_por_ticker[ticker] = df_t

    df_resumen = pd.DataFrame(resumen)
    st.dataframe(df_resumen)

    # Descargar resumen como CSV
    resumen_csv = df_resumen.to_csv(index=False).encode("utf-8")
    st.download_button("📥 Descargar resumen de señales", resumen_csv, "resumen_senales.csv", "text/csv")

    # Seleccionar activo para análisis
    ticker_sel = st.selectbox("Selecciona un activo para ver el análisis detallado:", df["Ticker"].unique())

    if ticker_sel in st.session_state.dfs_por_ticker:
        df = st.session_state.dfs_por_ticker[ticker_sel]
    else:
        st.warning("No se ha generado el análisis para este activo todavía.")
        st.stop()

# === Caso de un único activo (sin columna Ticker) ===
else:
    st.subheader("📋 Resumen de señales de hoy")

    ultimo_año = df['Fecha'].dt.year.max()
    df_train = df[df['Fecha'].dt.year < ultimo_año].copy()

    delta = df_train['Último'].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=5).mean()
    avg_loss = loss.rolling(window=5).mean()
    rs = avg_gain / avg_loss
    df_train['RSI'] = 100 - (100 / (1 + rs))

    df_train['Target'] = ((df_train['Último'].shift(-1) - (df_train['Último'] + spread)) > 0).astype(int)
    df_train.dropna(subset=['RSI', 'Vol.', 'Target'], inplace=True)

    if len(df_train) < 10:
        st.warning("No hay suficientes datos para entrenar el modelo.")
        st.stop()

    X = df_train[['RSI', 'Vol.']]
    y = df_train['Target']
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)

    # Predecir en todo df (incluyendo último año)
    delta_all = df['Último'].diff()
    gain_all = delta_all.clip(lower=0)
    loss_all = -delta_all.clip(upper=0)
    avg_gain_all = gain_all.rolling(window=5).mean()
    avg_loss_all = loss_all.rolling(window=5).mean()
    rs_all = avg_gain_all / avg_loss_all
    df['RSI'] = 100 - (100 / (1 + rs_all))

    df['Target'] = ((df['Último'].shift(-1) - (df['Último'] + spread)) > 0).astype(int)
    df.dropna(subset=['RSI', 'Vol.', 'Target'], inplace=True)

    df['Predicción'] = model.predict(df[['RSI', 'Vol.']])
    df['Señal'] = df['Predicción'].apply(lambda x: '📈 Comprar' if x == 1 else '📉 Vender')

    ultima = df.iloc[-1]

    resumen = {
        "Fecha": ultima["Fecha"].date(),
        "Precio": round(ultima["Último"], 2),
        "RSI": round(ultima["RSI"], 2),
        "Volumen": round(ultima["Vol."], 0),
        "Señal": ultima["Señal"]
    }
    st.dataframe(pd.DataFrame([resumen]))

    st.session_state.dfs_por_ticker = {"_único": df}  # Guarda el df para simulación

# === Análisis detallado ===
st.markdown("## 📌 Señal para hoy")

dias = st.slider("¿Cuántos días mostrar?", 30, 180, 90)

df_viz = df.tail(dias)

fig, ax1 = plt.subplots(figsize=(12, 5))
line1, = ax1.plot(df_viz['Fecha'], df_viz['RSI'], color='blue', label='RSI')
ax1.axhline(70, color='red', linestyle='--')
ax1.axhline(30, color='green', linestyle='--')
ax1.set_ylabel('RSI')

ax2 = ax1.twinx()
line2, = ax2.plot(df_viz['Fecha'], df_viz['Último'], color='orange', label='Precio')
ax2.set_ylabel('Precio')

ax1.legend([line1, line2], ['RSI', 'Precio'], loc='upper left')
st.pyplot(fig)

# Tabla de señales recientes
st.subheader("🔮 Señales recientes")
st.dataframe(df[['Fecha', 'Último', 'RSI', 'Vol.', 'Predicción', 'Señal']].tail(10))

# Descargar CSV del análisis detallado
csv_out = df.to_csv(index=False).encode('utf-8')
st.download_button("📥 Descargar CSV del activo", csv_out, "predicciones.csv", "text/csv")
