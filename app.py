# plik: app.py
import streamlit as st
import yfinance as yf
from prophet import Prophet
from prophet.plot import plot_plotly
import plotly.graph_objects as go
import pandas as pd
import openai

# ====== Ustawienia OpenAI (jeśli chcesz raport AI) ======
openai.api_key = "TWÓJ_OPENAI_API_KEY"  # wstaw swój klucz

# ====== Tytuł aplikacji ======
st.title("Prototyp Aplikacji Giełdowej – Prognoza Dzienna")

# ====== Wybór akcji ======
ticker = st.text_input("Wpisz symbol akcji (np. AAPL, TSLA, MSFT):", "AAPL")
days = st.slider("Liczba dni prognozy:", min_value=1, max_value=7, value=1)

# ====== Pobranie danych ======
@st.cache_data
def get_data(symbol):
    data = yf.download(symbol, period="2y", interval="1d")
    data.reset_index(inplace=True)
    return data

data = get_data(ticker)

st.subheader("Dane historyczne")
st.line_chart(data['Close'])

# ====== Przygotowanie danych dla Prophet ======
df_prophet = data[['Date', 'Close']].rename(columns={'Date':'ds', 'Close':'y'})

# ====== Model i prognoza ======
model = Prophet(daily_seasonality=True)
model.fit(df_prophet)
future = model.make_future_dataframe(periods=days)
forecast = model.predict(future)

st.subheader(f"Prognoza na {days} dzień/dni")
fig1 = plot_plotly(model, forecast)
st.plotly_chart(fig1)

# ====== Prognoza ceny na jutro ======
predicted_price = forecast['yhat'].iloc[-days]
st.write(f"Przewidywana cena akcji {ticker} za {days} dzień/dni: **{predicted_price:.2f} USD**")

# ====== Raport AI (opcjonalnie) ======
if st.button("Generuj raport AI"):
    prompt = f"""
    Napisz krótkie podsumowanie i prognozę dla akcji {ticker}.
    Ostatnie ceny: {data['Close'].tail(10).tolist()}.
    Przewidywana cena na następny dzień: {predicted_price:.2f} USD.
    Podaj możliwe czynniki wzrostu lub spadku.
    """
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[{"role":"user","content":prompt}],
        temperature=0.7,
        max_tokens=150
    )
    st.subheader("Raport AI")
    st.write(response['choices'][0]['message']['content'])