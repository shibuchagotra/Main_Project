import streamlit as st

st.set_page_config(
    page_title="Air Quality Dashboard",
    page_icon="🌍",
    layout="wide"
)

st.title("🌍 Air Quality Prediction & Analysis Dashboard")
st.markdown("""
Welcome to the **Air Quality Monitoring System**!

Use the sidebar to navigate:
- 🔮 **Prediction:** Forecast next-day pollutant levels (PM2.5, PM10,O3)
- 🤖 **Analysis Chatbot:** Ask questions about your air quality data

---
""")

st.image("https://cdn-icons-png.flaticon.com/512/2904/2904976.png", width=200)
