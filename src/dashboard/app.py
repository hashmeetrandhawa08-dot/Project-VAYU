# ==========================================================
# PROJECT VAYU 2.0
# PART 1
# ==========================================================

import os
import joblib
import requests
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import google.generativeai as genai

from dotenv import load_dotenv
from geopy.geocoders import Nominatim
from streamlit_option_menu import option_menu
from streamlit_autorefresh import st_autorefresh

# ==========================================================
# PAGE CONFIG
# ==========================================================

st.set_page_config(
    page_title="Project VAYU",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================================================
# AUTO REFRESH
# ==========================================================

st_autorefresh(
    interval=60000,
    key="refresh"
)

# ==========================================================
# LOAD ENV
# ==========================================================

load_dotenv()

OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# ==========================================================
# GEMINI
# ==========================================================

genai.configure(api_key=GEMINI_API_KEY)

gemini = genai.GenerativeModel("gemini-2.5-flash")

# ==========================================================
# PATHS
# ==========================================================

MODEL_PATH = "src/models/air_quality_model.pkl"
DATA_PATH = "data/processed/clean_air_quality.csv"

# ==========================================================
# LOAD MODEL
# ==========================================================

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

model = load_model()

# ==========================================================
# LOAD DATA
# ==========================================================

@st.cache_data
def load_data():

    df = pd.read_csv(DATA_PATH)

    df["Date"] = pd.to_datetime(df["Date"])

    return df

df = load_data()

# ==========================================================
# MODEL FEATURE ORDER
# ==========================================================

feature_order = [
    "PM2.5",
    "PM10",
    "NO",
    "NO2",
    "NH3",
    "CO",
    "SO2",
    "O3"
]

# ==========================================================
# GEOLOCATION
# ==========================================================

geolocator = Nominatim(
    user_agent="project_vayu"
)

@st.cache_data
def get_coordinates(city):

    try:

        location = geolocator.geocode(
            f"{city}, India"
        )

        if location:

            return (
                location.latitude,
                location.longitude
            )

    except:

        pass

    return None, None

# ==========================================================
# WEATHER API
# ==========================================================

def get_weather(city):

    lat, lon = get_coordinates(city)

    if lat is None:
        return None

    url = (
        "https://api.openweathermap.org/data/2.5/weather"
        f"?lat={lat}"
        f"&lon={lon}"
        f"&appid={OPENWEATHER_API_KEY}"
        "&units=metric"
    )

    response = requests.get(url)

    if response.status_code != 200:
        return None

    return response.json()

# ==========================================================
# AIR POLLUTION API
# ==========================================================

def get_air(city):

    lat, lon = get_coordinates(city)

    if lat is None:
        return None

    url = (
        "https://api.openweathermap.org/data/2.5/air_pollution"
        f"?lat={lat}"
        f"&lon={lon}"
        f"&appid={OPENWEATHER_API_KEY}"
    )

    response = requests.get(url)

    if response.status_code != 200:
        return None

    return response.json()

# ==========================================================
# AQI CATEGORY
# ==========================================================

def get_aqi_category(aqi):

    if aqi <= 50:
        return "🟢 Good"

    elif aqi <= 100:
        return "🟡 Satisfactory"

    elif aqi <= 200:
        return "🟠 Moderate"

    elif aqi <= 300:
        return "🔴 Poor"

    elif aqi <= 400:
        return "🟣 Very Poor"

    return "⚫ Severe"

# ==========================================================
# HEALTH RECOMMENDATION
# ==========================================================

def health_recommendation(aqi):

    if aqi <= 50:

        return [
            "✅ Excellent air quality.",
            "Outdoor exercise is safe.",
            "No precautions required."
        ]

    elif aqi <= 100:

        return [
            "🙂 Air quality is satisfactory.",
            "Sensitive people should reduce prolonged exposure."
        ]

    elif aqi <= 200:

        return [
            "😷 Moderate pollution.",
            "Wear a mask outdoors.",
            "Avoid heavy exercise."
        ]

    elif aqi <= 300:

        return [
            "⚠ Poor air quality.",
            "Avoid outdoor activity.",
            "Wear an N95 mask."
        ]

    elif aqi <= 400:

        return [
            "🚨 Very Poor air quality.",
            "Stay indoors.",
            "Use an air purifier if available."
        ]

    return [
        "☠ Severe pollution.",
        "Avoid all outdoor activity.",
        "Follow emergency precautions."
    ]
    # ==========================================================
# PROJECT VAYU 2.0
# PART 2
# SIDEBAR + DASHBOARD
# ==========================================================

# ----------------------------------------------------------
# SIDEBAR
# ----------------------------------------------------------

with st.sidebar:

    st.title("🌍 Project VAYU")

    st.markdown("AI Powered Air Quality Intelligence Platform")

    page = option_menu(
        menu_title=None,
        options=[
            "Dashboard",
            "Analytics",
            "Ask VAYU"
        ],
        icons=[
            "house",
            "bar-chart",
            "robot"
        ],
        default_index=0
    )

    city = st.selectbox(
        "📍 Select City",
        sorted(df["City"].unique())
    )

# ==========================================================
# LOAD LIVE DATA
# ==========================================================

weather = get_weather(city)
air = get_air(city)

if weather is None:
    st.error("Unable to fetch weather.")
    st.stop()

if air is None:
    st.error("Unable to fetch AQI.")
    st.stop()

city_df = (
    df[df["City"] == city]
    .sort_values("Date")
    .copy()
)

latest = city_df.iloc[-1]

# ==========================================================
# WEATHER
# ==========================================================

temp = weather["main"]["temp"]
humidity = weather["main"]["humidity"]
pressure = weather["main"]["pressure"]
wind = weather["wind"]["speed"]
description = weather["weather"][0]["description"].title()

# ==========================================================
# LIVE POLLUTANTS
# ==========================================================

components = air["list"][0]["components"]

live_features = np.array([[
    components["pm2_5"],
    components["pm10"],
    components["no"],
    components["no2"],
    components["nh3"],
    components["co"],
    components["so2"],
    components["o3"]
]])

# ==========================================================
# AI PREDICTION
# ==========================================================

prediction = float(model.predict(live_features)[0])

prediction = max(0, min(500, prediction))

# ==========================================================
# OPENWEATHER AQI LEVEL
# ==========================================================

aqi_level = air["list"][0]["main"]["aqi"]

aqi_text = {
    1: "Good",
    2: "Fair",
    3: "Moderate",
    4: "Poor",
    5: "Very Poor"
}

# ==========================================================
# DASHBOARD
# ==========================================================

if page == "Dashboard":

    st.title("🌍 Project VAYU")

    st.caption(
        "Real-Time Air Quality Intelligence Dashboard"
    )

    st.markdown("---")

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.metric(
            "🤖 Predicted AQI",
            int(prediction)
        )

    with c2:
        st.metric(
            "🌡 Temperature",
            f"{temp:.1f} °C"
        )

    with c3:
        st.metric(
            "💧 Humidity",
            f"{humidity}%"
        )

    with c4:
        st.metric(
            "💨 Wind",
            f"{wind} m/s"
        )

    st.markdown("---")

    left, right = st.columns([2,1])

    with left:

        st.subheader("Current Weather")

        st.write(f"**Condition:** {description}")

        st.write(f"**Pressure:** {pressure} hPa")

        st.write(f"**Temperature:** {temp:.1f} °C")

        st.write(f"**Humidity:** {humidity}%")

        st.write(f"**Wind Speed:** {wind} m/s")

    with right:

        st.subheader("Air Quality")

        st.success(get_aqi_category(prediction))

        st.info(f"OpenWeather AQI Level: {aqi_level}/5 ({aqi_text[aqi_level]})")

    st.markdown("---")

    st.subheader("Live Pollutants")

    cols = st.columns(4)

    pollutants = [
        ("PM2.5", components["pm2_5"]),
        ("PM10", components["pm10"]),
        ("NO", components["no"]),
        ("NO₂", components["no2"]),
        ("NH₃", components["nh3"]),
        ("CO", components["co"]),
        ("SO₂", components["so2"]),
        ("O₃", components["o3"])
    ]

    for i, (name, value) in enumerate(pollutants):

        cols[i % 4].metric(
            name,
            round(value,2)
        )

    st.markdown("---")

    st.subheader("❤️ AI Health Recommendation")

    for advice in health_recommendation(prediction):

        st.write(advice)

    st.markdown("---")

    st.subheader("🤖 AI Summary")

    dominant = max(components, key=components.get).upper()

    st.write(f"""
The AI predicts an AQI of **{int(prediction)}**, which falls under **{get_aqi_category(prediction)}**.

Current weather is **{description}** with a temperature of **{temp:.1f}°C**.

The dominant pollutant is **{dominant}**.

These recommendations are generated using your trained LightGBM model combined with live OpenWeather environmental data.
""")
    # ==========================================================
# PROJECT VAYU 2.0
# PART 3
# ANALYTICS
# ==========================================================

if page == "Analytics":

    st.title("📊 Analytics Dashboard")

    st.markdown("---")

    st.subheader(f"Air Quality Analysis - {city}")

    c1, c2, c3 = st.columns(3)

    with c1:
        st.metric(
            "Average AQI",
            round(city_df["AQI"].mean(),2)
        )

    with c2:
        st.metric(
            "Maximum AQI",
            round(city_df["AQI"].max(),2)
        )

    with c3:
        st.metric(
            "Minimum AQI",
            round(city_df["AQI"].min(),2)
        )

# ==========================================================
# AQI HISTORY
# ==========================================================

    st.markdown("---")

    st.subheader("Historical AQI")

    fig = px.line(
        city_df,
        x="Date",
        y="AQI",
        markers=True
    )

    fig.update_layout(
        xaxis_title="Date",
        yaxis_title="AQI",
        template="plotly_white"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

# ==========================================================
# POLLUTANT TREND
# ==========================================================

    st.markdown("---")

    st.subheader("Pollutant Trend")

    pollutant = st.selectbox(

        "Choose Pollutant",

        [
            "PM2.5",
            "PM10",
            "NO",
            "NO2",
            "NH3",
            "CO",
            "SO2",
            "O3"
        ]

    )

    fig = px.line(

        city_df,

        x="Date",

        y=pollutant,

        markers=True

    )

    st.plotly_chart(

        fig,

        use_container_width=True

    )

# ==========================================================
# CURRENT POLLUTANTS
# ==========================================================

    st.markdown("---")

    st.subheader("Current Live Pollutants")

    poll_df = pd.DataFrame({

        "Pollutant":[
            "PM2.5",
            "PM10",
            "NO",
            "NO2",
            "NH3",
            "CO",
            "SO2",
            "O3"
        ],

        "Value":[
            components["pm2_5"],
            components["pm10"],
            components["no"],
            components["no2"],
            components["nh3"],
            components["co"],
            components["so2"],
            components["o3"]
        ]

    })

    fig = px.bar(

        poll_df,

        x="Pollutant",

        y="Value",

        color="Value",

        text_auto=".2f"

    )

    st.plotly_chart(

        fig,

        use_container_width=True

    )

# ==========================================================
# MONTHLY AQI
# ==========================================================

    st.markdown("---")

    temp_df = city_df.copy()

    temp_df["Month"] = temp_df["Date"].dt.strftime("%b")

    monthly = (

        temp_df

        .groupby("Month")["AQI"]

        .mean()

        .reset_index()

    )

    st.subheader("Monthly Average AQI")

    fig = px.bar(

        monthly,

        x="Month",

        y="AQI",

        color="AQI"

    )

    st.plotly_chart(

        fig,

        use_container_width=True

    )

# ==========================================================
# CORRELATION
# ==========================================================

    st.markdown("---")

    st.subheader("Feature Correlation")

    corr = city_df.corr(numeric_only=True)

    fig = px.imshow(

        corr,

        text_auto=True,

        aspect="auto",

        color_continuous_scale="RdBu"

    )

    st.plotly_chart(

        fig,

        use_container_width=True

    )

# ==========================================================
# CITY LOCATION
# ==========================================================

    st.markdown("---")

    st.subheader("City Location")

    lat, lon = get_coordinates(city)

    if lat is not None:

        map_df = pd.DataFrame({

            "lat":[lat],

            "lon":[lon]

        })

        st.map(map_df)

# ==========================================================
# DOWNLOAD
# ==========================================================

    st.markdown("---")

    st.subheader("Download Data")

    csv = city_df.to_csv(index=False)

    st.download_button(

        label="📥 Download CSV",

        data=csv,

        file_name=f"{city}_AQI.csv",

        mime="text/csv"

    )
    # ==========================================================
# PROJECT VAYU 2.0
# PART 4
# ASK VAYU
# ==========================================================

if page == "Ask VAYU":

    st.title("🤖 Ask VAYU")

    st.caption("AI Environmental Assistant")

    st.markdown("---")

    c1, c2, c3 = st.columns(3)

    with c1:
        st.metric(
            "Predicted AQI",
            int(prediction)
        )

    with c2:
        st.metric(
            "Temperature",
            f"{temp:.1f} °C"
        )

    with c3:
        st.metric(
            "Humidity",
            f"{humidity}%"
        )

    st.markdown("---")

    st.subheader("Suggested Questions")

    suggestions = [

        "Is it safe to go outside today?",

        "Can children play outside?",

        "Should I wear a mask?",

        "Why is AQI high?",

        "How can pollution be reduced?",

        "Explain today's air quality."

    ]

    for q in suggestions:

        if st.button(q):

            st.session_state["question"] = q

    question = st.text_area(

        "Ask VAYU",

        value=st.session_state.get("question",""),

        height=120,

        placeholder="Ask anything about today's air quality..."

    )

    if st.button("🚀 Ask AI"):

        if question.strip() == "":

            st.warning("Please enter a question.")

        else:

            dominant = max(
                components,
                key=components.get
            ).upper()

            prompt = f"""
You are Project VAYU AI.

Current City:
{city}

Current Weather

Temperature:
{temp}

Humidity:
{humidity}

Pressure:
{pressure}

Wind Speed:
{wind}

Weather:
{description}

Predicted AQI:
{int(prediction)}

AQI Category:
{get_aqi_category(prediction)}

Dominant Pollutant:
{dominant}

Pollutants

PM2.5 = {components['pm2_5']}

PM10 = {components['pm10']}

NO = {components['no']}

NO2 = {components['no2']}

NH3 = {components['nh3']}

CO = {components['co']}

SO2 = {components['so2']}

O3 = {components['o3']}

User Question:

{question}

Rules:

1. Explain in simple English.

2. Maximum 180 words.

3. Give practical advice.

4. Mention health precautions.

5. Mention dominant pollutant if relevant.

6. Answer only using the provided environmental data.
"""

            with st.spinner("VAYU is analyzing..."):

                response = gemini.generate_content(prompt)

            st.markdown("---")

            st.subheader("🤖 AI Response")

            st.write(response.text)

            st.download_button(

                "📄 Download Response",

                response.text,

                file_name="vayu_response.txt"

            )

# ==========================================================
# FOOTER
# ==========================================================

st.markdown("---")

st.markdown(
"""
<center>

## 🌍 Project VAYU

AI Powered Urban Air Quality Intelligence Platform

Built using

✅ LightGBM

✅ Streamlit

✅ OpenWeather API

✅ Google Gemini

© 2026

</center>
""",
unsafe_allow_html=True
)