from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import streamlit as st

try:
    import plotly.express as px
    import plotly.graph_objects as go
    PLOTLY_AVAILABLE = True
except Exception:
    PLOTLY_AVAILABLE = False

try:
    from groq import Groq
    GROQ_AVAILABLE = True
except Exception:
    GROQ_AVAILABLE = False

try:
    from sklearn.compose import ColumnTransformer
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder
    SKLEARN_AVAILABLE = True
except Exception:
    SKLEARN_AVAILABLE = False


# -----------------------------------------------------------------------------
# Basic settings
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Food Waste Rescue Radar",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="collapsed",
)

DATA_FILE = Path("food_waste_records.csv")

EVENT_TYPES = [
    "School cafeteria lunch",
    "Breakfast program",
    "After-school club",
    "Sports event",
    "Community event",
    "Fundraiser",
    "Food donation program",
]

LOCATIONS = [
    "Cafeteria",
    "Classroom",
    "Gym",
    "Library / common area",
    "Outdoor area",
    "Community hall",
    "Event venue",
]

MEAL_TIMES = ["Breakfast", "Lunch", "Snack", "Dinner", "All-day event"]
DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
WEATHER = ["Normal", "Sunny", "Cloudy", "Rainy", "Stormy", "Very hot", "Very cold"]
CONFIDENCE = ["High", "Medium", "Low"]
INTERVENTIONS = [
    "None yet",
    "Pre-order form",
    "Attendance confirmation",
    "Smaller first batch",
    "Menu preference survey",
    "Donation partner ready",
    "Compost plan",
    "Mixed intervention",
]

COLUMNS = [
    "Time",
    "Event Type",
    "Location",
    "Day",
    "Meal Time",
    "Meal / Food",
    "Menu Popularity",
    "Weather",
    "Attendance Confidence",
    "Expected Attendance",
    "Actual Attendance",
    "Food Prepared",
    "Leftover Portions",
    "Waste Rate",
    "Predicted Waste Rate",
    "Risk Score",
    "Risk Level",
    "Recommended Min",
    "Recommended Max",
    "Donation Route",
    "Batch Cooking",
    "Intervention",
    "Cost per Portion",
    "CO2e per Portion",
    "Cost Impact",
    "CO2e Impact",
    "Potential Meals Rescued",
    "Notes",
]


# -----------------------------------------------------------------------------
# Styling: clean environmental luxury
# -----------------------------------------------------------------------------
def css() -> None:
    st.markdown(
        """
        <style>
        :root {
            --cream: #FBF7EF;
            --cream-2: #F5F0E5;
            --paper: #FFFFFF;
            --paper-soft: #FCFAF5;
            --sage-50: #F2F7F1;
            --sage-100: #E6F0E4;
            --sage-200: #D5E4D2;
            --sage-300: #B9CEB4;
            --green-900: #0F2F22;
            --green-800: #153D2C;
            --green-700: #1E5B3B;
            --green-600: #277447;
            --green-500: #3B8B5A;
            --gold: #B8954A;
            --gold-2: #D5C08A;
            --ink: #173528;
            --ink-soft: #4A665A;
            --muted: #728377;
            --line: rgba(21, 61, 44, .13);
            --shadow: 0 20px 60px rgba(15, 47, 34, .09);
            --shadow-soft: 0 10px 28px rgba(15, 47, 34, .07);
        }

        html, body, [class*="css"] {
            font-family: Inter, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
        }

        .stApp {
            color: var(--ink);
            background:
                radial-gradient(circle at 8% 0%, rgba(184, 149, 74, .14), transparent 27%),
                radial-gradient(circle at 88% 8%, rgba(59, 139, 90, .16), transparent 32%),
                linear-gradient(135deg, #FBF7EF 0%, #F2F7F1 52%, #E6F0E4 100%);
        }

        [data-testid="stHeader"] {
            background: rgba(251, 247, 239, .78);
            backdrop-filter: blur(18px);
            border-bottom: 1px solid rgba(21, 61, 44, .08);
        }

        .block-container {
            max-width: 1180px;
            padding-top: 1.6rem;
            padding-bottom: 4rem;
        }

        h1 {
            color: var(--green-900);
            letter-spacing: -.055em;
            font-weight: 850 !important;
            line-height: .96 !important;
            font-size: clamp(2.75rem, 6vw, 5.6rem) !important;
            margin-bottom: .7rem !important;
        }

        h2 {
            color: var(--green-900);
            letter-spacing: -.035em;
            font-weight: 800 !important;
            font-size: clamp(1.45rem, 2.7vw, 2.3rem) !important;
        }

        h3 {
            color: var(--green-800);
            font-weight: 760 !important;
            letter-spacing: -.02em;
        }

        p, li, label, div, span {
            color: var(--ink);
        }

        .app-shell {
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 1rem;
            padding: .78rem .92rem;
            border: 1px solid var(--line);
            border-radius: 999px;
            background: rgba(255,255,255,.72);
            box-shadow: var(--shadow-soft);
            margin-bottom: 1.1rem;
        }

        .brand {
            display: flex;
            align-items: center;
            gap: .72rem;
            font-weight: 820;
            color: var(--green-900);
            letter-spacing: -.02em;
        }

        .brand-icon {
            width: 2.15rem;
            height: 2.15rem;
            border-radius: 999px;
            background: linear-gradient(135deg, var(--green-800), var(--green-500));
            color: #FFF8E8;
            display: grid;
            place-items: center;
            box-shadow: 0 10px 22px rgba(39,116,71,.22);
        }

        .ai-status-ok, .ai-status-basic {
            font-size: .82rem;
            font-weight: 750;
            border-radius: 999px;
            padding: .48rem .72rem;
            white-space: nowrap;
        }

        .ai-status-ok {
            background: rgba(39,116,71,.11);
            color: var(--green-800);
            border: 1px solid rgba(39,116,71,.22);
        }

        .ai-status-basic {
            background: rgba(184,149,74,.14);
            color: #705722;
            border: 1px solid rgba(184,149,74,.25);
        }

        .hero {
            position: relative;
            overflow: hidden;
            border-radius: 36px;
            padding: clamp(2rem, 4vw, 3.4rem);
            background:
                linear-gradient(135deg, rgba(255,255,255,.88), rgba(246,250,243,.82)),
                radial-gradient(circle at 90% 12%, rgba(213,192,138,.30), transparent 30%);
            border: 1px solid rgba(21,61,44,.12);
            box-shadow: var(--shadow);
            margin-bottom: 1.1rem;
        }

        .hero:after {
            content: "";
            position: absolute;
            right: -120px;
            bottom: -140px;
            width: 360px;
            height: 360px;
            border-radius: 50%;
            background: radial-gradient(circle, rgba(59,139,90,.18), transparent 68%);
            pointer-events: none;
        }

        .eyebrow {
            display: inline-flex;
            align-items: center;
            gap: .5rem;
            color: var(--gold);
            font-size: .78rem;
            font-weight: 880;
            letter-spacing: .18em;
            text-transform: uppercase;
            margin-bottom: 1rem;
        }

        .subtitle {
            max-width: 760px;
            font-size: clamp(1.05rem, 1.55vw, 1.28rem);
            line-height: 1.68;
            color: var(--ink-soft);
            margin-bottom: 1.15rem;
        }

        .chips {
            display: flex;
            flex-wrap: wrap;
            gap: .6rem;
            margin-top: .9rem;
        }

        .chip {
            display: inline-flex;
            align-items: center;
            gap: .42rem;
            padding: .55rem .78rem;
            border-radius: 999px;
            background: rgba(255,255,255,.74);
            border: 1px solid rgba(21,61,44,.12);
            box-shadow: 0 8px 22px rgba(15,47,34,.055);
            color: var(--green-800);
            font-size: .88rem;
            font-weight: 700;
        }

        .guide-grid {
            display: grid;
            grid-template-columns: repeat(4, minmax(0,1fr));
            gap: .85rem;
            margin: 1rem 0 1.2rem;
        }

        .guide-step {
            background: rgba(255,255,255,.76);
            border: 1px solid rgba(21,61,44,.12);
            box-shadow: var(--shadow-soft);
            border-radius: 24px;
            padding: 1rem;
        }

        .step-num {
            width: 2rem;
            height: 2rem;
            display: grid;
            place-items: center;
            border-radius: 999px;
            background: var(--sage-100);
            color: var(--green-800);
            font-weight: 850;
            margin-bottom: .7rem;
        }

        .step-title {
            font-weight: 820;
            color: var(--green-900);
            margin-bottom: .25rem;
        }

        .step-body {
            color: var(--ink-soft);
            font-size: .92rem;
            line-height: 1.5;
        }

        .panel {
            background: rgba(255,255,255,.76);
            border: 1px solid rgba(21,61,44,.12);
            box-shadow: var(--shadow-soft);
            border-radius: 28px;
            padding: 1.25rem;
            margin-bottom: 1rem;
        }

        .soft-panel {
            background: rgba(242,247,241,.72);
            border: 1px solid rgba(21,61,44,.10);
            border-radius: 26px;
            padding: 1.15rem;
            margin-bottom: 1rem;
        }

        .result-panel {
            background:
                linear-gradient(135deg, rgba(255,255,255,.84), rgba(242,247,241,.78));
            border: 1px solid rgba(39,116,71,.16);
            box-shadow: var(--shadow);
            border-radius: 30px;
            padding: 1.35rem;
            margin: 1rem 0;
        }

        .metric-row {
            display: grid;
            grid-template-columns: repeat(4, minmax(0,1fr));
            gap: .85rem;
            margin: 1rem 0;
        }

        .metric-card {
            background: linear-gradient(180deg, #FFFFFF, #F8FBF6);
            border: 1px solid rgba(21,61,44,.105);
            border-radius: 23px;
            padding: 1rem;
            box-shadow: 0 12px 28px rgba(15,47,34,.06);
        }

        .metric-label {
            color: var(--muted);
            font-size: .76rem;
            font-weight: 830;
            letter-spacing: .08em;
            text-transform: uppercase;
            margin-bottom: .34rem;
        }

        .metric-value {
            color: var(--green-900);
            font-size: 1.55rem;
            font-weight: 870;
            letter-spacing: -.04em;
        }

        .metric-note {
            color: var(--ink-soft);
            font-size: .84rem;
            margin-top: .22rem;
            line-height: 1.36;
        }

        .section-title {
            display: flex;
            align-items: center;
            gap: .72rem;
            margin: 1.2rem 0 .65rem;
        }

        .section-dot {
            width: 2.15rem;
            height: 2.15rem;
            border-radius: 999px;
            display: grid;
            place-items: center;
            background: linear-gradient(135deg, var(--green-800), var(--green-500));
            color: #FFF8E8;
            font-weight: 870;
            box-shadow: 0 10px 24px rgba(39,116,71,.2);
        }

        .risk-low, .risk-medium, .risk-high {
            display: inline-flex;
            align-items: center;
            gap: .45rem;
            border-radius: 999px;
            padding: .54rem .82rem;
            font-weight: 850;
            border: 1px solid;
        }
        .risk-low {
            color: #145B38;
            background: rgba(39,116,71,.10);
            border-color: rgba(39,116,71,.20);
        }
        .risk-medium {
            color: #715514;
            background: rgba(184,149,74,.16);
            border-color: rgba(184,149,74,.25);
        }
        .risk-high {
            color: #8D332B;
            background: rgba(183,74,61,.11);
            border-color: rgba(183,74,61,.20);
        }

        .insight {
            padding: .9rem 1rem;
            border-radius: 19px;
            background: rgba(255,255,255,.78);
            border: 1px solid rgba(21,61,44,.11);
            border-left: 5px solid var(--green-500);
            margin: .52rem 0;
            box-shadow: 0 8px 22px rgba(15,47,34,.045);
            color: var(--ink-soft);
            line-height: 1.52;
        }

        .action {
            display: flex;
            gap: .85rem;
            align-items: flex-start;
            padding: .95rem 1rem;
            border-radius: 20px;
            background: #FFFFFF;
            border: 1px solid rgba(21,61,44,.11);
            margin: .5rem 0;
            box-shadow: 0 8px 20px rgba(15,47,34,.045);
        }

        .action-num {
            width: 1.85rem;
            height: 1.85rem;
            flex: 0 0 auto;
            display: grid;
            place-items: center;
            border-radius: 999px;
            background: var(--sage-100);
            color: var(--green-800);
            font-weight: 850;
        }

        .action-title {
            font-weight: 820;
            color: var(--green-900);
        }

        .action-body {
            color: var(--ink-soft);
            line-height: 1.5;
            margin-top: .1rem;
        }

        .notice {
            padding: 1rem 1.1rem;
            border-radius: 22px;
            background: rgba(255,255,255,.65);
            border: 1px solid rgba(21,61,44,.11);
            color: var(--ink-soft);
            line-height: 1.58;
            margin: 1rem 0;
        }

        .small-muted {
            color: var(--muted);
            font-size: .88rem;
        }

        .stButton > button {
            background: linear-gradient(135deg, #153D2C, #277447) !important;
            color: #FFF8E8 !important;
            border: 1px solid rgba(21,61,44,.16) !important;
            border-radius: 999px !important;
            font-weight: 820 !important;
            padding: .76rem 1.05rem !important;
            box-shadow: 0 12px 26px rgba(39,116,71,.20) !important;
        }

        .stButton > button:hover {
            filter: brightness(1.03);
            transform: translateY(-1px);
        }

        .stButton > button:focus:not(:active) {
            color: #FFF8E8 !important;
            border-color: rgba(21,61,44,.24) !important;
        }

        div[data-testid="stDownloadButton"] > button {
            background: linear-gradient(135deg, #B8954A, #D5C08A) !important;
            color: #173528 !important;
            border-radius: 999px !important;
            font-weight: 850 !important;
            border: 1px solid rgba(184,149,74,.25) !important;
        }

        div[data-testid="stRadio"] {
            background: rgba(255,255,255,.60);
            border: 1px solid rgba(21,61,44,.10);
            border-radius: 999px;
            padding: .35rem .5rem;
            margin-bottom: 1rem;
            box-shadow: 0 8px 22px rgba(15,47,34,.045);
        }

        div[data-testid="stRadio"] label {
            color: var(--green-900) !important;
            font-weight: 780 !important;
        }

        div[data-testid="stTextInput"] input,
        div[data-testid="stNumberInput"] input,
        div[data-testid="stSelectbox"] div[data-baseweb="select"] > div,
        div[data-testid="stTextArea"] textarea {
            color: var(--green-900) !important;
            background: #FFFFFF !important;
            border: 1px solid rgba(21,61,44,.16) !important;
            border-radius: 16px !important;
            box-shadow: none !important;
        }

        div[data-testid="stTextInput"] input::placeholder,
        div[data-testid="stTextArea"] textarea::placeholder {
            color: #8C9B91 !important;
        }

        div[data-testid="stSelectbox"] span {
            color: var(--green-900) !important;
        }

        .stSlider label {
            color: var(--green-900) !important;
        }

        [data-testid="stExpander"] {
            background: rgba(255,255,255,.63) !important;
            border: 1px solid rgba(21,61,44,.11) !important;
            border-radius: 22px !important;
            box-shadow: 0 8px 20px rgba(15,47,34,.045) !important;
        }

        [data-testid="stDataFrame"] {
            border-radius: 20px;
            overflow: hidden;
        }

        @media (max-width: 980px) {
            .guide-grid { grid-template-columns: repeat(2, minmax(0,1fr)); }
            .metric-row { grid-template-columns: repeat(2, minmax(0,1fr)); }
            .app-shell { border-radius: 26px; align-items: flex-start; flex-direction: column; }
        }

        @media (max-width: 640px) {
            .guide-grid { grid-template-columns: 1fr; }
            .metric-row { grid-template-columns: 1fr; }
            h1 { font-size: 2.6rem !important; }
            .hero { border-radius: 28px; padding: 1.55rem; }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


# -----------------------------------------------------------------------------
# Data utilities
# -----------------------------------------------------------------------------
def blank_data() -> pd.DataFrame:
    return pd.DataFrame(columns=COLUMNS)


def load_data() -> pd.DataFrame:
    if not DATA_FILE.exists():
        return blank_data()
    df = pd.read_csv(DATA_FILE)
    for col in COLUMNS:
        if col not in df.columns:
            df[col] = np.nan
    return df[COLUMNS]


def save_data(df: pd.DataFrame) -> None:
    df[COLUMNS].to_csv(DATA_FILE, index=False)


def add_record(record: dict[str, Any]) -> None:
    df = load_data()
    new = pd.DataFrame([record])
    df = pd.concat([df, new], ignore_index=True)
    save_data(df)


def num(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    out = df.copy()
    for col in cols:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


def sample_data() -> pd.DataFrame:
    rows = [
        ("School cafeteria lunch", "Cafeteria", "Monday", "Lunch", "Pasta", 3, "Normal", "Medium", 118, 108, 132, 29, "None yet"),
        ("School cafeteria lunch", "Cafeteria", "Tuesday", "Lunch", "Chicken rice bowl", 5, "Sunny", "High", 120, 123, 126, 6, "Attendance confirmation"),
        ("School cafeteria lunch", "Cafeteria", "Wednesday", "Lunch", "Vegetarian chili", 2, "Cloudy", "Medium", 115, 100, 126, 34, "Menu preference survey"),
        ("After-school club", "Classroom", "Thursday", "Snack", "Fruit cups", 4, "Normal", "High", 42, 39, 45, 6, "Pre-order form"),
        ("Sports event", "Gym", "Friday", "Dinner", "Sandwiches", 3, "Rainy", "Low", 95, 72, 120, 43, "None yet"),
        ("Breakfast program", "Cafeteria", "Monday", "Breakfast", "Bagels", 4, "Very cold", "Medium", 70, 58, 80, 19, "Smaller first batch"),
        ("Community event", "Community hall", "Saturday", "All-day event", "Pizza", 5, "Sunny", "Medium", 160, 148, 170, 18, "Donation partner ready"),
        ("Fundraiser", "Event venue", "Sunday", "Dinner", "Hot dogs", 3, "Stormy", "Low", 140, 91, 160, 62, "Compost plan"),
        ("School cafeteria lunch", "Cafeteria", "Friday", "Lunch", "Fish tacos", 2, "Normal", "Medium", 117, 99, 128, 31, "None yet"),
        ("Food donation program", "Library / common area", "Wednesday", "Snack", "Bakery items", 4, "Cloudy", "Medium", 60, 52, 70, 12, "Donation partner ready"),
        ("School cafeteria lunch", "Cafeteria", "Thursday", "Lunch", "Rice noodle bowl", 4, "Sunny", "High", 122, 121, 126, 8, "Attendance confirmation"),
        ("Community event", "Outdoor area", "Saturday", "Lunch", "BBQ plates", 5, "Very hot", "Medium", 180, 151, 195, 39, "Smaller first batch"),
        ("After-school club", "Classroom", "Tuesday", "Snack", "Granola bars", 5, "Normal", "High", 36, 35, 38, 2, "Pre-order form"),
        ("School cafeteria lunch", "Cafeteria", "Wednesday", "Lunch", "Mac and cheese", 5, "Normal", "High", 125, 129, 130, 5, "Attendance confirmation"),
        ("Sports event", "Gym", "Friday", "Dinner", "Wraps", 3, "Cloudy", "Low", 110, 88, 135, 41, "None yet"),
        ("Breakfast program", "Cafeteria", "Thursday", "Breakfast", "Oatmeal cups", 2, "Normal", "Medium", 72, 63, 82, 22, "Menu preference survey"),
    ]

    records = []
    for i, r in enumerate(rows):
        (
            event_type, location, day, meal_time, food, popularity, weather, confidence,
            expected, actual, prepared, leftover, intervention
        ) = r
        waste_rate = leftover / prepared * 100
        risk_score = min(96, max(8, waste_rate * 2.35 + max(prepared - actual, 0) / prepared * 24))
        risk_level = level_from_score(risk_score)
        cost = leftover * 2.5
        co2 = leftover * 0.5

        records.append({
            "Time": (pd.Timestamp.now() - pd.Timedelta(days=18 - i)).strftime("%Y-%m-%d %H:%M:%S"),
            "Event Type": event_type,
            "Location": location,
            "Day": day,
            "Meal Time": meal_time,
            "Meal / Food": food,
            "Menu Popularity": popularity,
            "Weather": weather,
            "Attendance Confidence": confidence,
            "Expected Attendance": expected,
            "Actual Attendance": actual,
            "Food Prepared": prepared,
            "Leftover Portions": leftover,
            "Waste Rate": round(waste_rate, 2),
            "Predicted Waste Rate": round(max(0, waste_rate + np.random.uniform(-4, 4)), 2),
            "Risk Score": round(risk_score, 1),
            "Risk Level": risk_level,
            "Recommended Min": max(1, int(actual * 0.94)),
            "Recommended Max": max(1, int(actual * 1.05)),
            "Donation Route": "Yes" if intervention == "Donation partner ready" else "No",
            "Batch Cooking": "Yes" if intervention == "Smaller first batch" else "No",
            "Intervention": intervention,
            "Cost per Portion": 2.5,
            "CO2e per Portion": 0.5,
            "Cost Impact": round(cost, 2),
            "CO2e Impact": round(co2, 2),
            "Potential Meals Rescued": min(leftover, 20 if intervention == "Donation partner ready" else 0),
            "Notes": "Sample record for demo and judging.",
        })
    return pd.DataFrame(records, columns=COLUMNS)


# -----------------------------------------------------------------------------
# AI and prediction logic
# -----------------------------------------------------------------------------
def groq_client():
    if not GROQ_AVAILABLE:
        return None
    try:
        key = st.secrets.get("GROQ_API_KEY")
    except Exception:
        key = None
    if not key:
        return None
    return Groq(api_key=key)


def ai_status() -> tuple[bool, str]:
    if groq_client() is not None:
        return True, "Groq AI Advisor on"
    return False, "Built-in AI engine on"


def ask_ai(prompt: str) -> str | None:
    client = groq_client()
    if client is None:
        return None
    try:
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are an AI food waste planning assistant for high school students. "
                        "Be practical, concise, and safe. Do not make food safety decisions. "
                        "Always remind that human staff must inspect food and follow local rules."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.35,
            max_tokens=650,
        )
        return response.choices[0].message.content
    except Exception:
        return None


def level_from_score(score: float) -> str:
    if score >= 70:
        return "High"
    if score >= 42:
        return "Medium"
    return "Low"


def demand_multiplier(weather: str, confidence: str, popularity: int, day: str) -> float:
    weather_factor = {
        "Normal": 1.00,
        "Sunny": 1.02,
        "Cloudy": 0.99,
        "Rainy": 0.94,
        "Stormy": 0.88,
        "Very hot": 0.93,
        "Very cold": 0.92,
    }.get(weather, 1.0)

    confidence_factor = {
        "High": 1.02,
        "Medium": 0.98,
        "Low": 0.91,
    }.get(confidence, 0.98)

    popularity_factor = {
        1: 0.86,
        2: 0.92,
        3: 0.98,
        4: 1.03,
        5: 1.07,
    }.get(popularity, 0.98)

    day_factor = {
        "Monday": 0.97,
        "Tuesday": 1.00,
        "Wednesday": 1.00,
        "Thursday": 1.01,
        "Friday": 0.96,
        "Saturday": 0.95,
        "Sunday": 0.93,
    }.get(day, 1.0)

    return float(np.clip(weather_factor * confidence_factor * popularity_factor * day_factor, 0.68, 1.16))


def recommended_range(expected: int, weather: str, confidence: str, popularity: int, day: str, batch: bool) -> tuple[int, int, int]:
    expected = max(int(expected), 1)
    demand = expected * demand_multiplier(weather, confidence, popularity, day)
    if batch:
        low, high = demand * 0.92, demand * 1.02
    else:
        low, high = demand * 0.96, demand * 1.07
    return max(1, round(low)), max(1, round(high)), max(1, round(demand))


def history_signal(df: pd.DataFrame, event_type: str, food: str) -> tuple[float | None, str]:
    if df.empty:
        return None, "No historical records yet."

    df = num(df, ["Waste Rate"]).dropna(subset=["Waste Rate"])
    if df.empty:
        return None, "No usable waste-rate history yet."

    food_lower = str(food).strip().lower()
    if food_lower:
        same_food = df[df["Meal / Food"].astype(str).str.lower().str.contains(food_lower, regex=False, na=False)]
        if len(same_food) >= 2:
            avg = float(same_food["Waste Rate"].mean())
            return avg, f"Similar food has averaged {avg:.1f}% waste."

    same_event = df[df["Event Type"].astype(str) == event_type]
    if len(same_event) >= 2:
        avg = float(same_event["Waste Rate"].mean())
        return avg, f"Similar event type has averaged {avg:.1f}% waste."

    avg = float(df["Waste Rate"].tail(8).mean())
    return avg, f"Recent records have averaged {avg:.1f}% waste."


def train_model(df: pd.DataFrame):
    if not SKLEARN_AVAILABLE:
        return None, "ML model unavailable because scikit-learn is not installed."

    df = df.copy()
    df = num(df, ["Expected Attendance", "Food Prepared", "Menu Popularity", "Waste Rate"])
    df = df.dropna(subset=["Waste Rate", "Expected Attendance", "Food Prepared"])

    if len(df) < 12:
        return None, "Add 12+ logged events to unlock historical ML prediction."
    if df["Waste Rate"].nunique() < 3:
        return None, "More varied records are needed before ML prediction is useful."

    features = [
        "Event Type", "Location", "Day", "Meal Time", "Meal / Food",
        "Menu Popularity", "Weather", "Attendance Confidence",
        "Expected Attendance", "Food Prepared", "Batch Cooking", "Intervention"
    ]
    for col in features:
        if col not in df.columns:
            df[col] = ""

    X = df[features]
    y = df["Waste Rate"].clip(0, 100)

    categorical = [
        "Event Type", "Location", "Day", "Meal Time", "Meal / Food",
        "Weather", "Attendance Confidence", "Batch Cooking", "Intervention"
    ]
    numeric = ["Menu Popularity", "Expected Attendance", "Food Prepared"]

    pre = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical),
        ("num", "passthrough", numeric),
    ])

    model = Pipeline([
        ("pre", pre),
        ("rf", RandomForestRegressor(n_estimators=160, min_samples_leaf=2, random_state=7)),
    ])
    model.fit(X, y)
    return model, f"ML prediction active using {len(df)} event records."


def ml_predict(model, event: dict[str, Any]) -> float | None:
    if model is None:
        return None
    features = [
        "Event Type", "Location", "Day", "Meal Time", "Meal / Food",
        "Menu Popularity", "Weather", "Attendance Confidence",
        "Expected Attendance", "Food Prepared", "Batch Cooking", "Intervention"
    ]
    try:
        X = pd.DataFrame([{col: event.get(col, "") for col in features}])
        return float(model.predict(X)[0])
    except Exception:
        return None


def scan_plan(event: dict[str, Any], history: pd.DataFrame) -> dict[str, Any]:
    expected = max(int(event["Expected Attendance"]), 1)
    prepared = max(int(event["Food Prepared"]), 1)
    popularity = int(event["Menu Popularity"])
    batch = event["Batch Cooking"] == "Yes"

    rec_min, rec_max, demand = recommended_range(
        expected,
        event["Weather"],
        event["Attendance Confidence"],
        popularity,
        event["Day"],
        batch,
    )

    predicted_leftover = max(prepared - demand, 0)
    predicted_waste = predicted_leftover / prepared * 100
    overprep = max(prepared - expected, 0) / expected * 100

    score = 18 + predicted_waste * 1.18 + overprep * 0.32
    drivers = []

    if prepared > rec_max:
        score += min(22, (prepared - rec_max) / expected * 70)
        drivers.append(f"Prepared food is above the recommended range of {rec_min}–{rec_max} portions.")

    if event["Attendance Confidence"] == "Low":
        score += 15
        drivers.append("Attendance confidence is low, so over-preparation risk is higher.")
    elif event["Attendance Confidence"] == "Medium":
        score += 6
        drivers.append("Attendance confidence is medium; a quick RSVP check could improve accuracy.")

    if event["Weather"] in ["Rainy", "Stormy", "Very hot", "Very cold"]:
        score += {"Rainy": 8, "Stormy": 15, "Very hot": 10, "Very cold": 10}[event["Weather"]]
        drivers.append(f"{event['Weather']} conditions may lower attendance or appetite.")

    if popularity <= 2:
        score += 14 if popularity == 1 else 9
        drivers.append("Menu popularity is low, which can increase leftovers.")
    elif popularity >= 4:
        score -= 4
        drivers.append("Menu popularity is strong, which lowers leftover risk.")

    if event["Day"] in ["Monday", "Friday", "Saturday", "Sunday"]:
        score += 4
        drivers.append(f"{event['Day']} demand can be less predictable.")

    if event["Batch Cooking"] == "Yes":
        score -= 9
        drivers.append("Batch cooking lowers risk because food can be prepared in stages.")

    if event["Intervention"] != "None yet":
        score -= 6
        drivers.append(f"Planned intervention: {event['Intervention']}.")

    hist_avg, hist_text = history_signal(history, event["Event Type"], event["Meal / Food"])
    if hist_avg is not None:
        if hist_avg >= 25:
            score += 10
        elif hist_avg >= 12:
            score += 4
        elif hist_avg < 8:
            score -= 4
        drivers.append(hist_text)

    model, model_msg = train_model(history)
    ml_waste = ml_predict(model, event)

    if ml_waste is not None:
        predicted_waste = predicted_waste * 0.58 + ml_waste * 0.42
        score = score * 0.70 + predicted_waste * 1.45 * 0.30
        model_status = f"Built-in rules + historical ML. ML estimate: {ml_waste:.1f}% waste."
    else:
        model_status = f"Built-in AI risk engine. {model_msg}"

    score = float(np.clip(score, 5, 96))
    level = level_from_score(score)

    return {
        "Risk Score": round(score, 1),
        "Risk Level": level,
        "Predicted Waste Rate": round(float(np.clip(predicted_waste, 0, 100)), 1),
        "Predicted Demand": int(demand),
        "Predicted Leftovers": int(round(max(prepared - demand, 0))),
        "Recommended Min": int(rec_min),
        "Recommended Max": int(rec_max),
        "Drivers": drivers[:5] if drivers else ["No major red flags. Continue tracking the result after the event."],
        "Model Status": model_status,
    }


def action_plan(event: dict[str, Any], result: dict[str, Any]) -> list[tuple[str, str]]:
    actions = [
        ("Set a smarter prep range", f"Use {result['Recommended Min']}–{result['Recommended Max']} portions as the planning range."),
        ("Confirm attendance", "Ask homerooms, club members, or event attendees to confirm numbers before food is prepared."),
    ]

    if result["Risk Level"] == "High":
        actions.insert(1, ("Do not cook everything at once", "Start with 70–80% of the expected demand and keep the rest as backup."))
    elif result["Risk Level"] == "Medium":
        actions.insert(1, ("Use a smaller buffer", "Keep a small safety buffer, but avoid preparing the full surplus upfront."))

    if event["Batch Cooking"] == "No":
        actions.append(("Add batch control", "If possible, split food into smaller batches so staff can stop early if turnout is low."))

    if event["Donation Route"] == "Yes":
        actions.append(("Prepare a rescue route", "If food is still suitable after human inspection, contact the verified redistribution route."))
    else:
        actions.append(("Plan the end route", "Before the event, decide whether safe leftovers can become staff meals, donation review, or compost."))

    if event["Intervention"] == "None yet":
        actions.append(("Choose one experiment", "Try one intervention this event so the dashboard can compare whether it helped."))

    return actions[:6]


# -----------------------------------------------------------------------------
# UI helpers
# -----------------------------------------------------------------------------
def app_bar() -> None:
    ok, label = ai_status()
    status_class = "ai-status-ok" if ok else "ai-status-basic"
    status_note = "Optional Groq connected" if ok else "No API key needed"
    st.markdown(
        f"""
        <div class="app-shell">
            <div class="brand"><span class="brand-icon">🌿</span> Food Waste Rescue Radar</div>
            <div class="{status_class}">{label} · {status_note}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def hero(kicker: str, title: str, subtitle: str, chips: list[str] | None = None) -> None:
    chips_html = ""
    if chips:
        chips_html = '<div class="chips">' + "".join(f'<span class="chip">{c}</span>' for c in chips) + "</div>"

    st.markdown(
        f"""
        <div class="hero">
            <div class="eyebrow">{kicker}</div>
            <h1>{title}</h1>
            <div class="subtitle">{subtitle}</div>
            {chips_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def guide_steps() -> None:
    st.markdown(
        """
        <div class="guide-grid">
            <div class="guide-step">
                <div class="step-num">1</div>
                <div class="step-title">Plan before food is made</div>
                <div class="step-body">Use the scanner to estimate waste risk and recommended portions.</div>
            </div>
            <div class="guide-step">
                <div class="step-num">2</div>
                <div class="step-title">Take one action</div>
                <div class="step-body">Try RSVP, smaller first batch, menu survey, or donation planning.</div>
            </div>
            <div class="guide-step">
                <div class="step-num">3</div>
                <div class="step-title">Log the real result</div>
                <div class="step-body">After the event, record attendance, portions prepared, and leftovers.</div>
            </div>
            <div class="guide-step">
                <div class="step-num">4</div>
                <div class="step-title">Learn from patterns</div>
                <div class="step-body">Use the dashboard to find high-waste meals, days, and interventions.</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def section(n: str, title: str) -> None:
    st.markdown(
        f"""
        <div class="section-title">
            <div class="section-dot">{n}</div>
            <h2 style="margin:0;">{title}</h2>
        </div>
        """,
        unsafe_allow_html=True,
    )


def metrics(cards: list[tuple[str, str, str]]) -> None:
    html = '<div class="metric-row">'
    for label, value, note in cards:
        html += f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
            <div class="metric-note">{note}</div>
        </div>
        """
    html += "</div>"
    st.markdown(html, unsafe_allow_html=True)


def risk_badge(level: str) -> str:
    cls = {"Low": "risk-low", "Medium": "risk-medium", "High": "risk-high"}.get(level, "risk-medium")
    return f'<span class="{cls}">● {level} Risk</span>'


def insight(text: str) -> None:
    st.markdown(f'<div class="insight">{text}</div>', unsafe_allow_html=True)


def action_item(i: int, title: str, body: str) -> None:
    st.markdown(
        f"""
        <div class="action">
            <div class="action-num">{i}</div>
            <div>
                <div class="action-title">{title}</div>
                <div class="action-body">{body}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def notice(text: str) -> None:
    st.markdown(f'<div class="notice">{text}</div>', unsafe_allow_html=True)


def collect_inputs(prefix: str, include_actual: bool) -> dict[str, Any]:
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown("### Event details")
    st.caption("Fill only what you know. Use estimates before the event and real numbers after the event.")

    c1, c2, c3 = st.columns(3, gap="medium")
    with c1:
        event_type = st.selectbox("What kind of event is it?", EVENT_TYPES, key=f"{prefix}_event")
        location = st.selectbox("Where will food be served?", LOCATIONS, key=f"{prefix}_location")
        day = st.selectbox("Day", DAYS, index=4, key=f"{prefix}_day")
    with c2:
        meal_time = st.selectbox("Meal time", MEAL_TIMES, index=1, key=f"{prefix}_meal_time")
        food = st.text_input("Food / menu item", placeholder="Example: pasta, pizza, sandwiches", key=f"{prefix}_food")
        weather = st.selectbox("Weather / condition", WEATHER, key=f"{prefix}_weather")
    with c3:
        popularity = st.slider("Menu popularity", 1, 5, 3, help="1 = unpopular, 5 = very popular", key=f"{prefix}_popular")
        confidence = st.selectbox("How confident is attendance?", CONFIDENCE, index=1, key=f"{prefix}_confidence")
        intervention = st.selectbox("Action planned or used", INTERVENTIONS, key=f"{prefix}_intervention")

    st.markdown("### Numbers")
    n1, n2, n3, n4 = st.columns(4, gap="medium")
    with n1:
        expected = st.number_input("Expected attendance", min_value=1, value=100, step=1, key=f"{prefix}_expected")
    with n2:
        prepared = st.number_input("Food prepared / planned", min_value=1, value=110, step=1, key=f"{prefix}_prepared")
    with n3:
        cost = st.number_input("Cost per portion ($)", min_value=0.0, value=2.50, step=0.10, format="%.2f", key=f"{prefix}_cost")
    with n4:
        co2 = st.number_input("CO₂e per portion (kg)", min_value=0.0, value=0.50, step=0.05, format="%.2f", key=f"{prefix}_co2")

    actual = np.nan
    leftover = np.nan
    if include_actual:
        a1, a2 = st.columns(2, gap="medium")
        with a1:
            actual = st.number_input("Actual attendance", min_value=0, value=90, step=1, key=f"{prefix}_actual")
        with a2:
            leftover = st.number_input("Leftover portions", min_value=0, value=18, step=1, key=f"{prefix}_leftover")

    with st.expander("Optional details for better prediction", expanded=False):
        o1, o2 = st.columns(2, gap="medium")
        with o1:
            donation_route = st.toggle("A verified donation / redistribution route is available", value=False, key=f"{prefix}_donation")
        with o2:
            batch = st.toggle("Food can be prepared in smaller batches", value=True, key=f"{prefix}_batch")
        notes = st.text_area("Notes", placeholder="Example: exam week, rainy day, unpopular menu last time...", key=f"{prefix}_notes")

    st.markdown("</div>", unsafe_allow_html=True)

    return {
        "Event Type": event_type,
        "Location": location,
        "Day": day,
        "Meal Time": meal_time,
        "Meal / Food": food.strip() or "Unknown",
        "Menu Popularity": int(popularity),
        "Weather": weather,
        "Attendance Confidence": confidence,
        "Expected Attendance": int(expected),
        "Actual Attendance": actual,
        "Food Prepared": int(prepared),
        "Leftover Portions": leftover,
        "Donation Route": "Yes" if donation_route else "No",
        "Batch Cooking": "Yes" if batch else "No",
        "Intervention": intervention,
        "Cost per Portion": float(cost),
        "CO2e per Portion": float(co2),
        "Notes": notes.strip(),
    }


# -----------------------------------------------------------------------------
# Pages
# -----------------------------------------------------------------------------
def home_page() -> None:
    hero(
        "High School Track · AI for Everyday Good",
        "A calm, premium AI tool for reducing food waste.",
        (
            "This app helps a school or community team predict food waste risk, choose practical actions, "
            "log real event results, and show environmental impact in a simple dashboard."
        ),
        ["Built-in AI risk engine", "Optional Groq AI Advisor", "What-if simulator", "Impact dashboard"],
    )
    guide_steps()

    df = load_data()
    df_num = num(df, ["Waste Rate", "Leftover Portions", "Cost Impact", "CO2e Impact", "Potential Meals Rescued"])

    if df_num.empty:
        metrics([
            ("Records", "0", "Start with sample data or log your first event"),
            ("AI mode", "Ready", "Built-in risk engine works without an API key"),
            ("Dashboard", "Empty", "Needs event records"),
            ("Next step", "Scan", "Open the planner below"),
        ])
    else:
        metrics([
            ("Records", f"{len(df_num)}", "Saved event results"),
            ("Average waste", f"{df_num['Waste Rate'].mean():.1f}%", "Across records"),
            ("CO₂e impact", f"{df_num['CO2e Impact'].sum():.1f} kg", "Estimated from leftovers"),
            ("Meals rescued", f"{df_num['Potential Meals Rescued'].sum():.0f}", "Potential portions"),
        ])

    c1, c2, c3 = st.columns(3, gap="medium")
    with c1:
        if st.button("Open Waste Scanner", use_container_width=True):
            st.session_state.page = "1 · Scan"
            st.rerun()
    with c2:
        if st.button("Load Sample Data", use_container_width=True):
            save_data(sample_data())
            st.success("Sample data loaded. Open Dashboard to see patterns.")
            st.rerun()
    with c3:
        if st.button("Open Dashboard", use_container_width=True):
            st.session_state.page = "3 · Dashboard"
            st.rerun()

    section("?", "What counts as AI here?")
    notice(
        "<b>Built-in AI risk engine:</b> always on. It uses rules, historical patterns, and a machine-learning model when enough data exists. "
        "<br><br><b>Groq AI Advisor:</b> optional. If you add <code>GROQ_API_KEY</code> in Streamlit Secrets, the app will also generate written recommendations. "
        "If you do not add a key, the app still works normally."
    )


def scan_page() -> None:
    hero(
        "Step 1 · Before the event",
        "Scan the food plan before waste happens.",
        (
            "Use this page before a school lunch, club event, or community program. "
            "The app estimates waste risk and gives a practical portion range."
        ),
        ["Takes 2 minutes", "No coding needed", "Useful before ordering food"],
    )

    event = collect_inputs("scan", include_actual=False)

    if st.button("Generate Waste Risk Plan", use_container_width=True):
        history = load_data()
        result = scan_plan(event, history)

        section("1", "Risk result")
        st.markdown(risk_badge(result["Risk Level"]), unsafe_allow_html=True)
        metrics([
            ("Risk score", f"{result['Risk Score']:.0f}/100", "Higher means more likely to waste"),
            ("Predicted waste", f"{result['Predicted Waste Rate']:.1f}%", "Estimated before event"),
            ("Recommended prep", f"{result['Recommended Min']}–{result['Recommended Max']}", "Suggested portion range"),
            ("Model", "AI engine", result["Model Status"]),
        ])

        section("2", "Why the app thinks this")
        for d in result["Drivers"]:
            insight(d)

        section("3", "Action plan for students")
        for i, (title, body) in enumerate(action_plan(event, result), 1):
            action_item(i, title, body)

        section("4", "What-if simulator")
        st.caption("Test different preparation amounts and see how predicted waste changes.")
        low = max(1, int(event["Expected Attendance"] * 0.65))
        high = max(low + 1, int(event["Expected Attendance"] * 1.45))
        test_portions = st.slider("Test prepared portions", low, high, event["Food Prepared"], key="sim_slider")

        rows = []
        for p in range(low, high + 1):
            e2 = dict(event)
            e2["Food Prepared"] = p
            r2 = scan_plan(e2, history)
            shortage = max(r2["Predicted Demand"] - p, 0)
            rows.append({
                "Prepared Portions": p,
                "Predicted Waste %": r2["Predicted Waste Rate"],
                "Estimated Leftovers": r2["Predicted Leftovers"],
                "Shortage Risk": shortage,
            })
        sim = pd.DataFrame(rows)
        chosen = sim[sim["Prepared Portions"] == test_portions].iloc[0]

        metrics([
            ("Test amount", f"{test_portions}", "Prepared portions"),
            ("Waste estimate", f"{chosen['Predicted Waste %']:.1f}%", "At test amount"),
            ("Leftovers", f"{chosen['Estimated Leftovers']:.0f}", "Estimated portions"),
            ("Shortage risk", f"{chosen['Shortage Risk']:.0f}", "Estimated portions"),
        ])

        if PLOTLY_AVAILABLE:
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=sim["Prepared Portions"],
                y=sim["Predicted Waste %"],
                mode="lines",
                line=dict(color="#277447", width=4),
                fill="tozeroy",
                fillcolor="rgba(39,116,71,.13)",
                name="Predicted waste",
            ))
            fig.add_vrect(
                x0=result["Recommended Min"],
                x1=result["Recommended Max"],
                fillcolor="rgba(184,149,74,.16)",
                line_width=0,
                annotation_text="recommended range",
                annotation_position="top left",
            )
            fig.update_layout(
                height=330,
                margin=dict(l=10, r=10, t=30, b=10),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(255,255,255,.72)",
                font=dict(color="#173528"),
                xaxis_title="Prepared portions",
                yaxis_title="Predicted waste %",
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.line_chart(sim.set_index("Prepared Portions")["Predicted Waste %"])

        section("5", "Optional Groq AI Advisor")
        ai_text = ask_ai(
            "Create a short food waste prevention plan for high school students using this data:\n"
            + json.dumps({**event, **result}, indent=2)
        )
        if ai_text:
            st.markdown('<div class="soft-panel">', unsafe_allow_html=True)
            st.write(ai_text)
            st.markdown("</div>", unsafe_allow_html=True)
        else:
            notice(
                "Groq AI Advisor is not connected. This is okay for the MVP: the built-in AI risk engine, simulator, and dashboard still work. "
                "To enable written AI advice, add GROQ_API_KEY in Streamlit Secrets."
            )

        notice(
            "<b>Responsible AI:</b> The app predicts planning risk only. Human staff must inspect food and follow local food safety rules before donation, reuse, composting, or disposal."
        )


def log_page() -> None:
    hero(
        "Step 2 · After the event",
        "Log the real result.",
        (
            "After food is served, record actual attendance and leftovers. "
            "This is what makes the dashboard and historical prediction smarter."
        ),
        ["Actual attendance", "Leftover portions", "Impact receipt"],
    )

    event = collect_inputs("log", include_actual=True)

    if st.button("Save Event Result", use_container_width=True):
        prepared = int(event["Food Prepared"])
        leftover = int(event["Leftover Portions"])
        actual = int(event["Actual Attendance"])

        if leftover > prepared:
            st.error("Leftover portions cannot be greater than food prepared.")
            return

        history = load_data()
        result = scan_plan(event, history)

        waste = leftover / prepared * 100
        cost_impact = leftover * event["Cost per Portion"]
        co2_impact = leftover * event["CO2e per Portion"]
        score = float(np.clip(waste * 2.35 + max(prepared - actual, 0) / prepared * 24, 5, 96))
        level = level_from_score(score)

        record = {
            "Time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            **event,
            "Waste Rate": round(waste, 2),
            "Predicted Waste Rate": result["Predicted Waste Rate"],
            "Risk Score": round(score, 1),
            "Risk Level": level,
            "Recommended Min": result["Recommended Min"],
            "Recommended Max": result["Recommended Max"],
            "Cost Impact": round(cost_impact, 2),
            "CO2e Impact": round(co2_impact, 2),
            "Potential Meals Rescued": min(leftover, 20 if event["Donation Route"] == "Yes" else 0),
        }
        add_record(record)

        st.success("Saved. Open the dashboard to see updated patterns.")
        section("✓", "Impact receipt")
        st.markdown(risk_badge(level), unsafe_allow_html=True)
        metrics([
            ("Actual waste", f"{waste:.1f}%", "Leftover / prepared"),
            ("Leftovers", f"{leftover}", "Portions"),
            ("Cost impact", f"${cost_impact:.0f}", "Estimated"),
            ("CO₂e impact", f"{co2_impact:.1f} kg", "Estimated"),
        ])

        if waste >= 25:
            insight("This was a high-waste event. Next time, focus on attendance confirmation and smaller first batches.")
        elif waste >= 10:
            insight("This was a moderate-waste event. A smaller buffer or menu preference survey may help.")
        else:
            insight("This was a low-waste event. Keep this record as a good planning example.")


def dashboard_page() -> None:
    hero(
        "Step 3 · Pattern detection",
        "Understand what creates food waste.",
        (
            "The dashboard turns logged events into clear patterns: high-waste foods, risky days, cost impact, CO₂e impact, and intervention results."
        ),
        ["Environmental impact", "Cost impact", "Pattern detection"],
    )

    df = load_data()
    if df.empty:
        notice("No records yet. Click <b>Load Sample Data</b> on the Home page or log your first event.")
        return

    df = num(df, ["Waste Rate", "Leftover Portions", "Cost Impact", "CO2e Impact", "Potential Meals Rescued"])
    df["Time"] = pd.to_datetime(df["Time"], errors="coerce")
    df = df.sort_values("Time")

    metrics([
        ("Events logged", f"{len(df)}", "Total records"),
        ("Average waste", f"{df['Waste Rate'].mean():.1f}%", "Across all events"),
        ("Cost impact", f"${df['Cost Impact'].sum():.0f}", "Estimated total"),
        ("CO₂e impact", f"{df['CO2e Impact'].sum():.1f} kg", "Estimated total"),
    ])

    model, model_msg = train_model(df)
    insight(f"<b>AI model status:</b> {model_msg}")

    c1, c2 = st.columns([1.15, .85], gap="large")
    with c1:
        section("1", "Waste trend")
        trend = df.dropna(subset=["Time", "Waste Rate"])
        if PLOTLY_AVAILABLE and not trend.empty:
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=trend["Time"],
                y=trend["Waste Rate"],
                mode="lines+markers",
                line=dict(color="#277447", width=4),
                marker=dict(color="#B8954A", size=8),
                name="Waste rate",
            ))
            fig.update_layout(
                height=340,
                margin=dict(l=10, r=10, t=25, b=10),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(255,255,255,.72)",
                font=dict(color="#173528"),
                xaxis_title="Date",
                yaxis_title="Waste rate %",
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.line_chart(trend.set_index("Time")["Waste Rate"])

    with c2:
        section("2", "Risk mix")
        mix = df["Risk Level"].value_counts().reset_index()
        mix.columns = ["Risk Level", "Events"]
        if PLOTLY_AVAILABLE and not mix.empty:
            fig = px.pie(
                mix,
                values="Events",
                names="Risk Level",
                hole=.58,
                color="Risk Level",
                color_discrete_map={"Low": "#277447", "Medium": "#B8954A", "High": "#B75A4A"},
            )
            fig.update_layout(
                height=340,
                margin=dict(l=10, r=10, t=25, b=10),
                paper_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#173528"),
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.dataframe(mix, use_container_width=True)

    section("3", "Main waste drivers")
    d1, d2 = st.columns(2, gap="large")

    with d1:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.markdown("### Highest-waste foods")
        food_avg = df.groupby("Meal / Food")["Waste Rate"].mean().sort_values(ascending=False).head(8).reset_index()
        if PLOTLY_AVAILABLE and not food_avg.empty:
            fig = px.bar(
                food_avg,
                x="Waste Rate",
                y="Meal / Food",
                orientation="h",
                color="Waste Rate",
                color_continuous_scale=["#DDEBDD", "#3B8B5A", "#153D2C"],
            )
            fig.update_layout(
                height=345,
                margin=dict(l=10, r=10, t=10, b=10),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(255,255,255,.72)",
                font=dict(color="#173528"),
                coloraxis_showscale=False,
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.dataframe(food_avg, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with d2:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.markdown("### Waste by day")
        day_avg = df.groupby("Day")["Waste Rate"].mean().reindex(DAYS).dropna().reset_index()
        if PLOTLY_AVAILABLE and not day_avg.empty:
            fig = px.bar(
                day_avg,
                x="Day",
                y="Waste Rate",
                color="Waste Rate",
                color_continuous_scale=["#DDEBDD", "#3B8B5A", "#153D2C"],
            )
            fig.update_layout(
                height=345,
                margin=dict(l=10, r=10, t=10, b=10),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(255,255,255,.72)",
                font=dict(color="#173528"),
                coloraxis_showscale=False,
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.dataframe(day_avg, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    section("4", "Intervention comparison")
    inter = df.groupby("Intervention")["Waste Rate"].agg(["count", "mean"]).reset_index()
    inter.columns = ["Intervention", "Events", "Average Waste Rate"]
    inter = inter.sort_values("Average Waste Rate")
    st.dataframe(inter, use_container_width=True)

    if len(inter) > 0:
        best = inter.iloc[0]
        insight(
            f"<b>Current best signal:</b> {best['Intervention']} has the lowest average waste in this dataset "
            f"({best['Average Waste Rate']:.1f}%). Treat this as a clue, not a final scientific conclusion."
        )

    section("5", "Recent records")
    show_cols = [
        "Time", "Event Type", "Meal / Food", "Day", "Expected Attendance",
        "Actual Attendance", "Food Prepared", "Leftover Portions", "Waste Rate", "Risk Level", "Intervention"
    ]
    st.dataframe(df[show_cols].tail(12), use_container_width=True)


def export_page() -> None:
    hero(
        "Step 4 · Share the results",
        "Export data and impact report.",
        (
            "Download the raw data or a short report that can be used in a presentation, judging demo, or school sustainability meeting."
        ),
        ["CSV", "Markdown report", "Transparent records"],
    )

    df = load_data()
    if df.empty:
        notice("No records yet. Load sample data or log an event first.")
        return

    df_num = num(df, ["Waste Rate", "Leftover Portions", "Cost Impact", "CO2e Impact", "Potential Meals Rescued"])

    report = [
        "# Food Waste Rescue Radar · Impact Report",
        "",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        "",
        "## Summary",
        f"- Events logged: {len(df_num)}",
        f"- Average waste rate: {df_num['Waste Rate'].mean():.1f}%",
        f"- Total leftover portions: {df_num['Leftover Portions'].sum():.0f}",
        f"- Estimated cost impact: ${df_num['Cost Impact'].sum():.0f}",
        f"- Estimated CO2e impact: {df_num['CO2e Impact'].sum():.1f} kg",
        f"- Potential meals rescued: {df_num['Potential Meals Rescued'].sum():.0f}",
        "",
        "## Recommended next actions",
        "- Confirm attendance before food is prepared.",
        "- Use smaller first batches for high-risk events.",
        "- Track menu popularity and compare it to leftovers.",
        "- Plan a safe donation, redistribution, or compost route before large events.",
        "- Keep all food safety decisions with trained human staff.",
        "",
        "Responsible AI note: This app supports planning only and does not decide whether food is safe to serve or donate.",
    ]
    report_text = "\n".join(report)

    c1, c2 = st.columns(2, gap="medium")
    with c1:
        st.download_button(
            "Download CSV",
            data=df.to_csv(index=False).encode("utf-8"),
            file_name="food_waste_records.csv",
            mime="text/csv",
            use_container_width=True,
        )
    with c2:
        st.download_button(
            "Download Report",
            data=report_text.encode("utf-8"),
            file_name="food_waste_impact_report.md",
            mime="text/markdown",
            use_container_width=True,
        )

    section("1", "Report preview")
    st.markdown(report_text)

    section("2", "Raw data")
    st.dataframe(df.tail(50), use_container_width=True)


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main() -> None:
    css()
    app_bar()

    if "page" not in st.session_state:
        st.session_state.page = "Home"

    pages = ["Home", "1 · Scan", "2 · Log", "3 · Dashboard", "4 · Export"]
    default_index = pages.index(st.session_state.page) if st.session_state.page in pages else 0

    page = st.radio("Navigation", pages, index=default_index, horizontal=True, label_visibility="collapsed")
    st.session_state.page = page

    if page == "Home":
        home_page()
    elif page == "1 · Scan":
        scan_page()
    elif page == "2 · Log":
        log_page()
    elif page == "3 · Dashboard":
        dashboard_page()
    elif page == "4 · Export":
        export_page()


if __name__ == "__main__":
    main()
