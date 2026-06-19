import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

try:
    import plotly.graph_objects as go
    import plotly.express as px
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
# Page setup
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Food Waste Rescue Radar",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="collapsed",
)


APP_TITLE = "Food Waste Rescue Radar"
HISTORY_FILE = Path("historical_data.csv")

EVENT_TYPES = [
    "School lunch",
    "Breakfast program",
    "After-school club",
    "Sports event",
    "Community event",
    "Fundraiser",
    "Donation program",
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

WEATHER_OPTIONS = [
    "Normal",
    "Sunny",
    "Cloudy",
    "Rainy",
    "Stormy",
    "Very hot",
    "Very cold",
]

CONFIDENCE_OPTIONS = ["High", "Medium", "Low"]
INTERVENTIONS = [
    "None yet",
    "Pre-order form",
    "Attendance confirmation",
    "Smaller first batch",
    "Menu preference survey",
    "Donation partner ready",
    "Reusable serving station",
    "Compost plan",
    "Mixed intervention",
]

HISTORY_COLUMNS = [
    "Username",
    "Time",
    "Event Type",
    "Location",
    "Day of Week",
    "Meal Time",
    "Meal / Food Type",
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
    "Recommended Min Portions",
    "Recommended Max Portions",
    "Donation Partner Available",
    "Batch Cooking Available",
    "Reusable Serving Available",
    "Intervention Used",
    "Cost per Portion",
    "CO2e per Portion",
    "Estimated Cost Impact",
    "Estimated CO2e Impact",
    "Potential Meals Rescued",
    "Notes",
]


# -----------------------------------------------------------------------------
# Luxury emerald CSS
# -----------------------------------------------------------------------------
def inject_css() -> None:
    st.markdown(
        """
        <style>
        :root {
            --forest: #123D2B;
            --deep: #0B2A1D;
            --emerald: #1F7A4D;
            --moss: #6F8F5E;
            --sage: #DCEBDD;
            --sage-2: #EAF4E8;
            --ivory: #FBF8EF;
            --paper: rgba(255, 255, 250, 0.76);
            --paper-solid: #FFFDF5;
            --gold: #B9903D;
            --gold-soft: #E8D8AE;
            --ink: #153729;
            --muted: #607469;
            --line: rgba(31, 122, 77, 0.17);
            --shadow: 0 22px 65px rgba(18, 61, 43, 0.12);
            --soft-shadow: 0 14px 36px rgba(18, 61, 43, 0.09);
        }

        html, body, [class*="css"] {
            font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
        }

        .stApp {
            background:
                radial-gradient(circle at 8% 3%, rgba(185, 144, 61, 0.12), transparent 31%),
                radial-gradient(circle at 88% 9%, rgba(31, 122, 77, 0.16), transparent 34%),
                linear-gradient(135deg, #F7FBF4 0%, #EFF7EE 40%, #E2F0E1 100%);
            color: var(--ink);
        }

        [data-testid="stHeader"] {
            background: rgba(247, 251, 244, 0.72);
            backdrop-filter: blur(18px);
            border-bottom: 1px solid rgba(31, 122, 77, 0.10);
        }

        [data-testid="stToolbar"] { right: 1.2rem; }

        .block-container {
            padding-top: 1.7rem;
            padding-bottom: 4rem;
            max-width: 1160px;
        }

        section[data-testid="stSidebar"] {
            background: linear-gradient(180deg, #F7FBF4, #E7F2E3);
            border-right: 1px solid rgba(31, 122, 77, 0.16);
        }

        h1, h2, h3 {
            color: var(--forest);
            letter-spacing: -0.035em;
        }

        h1 {
            font-size: clamp(2.65rem, 6vw, 5.65rem) !important;
            line-height: 0.92 !important;
            font-weight: 830 !important;
        }

        h2 {
            font-size: clamp(1.7rem, 3vw, 2.55rem) !important;
            font-weight: 780 !important;
        }

        h3 {
            font-size: 1.24rem !important;
            font-weight: 740 !important;
        }

        p, li, label, .stMarkdown, [data-testid="stMetricLabel"] {
            color: var(--ink);
        }

        .muted, .muted p {
            color: var(--muted) !important;
        }

        .topbar {
            display: flex;
            justify-content: space-between;
            align-items: center;
            gap: 1rem;
            margin-bottom: 1.1rem;
        }

        .brand {
            display: inline-flex;
            align-items: center;
            gap: .65rem;
            padding: .56rem .82rem;
            border: 1px solid rgba(31, 122, 77, .18);
            border-radius: 999px;
            background: rgba(255, 255, 250, .68);
            box-shadow: 0 10px 30px rgba(31, 122, 77, .06);
            color: var(--forest);
            font-weight: 740;
            letter-spacing: -.01em;
        }

        .brand-mark {
            width: 2rem;
            height: 2rem;
            border-radius: 999px;
            display: grid;
            place-items: center;
            background: linear-gradient(135deg, var(--forest), var(--emerald));
            color: #F6F0DE;
            box-shadow: 0 8px 24px rgba(31, 122, 77, .24);
        }

        .user-pill {
            padding: .62rem .9rem;
            border-radius: 999px;
            background: rgba(18, 61, 43, .07);
            color: var(--forest);
            border: 1px solid rgba(31, 122, 77, .13);
            font-size: .86rem;
            font-weight: 650;
        }

        .hero {
            position: relative;
            overflow: hidden;
            padding: clamp(2rem, 4vw, 3.7rem);
            border-radius: 34px;
            border: 1px solid rgba(31, 122, 77, 0.17);
            background:
                linear-gradient(130deg, rgba(255, 253, 245, .88), rgba(232, 244, 228, .78)),
                radial-gradient(circle at top right, rgba(185, 144, 61, .22), transparent 33%);
            box-shadow: var(--shadow);
            margin-bottom: 1.25rem;
        }

        .hero:after {
            content: "";
            position: absolute;
            right: -115px;
            top: -115px;
            width: 310px;
            height: 310px;
            border-radius: 50%;
            background: radial-gradient(circle, rgba(31,122,77,.24), transparent 68%);
            filter: blur(2px);
        }

        .eyebrow {
            display: inline-flex;
            align-items: center;
            gap: .55rem;
            color: var(--gold);
            font-size: .78rem;
            font-weight: 850;
            letter-spacing: .18em;
            text-transform: uppercase;
            margin-bottom: 1rem;
        }

        .hero-subtitle {
            max-width: 740px;
            font-size: clamp(1.08rem, 1.6vw, 1.32rem);
            line-height: 1.65;
            color: #416055;
            margin-top: 1.1rem;
            margin-bottom: 1.4rem;
        }

        .hero-chips {
            display: flex;
            flex-wrap: wrap;
            gap: .65rem;
            margin-top: 1rem;
        }

        .chip {
            display: inline-flex;
            align-items: center;
            gap: .45rem;
            padding: .58rem .78rem;
            border-radius: 999px;
            background: rgba(255,255,250,.72);
            border: 1px solid rgba(31,122,77,.16);
            color: var(--forest);
            font-weight: 650;
            font-size: .88rem;
            box-shadow: 0 8px 22px rgba(18, 61, 43, .05);
        }

        .panel {
            padding: 1.35rem;
            border-radius: 24px;
            background: rgba(255, 253, 245, .72);
            border: 1px solid rgba(31, 122, 77, 0.16);
            box-shadow: var(--soft-shadow);
            backdrop-filter: blur(20px);
            margin-bottom: 1rem;
        }

        .panel-soft {
            padding: 1.25rem;
            border-radius: 22px;
            background: rgba(236, 247, 232, .62);
            border: 1px solid rgba(31, 122, 77, 0.13);
            margin-bottom: 1rem;
        }

        .metric-grid {
            display: grid;
            grid-template-columns: repeat(4, minmax(0, 1fr));
            gap: .9rem;
            margin: .65rem 0 1rem 0;
        }

        .metric-card {
            padding: 1.05rem;
            border-radius: 22px;
            background: linear-gradient(180deg, rgba(255,253,245,.88), rgba(239,247,236,.76));
            border: 1px solid rgba(31,122,77,.14);
            box-shadow: 0 14px 30px rgba(18, 61, 43, .07);
        }

        .metric-label {
            color: var(--muted);
            font-size: .76rem;
            font-weight: 750;
            text-transform: uppercase;
            letter-spacing: .08em;
            margin-bottom: .35rem;
        }

        .metric-value {
            color: var(--forest);
            font-size: 1.55rem;
            font-weight: 820;
            letter-spacing: -.04em;
        }

        .metric-caption {
            color: #73867C;
            font-size: .82rem;
            margin-top: .3rem;
        }

        .section-title {
            margin-top: 1.2rem;
            margin-bottom: .8rem;
            display: flex;
            align-items: center;
            gap: .65rem;
        }

        .section-number {
            display: inline-grid;
            place-items: center;
            width: 2rem;
            height: 2rem;
            border-radius: 999px;
            background: linear-gradient(135deg, var(--forest), var(--emerald));
            color: #FFFBEA;
            font-size: .82rem;
            font-weight: 850;
        }

        .risk-low, .risk-medium, .risk-high {
            display: inline-flex;
            align-items: center;
            gap: .42rem;
            padding: .48rem .72rem;
            border-radius: 999px;
            font-weight: 800;
            font-size: .84rem;
            border: 1px solid;
        }

        .risk-low {
            color: #16603F;
            background: rgba(31, 122, 77, .11);
            border-color: rgba(31, 122, 77, .22);
        }

        .risk-medium {
            color: #8A641B;
            background: rgba(185, 144, 61, .15);
            border-color: rgba(185, 144, 61, .28);
        }

        .risk-high {
            color: #8B2F27;
            background: rgba(176, 79, 65, .12);
            border-color: rgba(176, 79, 65, .22);
        }

        .insight-card {
            padding: 1rem 1rem 1rem 1.05rem;
            border-radius: 18px;
            background: rgba(255,255,250,.70);
            border: 1px solid rgba(31,122,77,.13);
            border-left: 5px solid var(--emerald);
            margin: .55rem 0;
            box-shadow: 0 10px 24px rgba(18,61,43,.055);
        }

        .action-card {
            padding: .92rem 1rem;
            border-radius: 18px;
            background: rgba(255, 253, 245, .74);
            border: 1px solid rgba(31,122,77,.13);
            margin: .45rem 0;
            display: flex;
            gap: .75rem;
            align-items: flex-start;
        }

        .action-icon {
            flex: 0 0 auto;
            display: grid;
            place-items: center;
            width: 1.75rem;
            height: 1.75rem;
            border-radius: 999px;
            background: rgba(31,122,77,.12);
            color: var(--forest);
            font-weight: 850;
        }

        .footer-note {
            margin-top: 1.5rem;
            padding: 1rem 1.15rem;
            border-radius: 20px;
            background: rgba(18, 61, 43, .06);
            border: 1px solid rgba(18, 61, 43, .10);
            color: #607469;
            font-size: .92rem;
        }

        .stButton > button {
            border-radius: 999px !important;
            border: 1px solid rgba(18, 61, 43, .10) !important;
            background: linear-gradient(135deg, #123D2B 0%, #1F7A4D 100%) !important;
            color: #FFFBEA !important;
            font-weight: 760 !important;
            padding: .78rem 1.15rem !important;
            box-shadow: 0 14px 30px rgba(31, 122, 77, .18) !important;
            transition: transform .16s ease, box-shadow .16s ease, filter .16s ease !important;
        }

        .stButton > button:hover {
            transform: translateY(-1px);
            filter: brightness(1.03);
            box-shadow: 0 18px 42px rgba(31, 122, 77, .24) !important;
        }

        div[data-testid="stDownloadButton"] > button {
            border-radius: 999px !important;
            background: linear-gradient(135deg, #B9903D 0%, #D9BE79 100%) !important;
            color: #143425 !important;
            border: 1px solid rgba(185,144,61,.25) !important;
            font-weight: 800 !important;
        }

        div[data-testid="stTextInput"] input,
        div[data-testid="stNumberInput"] input,
        div[data-testid="stSelectbox"] div[data-baseweb="select"] > div,
        div[data-testid="stTextArea"] textarea {
            border-radius: 15px !important;
            border: 1px solid rgba(31,122,77,.20) !important;
            background-color: rgba(255, 253, 245, .92) !important;
            color: var(--ink) !important;
            box-shadow: none !important;
        }

        div[data-testid="stTextInput"] input:focus,
        div[data-testid="stNumberInput"] input:focus,
        div[data-testid="stTextArea"] textarea:focus {
            border-color: rgba(31,122,77,.45) !important;
            box-shadow: 0 0 0 4px rgba(31,122,77,.09) !important;
        }

        div[data-testid="stSlider"] div[role="slider"] {
            background-color: var(--emerald) !important;
            border-color: var(--emerald) !important;
        }

        div[data-testid="stRadio"] label {
            border-radius: 999px;
        }

        .stTabs [data-baseweb="tab-list"] {
            gap: .55rem;
            background: rgba(255,253,245,.6);
            padding: .45rem;
            border-radius: 999px;
            border: 1px solid rgba(31,122,77,.14);
        }

        .stTabs [data-baseweb="tab"] {
            border-radius: 999px;
            color: var(--forest);
            font-weight: 750;
        }

        .stTabs [aria-selected="true"] {
            background: linear-gradient(135deg, rgba(31,122,77,.14), rgba(185,144,61,.14));
        }

        [data-testid="stExpander"] {
            border: 1px solid rgba(31,122,77,.14) !important;
            border-radius: 19px !important;
            background: rgba(255,253,245,.55) !important;
            box-shadow: 0 10px 24px rgba(18, 61, 43, .05) !important;
        }

        .dataframe {
            border-radius: 18px !important;
        }

        @media (max-width: 920px) {
            .metric-grid { grid-template-columns: repeat(2, minmax(0, 1fr)); }
            .topbar { flex-direction: column; align-items: flex-start; }
        }

        @media (max-width: 640px) {
            .metric-grid { grid-template-columns: 1fr; }
            h1 { font-size: 2.55rem !important; }
            .hero { padding: 1.5rem; border-radius: 26px; }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


# -----------------------------------------------------------------------------
# Data functions
# -----------------------------------------------------------------------------
def normalize_username(name: str) -> str:
    name = (name or "").strip().lower().replace(" ", "_")
    return "".join(ch for ch in name if ch.isalnum() or ch in ["_", "-", "."])


def load_history() -> pd.DataFrame:
    if not HISTORY_FILE.exists():
        return pd.DataFrame(columns=HISTORY_COLUMNS)

    history = pd.read_csv(HISTORY_FILE)

    for col in HISTORY_COLUMNS:
        if col not in history.columns:
            history[col] = np.nan

    return history[HISTORY_COLUMNS]


def save_record(row: dict) -> None:
    history = load_history()
    row_df = pd.DataFrame([row])
    updated = pd.concat([history, row_df], ignore_index=True)
    updated.to_csv(HISTORY_FILE, index=False)


def save_many(rows: list[dict]) -> None:
    history = load_history()
    rows_df = pd.DataFrame(rows)
    updated = pd.concat([history, rows_df], ignore_index=True)
    updated.to_csv(HISTORY_FILE, index=False)


def get_user_history() -> pd.DataFrame:
    username = st.session_state.get("username")
    history = load_history()
    if not username:
        return pd.DataFrame(columns=HISTORY_COLUMNS)
    return history[history["Username"] == username].copy()


def clear_user_history() -> None:
    username = st.session_state.get("username")
    history = load_history()
    history = history[history["Username"] != username]
    history.to_csv(HISTORY_FILE, index=False)


def to_numeric(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    for col in columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


# -----------------------------------------------------------------------------
# AI and model logic
# -----------------------------------------------------------------------------
def get_groq_client():
    if not GROQ_AVAILABLE:
        return None
    try:
        api_key = st.secrets.get("GROQ_API_KEY", None)
    except Exception:
        api_key = None
    if not api_key:
        return None
    return Groq(api_key=api_key)


def ask_ai(prompt: str) -> str | None:
    client = get_groq_client()
    if client is None:
        return None

    try:
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are an AI assistant for a high school environmental impact MVP. "
                        "Give practical, realistic, concise recommendations for reducing food waste. "
                        "Do not guarantee food safety, do not decide donation safety, and always remind users "
                        "that human staff must follow local food safety rules."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.35,
            max_tokens=720,
        )
        return response.choices[0].message.content
    except Exception:
        return None


def risk_level_from_score(score: float) -> str:
    if score >= 70:
        return "High"
    if score >= 42:
        return "Medium"
    return "Low"


def risk_badge(level: str) -> str:
    css_class = {
        "Low": "risk-low",
        "Medium": "risk-medium",
        "High": "risk-high",
    }.get(level, "risk-medium")
    icon = {"Low": "●", "Medium": "●", "High": "●"}.get(level, "●")
    return f'<span class="{css_class}">{icon} {level} Risk</span>'


def expected_attendance_multiplier(weather: str, confidence: str, popularity: int, day: str) -> float:
    multiplier = 1.0

    weather_factors = {
        "Normal": 1.00,
        "Sunny": 1.02,
        "Cloudy": 0.99,
        "Rainy": 0.94,
        "Stormy": 0.88,
        "Very hot": 0.93,
        "Very cold": 0.92,
    }
    confidence_factors = {
        "High": 1.02,
        "Medium": 0.98,
        "Low": 0.91,
    }
    popularity_factors = {
        1: 0.86,
        2: 0.92,
        3: 0.98,
        4: 1.03,
        5: 1.07,
    }
    day_factors = {
        "Monday": 0.97,
        "Tuesday": 1.00,
        "Wednesday": 1.00,
        "Thursday": 1.01,
        "Friday": 0.96,
        "Saturday": 0.95,
        "Sunday": 0.93,
    }

    multiplier *= weather_factors.get(weather, 1.0)
    multiplier *= confidence_factors.get(confidence, 0.98)
    multiplier *= popularity_factors.get(int(popularity), 0.98)
    multiplier *= day_factors.get(day, 1.0)

    return max(0.68, min(1.16, multiplier))


def calculate_recommendation(
    expected_attendance: int,
    weather: str,
    confidence: str,
    menu_popularity: int,
    day_of_week: str,
    batch_cooking: bool,
) -> tuple[int, int, int]:
    expected_attendance = max(int(expected_attendance), 1)
    multiplier = expected_attendance_multiplier(weather, confidence, menu_popularity, day_of_week)
    target = expected_attendance * multiplier

    if batch_cooking:
        low = target * 0.92
        high = target * 1.01
    else:
        low = target * 0.96
        high = target * 1.07

    return max(1, int(round(low))), max(1, int(round(high))), max(1, int(round(target)))


def history_similarity_signal(history: pd.DataFrame, event_type: str, meal_time: str, meal_type: str) -> tuple[float | None, str | None]:
    if history.empty:
        return None, None

    history = to_numeric(history.copy(), ["Waste Rate"])
    relevant = history.dropna(subset=["Waste Rate"])

    if relevant.empty:
        return None, None

    meal_type_lower = str(meal_type or "").strip().lower()

    filters = [
        (relevant["Event Type"].astype(str) == event_type) & (relevant["Meal Time"].astype(str) == meal_time),
        relevant["Event Type"].astype(str) == event_type,
    ]

    if meal_type_lower:
        filters.insert(
            0,
            relevant["Meal / Food Type"].astype(str).str.lower().str.contains(meal_type_lower, regex=False, na=False),
        )

    for mask in filters:
        subset = relevant[mask]
        if len(subset) >= 2:
            avg = float(subset["Waste Rate"].mean())
            label = f"similar records average {avg:.1f}% waste"
            return avg, label

    avg = float(relevant["Waste Rate"].tail(8).mean())
    return avg, f"recent records average {avg:.1f}% waste"


def train_historical_model(history: pd.DataFrame):
    if not SKLEARN_AVAILABLE:
        return None, "scikit-learn is not installed"

    history = history.copy()
    numeric_cols = [
        "Expected Attendance",
        "Food Prepared",
        "Menu Popularity",
        "Cost per Portion",
        "CO2e per Portion",
        "Waste Rate",
    ]
    history = to_numeric(history, numeric_cols)
    history = history.dropna(subset=["Waste Rate", "Expected Attendance", "Food Prepared"])

    if len(history) < 12:
        return None, "Add at least 12 logged records to unlock historical learning"
    if history["Waste Rate"].nunique() < 3:
        return None, "Historical data needs more variation before a model is useful"

    feature_cols = [
        "Event Type",
        "Location",
        "Day of Week",
        "Meal Time",
        "Meal / Food Type",
        "Menu Popularity",
        "Weather",
        "Attendance Confidence",
        "Expected Attendance",
        "Food Prepared",
        "Batch Cooking Available",
        "Intervention Used",
    ]
    for col in feature_cols:
        if col not in history.columns:
            history[col] = ""

    X = history[feature_cols].copy()
    y = history["Waste Rate"].clip(0, 100)

    categorical = [
        "Event Type",
        "Location",
        "Day of Week",
        "Meal Time",
        "Meal / Food Type",
        "Weather",
        "Attendance Confidence",
        "Batch Cooking Available",
        "Intervention Used",
    ]
    numeric = ["Menu Popularity", "Expected Attendance", "Food Prepared"]

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical),
            ("num", "passthrough", numeric),
        ]
    )

    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "model",
                RandomForestRegressor(
                    n_estimators=180,
                    min_samples_leaf=2,
                    random_state=42,
                ),
            ),
        ]
    )

    model.fit(X, y)
    return model, f"Historical learning enabled using {len(history)} records"


def predict_with_historical_model(model, row: dict) -> float | None:
    if model is None:
        return None
    feature_cols = [
        "Event Type",
        "Location",
        "Day of Week",
        "Meal Time",
        "Meal / Food Type",
        "Menu Popularity",
        "Weather",
        "Attendance Confidence",
        "Expected Attendance",
        "Food Prepared",
        "Batch Cooking Available",
        "Intervention Used",
    ]
    X = pd.DataFrame([{col: row.get(col, "") for col in feature_cols}])
    try:
        return float(model.predict(X)[0])
    except Exception:
        return None


def calculate_rule_based_prediction(
    *,
    event_type: str,
    location: str,
    day_of_week: str,
    meal_time: str,
    meal_type: str,
    menu_popularity: int,
    weather: str,
    confidence: str,
    expected_attendance: int,
    food_prepared: int,
    donation_available: bool,
    batch_cooking: bool,
    reusable_serving: bool,
    intervention: str,
    history: pd.DataFrame,
) -> dict:
    expected_attendance = max(int(expected_attendance), 1)
    food_prepared = max(int(food_prepared), 1)

    rec_min, rec_max, predicted_demand = calculate_recommendation(
        expected_attendance,
        weather,
        confidence,
        menu_popularity,
        day_of_week,
        batch_cooking,
    )

    predicted_leftover = max(food_prepared - predicted_demand, 0)
    predicted_waste_rate = predicted_leftover / food_prepared * 100

    overprep_rate = max(food_prepared - expected_attendance, 0) / expected_attendance * 100
    score = 18 + predicted_waste_rate * 1.18 + overprep_rate * 0.33
    drivers = []

    if food_prepared > rec_max:
        score += min(22, (food_prepared - rec_max) / expected_attendance * 70)
        drivers.append(f"Planned portions are above the recommended range ({rec_min}–{rec_max}).")

    if confidence == "Low":
        score += 15
        drivers.append("Attendance confidence is low, so over-preparation risk is higher.")
    elif confidence == "Medium":
        score += 6
        drivers.append("Attendance confidence is only medium; confirmation could improve accuracy.")

    if weather in ["Rainy", "Stormy", "Very hot", "Very cold"]:
        impact = {"Rainy": 8, "Stormy": 15, "Very hot": 10, "Very cold": 10}[weather]
        score += impact
        drivers.append(f"{weather} conditions may reduce turnout or appetite.")

    if int(menu_popularity) <= 2:
        score += 14 if int(menu_popularity) == 1 else 9
        drivers.append("Menu popularity is low, which often increases leftovers.")
    elif int(menu_popularity) >= 4:
        score -= 4

    if day_of_week in ["Friday", "Monday"]:
        score += 4
        drivers.append(f"{day_of_week} can be less predictable for school attendance and meal demand.")

    if event_type in ["Community event", "Fundraiser", "Donation program"]:
        score += 4
        drivers.append("Community events often have more uncertain attendance than regular lunches.")

    if batch_cooking:
        score -= 9
        drivers.append("Batch cooking lowers risk because food can be prepared in stages.")
    if reusable_serving:
        score -= 4
    if intervention != "None yet":
        score -= 6
        drivers.append(f"Intervention planned: {intervention}.")

    history_avg, history_label = history_similarity_signal(history, event_type, meal_time, meal_type)
    if history_avg is not None:
        if history_avg >= 25:
            score += 10
            drivers.append(f"Historical signal: {history_label}.")
        elif history_avg >= 12:
            score += 4
            drivers.append(f"Historical signal: {history_label}.")
        elif history_avg < 8:
            score -= 4
            drivers.append(f"Historical signal: {history_label}.")

    score = float(np.clip(score, 5, 96))
    risk_level = risk_level_from_score(score)

    rescue_capacity = min(predicted_leftover, 20 if donation_available else 0)

    return {
        "risk_score": round(score, 1),
        "risk_level": risk_level,
        "predicted_demand": int(predicted_demand),
        "predicted_leftover": int(round(predicted_leftover)),
        "predicted_waste_rate": round(predicted_waste_rate, 1),
        "recommended_min": int(rec_min),
        "recommended_max": int(rec_max),
        "rescue_capacity": int(rescue_capacity),
        "drivers": drivers[:5] if drivers else ["No major red flags. Continue tracking actual attendance and leftovers."],
    }


def generate_action_plan(result: dict, donation_available: bool, batch_cooking: bool, intervention: str) -> list[tuple[str, str]]:
    score = result["risk_score"]
    rec_min = result["recommended_min"]
    rec_max = result["recommended_max"]
    predicted_left = result["predicted_leftover"]

    actions = [
        ("Set the prep range", f"Prepare about {rec_min}–{rec_max} portions instead of treating the original plan as fixed."),
        ("Confirm attendance", "Use a quick form, homeroom count, or club RSVP 24 hours before food is prepared."),
    ]

    if score >= 70:
        actions.insert(1, ("Reduce first batch", "Start with 70–80% of expected demand and hold the rest as a flexible backup."))
    elif score >= 42:
        actions.insert(1, ("Keep a controlled buffer", "Prepare a modest buffer, but avoid cooking the full surplus upfront."))

    if not batch_cooking:
        actions.append(("Add batch control", "Split preparation into smaller batches so staff can stop early if turnout is lower than expected."))

    if predicted_left > 0:
        if donation_available:
            actions.append(("Prepare rescue route", "Pre-label possible surplus for human review and contact the donation partner only if staff confirm safety."))
        else:
            actions.append(("Create a rescue backup", "Identify a verified donation, staff meal, or compost route before the event starts."))

    if intervention == "None yet":
        actions.append(("Track one intervention", "Choose one simple intervention this time so the dashboard can compare before vs. after results."))

    return actions[:6]


def calculate_actual_waste(food_prepared: int, leftover: int) -> float:
    food_prepared = max(int(food_prepared), 1)
    leftover = max(int(leftover), 0)
    return leftover / food_prepared * 100


def create_demo_rows(username: str) -> list[dict]:
    base = [
        ("School lunch", "Cafeteria", "Monday", "Lunch", "Pasta", 3, "Normal", "Medium", 118, 111, 132, 28, "None yet"),
        ("School lunch", "Cafeteria", "Tuesday", "Lunch", "Chicken rice bowl", 5, "Sunny", "High", 120, 123, 124, 7, "Attendance confirmation"),
        ("School lunch", "Cafeteria", "Wednesday", "Lunch", "Vegetarian chili", 2, "Cloudy", "Medium", 115, 100, 126, 34, "Menu preference survey"),
        ("After-school club", "Classroom", "Thursday", "Snack", "Fruit cups", 4, "Normal", "High", 42, 39, 45, 6, "Pre-order form"),
        ("Sports event", "Gym", "Friday", "Dinner", "Sandwiches", 3, "Rainy", "Low", 95, 72, 120, 43, "None yet"),
        ("Breakfast program", "Cafeteria", "Monday", "Breakfast", "Bagels", 4, "Very cold", "Medium", 70, 58, 80, 19, "Smaller first batch"),
        ("Community event", "Community hall", "Saturday", "All-day event", "Pizza", 5, "Sunny", "Medium", 160, 148, 170, 18, "Donation partner ready"),
        ("Fundraiser", "Event venue", "Sunday", "Dinner", "Hot dogs", 3, "Stormy", "Low", 140, 91, 160, 62, "Compost plan"),
        ("School lunch", "Cafeteria", "Friday", "Lunch", "Fish tacos", 2, "Normal", "Medium", 117, 99, 128, 31, "None yet"),
        ("Donation program", "Library / common area", "Wednesday", "Snack", "Bakery items", 4, "Cloudy", "Medium", 60, 52, 70, 12, "Donation partner ready"),
        ("School lunch", "Cafeteria", "Thursday", "Lunch", "Rice noodle bowl", 4, "Sunny", "High", 122, 121, 126, 8, "Attendance confirmation"),
        ("Community event", "Outdoor area", "Saturday", "Lunch", "BBQ plates", 5, "Very hot", "Medium", 180, 151, 195, 39, "Smaller first batch"),
        ("After-school club", "Classroom", "Tuesday", "Snack", "Granola bars", 5, "Normal", "High", 36, 35, 38, 2, "Pre-order form"),
        ("School lunch", "Cafeteria", "Wednesday", "Lunch", "Mac and cheese", 5, "Normal", "High", 125, 129, 130, 5, "Attendance confirmation"),
        ("Sports event", "Gym", "Friday", "Dinner", "Wraps", 3, "Cloudy", "Low", 110, 88, 135, 41, "None yet"),
        ("Breakfast program", "Cafeteria", "Thursday", "Breakfast", "Oatmeal cups", 2, "Normal", "Medium", 72, 63, 82, 22, "Menu preference survey"),
    ]

    rows = []
    for i, item in enumerate(base):
        (
            event_type,
            location,
            day,
            meal_time,
            meal_type,
            popularity,
            weather,
            confidence,
            expected,
            actual,
            prepared,
            leftover,
            intervention,
        ) = item

        waste_rate = calculate_actual_waste(prepared, leftover)
        risk_score = min(96, max(5, waste_rate * 2.5 + (prepared - actual) / max(prepared, 1) * 25))
        risk_level = risk_level_from_score(risk_score)
        cost = leftover * 2.5
        co2 = leftover * 0.5

        rows.append(
            {
                "Username": username,
                "Time": (pd.Timestamp.now() - pd.Timedelta(days=20 - i)).strftime("%Y-%m-%d %H:%M:%S"),
                "Event Type": event_type,
                "Location": location,
                "Day of Week": day,
                "Meal Time": meal_time,
                "Meal / Food Type": meal_type,
                "Menu Popularity": popularity,
                "Weather": weather,
                "Attendance Confidence": confidence,
                "Expected Attendance": expected,
                "Actual Attendance": actual,
                "Food Prepared": prepared,
                "Leftover Portions": leftover,
                "Waste Rate": round(waste_rate, 2),
                "Predicted Waste Rate": round(waste_rate * np.random.uniform(0.82, 1.12), 2),
                "Risk Score": round(risk_score, 1),
                "Risk Level": risk_level,
                "Recommended Min Portions": max(1, int(actual * 0.94)),
                "Recommended Max Portions": max(1, int(actual * 1.05)),
                "Donation Partner Available": "Yes" if intervention == "Donation partner ready" else "No",
                "Batch Cooking Available": "Yes" if intervention == "Smaller first batch" else "No",
                "Reusable Serving Available": "Yes" if i % 4 == 0 else "No",
                "Intervention Used": intervention,
                "Cost per Portion": 2.5,
                "CO2e per Portion": 0.5,
                "Estimated Cost Impact": round(cost, 2),
                "Estimated CO2e Impact": round(co2, 2),
                "Potential Meals Rescued": min(leftover, 20 if intervention == "Donation partner ready" else 0),
                "Notes": "Demo record for judging and testing.",
            }
        )
    return rows


# -----------------------------------------------------------------------------
# UI helpers
# -----------------------------------------------------------------------------
def render_topbar() -> None:
    username = st.session_state.get("username")
    st.markdown(
        f"""
        <div class="topbar">
            <div class="brand"><span class="brand-mark">🌿</span><span>{APP_TITLE}</span></div>
            <div class="user-pill">Signed in as <b>{username}</b></div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_hero(kicker: str, title: str, subtitle: str, chips: list[str] | None = None) -> None:
    chip_html = ""
    if chips:
        chip_html = '<div class="hero-chips">' + "".join([f'<span class="chip">{chip}</span>' for chip in chips]) + "</div>"

    st.markdown(
        f"""
        <div class="hero">
            <div class="eyebrow">{kicker}</div>
            <h1>{title}</h1>
            <div class="hero-subtitle">{subtitle}</div>
            {chip_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def metric_card(label: str, value: str, caption: str = "") -> str:
    return f"""
    <div class="metric-card">
        <div class="metric-label">{label}</div>
        <div class="metric-value">{value}</div>
        <div class="metric-caption">{caption}</div>
    </div>
    """


def metric_grid(cards: list[tuple[str, str, str]]) -> None:
    st.markdown(
        '<div class="metric-grid">' + "".join(metric_card(*card) for card in cards) + "</div>",
        unsafe_allow_html=True,
    )


def section_header(number: str, title: str) -> None:
    st.markdown(
        f"""
        <div class="section-title">
            <span class="section-number">{number}</span>
            <h2 style="margin:0;">{title}</h2>
        </div>
        """,
        unsafe_allow_html=True,
    )


def insight(text: str) -> None:
    st.markdown(f'<div class="insight-card">{text}</div>', unsafe_allow_html=True)


def action_card(i: int, title: str, body: str) -> None:
    st.markdown(
        f"""
        <div class="action-card">
            <div class="action-icon">{i}</div>
            <div><b>{title}</b><br><span class="muted">{body}</span></div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def get_default_username_page() -> None:
    inject_css()
    st.markdown("<br>", unsafe_allow_html=True)
    render_hero(
        "High School Track · AI for Everyday Good",
        "Food waste intelligence, made calm.",
        (
            "A premium environmental MVP for schools and community groups. "
            "Predict high-risk meals, understand why waste happens, and turn leftovers into a practical rescue plan."
        ),
        ["AI risk scoring", "Pattern detection", "Rescue planning", "Impact dashboard"],
    )

    col1, col2 = st.columns([1.05, 0.95], gap="large")
    with col1:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.subheader("Start a workspace")
        username = st.text_input("Workspace name", placeholder="Example: green_school_team")
        c1, c2 = st.columns([1, 1])
        with c1:
            if st.button("Enter app", use_container_width=True):
                username = normalize_username(username)
                if not username:
                    st.error("Please enter a workspace name.")
                else:
                    st.session_state.username = username
                    st.session_state.page = "Mission Control"
                    st.rerun()
        with c2:
            if st.button("Try demo workspace", use_container_width=True):
                st.session_state.username = "demo_green_team"
                st.session_state.page = "Mission Control"
                existing = get_user_history()
                if existing.empty:
                    save_many(create_demo_rows("demo_green_team"))
                st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="panel-soft">', unsafe_allow_html=True)
        st.markdown("### What makes it realistic")
        st.markdown(
            """
            - Uses attendance, weather, menu popularity, preparation amount, and past records.
            - Gives a recommended portion range instead of a vague warning.
            - Separates food-waste planning from food-safety decisions.
            - Learns from logged results when enough history exists.
            """
        )
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown(
        '<div class="footer-note">Responsible AI note: this app supports planning only. Human staff must inspect food, follow local food safety rules, and decide whether leftovers can be served, donated, composted, or discarded.</div>',
        unsafe_allow_html=True,
    )


def navigation() -> str:
    pages = [
        "Mission Control",
        "Waste Risk Scanner",
        "Event Result Logger",
        "Impact Intelligence",
        "Data & Export",
    ]
    current = st.session_state.get("page", "Mission Control")
    if current not in pages:
        current = "Mission Control"

    choice = st.radio(
        "Navigation",
        pages,
        index=pages.index(current),
        horizontal=True,
        label_visibility="collapsed",
    )
    st.session_state.page = choice
    return choice


# -----------------------------------------------------------------------------
# Pages
# -----------------------------------------------------------------------------
def page_mission_control() -> None:
    render_hero(
        "AI for Everyday Good",
        "Predict less. Waste less. Rescue more.",
        (
            "A clean decision system for cafeterias, clubs, events, and donation programs. "
            "Use the scanner before food is prepared, log real results after the event, then let the dashboard reveal patterns."
        ),
        ["Luxury green interface", "Simple for students", "Advanced under the hood"],
    )

    history = get_user_history()
    history_num = to_numeric(
        history.copy(),
        ["Waste Rate", "Leftover Portions", "Estimated Cost Impact", "Estimated CO2e Impact", "Potential Meals Rescued"],
    )

    if history_num.empty:
        metric_grid([
            ("Records", "0", "Load demo data or log an event"),
            ("Avg waste", "—", "No records yet"),
            ("Meals rescued", "—", "Needs results"),
            ("Model mode", "Starter", "Rule-based engine active"),
        ])
    else:
        avg_waste = history_num["Waste Rate"].mean()
        total_left = history_num["Leftover Portions"].sum()
        total_rescued = history_num["Potential Meals Rescued"].sum()
        model, model_msg = train_historical_model(history_num)
        metric_grid([
            ("Records", f"{len(history_num)}", "Saved event results"),
            ("Avg waste", f"{avg_waste:.1f}%", "Across logged events"),
            ("Leftovers", f"{total_left:.0f}", "Total portions"),
            ("Model mode", "Learning" if model else "Starter", model_msg),
        ])

    col1, col2, col3 = st.columns(3, gap="medium")
    with col1:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.markdown("### 01 · Scan")
        st.write("Forecast waste risk before food is prepared.")
        if st.button("Open Risk Scanner", use_container_width=True):
            st.session_state.page = "Waste Risk Scanner"
            st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.markdown("### 02 · Log")
        st.write("Record actual attendance and leftovers after the event.")
        if st.button("Log Event Result", use_container_width=True):
            st.session_state.page = "Event Result Logger"
            st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)

    with col3:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.markdown("### 03 · Learn")
        st.write("Reveal the meals, days, and events that create waste.")
        if st.button("View Intelligence", use_container_width=True):
            st.session_state.page = "Impact Intelligence"
            st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)

    section_header("A", "Quick setup")
    c1, c2, c3 = st.columns(3, gap="medium")
    with c1:
        if st.button("Load demo data", use_container_width=True):
            save_many(create_demo_rows(st.session_state.username))
            st.success("Demo records added. Open Impact Intelligence to see the dashboard.")
    with c2:
        if st.button("Switch workspace", use_container_width=True):
            st.session_state.username = None
            st.session_state.page = "Mission Control"
            st.rerun()
    with c3:
        if st.button("Clear this workspace data", use_container_width=True):
            clear_user_history()
            st.success("Workspace data cleared.")
            st.rerun()

    if not history_num.empty:
        section_header("B", "Latest insight")
        worst = history_num.dropna(subset=["Waste Rate"]).sort_values("Waste Rate", ascending=False).head(1)
        if not worst.empty:
            r = worst.iloc[0]
            insight(
                f"<b>Highest recent waste:</b> {r['Meal / Food Type']} at {r['Event Type']} "
                f"had <b>{float(r['Waste Rate']):.1f}%</b> waste. "
                f"Check whether attendance confirmation, menu preference, or batch cooking could reduce this pattern."
            )


def collect_event_inputs(prefix: str = "scan", actual: bool = False) -> dict:
    with st.container():
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.markdown("### Event essentials")
        c1, c2, c3 = st.columns(3, gap="medium")

        with c1:
            event_type = st.selectbox("Event type", EVENT_TYPES, key=f"{prefix}_event_type")
            location = st.selectbox("Location", LOCATIONS, key=f"{prefix}_location")
            day = st.selectbox("Day of week", DAYS, index=4, key=f"{prefix}_day")

        with c2:
            meal_time = st.selectbox("Meal time", MEAL_TIMES, index=1, key=f"{prefix}_meal_time")
            meal_type = st.text_input("Meal / food type", placeholder="Example: pizza, pasta, sandwiches", key=f"{prefix}_meal_type")
            weather = st.selectbox("Expected weather / condition", WEATHER_OPTIONS, key=f"{prefix}_weather")

        with c3:
            popularity = st.slider("Menu popularity", 1, 5, 3, help="1 = unpopular, 5 = very popular", key=f"{prefix}_pop")
            confidence = st.selectbox("Attendance confidence", CONFIDENCE_OPTIONS, index=1, key=f"{prefix}_confidence")
            intervention = st.selectbox("Intervention planned / used", INTERVENTIONS, key=f"{prefix}_intervention")

        st.markdown("### Planning numbers")
        n1, n2, n3, n4 = st.columns(4, gap="medium")
        with n1:
            expected = st.number_input("Expected attendance", min_value=1, value=100, step=1, key=f"{prefix}_expected")
        with n2:
            prepared = st.number_input("Food prepared / planned", min_value=1, value=110, step=1, key=f"{prefix}_prepared")
        with n3:
            cost = st.number_input("Cost per portion ($)", min_value=0.0, value=2.50, step=0.10, format="%.2f", key=f"{prefix}_cost")
        with n4:
            co2 = st.number_input("CO₂e per portion (kg)", min_value=0.0, value=0.50, step=0.05, format="%.2f", key=f"{prefix}_co2")

        actual_attendance = None
        leftover = None
        if actual:
            a1, a2 = st.columns(2, gap="medium")
            with a1:
                actual_attendance = st.number_input("Actual attendance", min_value=0, value=max(1, int(expected * 0.9)), step=1, key=f"{prefix}_actual")
            with a2:
                leftover = st.number_input("Leftover portions", min_value=0, value=max(0, int(prepared * 0.15)), step=1, key=f"{prefix}_leftover")

        with st.expander("Advanced operational signals", expanded=False):
            e1, e2, e3 = st.columns(3, gap="medium")
            with e1:
                donation_available = st.toggle("Verified donation / redistribution route exists", value=False, key=f"{prefix}_donation")
            with e2:
                batch_cooking = st.toggle("Food can be prepared in smaller batches", value=True, key=f"{prefix}_batch")
            with e3:
                reusable_serving = st.toggle("Reusable / adjustable serving is available", value=False, key=f"{prefix}_reusable")

            notes = st.text_area("Notes", placeholder="Example: exam week, club meeting, unpopular menu last time...", key=f"{prefix}_notes")
        st.markdown("</div>", unsafe_allow_html=True)

    return {
        "Event Type": event_type,
        "Location": location,
        "Day of Week": day,
        "Meal Time": meal_time,
        "Meal / Food Type": meal_type.strip() if meal_type else "Unknown",
        "Menu Popularity": int(popularity),
        "Weather": weather,
        "Attendance Confidence": confidence,
        "Expected Attendance": int(expected),
        "Actual Attendance": actual_attendance,
        "Food Prepared": int(prepared),
        "Leftover Portions": leftover,
        "Donation Partner Available": "Yes" if donation_available else "No",
        "Batch Cooking Available": "Yes" if batch_cooking else "No",
        "Reusable Serving Available": "Yes" if reusable_serving else "No",
        "Intervention Used": intervention,
        "Cost per Portion": float(cost),
        "CO2e per Portion": float(co2),
        "Notes": notes,
    }


def page_waste_risk_scanner() -> None:
    render_hero(
        "Pre-event planning",
        "Waste Risk Scanner",
        (
            "Enter a simple food plan. The scanner estimates the risk of over-preparation, "
            "recommends a smarter portion range, and builds an action plan that students can actually use."
        ),
        ["Before the event", "AI + rules + history", "What-if simulator"],
    )

    inputs = collect_event_inputs("scan", actual=False)

    history = get_user_history()
    model, model_msg = train_historical_model(history)

    if st.button("Scan this food plan", use_container_width=True):
        result = calculate_rule_based_prediction(
            event_type=inputs["Event Type"],
            location=inputs["Location"],
            day_of_week=inputs["Day of Week"],
            meal_time=inputs["Meal Time"],
            meal_type=inputs["Meal / Food Type"],
            menu_popularity=inputs["Menu Popularity"],
            weather=inputs["Weather"],
            confidence=inputs["Attendance Confidence"],
            expected_attendance=inputs["Expected Attendance"],
            food_prepared=inputs["Food Prepared"],
            donation_available=inputs["Donation Partner Available"] == "Yes",
            batch_cooking=inputs["Batch Cooking Available"] == "Yes",
            reusable_serving=inputs["Reusable Serving Available"] == "Yes",
            intervention=inputs["Intervention Used"],
            history=history,
        )

        ml_pred = predict_with_historical_model(model, inputs)
        if ml_pred is not None:
            blended_pred = result["predicted_waste_rate"] * 0.55 + ml_pred * 0.45
            result["predicted_waste_rate"] = round(float(np.clip(blended_pred, 0, 100)), 1)
            result["risk_score"] = round(float(np.clip(result["risk_score"] * 0.65 + blended_pred * 1.6 * 0.35, 5, 96)), 1)
            result["risk_level"] = risk_level_from_score(result["risk_score"])
            model_label = f"Historical learning active · model predicts {ml_pred:.1f}% waste"
        else:
            model_label = model_msg

        section_header("1", "Risk result")
        st.markdown(risk_badge(result["risk_level"]), unsafe_allow_html=True)
        metric_grid([
            ("Risk score", f"{result['risk_score']:.0f}/100", "Higher means more likely to waste"),
            ("Predicted waste", f"{result['predicted_waste_rate']:.1f}%", "Estimated before the event"),
            ("Recommended prep", f"{result['recommended_min']}–{result['recommended_max']}", "Suggested portion range"),
            ("Model mode", "Learning" if ml_pred is not None else "Starter", model_label),
        ])

        section_header("2", "Why this is happening")
        for item in result["drivers"]:
            insight(item)

        section_header("3", "Rescue action plan")
        actions = generate_action_plan(
            result,
            inputs["Donation Partner Available"] == "Yes",
            inputs["Batch Cooking Available"] == "Yes",
            inputs["Intervention Used"],
        )
        for i, (title, body) in enumerate(actions, start=1):
            action_card(i, title, body)

        section_header("4", "What-if simulator")
        st.write("Compare the current plan against a lower-waste prep range.")
        min_sim = max(1, int(inputs["Expected Attendance"] * 0.65))
        max_sim = max(min_sim + 1, int(inputs["Expected Attendance"] * 1.45))
        selected = st.slider(
            "Test a different preparation amount",
            min_value=min_sim,
            max_value=max_sim,
            value=int(inputs["Food Prepared"]),
            step=1,
            key="what_if_slider",
        )

        sim_rows = []
        for portions in range(min_sim, max_sim + 1):
            sim_result = calculate_rule_based_prediction(
                event_type=inputs["Event Type"],
                location=inputs["Location"],
                day_of_week=inputs["Day of Week"],
                meal_time=inputs["Meal Time"],
                meal_type=inputs["Meal / Food Type"],
                menu_popularity=inputs["Menu Popularity"],
                weather=inputs["Weather"],
                confidence=inputs["Attendance Confidence"],
                expected_attendance=inputs["Expected Attendance"],
                food_prepared=portions,
                donation_available=inputs["Donation Partner Available"] == "Yes",
                batch_cooking=inputs["Batch Cooking Available"] == "Yes",
                reusable_serving=inputs["Reusable Serving Available"] == "Yes",
                intervention=inputs["Intervention Used"],
                history=history,
            )
            shortage = max(sim_result["predicted_demand"] - portions, 0)
            sim_rows.append(
                {
                    "Prepared Portions": portions,
                    "Predicted Waste %": sim_result["predicted_waste_rate"],
                    "Estimated Leftovers": sim_result["predicted_leftover"],
                    "Shortage Risk Portions": shortage,
                }
            )

        sim_df = pd.DataFrame(sim_rows)
        selected_row = sim_df[sim_df["Prepared Portions"] == selected].iloc[0]
        metric_grid([
            ("Tested prep", f"{selected}", "Portions"),
            ("Waste estimate", f"{selected_row['Predicted Waste %']:.1f}%", "For tested prep"),
            ("Leftovers", f"{selected_row['Estimated Leftovers']:.0f}", "Estimated portions"),
            ("Shortage risk", f"{selected_row['Shortage Risk Portions']:.0f}", "Estimated portions"),
        ])

        if PLOTLY_AVAILABLE:
            fig = go.Figure()
            fig.add_trace(
                go.Scatter(
                    x=sim_df["Prepared Portions"],
                    y=sim_df["Predicted Waste %"],
                    mode="lines",
                    name="Predicted waste %",
                    line=dict(color="#1F7A4D", width=4),
                    fill="tozeroy",
                    fillcolor="rgba(31,122,77,0.12)",
                )
            )
            fig.add_vline(x=result["recommended_min"], line_width=2, line_dash="dash", line_color="#B9903D")
            fig.add_vline(x=result["recommended_max"], line_width=2, line_dash="dash", line_color="#B9903D")
            fig.update_layout(
                height=360,
                margin=dict(l=10, r=10, t=20, b=10),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(255,255,250,0.55)",
                font=dict(color="#153729"),
                xaxis_title="Prepared portions",
                yaxis_title="Predicted waste rate",
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.line_chart(sim_df.set_index("Prepared Portions")["Predicted Waste %"])

        section_header("5", "Optional AI coach")
        ai_prompt = f"""
        Create a concise food waste prevention plan for this school/community event.
        Keep it practical for high school students.

        Data:
        {json.dumps({**inputs, **result}, indent=2)}
        """
        ai_answer = ask_ai(ai_prompt)
        if ai_answer:
            st.markdown('<div class="panel-soft">', unsafe_allow_html=True)
            st.write(ai_answer)
            st.markdown("</div>", unsafe_allow_html=True)
        else:
            st.info("AI Coach is optional. Add GROQ_API_KEY in Streamlit Secrets to enable it. The app still runs with the built-in risk engine.")

        st.markdown(
            '<div class="footer-note">Responsible AI note: predicted leftovers are planning estimates only. Human staff must inspect food and follow local food safety rules before donation, reuse, composting, or disposal.</div>',
            unsafe_allow_html=True,
        )


def page_event_result_logger() -> None:
    render_hero(
        "Learn from real results",
        "Event Result Logger",
        (
            "Use this after an event to save what actually happened. These records power pattern detection, impact reporting, and the historical learning model."
        ),
        ["After the event", "Actual leftovers", "Dashboard data"],
    )

    inputs = collect_event_inputs("logger", actual=True)

    if st.button("Save event result", use_container_width=True):
        food_prepared = int(inputs["Food Prepared"])
        leftover = int(inputs["Leftover Portions"] or 0)
        actual_attendance = int(inputs["Actual Attendance"] or 0)

        if leftover > food_prepared:
            st.error("Leftover portions cannot be greater than food prepared.")
            return

        history = get_user_history()
        result = calculate_rule_based_prediction(
            event_type=inputs["Event Type"],
            location=inputs["Location"],
            day_of_week=inputs["Day of Week"],
            meal_time=inputs["Meal Time"],
            meal_type=inputs["Meal / Food Type"],
            menu_popularity=inputs["Menu Popularity"],
            weather=inputs["Weather"],
            confidence=inputs["Attendance Confidence"],
            expected_attendance=inputs["Expected Attendance"],
            food_prepared=food_prepared,
            donation_available=inputs["Donation Partner Available"] == "Yes",
            batch_cooking=inputs["Batch Cooking Available"] == "Yes",
            reusable_serving=inputs["Reusable Serving Available"] == "Yes",
            intervention=inputs["Intervention Used"],
            history=history,
        )

        waste_rate = calculate_actual_waste(food_prepared, leftover)
        risk_score = float(np.clip(waste_rate * 2.45 + max(food_prepared - actual_attendance, 0) / max(food_prepared, 1) * 22, 5, 96))
        risk_level = risk_level_from_score(risk_score)
        cost_impact = leftover * float(inputs["Cost per Portion"])
        co2_impact = leftover * float(inputs["CO2e per Portion"])
        potential_rescued = min(leftover, 20 if inputs["Donation Partner Available"] == "Yes" else 0)

        row = {
            "Username": st.session_state.username,
            "Time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            **inputs,
            "Waste Rate": round(waste_rate, 2),
            "Predicted Waste Rate": result["predicted_waste_rate"],
            "Risk Score": round(risk_score, 1),
            "Risk Level": risk_level,
            "Recommended Min Portions": result["recommended_min"],
            "Recommended Max Portions": result["recommended_max"],
            "Estimated Cost Impact": round(cost_impact, 2),
            "Estimated CO2e Impact": round(co2_impact, 2),
            "Potential Meals Rescued": int(potential_rescued),
        }
        save_record(row)

        st.success("Event result saved.")
        section_header("1", "Impact receipt")
        st.markdown(risk_badge(risk_level), unsafe_allow_html=True)
        attendance_gap = int(inputs["Expected Attendance"]) - actual_attendance
        metric_grid([
            ("Actual waste", f"{waste_rate:.1f}%", "Leftover / prepared"),
            ("Attendance gap", f"{attendance_gap}", "Expected minus actual"),
            ("Cost impact", f"${cost_impact:.0f}", "Estimated"),
            ("CO₂e impact", f"{co2_impact:.1f} kg", "Estimated"),
        ])

        if waste_rate >= 25:
            insight("This event should be reviewed. The highest-value next step is to reduce the first batch and confirm attendance closer to the event.")
        elif waste_rate >= 10:
            insight("This event had moderate waste. Try a smaller buffer, pre-orders, or menu preference tracking next time.")
        else:
            insight("This result is strong. Keep tracking it so the app can learn what low-waste planning looks like.")


def page_impact_intelligence() -> None:
    render_hero(
        "Pattern detection",
        "Impact Intelligence",
        (
            "A dashboard that turns event logs into environmental, cost, and planning insights. "
            "Use this page to prove that the MVP understands impact, not just inputs."
        ),
        ["Patterns", "Charts", "Intervention ROI", "Historical learning"],
    )

    history = get_user_history()
    if history.empty:
        st.info("No records yet. Load demo data from Mission Control or log an event result.")
        return

    num_cols = [
        "Expected Attendance",
        "Actual Attendance",
        "Food Prepared",
        "Leftover Portions",
        "Waste Rate",
        "Predicted Waste Rate",
        "Risk Score",
        "Estimated Cost Impact",
        "Estimated CO2e Impact",
        "Potential Meals Rescued",
    ]
    history = to_numeric(history.copy(), num_cols)
    history["Time"] = pd.to_datetime(history["Time"], errors="coerce")
    history = history.sort_values("Time")

    total_records = len(history)
    avg_waste = history["Waste Rate"].mean()
    total_cost = history["Estimated Cost Impact"].sum()
    total_co2 = history["Estimated CO2e Impact"].sum()
    total_rescued = history["Potential Meals Rescued"].sum()

    metric_grid([
        ("Records", f"{total_records}", "Logged events"),
        ("Average waste", f"{avg_waste:.1f}%", "Across records"),
        ("Cost impact", f"${total_cost:.0f}", "Estimated waste cost"),
        ("Meals rescued", f"{total_rescued:.0f}", "Potential portions"),
    ])

    model, model_msg = train_historical_model(history)
    insight(f"<b>Model status:</b> {model_msg}.")

    c1, c2 = st.columns([1.1, 0.9], gap="large")
    with c1:
        section_header("1", "Waste trend")
        if PLOTLY_AVAILABLE:
            chart_df = history.dropna(subset=["Time", "Waste Rate"])
            fig = go.Figure()
            fig.add_trace(
                go.Scatter(
                    x=chart_df["Time"],
                    y=chart_df["Waste Rate"],
                    mode="lines+markers",
                    name="Waste rate",
                    line=dict(color="#1F7A4D", width=4),
                    marker=dict(size=8, color="#B9903D"),
                )
            )
            fig.update_layout(
                height=350,
                margin=dict(l=10, r=10, t=20, b=10),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(255,255,250,0.55)",
                font=dict(color="#153729"),
                xaxis_title="Date",
                yaxis_title="Waste rate %",
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.line_chart(history.set_index("Time")["Waste Rate"])

    with c2:
        section_header("2", "Risk mix")
        mix = history["Risk Level"].fillna("Unknown").value_counts().reset_index()
        mix.columns = ["Risk Level", "Events"]
        if PLOTLY_AVAILABLE and not mix.empty:
            fig = px.pie(
                mix,
                values="Events",
                names="Risk Level",
                hole=0.62,
                color="Risk Level",
                color_discrete_map={"Low": "#1F7A4D", "Medium": "#B9903D", "High": "#B85C4B"},
            )
            fig.update_layout(
                height=350,
                margin=dict(l=10, r=10, t=20, b=10),
                paper_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#153729"),
                showlegend=True,
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.dataframe(mix, use_container_width=True)

    section_header("3", "Waste drivers")
    g1, g2 = st.columns(2, gap="large")

    with g1:
        meal_chart = history.groupby("Meal / Food Type", dropna=False)["Waste Rate"].mean().sort_values(ascending=False).head(8)
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.markdown("### Highest-waste foods")
        if PLOTLY_AVAILABLE and not meal_chart.empty:
            fig = px.bar(
                meal_chart.reset_index(),
                x="Waste Rate",
                y="Meal / Food Type",
                orientation="h",
                color="Waste Rate",
                color_continuous_scale=["#DCEBDD", "#1F7A4D", "#123D2B"],
            )
            fig.update_layout(
                height=360,
                margin=dict(l=10, r=10, t=20, b=10),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(255,255,250,0.50)",
                font=dict(color="#153729"),
                coloraxis_showscale=False,
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.bar_chart(meal_chart)
        st.markdown("</div>", unsafe_allow_html=True)

    with g2:
        day_chart = history.groupby("Day of Week", dropna=False)["Waste Rate"].mean().reindex(DAYS).dropna()
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.markdown("### Waste by day")
        if PLOTLY_AVAILABLE and not day_chart.empty:
            fig = px.bar(
                day_chart.reset_index(),
                x="Day of Week",
                y="Waste Rate",
                color="Waste Rate",
                color_continuous_scale=["#DCEBDD", "#1F7A4D", "#123D2B"],
            )
            fig.update_layout(
                height=360,
                margin=dict(l=10, r=10, t=20, b=10),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(255,255,250,0.50)",
                font=dict(color="#153729"),
                coloraxis_showscale=False,
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.bar_chart(day_chart)
        st.markdown("</div>", unsafe_allow_html=True)

    section_header("4", "Intervention signal")
    intervention = history.groupby("Intervention Used", dropna=False)["Waste Rate"].agg(["count", "mean"]).sort_values("mean")
    intervention = intervention.rename(columns={"count": "Events", "mean": "Avg Waste Rate"})
    st.dataframe(intervention, use_container_width=True)

    if not intervention.empty:
        best = intervention.head(1).index[0]
        worst = intervention.tail(1).index[0]
        insight(f"<b>Best current intervention:</b> {best}. <b>Highest-waste category:</b> {worst}. Use this as a discussion point, not a final conclusion, because small datasets can be noisy.")

    section_header("5", "Recent records")
    display_cols = [
        "Time",
        "Event Type",
        "Meal / Food Type",
        "Day of Week",
        "Expected Attendance",
        "Actual Attendance",
        "Food Prepared",
        "Leftover Portions",
        "Waste Rate",
        "Risk Level",
        "Intervention Used",
    ]
    st.dataframe(history[display_cols].tail(12), use_container_width=True)


def create_report(history: pd.DataFrame) -> str:
    if history.empty:
        return "No records available yet."

    history = to_numeric(
        history.copy(),
        ["Waste Rate", "Leftover Portions", "Estimated Cost Impact", "Estimated CO2e Impact", "Potential Meals Rescued"],
    )
    avg = history["Waste Rate"].mean()
    total_left = history["Leftover Portions"].sum()
    total_cost = history["Estimated Cost Impact"].sum()
    total_co2 = history["Estimated CO2e Impact"].sum()
    total_rescued = history["Potential Meals Rescued"].sum()

    worst_meal = history.groupby("Meal / Food Type")["Waste Rate"].mean().sort_values(ascending=False).head(1)
    worst_day = history.groupby("Day of Week")["Waste Rate"].mean().sort_values(ascending=False).head(1)

    lines = [
        "# Food Waste Rescue Radar · Impact Report",
        "",
        f"Workspace: {st.session_state.username}",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        "",
        "## Summary",
        f"- Total logged events: {len(history)}",
        f"- Average waste rate: {avg:.1f}%",
        f"- Total leftover portions: {total_left:.0f}",
        f"- Estimated cost impact: ${total_cost:.0f}",
        f"- Estimated CO₂e impact: {total_co2:.1f} kg",
        f"- Potential meals rescued: {total_rescued:.0f}",
        "",
        "## Pattern insights",
    ]

    if not worst_meal.empty:
        lines.append(f"- Highest-waste food type: {worst_meal.index[0]} ({worst_meal.iloc[0]:.1f}% average waste)")
    if not worst_day.empty:
        lines.append(f"- Highest-waste day: {worst_day.index[0]} ({worst_day.iloc[0]:.1f}% average waste)")

    lines.extend(
        [
            "",
            "## Recommended next actions",
            "- Confirm attendance 24 hours before food preparation.",
            "- Use smaller first batches for high-risk events.",
            "- Track menu popularity and compare it with leftover patterns.",
            "- Prepare a verified donation, redistribution, or compost route before large events.",
            "- Keep human food-safety review separate from AI planning recommendations.",
            "",
            "Responsible AI note: This report supports planning only. Human staff must follow local food safety rules.",
        ]
    )

    return "\n".join(lines)


def page_data_export() -> None:
    render_hero(
        "Evidence and reporting",
        "Data & Export",
        (
            "Review the raw records, download CSV data, and generate a clean markdown report for presentations or judging."
        ),
        ["CSV export", "Impact report", "Transparent data"],
    )

    history = get_user_history()
    if history.empty:
        st.info("No records yet. Load demo data or log event results first.")
        return

    section_header("1", "Data table")
    st.dataframe(history.tail(50), use_container_width=True)

    csv = history.to_csv(index=False).encode("utf-8")
    report = create_report(history).encode("utf-8")

    c1, c2 = st.columns(2, gap="medium")
    with c1:
        st.download_button(
            "Download CSV data",
            data=csv,
            file_name=f"{st.session_state.username}_food_waste_data.csv",
            mime="text/csv",
            use_container_width=True,
        )
    with c2:
        st.download_button(
            "Download impact report",
            data=report,
            file_name=f"{st.session_state.username}_impact_report.md",
            mime="text/markdown",
            use_container_width=True,
        )

    section_header("2", "Report preview")
    st.markdown(create_report(history))


# -----------------------------------------------------------------------------
# Main app
# -----------------------------------------------------------------------------
def main() -> None:
    inject_css()

    if "username" not in st.session_state:
        st.session_state.username = None
    if "page" not in st.session_state:
        st.session_state.page = "Mission Control"

    if st.session_state.username is None:
        get_default_username_page()
        return

    render_topbar()
    page = navigation()

    if page == "Mission Control":
        page_mission_control()
    elif page == "Waste Risk Scanner":
        page_waste_risk_scanner()
    elif page == "Event Result Logger":
        page_event_result_logger()
    elif page == "Impact Intelligence":
        page_impact_intelligence()
    elif page == "Data & Export":
        page_data_export()


if __name__ == "__main__":
    main()
