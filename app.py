from __future__ import annotations

import math
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

try:
    from groq import Groq
except Exception:  # pragma: no cover - app still works without the package/key
    Groq = None

try:
    from sklearn.compose import ColumnTransformer
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder
except Exception:  # pragma: no cover - app still works in rule-based mode
    ColumnTransformer = None
    RandomForestRegressor = None
    Pipeline = None
    OneHotEncoder = None


# ------------------------------------------------------------
# Page setup
# ------------------------------------------------------------
st.set_page_config(
    page_title="Food Waste Rescue Radar",
    page_icon="♻️",
    layout="wide",
    initial_sidebar_state="expanded",
)

APP_NAME = "Food Waste Rescue Radar"
TAGLINE = "AI-powered food waste intelligence for schools and community events."
HISTORY_FILE = Path("historical_data.csv")

HISTORY_COLUMNS = [
    "Username",
    "Timestamp",
    "Record Type",
    "Event Type",
    "Location",
    "Day of Week",
    "Meal Time",
    "Meal Type",
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
    "Donation Partner Available",
    "Donation Capacity",
    "Batch Cooking Available",
    "Storage Capacity",
    "Intervention Used",
    "Potential Meals Rescued",
    "Estimated CO2 Kg",
    "Estimated Cost CAD",
    "Estimated Cost Saved CAD",
    "Notes",
]

EVENT_TYPES = [
    "School lunch",
    "Breakfast program",
    "After-school club",
    "Sports event",
    "Community event",
    "Fundraiser / banquet",
    "Grocery donation program",
]

LOCATIONS = [
    "Cafeteria",
    "Classroom",
    "Gym / Hall",
    "Outdoor field",
    "Community centre",
    "Donation pickup point",
    "Party / Event venue",
]

DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
MEAL_TIMES = ["Breakfast", "Lunch", "Dinner", "Snack / Break", "After-school", "Full-day event"]
WEATHER_OPTIONS = ["Normal", "Sunny", "Cloudy", "Rainy", "Stormy", "Very hot", "Snowy / icy"]
ATTENDANCE_CONFIDENCE = ["High", "Medium", "Low"]
INTERVENTIONS = [
    "None yet",
    "Pre-order / RSVP",
    "Smaller first batch",
    "Menu popularity check",
    "Donation partner on standby",
    "Compost plan",
    "Student awareness campaign",
    "Portion size adjustment",
]

WEATHER_ATTENDANCE_MODIFIER = {
    "Normal": 1.00,
    "Sunny": 1.02,
    "Cloudy": 0.98,
    "Rainy": 0.90,
    "Stormy": 0.78,
    "Very hot": 0.92,
    "Snowy / icy": 0.86,
}

DAY_ATTENDANCE_MODIFIER = {
    "Monday": 0.97,
    "Tuesday": 1.00,
    "Wednesday": 1.00,
    "Thursday": 0.99,
    "Friday": 0.93,
    "Saturday": 0.91,
    "Sunday": 0.88,
}

MEAL_TIME_MODIFIER = {
    "Breakfast": 0.88,
    "Lunch": 1.00,
    "Dinner": 0.96,
    "Snack / Break": 0.92,
    "After-school": 0.87,
    "Full-day event": 0.95,
}

CONFIDENCE_RISK_POINTS = {"High": 2, "Medium": 10, "Low": 20}
WEATHER_RISK_POINTS = {
    "Normal": 0,
    "Sunny": 1,
    "Cloudy": 3,
    "Rainy": 11,
    "Stormy": 20,
    "Very hot": 12,
    "Snowy / icy": 15,
}

FOOD_SAFETY_NOTE = (
    "Responsible AI note: this app supports planning only. Human staff must inspect food, "
    "follow local food safety rules, and decide whether leftovers can be served, stored, "
    "donated, composted, or discarded."
)


# ------------------------------------------------------------
# Premium CSS
# ------------------------------------------------------------
def inject_css() -> None:
    st.markdown(
        """
        <style>
        :root {
            --bg: #0E1111;
            --panel: rgba(255,255,255,0.055);
            --panel-strong: rgba(255,255,255,0.09);
            --text: #F8F4EC;
            --muted: #B8B0A4;
            --gold: #C8A45D;
            --sage: #AFC7A3;
            --green: #1F4D3A;
            --danger: #E57373;
            --warning: #E6B85C;
        }
        .stApp {
            background:
                radial-gradient(circle at top left, rgba(200,164,93,0.16), transparent 32%),
                radial-gradient(circle at top right, rgba(31,77,58,0.20), transparent 38%),
                linear-gradient(135deg, #0E1111 0%, #111715 46%, #0A0B0B 100%);
            color: var(--text);
        }
        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, rgba(15,20,18,0.98), rgba(7,8,8,0.98));
            border-right: 1px solid rgba(200,164,93,0.18);
        }
        h1, h2, h3 {
            letter-spacing: -0.03em;
        }
        .hero {
            padding: 2.3rem 2rem;
            border: 1px solid rgba(200,164,93,0.25);
            border-radius: 28px;
            background: linear-gradient(135deg, rgba(255,255,255,0.08), rgba(255,255,255,0.025));
            box-shadow: 0 24px 80px rgba(0,0,0,0.36);
            margin-bottom: 1.3rem;
        }
        .eyebrow {
            color: var(--gold);
            font-size: 0.78rem;
            font-weight: 700;
            letter-spacing: .18em;
            text-transform: uppercase;
            margin-bottom: .6rem;
        }
        .hero-title {
            font-size: clamp(2.2rem, 5vw, 4.4rem);
            line-height: 0.95;
            font-weight: 800;
            color: #FFF9ED;
            margin-bottom: 1rem;
        }
        .hero-subtitle {
            max-width: 880px;
            color: #D8D0C3;
            font-size: 1.08rem;
            line-height: 1.65;
        }
        .glass-card {
            padding: 1.15rem 1.25rem;
            border-radius: 22px;
            border: 1px solid rgba(255,255,255,0.11);
            background: linear-gradient(145deg, rgba(255,255,255,0.08), rgba(255,255,255,0.025));
            box-shadow: 0 18px 42px rgba(0,0,0,0.18);
            height: 100%;
        }
        .card-label {
            color: var(--gold);
            font-size: .78rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: .14em;
        }
        .card-title {
            font-size: 1.22rem;
            font-weight: 780;
            margin: .35rem 0 .45rem 0;
            color: #FFF9ED;
        }
        .card-copy {
            color: #CFC7B9;
            font-size: .96rem;
            line-height: 1.55;
        }
        .lux-note {
            color: #D8D0C3;
            border-left: 3px solid var(--gold);
            padding: .72rem 1rem;
            background: rgba(200,164,93,0.08);
            border-radius: 14px;
        }
        div[data-testid="stMetric"] {
            border: 1px solid rgba(255,255,255,0.10);
            background: rgba(255,255,255,0.055);
            border-radius: 18px;
            padding: 1rem;
            box-shadow: 0 10px 32px rgba(0,0,0,0.15);
        }
        div[data-testid="stMetricLabel"] p { color: #C8A45D !important; }
        div[data-testid="stMetricValue"] { color: #FFF9ED !important; }
        .stButton > button {
            border-radius: 999px;
            border: 1px solid rgba(200,164,93,.45);
            background: linear-gradient(135deg, rgba(200,164,93,.90), rgba(150,112,45,.88));
            color: #111 !important;
            font-weight: 750;
            box-shadow: 0 12px 28px rgba(200,164,93,.16);
        }
        .stButton > button:hover {
            border-color: #F2D28D;
            filter: brightness(1.06);
        }
        div[data-baseweb="select"] > div, .stTextInput input, .stNumberInput input, textarea {
            border-radius: 14px !important;
        }
        .small-muted { color: #B8B0A4; font-size: .9rem; }
        .status-high { color: #E57373; font-weight: 800; }
        .status-medium { color: #E6B85C; font-weight: 800; }
        .status-low { color: #AFC7A3; font-weight: 800; }
        .divider { height: 1px; background: rgba(200,164,93,0.18); margin: 1.1rem 0; }
        footer {visibility: hidden;}
        </style>
        """,
        unsafe_allow_html=True,
    )


# ------------------------------------------------------------
# Storage helpers
# ------------------------------------------------------------
def clean_username(username: str) -> str:
    return username.strip().lower().replace(" ", "_")


def load_history() -> pd.DataFrame:
    if not HISTORY_FILE.exists():
        return pd.DataFrame(columns=HISTORY_COLUMNS)

    try:
        df = pd.read_csv(HISTORY_FILE)
    except Exception:
        return pd.DataFrame(columns=HISTORY_COLUMNS)

    for col in HISTORY_COLUMNS:
        if col not in df.columns:
            df[col] = np.nan

    return df[HISTORY_COLUMNS]


def save_record(row: Dict[str, Any]) -> None:
    history = load_history()
    complete = {col: row.get(col, np.nan) for col in HISTORY_COLUMNS}
    updated = pd.concat([history, pd.DataFrame([complete])], ignore_index=True)
    updated.to_csv(HISTORY_FILE, index=False)


def save_many(rows: List[Dict[str, Any]]) -> None:
    for row in rows:
        save_record(row)


def get_user_history(username: Optional[str] = None) -> pd.DataFrame:
    history = load_history()
    user = username or st.session_state.get("username")
    if not user:
        return history.iloc[0:0]
    return history[history["Username"].astype(str) == user].copy()


def numeric(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    out = df.copy()
    for col in columns:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


def bool_label(value: bool) -> str:
    return "Yes" if value else "No"


def yes_no_to_bool(value: Any) -> bool:
    return str(value).strip().lower() in {"yes", "true", "1", "available"}


# ------------------------------------------------------------
# AI helper
# ------------------------------------------------------------
def get_groq_client() -> Optional[Any]:
    if Groq is None:
        return None

    try:
        api_key = st.secrets.get("GROQ_API_KEY", None)
    except Exception:
        api_key = None

    if not api_key:
        return None

    try:
        return Groq(api_key=api_key)
    except Exception:
        return None


def ask_ai(prompt: str) -> Optional[str]:
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
                        "You are an expert but practical food waste reduction advisor for schools. "
                        "Be specific, realistic, concise, and safe. Do not give food safety guarantees. "
                        "Always remind users that human staff must make food safety decisions."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.35,
            max_tokens=900,
        )
        return response.choices[0].message.content
    except Exception:
        return None


# ------------------------------------------------------------
# Risk engine and model
# ------------------------------------------------------------
def get_risk_level(score: float) -> str:
    if score >= 70:
        return "High"
    if score >= 40:
        return "Medium"
    return "Low"


def get_status_class(level: str) -> str:
    if level == "High":
        return "status-high"
    if level == "Medium":
        return "status-medium"
    return "status-low"


def popularity_modifier(menu_popularity: int) -> float:
    return {1: 0.68, 2: 0.82, 3: 0.94, 4: 1.00, 5: 1.06}.get(int(menu_popularity), 0.94)


def confidence_buffer(confidence: str) -> float:
    return {"High": 0.04, "Medium": 0.08, "Low": 0.13}.get(confidence, 0.08)


def build_current_feature_row(inputs: Dict[str, Any]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "Event Type": inputs.get("event_type"),
                "Location": inputs.get("location"),
                "Day of Week": inputs.get("day_of_week"),
                "Meal Time": inputs.get("meal_time"),
                "Meal Type": inputs.get("meal_type") or "Unknown",
                "Menu Popularity": inputs.get("menu_popularity", 3),
                "Weather": inputs.get("weather"),
                "Attendance Confidence": inputs.get("attendance_confidence"),
                "Expected Attendance": inputs.get("expected_attendance", 0),
                "Food Prepared": inputs.get("food_prepared", 0),
                "Donation Partner Available": bool_label(inputs.get("donation_partner_available", False)),
                "Batch Cooking Available": bool_label(inputs.get("batch_cooking_available", False)),
                "Storage Capacity": inputs.get("storage_capacity", 0),
            }
        ]
    )


def train_historical_model(history: pd.DataFrame) -> Tuple[Optional[Any], str, int]:
    if any(x is None for x in [ColumnTransformer, RandomForestRegressor, Pipeline, OneHotEncoder]):
        return None, "Rule-based mode", 0

    df = history.copy()
    df = df[df["Record Type"].isin(["Actual Result", "Demo Actual Result"])]
    df = numeric(
        df,
        [
            "Menu Popularity",
            "Expected Attendance",
            "Food Prepared",
            "Waste Rate",
            "Storage Capacity",
        ],
    )
    df = df.dropna(subset=["Waste Rate", "Expected Attendance", "Food Prepared"])

    if len(df) < 10:
        return None, f"Rule-based mode — add {10 - len(df)} more actual records to unlock learning", len(df)

    features = [
        "Event Type",
        "Location",
        "Day of Week",
        "Meal Time",
        "Meal Type",
        "Menu Popularity",
        "Weather",
        "Attendance Confidence",
        "Expected Attendance",
        "Food Prepared",
        "Donation Partner Available",
        "Batch Cooking Available",
        "Storage Capacity",
    ]

    for col in features:
        if col not in df.columns:
            df[col] = "Unknown"

    X = df[features]
    y = df["Waste Rate"].clip(lower=0, upper=100)

    categorical = [
        "Event Type",
        "Location",
        "Day of Week",
        "Meal Time",
        "Meal Type",
        "Weather",
        "Attendance Confidence",
        "Donation Partner Available",
        "Batch Cooking Available",
    ]
    numerical = ["Menu Popularity", "Expected Attendance", "Food Prepared", "Storage Capacity"]

    preprocessor = ColumnTransformer(
        transformers=[
            ("categorical", OneHotEncoder(handle_unknown="ignore"), categorical),
            ("numerical", "passthrough", numerical),
        ]
    )

    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "model",
                RandomForestRegressor(
                    n_estimators=220,
                    random_state=42,
                    min_samples_leaf=2,
                    max_depth=8,
                ),
            ),
        ]
    )

    model.fit(X, y)
    return model, f"Hybrid learning mode — trained on {len(df)} actual records", len(df)


def rule_based_forecast(inputs: Dict[str, Any]) -> Dict[str, Any]:
    expected_attendance = max(float(inputs.get("expected_attendance", 0)), 0)
    planned = max(float(inputs.get("food_prepared", 0)), 0)
    menu_popularity = int(inputs.get("menu_popularity", 3))
    weather = inputs.get("weather", "Normal")
    day = inputs.get("day_of_week", "Tuesday")
    meal_time = inputs.get("meal_time", "Lunch")
    confidence = inputs.get("attendance_confidence", "Medium")
    donation_available = bool(inputs.get("donation_partner_available", False))
    batch_available = bool(inputs.get("batch_cooking_available", False))
    donation_capacity = max(float(inputs.get("donation_capacity", 0)), 0)
    storage_capacity = max(float(inputs.get("storage_capacity", 0)), 0)
    cost_per_portion = max(float(inputs.get("cost_per_portion", 2.5)), 0)
    co2_per_portion = max(float(inputs.get("co2_per_portion", 0.5)), 0)

    attendance_modifier = (
        WEATHER_ATTENDANCE_MODIFIER.get(weather, 1.0)
        * DAY_ATTENDANCE_MODIFIER.get(day, 1.0)
        * MEAL_TIME_MODIFIER.get(meal_time, 1.0)
    )
    predicted_attendance = expected_attendance * attendance_modifier
    predicted_consumption = predicted_attendance * popularity_modifier(menu_popularity)

    if planned <= 0:
        rule_waste_rate = 0.0
        overprep_portions = 0.0
        shortage_risk = max(predicted_consumption, 0.0)
    else:
        overprep_portions = max(planned - predicted_consumption, 0.0)
        shortage_risk = max(predicted_consumption - planned, 0.0)
        rule_waste_rate = min(max(overprep_portions / planned * 100, 0.0), 90.0)

    buffer = confidence_buffer(confidence)
    if batch_available:
        recommended_min = math.ceil(predicted_consumption * 0.92)
        recommended_max = math.ceil(predicted_consumption * (1.02 + buffer * 0.35))
    else:
        recommended_min = math.ceil(predicted_consumption * 0.95)
        recommended_max = math.ceil(predicted_consumption * (1.05 + buffer))

    recommended_min = max(recommended_min, 0)
    recommended_max = max(recommended_max, recommended_min)

    overprep_rate_vs_recommended = 0.0
    if recommended_max > 0:
        overprep_rate_vs_recommended = max((planned - recommended_max) / recommended_max * 100, 0)

    drivers: Dict[str, float] = {}
    drivers["Predicted leftover rate"] = rule_waste_rate * 1.35
    drivers["Attendance uncertainty"] = CONFIDENCE_RISK_POINTS.get(confidence, 10)
    drivers[f"Weather: {weather}"] = WEATHER_RISK_POINTS.get(weather, 0)
    drivers["Over-preparation vs smart range"] = min(overprep_rate_vs_recommended * 0.55, 22)
    drivers["Low menu popularity"] = max((4 - menu_popularity) * 7, 0)
    drivers["No batch cooking flexibility"] = 8 if not batch_available else 0
    drivers["No rescue partner on standby"] = 7 if not donation_available else 0
    drivers["Limited storage capacity"] = 6 if storage_capacity < max(overprep_portions * 0.35, 5) else 0

    risk_score = min(max(sum(drivers.values()), 0), 100)
    risk_level = get_risk_level(risk_score)

    potential_rescued = 0.0
    if donation_available:
        potential_rescued += min(overprep_portions, donation_capacity)
    potential_rescued += min(max(overprep_portions - potential_rescued, 0), storage_capacity)

    waste_after_rescue = max(overprep_portions - potential_rescued, 0)
    estimated_cost = overprep_portions * cost_per_portion
    estimated_cost_saved = min(potential_rescued, overprep_portions) * cost_per_portion
    estimated_co2 = overprep_portions * co2_per_portion

    sorted_drivers = sorted(
        [(k, round(v, 1)) for k, v in drivers.items() if v > 0.5],
        key=lambda x: x[1],
        reverse=True,
    )[:5]

    return {
        "rule_waste_rate": round(rule_waste_rate, 1),
        "predicted_waste_rate": round(rule_waste_rate, 1),
        "predicted_attendance": round(predicted_attendance, 1),
        "predicted_consumption": round(predicted_consumption, 1),
        "overprep_portions": round(overprep_portions, 1),
        "shortage_risk_portions": round(shortage_risk, 1),
        "recommended_min": recommended_min,
        "recommended_max": recommended_max,
        "risk_score": round(risk_score, 0),
        "risk_level": risk_level,
        "top_drivers": sorted_drivers,
        "potential_rescued": round(potential_rescued, 1),
        "waste_after_rescue": round(waste_after_rescue, 1),
        "estimated_cost": round(estimated_cost, 2),
        "estimated_cost_saved": round(estimated_cost_saved, 2),
        "estimated_co2": round(estimated_co2, 2),
        "model_mode": "Rule-based mode",
    }


def evaluate_event(inputs: Dict[str, Any], history: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
    metrics = rule_based_forecast(inputs)
    user_history = history if history is not None else get_user_history()
    model, mode, n_records = train_historical_model(user_history)

    if model is not None:
        feature_row = build_current_feature_row(inputs)
        try:
            ml_pred = float(model.predict(feature_row)[0])
            blended = (0.58 * ml_pred) + (0.42 * metrics["rule_waste_rate"])
            metrics["predicted_waste_rate"] = round(min(max(blended, 0), 90), 1)
            metrics["ml_predicted_waste_rate"] = round(ml_pred, 1)
            # Recalibrate risk score using learned prediction.
            recalibrated_score = min(max(metrics["risk_score"] * 0.55 + metrics["predicted_waste_rate"] * 1.65, 0), 100)
            metrics["risk_score"] = round(recalibrated_score, 0)
            metrics["risk_level"] = get_risk_level(recalibrated_score)
        except Exception:
            pass

    metrics["model_mode"] = mode
    metrics["training_records"] = n_records
    return metrics


def actual_waste_metrics(inputs: Dict[str, Any]) -> Dict[str, Any]:
    prepared = max(float(inputs.get("food_prepared", 0)), 0)
    leftover = max(float(inputs.get("leftover_portions", 0)), 0)
    actual_attendance = max(float(inputs.get("actual_attendance", 0)), 0)
    expected_attendance = max(float(inputs.get("expected_attendance", 0)), 0)
    cost_per_portion = max(float(inputs.get("cost_per_portion", 2.5)), 0)
    co2_per_portion = max(float(inputs.get("co2_per_portion", 0.5)), 0)
    donation_available = bool(inputs.get("donation_partner_available", False))
    donation_capacity = max(float(inputs.get("donation_capacity", 0)), 0)
    storage_capacity = max(float(inputs.get("storage_capacity", 0)), 0)

    waste_rate = 0 if prepared == 0 else min(leftover / prepared * 100, 100)
    attendance_gap = expected_attendance - actual_attendance
    rescued = 0.0
    if donation_available:
        rescued += min(leftover, donation_capacity)
    rescued += min(max(leftover - rescued, 0), storage_capacity)
    discarded_estimate = max(leftover - rescued, 0)

    risk_score = min(waste_rate * 2.2 + max(attendance_gap, 0) * 0.3, 100)

    return {
        "waste_rate": round(waste_rate, 1),
        "risk_score": round(risk_score, 0),
        "risk_level": get_risk_level(risk_score),
        "attendance_gap": round(attendance_gap, 1),
        "rescued": round(rescued, 1),
        "discarded_estimate": round(discarded_estimate, 1),
        "estimated_cost": round(leftover * cost_per_portion, 2),
        "estimated_co2": round(leftover * co2_per_portion, 2),
        "estimated_cost_saved": round(rescued * cost_per_portion, 2),
    }


def build_action_plan(inputs: Dict[str, Any], metrics: Dict[str, Any]) -> Dict[str, List[str]]:
    risk_level = metrics.get("risk_level", "Medium")
    planned = int(inputs.get("food_prepared", 0))
    recommended_min = int(metrics.get("recommended_min", 0))
    recommended_max = int(metrics.get("recommended_max", 0))
    overprep = metrics.get("overprep_portions", 0)
    donation_available = bool(inputs.get("donation_partner_available", False))
    batch_available = bool(inputs.get("batch_cooking_available", False))
    confidence = inputs.get("attendance_confidence", "Medium")

    before = [
        f"Set the smart preparation target to {recommended_min}–{recommended_max} portions instead of {planned} if feasible.",
        "Confirm attendance or collect simple pre-orders 24 hours before the event.",
        "Separate food into a first batch and a backup batch so the team can pause production if turnout is low.",
    ]
    during = [
        "Track attendance during the first 15–30 minutes and compare it with the expected count.",
        "Use smaller serving portions first, then offer seconds if demand is strong.",
        "Record which menu items are skipped or repeatedly returned uneaten.",
    ]
    after = [
        "Log actual attendance, prepared portions, and leftover portions in the dashboard immediately after the event.",
        "Label leftover categories for human review: unopened, untouched, opened, compost only, or discard.",
        "Use the dashboard to compare waste before and after interventions.",
    ]

    if risk_level == "High":
        before.insert(0, "High-risk alert: reduce over-preparation before the event rather than relying only on donation afterward.")
    if confidence == "Low":
        before.append("Because attendance confidence is low, prepare a conservative first batch and keep shelf-stable backup options available.")
    if not batch_available:
        during.append("Because batch cooking is unavailable, use smaller initial plating and avoid opening all items at once.")
    if donation_available:
        after.append(f"If human food-safety checks approve, route up to the available rescue capacity for donation/storage instead of disposal.")
    else:
        after.append("No donation partner is listed, so create a verified rescue contact list before the next high-risk event.")
    if overprep > 15:
        before.append(f"The plan may create about {overprep:.0f} excess portions; assign one student/staff member to monitor rescue options.")

    return {"Before event": before[:5], "During event": during[:5], "After event": after[:5]}


def scenario_table(inputs: Dict[str, Any], history: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    base_planned = int(inputs.get("food_prepared", 0))
    variants = []
    candidates = [
        ("Current plan", base_planned),
        ("Reduce 10%", max(int(base_planned * 0.90), 0)),
        ("Reduce 20%", max(int(base_planned * 0.80), 0)),
        ("Smart range low", None),
        ("Smart range high", None),
    ]
    base_metrics = evaluate_event(inputs, history)
    candidates[3] = ("Smart range low", int(base_metrics["recommended_min"]))
    candidates[4] = ("Smart range high", int(base_metrics["recommended_max"]))

    for label, prepared in candidates:
        scenario_inputs = inputs.copy()
        scenario_inputs["food_prepared"] = prepared
        metrics = evaluate_event(scenario_inputs, history)
        shortage = metrics.get("shortage_risk_portions", 0)
        variants.append(
            {
                "Scenario": label,
                "Prepared Portions": prepared,
                "Predicted Waste Rate": metrics["predicted_waste_rate"],
                "Risk Score": metrics["risk_score"],
                "Risk Level": metrics["risk_level"],
                "Expected Excess Portions": metrics["overprep_portions"],
                "Shortage Risk Portions": shortage,
                "Estimated Cost Exposure": metrics["estimated_cost"],
                "Potential Meals Rescued": metrics["potential_rescued"],
            }
        )
    return pd.DataFrame(variants)


# ------------------------------------------------------------
# Demo data
# ------------------------------------------------------------
def make_demo_rows(username: str) -> List[Dict[str, Any]]:
    base = datetime.now() - timedelta(days=58)
    rows: List[Dict[str, Any]] = []
    examples = [
        ("School lunch", "Cafeteria", "Monday", "Lunch", "Pasta", 3, "Normal", "High", 180, 172, 190, 31, "None yet"),
        ("School lunch", "Cafeteria", "Tuesday", "Lunch", "Chicken rice", 5, "Sunny", "High", 185, 188, 190, 8, "Menu popularity check"),
        ("Breakfast program", "Cafeteria", "Wednesday", "Breakfast", "Bagels", 4, "Cloudy", "Medium", 95, 84, 105, 18, "Pre-order / RSVP"),
        ("After-school club", "Classroom", "Wednesday", "After-school", "Fruit cups", 2, "Rainy", "Low", 55, 34, 70, 29, "None yet"),
        ("Sports event", "Gym / Hall", "Friday", "Dinner", "Pizza", 5, "Normal", "Medium", 120, 104, 140, 21, "Smaller first batch"),
        ("Community event", "Community centre", "Saturday", "Full-day event", "Sandwiches", 3, "Stormy", "Low", 160, 103, 190, 61, "Donation partner on standby"),
        ("Fundraiser / banquet", "Gym / Hall", "Friday", "Dinner", "Vegetarian curry", 2, "Very hot", "Medium", 145, 118, 170, 47, "Donation partner on standby"),
        ("School lunch", "Cafeteria", "Thursday", "Lunch", "Tacos", 5, "Sunny", "High", 190, 193, 198, 9, "Portion size adjustment"),
        ("School lunch", "Cafeteria", "Friday", "Lunch", "Fish sandwiches", 2, "Rainy", "Medium", 175, 139, 190, 52, "Student awareness campaign"),
        ("Breakfast program", "Cafeteria", "Monday", "Breakfast", "Cereal packs", 3, "Snowy / icy", "Low", 100, 71, 115, 37, "None yet"),
        ("Grocery donation program", "Donation pickup point", "Wednesday", "Full-day event", "Produce boxes", 4, "Normal", "Medium", 75, 73, 82, 6, "Donation partner on standby"),
        ("School lunch", "Cafeteria", "Tuesday", "Lunch", "Noodles", 4, "Cloudy", "High", 188, 181, 194, 14, "Smaller first batch"),
        ("After-school club", "Classroom", "Thursday", "After-school", "Muffins", 3, "Normal", "Medium", 48, 42, 55, 11, "Pre-order / RSVP"),
        ("Community event", "Outdoor field", "Sunday", "Snack / Break", "Hot dogs", 4, "Very hot", "Low", 210, 151, 230, 63, "Compost plan"),
        ("School lunch", "Cafeteria", "Wednesday", "Lunch", "Mac and cheese", 5, "Normal", "High", 190, 195, 198, 7, "Menu popularity check"),
        ("Sports event", "Gym / Hall", "Saturday", "Dinner", "Wraps", 3, "Rainy", "Medium", 130, 111, 145, 25, "Smaller first batch"),
    ]

    for i, item in enumerate(examples):
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
        inputs = {
            "event_type": event_type,
            "location": location,
            "day_of_week": day,
            "meal_time": meal_time,
            "meal_type": meal_type,
            "menu_popularity": popularity,
            "weather": weather,
            "attendance_confidence": confidence,
            "expected_attendance": expected,
            "actual_attendance": actual,
            "food_prepared": prepared,
            "leftover_portions": leftover,
            "donation_partner_available": intervention == "Donation partner on standby",
            "donation_capacity": 30,
            "batch_cooking_available": intervention in ["Smaller first batch", "Pre-order / RSVP"],
            "storage_capacity": 12,
            "cost_per_portion": 2.75,
            "co2_per_portion": 0.5,
        }
        actual = actual_waste_metrics(inputs)
        rows.append(
            {
                "Username": username,
                "Timestamp": (base + timedelta(days=i * 3)).strftime("%Y-%m-%d %H:%M:%S"),
                "Record Type": "Demo Actual Result",
                "Event Type": event_type,
                "Location": location,
                "Day of Week": day,
                "Meal Time": meal_time,
                "Meal Type": meal_type,
                "Menu Popularity": popularity,
                "Weather": weather,
                "Attendance Confidence": confidence,
                "Expected Attendance": expected,
                "Actual Attendance": inputs["actual_attendance"],
                "Food Prepared": prepared,
                "Leftover Portions": leftover,
                "Waste Rate": actual["waste_rate"],
                "Predicted Waste Rate": np.nan,
                "Risk Score": actual["risk_score"],
                "Risk Level": actual["risk_level"],
                "Recommended Min": np.nan,
                "Recommended Max": np.nan,
                "Donation Partner Available": bool_label(inputs["donation_partner_available"]),
                "Donation Capacity": inputs["donation_capacity"],
                "Batch Cooking Available": bool_label(inputs["batch_cooking_available"]),
                "Storage Capacity": inputs["storage_capacity"],
                "Intervention Used": intervention,
                "Potential Meals Rescued": actual["rescued"],
                "Estimated CO2 Kg": actual["estimated_co2"],
                "Estimated Cost CAD": actual["estimated_cost"],
                "Estimated Cost Saved CAD": actual["estimated_cost_saved"],
                "Notes": "Demo record for judging and dashboard testing.",
            }
        )
    return rows


def add_demo_data_for_current_user() -> None:
    username = st.session_state.get("username", "demo_school")
    history = get_user_history(username)
    existing_demo = history[history["Record Type"].astype(str).str.contains("Demo", na=False)]
    if not existing_demo.empty:
        st.info("Demo data is already loaded for this username.")
        return
    save_many(make_demo_rows(username))
    st.success("Demo school cafeteria data loaded. Go to Impact Intelligence to see charts and patterns.")


# ------------------------------------------------------------
# UI helpers
# ------------------------------------------------------------
def hero(title: str, subtitle: str, eyebrow: str = "AI FOR EVERYDAY GOOD") -> None:
    st.markdown(
        f"""
        <div class="hero">
            <div class="eyebrow">{eyebrow}</div>
            <div class="hero-title">{title}</div>
            <div class="hero-subtitle">{subtitle}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def card(label: str, title: str, copy: str) -> None:
    st.markdown(
        f"""
        <div class="glass-card">
            <div class="card-label">{label}</div>
            <div class="card-title">{title}</div>
            <div class="card-copy">{copy}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def risk_gauge(score: float, level: str) -> go.Figure:
    color = "#AFC7A3" if level == "Low" else "#E6B85C" if level == "Medium" else "#E57373"
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=score,
            number={"suffix": "/100", "font": {"size": 42}},
            title={"text": f"Waste Risk Score — {level}", "font": {"size": 20}},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": color},
                "bgcolor": "rgba(255,255,255,0.04)",
                "borderwidth": 0,
                "steps": [
                    {"range": [0, 40], "color": "rgba(175,199,163,0.22)"},
                    {"range": [40, 70], "color": "rgba(230,184,92,0.22)"},
                    {"range": [70, 100], "color": "rgba(229,115,115,0.22)"},
                ],
            },
        )
    )
    fig.update_layout(
        height=315,
        margin=dict(l=20, r=20, t=55, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#F8F4EC"),
    )
    return fig


def driver_bar(drivers: List[Tuple[str, float]]) -> Optional[go.Figure]:
    if not drivers:
        return None
    df = pd.DataFrame(drivers, columns=["Driver", "Points"]).sort_values("Points", ascending=True)
    fig = px.bar(df, x="Points", y="Driver", orientation="h", text="Points")
    fig.update_layout(
        height=320,
        margin=dict(l=10, r=10, t=30, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(255,255,255,0.03)",
        font=dict(color="#F8F4EC"),
        xaxis_title="Risk contribution",
        yaxis_title="",
    )
    fig.update_traces(marker_color="#C8A45D", textposition="outside")
    return fig


def show_action_plan(plan: Dict[str, List[str]]) -> None:
    cols = st.columns(3)
    for col, (phase, items) in zip(cols, plan.items()):
        with col:
            st.markdown(f"### {phase}")
            for item in items:
                st.write(f"• {item}")


def build_report_text(inputs: Dict[str, Any], metrics: Dict[str, Any], plan: Dict[str, List[str]]) -> str:
    lines = [
        f"{APP_NAME} — Rescue Action Report",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        "",
        "Event Snapshot",
        f"- Event type: {inputs.get('event_type')}",
        f"- Location: {inputs.get('location')}",
        f"- Meal: {inputs.get('meal_type') or 'Unknown'} at {inputs.get('meal_time')}",
        f"- Expected attendance: {inputs.get('expected_attendance')}",
        f"- Planned food: {inputs.get('food_prepared')} portions",
        "",
        "AI Risk Summary",
        f"- Waste Risk Score: {metrics.get('risk_score')}/100 ({metrics.get('risk_level')})",
        f"- Predicted waste rate: {metrics.get('predicted_waste_rate')}%",
        f"- Predicted attendance: {metrics.get('predicted_attendance')}",
        f"- Recommended preparation range: {metrics.get('recommended_min')}–{metrics.get('recommended_max')} portions",
        f"- Potential meals rescued: {metrics.get('potential_rescued')}",
        f"- Estimated cost exposure: ${metrics.get('estimated_cost')}",
        f"- Estimated CO2 exposure: {metrics.get('estimated_co2')} kg",
        "",
        "Top Risk Drivers",
    ]
    for driver, points in metrics.get("top_drivers", []):
        lines.append(f"- {driver}: {points} points")
    lines.append("")
    lines.append("Action Plan")
    for phase, items in plan.items():
        lines.append(f"{phase}:")
        for item in items:
            lines.append(f"- {item}")
    lines.append("")
    lines.append(FOOD_SAFETY_NOTE)
    return "\n".join(lines)


def common_event_form(prefix: str, include_actuals: bool = False) -> Dict[str, Any]:
    today_index = datetime.now().weekday()
    col1, col2, col3 = st.columns(3)
    with col1:
        event_type = st.selectbox("Event type", EVENT_TYPES, key=f"{prefix}_event_type")
        location = st.selectbox("Location", LOCATIONS, key=f"{prefix}_location")
        day_of_week = st.selectbox("Day of week", DAYS, index=today_index, key=f"{prefix}_day")
    with col2:
        meal_time = st.selectbox("Meal time", MEAL_TIMES, index=1, key=f"{prefix}_meal_time")
        meal_type = st.text_input("Meal / food type", placeholder="Example: pasta, pizza, sandwiches", key=f"{prefix}_meal")
        weather = st.selectbox("Expected weather / condition", WEATHER_OPTIONS, key=f"{prefix}_weather")
    with col3:
        menu_popularity = st.slider("Menu popularity", 1, 5, 3, help="1 = unpopular, 5 = very popular", key=f"{prefix}_pop")
        attendance_confidence = st.selectbox("Attendance confidence", ATTENDANCE_CONFIDENCE, index=1, key=f"{prefix}_confidence")
        intervention = st.selectbox("Intervention used / planned", INTERVENTIONS, key=f"{prefix}_intervention")

    st.markdown("#### Planning numbers")
    num_cols = st.columns(4)
    with num_cols[0]:
        expected_attendance = st.number_input("Expected attendance", min_value=0, step=1, value=100, key=f"{prefix}_expected")
    with num_cols[1]:
        food_prepared = st.number_input("Food prepared / planned", min_value=0, step=1, value=110, key=f"{prefix}_prepared")
    with num_cols[2]:
        cost_per_portion = st.number_input("Cost per portion ($)", min_value=0.0, step=0.25, value=2.50, key=f"{prefix}_cost")
    with num_cols[3]:
        co2_per_portion = st.number_input("CO₂e per portion (kg)", min_value=0.0, step=0.05, value=0.50, key=f"{prefix}_co2")

    actual_attendance = np.nan
    leftover_portions = np.nan
    if include_actuals:
        actual_cols = st.columns(2)
        with actual_cols[0]:
            actual_attendance = st.number_input("Actual attendance", min_value=0, step=1, value=90, key=f"{prefix}_actual")
        with actual_cols[1]:
            leftover_portions = st.number_input("Leftover portions", min_value=0, step=1, value=18, key=f"{prefix}_leftover")

    st.markdown("#### Rescue capacity")
    rescue_cols = st.columns(4)
    with rescue_cols[0]:
        donation_partner_available = st.toggle("Donation partner available", value=False, key=f"{prefix}_donation")
    with rescue_cols[1]:
        donation_capacity = st.number_input("Donation capacity", min_value=0, step=1, value=20, key=f"{prefix}_donation_capacity")
    with rescue_cols[2]:
        batch_cooking_available = st.toggle("Batch cooking available", value=True, key=f"{prefix}_batch")
    with rescue_cols[3]:
        storage_capacity = st.number_input("Safe storage capacity", min_value=0, step=1, value=10, key=f"{prefix}_storage")

    notes = st.text_area("Notes", placeholder="Example: exam week, club meeting changed rooms, vegetarian option unpopular...", key=f"{prefix}_notes")

    return {
        "event_type": event_type,
        "location": location,
        "day_of_week": day_of_week,
        "meal_time": meal_time,
        "meal_type": meal_type.strip() or "Unknown",
        "menu_popularity": menu_popularity,
        "weather": weather,
        "attendance_confidence": attendance_confidence,
        "expected_attendance": expected_attendance,
        "actual_attendance": actual_attendance,
        "food_prepared": food_prepared,
        "leftover_portions": leftover_portions,
        "cost_per_portion": cost_per_portion,
        "co2_per_portion": co2_per_portion,
        "donation_partner_available": donation_partner_available,
        "donation_capacity": donation_capacity,
        "batch_cooking_available": batch_cooking_available,
        "storage_capacity": storage_capacity,
        "intervention": intervention,
        "notes": notes,
    }


def record_from_forecast(inputs: Dict[str, Any], metrics: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "Username": st.session_state.username,
        "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "Record Type": "Forecast Scan",
        "Event Type": inputs["event_type"],
        "Location": inputs["location"],
        "Day of Week": inputs["day_of_week"],
        "Meal Time": inputs["meal_time"],
        "Meal Type": inputs["meal_type"],
        "Menu Popularity": inputs["menu_popularity"],
        "Weather": inputs["weather"],
        "Attendance Confidence": inputs["attendance_confidence"],
        "Expected Attendance": inputs["expected_attendance"],
        "Actual Attendance": np.nan,
        "Food Prepared": inputs["food_prepared"],
        "Leftover Portions": np.nan,
        "Waste Rate": np.nan,
        "Predicted Waste Rate": metrics["predicted_waste_rate"],
        "Risk Score": metrics["risk_score"],
        "Risk Level": metrics["risk_level"],
        "Recommended Min": metrics["recommended_min"],
        "Recommended Max": metrics["recommended_max"],
        "Donation Partner Available": bool_label(inputs["donation_partner_available"]),
        "Donation Capacity": inputs["donation_capacity"],
        "Batch Cooking Available": bool_label(inputs["batch_cooking_available"]),
        "Storage Capacity": inputs["storage_capacity"],
        "Intervention Used": inputs["intervention"],
        "Potential Meals Rescued": metrics["potential_rescued"],
        "Estimated CO2 Kg": metrics["estimated_co2"],
        "Estimated Cost CAD": metrics["estimated_cost"],
        "Estimated Cost Saved CAD": metrics["estimated_cost_saved"],
        "Notes": inputs["notes"],
    }


def record_from_actual(inputs: Dict[str, Any], actual: Dict[str, Any], forecast: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "Username": st.session_state.username,
        "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "Record Type": "Actual Result",
        "Event Type": inputs["event_type"],
        "Location": inputs["location"],
        "Day of Week": inputs["day_of_week"],
        "Meal Time": inputs["meal_time"],
        "Meal Type": inputs["meal_type"],
        "Menu Popularity": inputs["menu_popularity"],
        "Weather": inputs["weather"],
        "Attendance Confidence": inputs["attendance_confidence"],
        "Expected Attendance": inputs["expected_attendance"],
        "Actual Attendance": inputs["actual_attendance"],
        "Food Prepared": inputs["food_prepared"],
        "Leftover Portions": inputs["leftover_portions"],
        "Waste Rate": actual["waste_rate"],
        "Predicted Waste Rate": forecast["predicted_waste_rate"],
        "Risk Score": actual["risk_score"],
        "Risk Level": actual["risk_level"],
        "Recommended Min": forecast["recommended_min"],
        "Recommended Max": forecast["recommended_max"],
        "Donation Partner Available": bool_label(inputs["donation_partner_available"]),
        "Donation Capacity": inputs["donation_capacity"],
        "Batch Cooking Available": bool_label(inputs["batch_cooking_available"]),
        "Storage Capacity": inputs["storage_capacity"],
        "Intervention Used": inputs["intervention"],
        "Potential Meals Rescued": actual["rescued"],
        "Estimated CO2 Kg": actual["estimated_co2"],
        "Estimated Cost CAD": actual["estimated_cost"],
        "Estimated Cost Saved CAD": actual["estimated_cost_saved"],
        "Notes": inputs["notes"],
    }


# ------------------------------------------------------------
# Pages
# ------------------------------------------------------------
def show_login() -> None:
    hero(
        "Food Waste Rescue Radar",
        "A premium MVP that helps schools predict food waste, detect patterns, and create realistic rescue actions before meals are wasted.",
    )
    col1, col2 = st.columns([1.1, 0.9])
    with col1:
        st.markdown("### Enter a demo username")
        username = st.text_input("Username", placeholder="example: green_school_team")
        if st.button("Enter Command Center", use_container_width=True):
            username = clean_username(username)
            if not username:
                st.error("Please enter a username.")
            else:
                st.session_state.username = username
                st.session_state.page = "Mission Control"
                st.rerun()
    with col2:
        card(
            "Built for judges",
            "Practical, explainable, and safe",
            "The app combines rule-based forecasting, optional historical machine learning, AI-generated action plans, and human food-safety reminders.",
        )


def show_sidebar() -> None:
    st.sidebar.markdown(f"### ♻️ {APP_NAME}")
    st.sidebar.caption(f"Signed in as **{st.session_state.username}**")
    pages = [
        "Mission Control",
        "Waste Risk Scanner",
        "Event Result Logger",
        "Impact Intelligence",
        "Data & Export",
    ]
    current = st.session_state.get("page", pages[0])
    page = st.sidebar.radio("Navigation", pages, index=pages.index(current) if current in pages else 0)
    st.session_state.page = page
    st.sidebar.markdown("---")
    if st.sidebar.button("Load demo data", use_container_width=True):
        add_demo_data_for_current_user()
    if st.sidebar.button("Switch user", use_container_width=True):
        st.session_state.username = None
        st.session_state.page = "Mission Control"
        st.rerun()
    st.sidebar.markdown("---")
    st.sidebar.caption(FOOD_SAFETY_NOTE)


def show_mission_control() -> None:
    hero(
        "Food waste intelligence, not just a calculator.",
        "Predict high-risk meals, simulate smarter preparation ranges, identify recurring waste patterns, and build realistic rescue actions for schools and community events.",
        eyebrow="MISSION CONTROL",
    )

    history = get_user_history()
    actual = history[history["Record Type"].isin(["Actual Result", "Demo Actual Result"])].copy()
    actual = numeric(actual, ["Waste Rate", "Leftover Portions", "Estimated CO2 Kg", "Estimated Cost CAD", "Potential Meals Rescued"])

    if actual.empty:
        st.info("Start by loading demo data or logging your first event. The app becomes smarter as you add records.")
    else:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Actual records", f"{len(actual)}")
        c2.metric("Average waste", f"{actual['Waste Rate'].mean():.1f}%")
        c3.metric("Meals rescued", f"{actual['Potential Meals Rescued'].sum():.0f}")
        c4.metric("Cost exposure", f"${actual['Estimated Cost CAD'].sum():.0f}")

    st.markdown("### What this MVP does")
    cols = st.columns(4)
    with cols[0]:
        card("01 / Scan", "Predict waste risk", "Forecast risk using attendance, weather, day, menu popularity, food quantity, rescue capacity, and historical data.")
    with cols[1]:
        card("02 / Simulate", "Compare scenarios", "Run a What-if Simulator to see what happens if the team prepares 10% less, 20% less, or follows the smart range.")
    with cols[2]:
        card("03 / Rescue", "Action plan", "Generate before/during/after actions for source reduction, batch cooking, donation review, storage, and composting.")
    with cols[3]:
        card("04 / Learn", "Pattern detection", "Find waste hotspots by meal, day, weather, intervention, and attendance gap so teams can improve over time.")

    st.markdown("### Competition strengths")
    st.markdown(
        """
        <div class="lux-note">
        This version is designed to match the High School Track: it identifies waste patterns, predicts where waste is likely to happen, and suggests practical interventions. It remains easy for high school students because every advanced feature is wrapped in simple forms, cards, charts, and plain-language explanations.
        </div>
        """,
        unsafe_allow_html=True,
    )


def show_risk_scanner() -> None:
    hero(
        "Waste Risk Scanner",
        "Use this before a cafeteria lunch, school club event, fundraiser, or donation program to predict waste risk and choose a smarter preparation range.",
        eyebrow="PREDICT BEFORE WASTE HAPPENS",
    )

    history = get_user_history()
    with st.form("risk_scan_form"):
        inputs = common_event_form("scan", include_actuals=False)
        submitted = st.form_submit_button("Run AI Waste Risk Scan", use_container_width=True)

    if submitted:
        if inputs["expected_attendance"] <= 0 or inputs["food_prepared"] <= 0:
            st.error("Expected attendance and food prepared must be greater than 0.")
            return

        metrics = evaluate_event(inputs, history)
        plan = build_action_plan(inputs, metrics)
        st.session_state["last_forecast"] = {"inputs": inputs, "metrics": metrics, "plan": plan}

        left, right = st.columns([0.92, 1.08])
        with left:
            st.plotly_chart(risk_gauge(metrics["risk_score"], metrics["risk_level"]), use_container_width=True)
        with right:
            m1, m2, m3 = st.columns(3)
            m1.metric("Predicted waste", f"{metrics['predicted_waste_rate']}%")
            m2.metric("Smart range", f"{metrics['recommended_min']}–{metrics['recommended_max']}")
            m3.metric("Potential rescued", f"{metrics['potential_rescued']:.0f}")
            st.markdown(f"**Model mode:** {metrics['model_mode']}")
            st.markdown(
                f"The scanner estimates **{metrics['predicted_attendance']} attendees** and **{metrics['predicted_consumption']} consumed portions** based on the event context."
            )
            if metrics["shortage_risk_portions"] > 0:
                st.warning(f"Shortage watch: this plan may be about {metrics['shortage_risk_portions']:.0f} portions below predicted demand.")
            st.caption(FOOD_SAFETY_NOTE)

        st.markdown("### Top risk drivers")
        fig = driver_bar(metrics["top_drivers"])
        if fig:
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.success("No major risk drivers detected.")

        st.markdown("### Rescue Action Plan")
        show_action_plan(plan)

        st.markdown("### What-if Simulator")
        scenarios = scenario_table(inputs, history)
        st.dataframe(scenarios, use_container_width=True, hide_index=True)
        fig = px.scatter(
            scenarios,
            x="Prepared Portions",
            y="Predicted Waste Rate",
            size="Risk Score",
            color="Risk Level",
            hover_name="Scenario",
            title="Preparation level vs predicted waste",
        )
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(255,255,255,0.03)", font=dict(color="#F8F4EC"))
        st.plotly_chart(fig, use_container_width=True)

        report_text = build_report_text(inputs, metrics, plan)
        dl1, dl2 = st.columns(2)
        with dl1:
            if st.button("Save this forecast to dashboard", use_container_width=True):
                save_record(record_from_forecast(inputs, metrics))
                st.success("Forecast saved.")
        with dl2:
            st.download_button(
                "Download Rescue Action Report",
                data=report_text,
                file_name="food_waste_rescue_report.txt",
                mime="text/plain",
                use_container_width=True,
            )

        st.markdown("### Optional Groq AI Coach")
        prompt = f"""
        Create a concise rescue plan for a school/community organizer using this data.
        Use this format exactly:
        1. Risk diagnosis
        2. Why waste may happen
        3. Smart preparation plan
        4. Redistribution / storage / compost plan
        5. Human food-safety reminder

        Inputs: {inputs}
        Metrics: {metrics}
        {FOOD_SAFETY_NOTE}
        """
        ai = ask_ai(prompt)
        if ai:
            st.write(ai)
        else:
            st.info("Groq API key is not set, so the app used its built-in risk engine and action planner. Add GROQ_API_KEY in Streamlit Secrets to enable the AI Coach.")


def show_event_logger() -> None:
    hero(
        "Event Result Logger",
        "Use this after an event to record actual attendance and leftovers. These records power the dashboard and unlock the historical learning model.",
        eyebrow="LEARN FROM REAL RESULTS",
    )

    history = get_user_history()
    with st.form("actual_logger_form"):
        inputs = common_event_form("actual", include_actuals=True)
        submitted = st.form_submit_button("Analyze and Save Actual Result", use_container_width=True)

    if submitted:
        if inputs["food_prepared"] <= 0:
            st.error("Food prepared must be greater than 0.")
            return
        if inputs["leftover_portions"] > inputs["food_prepared"]:
            st.error("Leftover portions cannot be greater than food prepared.")
            return

        forecast = evaluate_event(inputs, history)
        actual = actual_waste_metrics(inputs)
        plan = build_action_plan(inputs, forecast)
        save_record(record_from_actual(inputs, actual, forecast))

        st.success("Actual result saved to the dashboard.")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Actual waste rate", f"{actual['waste_rate']}%")
        c2.metric("Risk level", actual["risk_level"])
        c3.metric("Attendance gap", f"{actual['attendance_gap']:.0f}")
        c4.metric("Cost exposure", f"${actual['estimated_cost']:.0f}")

        st.markdown("### Impact Receipt")
        receipt = pd.DataFrame(
            [
                ["Leftover portions", inputs["leftover_portions"]],
                ["Potential meals rescued", actual["rescued"]],
                ["Estimated discard after rescue", actual["discarded_estimate"]],
                ["Estimated CO₂e exposure", f"{actual['estimated_co2']} kg"],
                ["Estimated cost saved through rescue", f"${actual['estimated_cost_saved']}"] ,
            ],
            columns=["Metric", "Value"],
        )
        st.dataframe(receipt, hide_index=True, use_container_width=True)

        if actual["waste_rate"] >= 25:
            st.warning("This was a high-waste event. Use the action plan below before the next similar event.")
        elif actual["waste_rate"] >= 10:
            st.warning("This was a medium-waste event. Small planning changes may reduce waste next time.")
        else:
            st.success("This event performed well. Keep tracking to confirm the pattern.")

        st.markdown("### Next-time action plan")
        show_action_plan(plan)
        st.caption(FOOD_SAFETY_NOTE)


def show_dashboard() -> None:
    hero(
        "Impact Intelligence",
        "Find waste hotspots, measure the effect of interventions, and turn messy event records into clear decisions.",
        eyebrow="PATTERN DETECTION DASHBOARD",
    )

    history = get_user_history()
    if history.empty:
        st.info("No records yet. Load demo data from the sidebar or log an event first.")
        return

    df = numeric(
        history,
        [
            "Waste Rate",
            "Predicted Waste Rate",
            "Risk Score",
            "Expected Attendance",
            "Actual Attendance",
            "Food Prepared",
            "Leftover Portions",
            "Potential Meals Rescued",
            "Estimated CO2 Kg",
            "Estimated Cost CAD",
            "Estimated Cost Saved CAD",
            "Menu Popularity",
        ],
    )
    df["Timestamp Parsed"] = pd.to_datetime(df["Timestamp"], errors="coerce")
    actual = df[df["Record Type"].isin(["Actual Result", "Demo Actual Result"])].copy()
    forecasts = df[df["Record Type"].astype(str).str.contains("Forecast", na=False)].copy()

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Records", f"{len(df)}")
    c2.metric("Actual avg waste", "—" if actual.empty else f"{actual['Waste Rate'].mean():.1f}%")
    c3.metric("Forecasts", f"{len(forecasts)}")
    c4.metric("Meals rescued", "—" if actual.empty else f"{actual['Potential Meals Rescued'].sum():.0f}")
    c5.metric("Cost exposure", "—" if actual.empty else f"${actual['Estimated Cost CAD'].sum():.0f}")

    if not actual.empty:
        actual = actual.sort_values("Timestamp Parsed")
        st.markdown("### Waste trend over time")
        fig = px.line(actual, x="Timestamp Parsed", y="Waste Rate", markers=True, color="Event Type")
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(255,255,255,0.03)", font=dict(color="#F8F4EC"), xaxis_title="Date")
        st.plotly_chart(fig, use_container_width=True)

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("### Worst meal types")
            meal = actual.groupby("Meal Type", dropna=False)["Waste Rate"].mean().sort_values(ascending=False).head(8).reset_index()
            fig = px.bar(meal, x="Waste Rate", y="Meal Type", orientation="h", text="Waste Rate")
            fig.update_traces(marker_color="#C8A45D", texttemplate="%{text:.1f}%")
            fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(255,255,255,0.03)", font=dict(color="#F8F4EC"))
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            st.markdown("### Waste by day")
            day = actual.groupby("Day of Week", dropna=False)["Waste Rate"].mean().reindex(DAYS).dropna().reset_index()
            fig = px.bar(day, x="Day of Week", y="Waste Rate", text="Waste Rate")
            fig.update_traces(marker_color="#AFC7A3", texttemplate="%{text:.1f}%")
            fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(255,255,255,0.03)", font=dict(color="#F8F4EC"))
            st.plotly_chart(fig, use_container_width=True)

        st.markdown("### Hotspot heatmap")
        heat = actual.pivot_table(values="Waste Rate", index="Day of Week", columns="Meal Time", aggfunc="mean").reindex(DAYS)
        fig = px.imshow(heat, text_auto=".1f", aspect="auto", color_continuous_scale="RdYlGn_r")
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", font=dict(color="#F8F4EC"))
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("### Intervention effectiveness")
        intervention = actual.groupby("Intervention Used", dropna=False).agg(
            Average_Waste=("Waste Rate", "mean"),
            Records=("Waste Rate", "count"),
            Meals_Rescued=("Potential Meals Rescued", "sum"),
        ).sort_values("Average_Waste")
        st.dataframe(intervention, use_container_width=True)

        st.markdown("### Pattern insights")
        worst_meal = actual.groupby("Meal Type")["Waste Rate"].mean().sort_values(ascending=False).head(1)
        worst_day = actual.groupby("Day of Week")["Waste Rate"].mean().sort_values(ascending=False).head(1)
        worst_weather = actual.groupby("Weather")["Waste Rate"].mean().sort_values(ascending=False).head(1)
        insights = []
        if not worst_meal.empty:
            insights.append(f"Most waste-prone meal: **{worst_meal.index[0]}** with an average waste rate of **{worst_meal.iloc[0]:.1f}%**.")
        if not worst_day.empty:
            insights.append(f"Highest-risk day: **{worst_day.index[0]}** with an average waste rate of **{worst_day.iloc[0]:.1f}%**.")
        if not worst_weather.empty:
            insights.append(f"Weather condition linked to highest waste: **{worst_weather.index[0]}** at **{worst_weather.iloc[0]:.1f}%** average waste.")
        corr_df = actual.dropna(subset=["Menu Popularity", "Waste Rate"])
        if len(corr_df) >= 5:
            corr = corr_df["Menu Popularity"].corr(corr_df["Waste Rate"])
            if pd.notna(corr):
                direction = "lower" if corr < 0 else "higher"
                insights.append(f"Menu popularity appears linked to waste: {direction} popularity is associated with more waste in your current records.")
        for insight in insights:
            st.write(f"• {insight}")

    st.markdown("### Recent records")
    st.dataframe(df.sort_values("Timestamp Parsed", ascending=False).drop(columns=["Timestamp Parsed"]), use_container_width=True, hide_index=True)


def show_data_export() -> None:
    hero(
        "Data & Export",
        "Manage demo records, download CSV data, and generate a simple organization impact summary.",
        eyebrow="DATA OPERATIONS",
    )

    history = get_user_history()
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("Load demo data", use_container_width=True):
            add_demo_data_for_current_user()
    with col2:
        st.download_button(
            "Download my CSV",
            data=history.to_csv(index=False),
            file_name="food_waste_history.csv",
            mime="text/csv",
            use_container_width=True,
            disabled=history.empty,
        )
    with col3:
        if st.button("Clear my records", use_container_width=True):
            all_history = load_history()
            all_history = all_history[all_history["Username"].astype(str) != st.session_state.username]
            all_history.to_csv(HISTORY_FILE, index=False)
            st.success("Your records were cleared.")
            st.rerun()

    if history.empty:
        st.info("No data yet.")
        return

    df = numeric(history, ["Waste Rate", "Potential Meals Rescued", "Estimated CO2 Kg", "Estimated Cost CAD", "Estimated Cost Saved CAD"])
    actual = df[df["Record Type"].isin(["Actual Result", "Demo Actual Result"])].copy()

    if not actual.empty:
        summary = f"""{APP_NAME} — Organization Impact Summary
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}

Records analyzed: {len(actual)} actual events
Average waste rate: {actual['Waste Rate'].mean():.1f}%
Total leftover portions: {actual['Leftover Portions'].sum():.0f}
Potential meals rescued: {actual['Potential Meals Rescued'].sum():.0f}
Estimated CO2e exposure tracked: {actual['Estimated CO2 Kg'].sum():.1f} kg
Estimated cost exposure tracked: ${actual['Estimated Cost CAD'].sum():.2f}
Estimated cost saved through rescue capacity: ${actual['Estimated Cost Saved CAD'].sum():.2f}

Recommended next steps:
1. Focus first on meal/day combinations with the highest waste rate.
2. Use RSVP or pre-order forms for low-confidence attendance events.
3. Prepare smaller first batches for meals with low popularity.
4. Build a verified rescue partner list before high-risk events.
5. Keep human food-safety review separate from AI suggestions.

{FOOD_SAFETY_NOTE}
"""
        st.download_button(
            "Download impact summary",
            data=summary,
            file_name="food_waste_impact_summary.txt",
            mime="text/plain",
            use_container_width=True,
        )

    st.markdown("### Raw data preview")
    st.dataframe(history, use_container_width=True, hide_index=True)


def main() -> None:
    inject_css()

    if "username" not in st.session_state:
        st.session_state.username = None
    if "page" not in st.session_state:
        st.session_state.page = "Mission Control"

    if st.session_state.username is None:
        show_login()
        return

    show_sidebar()
    page = st.session_state.page

    if page == "Mission Control":
        show_mission_control()
    elif page == "Waste Risk Scanner":
        show_risk_scanner()
    elif page == "Event Result Logger":
        show_event_logger()
    elif page == "Impact Intelligence":
        show_dashboard()
    elif page == "Data & Export":
        show_data_export()


if __name__ == "__main__":
    main()
