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
    PLOTLY_OK = True
except Exception:
    PLOTLY_OK = False

try:
    from groq import Groq
    GROQ_OK = True
except Exception:
    GROQ_OK = False

try:
    from sklearn.compose import ColumnTransformer
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder
    SKLEARN_OK = True
except Exception:
    SKLEARN_OK = False


# =============================================================================
# App setup
# =============================================================================
st.set_page_config(
    page_title="Food Rescue Radar",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="collapsed",
)

DATA_FILE = Path("food_waste_records.csv")
PROJECT_FILE = Path("projects.json")
ROUTE_FILE = Path("rescue_routes.json")

EVENT_TYPES = [
    "School lunch",
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
    "Library or common area",
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
    "Attendance check",
    "Pre-order form",
    "Smaller first batch",
    "Menu preference survey",
    "Donation route ready",
    "Compost plan ready",
    "Mixed plan",
]

ROUTE_TYPES = [
    "Donation partner",
    "Student club pickup",
    "Staff meal review",
    "Community fridge",
    "Compost program",
    "Animal feed partner",
]

FOOD_FORMS = ["Prepared meals", "Packaged snacks", "Bakery items", "Produce", "Mixed food"]

COLUMNS = [
    "Project",
    "Event Name",
    "Time",
    "Event Type",
    "Location",
    "Day",
    "Meal Time",
    "Food",
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


# =============================================================================
# Visual style
# =============================================================================
def apply_style() -> None:
    st.markdown(
        """
        <style>
        :root {
            --bg: #F7F3EA;
            --panel: #FFFDF7;
            --panel2: #F1F7EE;
            --green900: #0E2F21;
            --green800: #17442F;
            --green700: #1F5F3E;
            --green600: #2E7D4F;
            --sage: #DCE9D7;
            --sage2: #EEF5EB;
            --gold: #B7964A;
            --text: #173528;
            --muted: #627568;
            --line: rgba(23, 68, 47, 0.14);
            --shadow: 0 16px 44px rgba(14, 47, 33, 0.10);
        }

        .stApp {
            background:
                radial-gradient(circle at 12% 4%, rgba(183,150,74,0.12), transparent 30%),
                radial-gradient(circle at 92% 8%, rgba(46,125,79,0.16), transparent 30%),
                linear-gradient(135deg, #F7F3EA 0%, #F1F7EE 55%, #E8F2E4 100%);
            color: var(--text);
        }

        [data-testid="stHeader"] {
            background: rgba(247, 243, 234, 0.78);
            backdrop-filter: blur(18px);
            border-bottom: 1px solid rgba(23, 68, 47, 0.08);
        }

        .block-container {
            max-width: 1160px;
            padding-top: 1.35rem;
            padding-bottom: 4rem;
        }

        h1 {
            color: var(--green900) !important;
            font-weight: 850 !important;
            letter-spacing: -0.045em !important;
            line-height: 1.02 !important;
            font-size: clamp(2.2rem, 5vw, 4.4rem) !important;
        }

        h2, h3 {
            color: var(--green900) !important;
            letter-spacing: -0.025em !important;
        }

        p, li, label, span, div {
            color: var(--text);
        }

        small, .stCaption, [data-testid="stCaptionContainer"] {
            color: var(--muted) !important;
        }

        div[data-testid="stVerticalBlockBorderWrapper"] {
            border: 1px solid var(--line) !important;
            border-radius: 24px !important;
            background: rgba(255, 253, 247, 0.82) !important;
            box-shadow: 0 10px 30px rgba(14, 47, 33, 0.07) !important;
        }

        .stButton > button {
            background: linear-gradient(135deg, #17442F, #2E7D4F) !important;
            color: #FFF8E8 !important;
            border: 1px solid rgba(23, 68, 47, .18) !important;
            border-radius: 999px !important;
            font-weight: 800 !important;
            min-height: 3.05rem !important;
            box-shadow: 0 12px 26px rgba(46, 125, 79, 0.22) !important;
        }

        .stButton > button p,
        .stButton > button span,
        .stButton > button div {
            color: #FFF8E8 !important;
        }

        .stButton > button:hover {
            filter: brightness(1.04);
            transform: translateY(-1px);
        }

        div[data-testid="stDownloadButton"] > button {
            background: linear-gradient(135deg, #B7964A, #DAC58C) !important;
            color: #173528 !important;
            border-radius: 999px !important;
            font-weight: 800 !important;
            min-height: 3.05rem !important;
            border: 1px solid rgba(183,150,74,.22) !important;
        }

        div[data-testid="stDownloadButton"] > button p,
        div[data-testid="stDownloadButton"] > button span {
            color: #173528 !important;
        }

        div[data-testid="stRadio"] {
            padding: .38rem .45rem;
            border: 1px solid var(--line);
            border-radius: 999px;
            background: rgba(255,253,247,.72);
            box-shadow: 0 8px 20px rgba(14,47,33,.05);
            margin-bottom: 1rem;
        }

        div[data-testid="stRadio"] label p {
            color: var(--green900) !important;
            font-weight: 750 !important;
        }

        input, textarea {
            color: var(--green900) !important;
        }

        div[data-baseweb="select"] span {
            color: var(--green900) !important;
        }

        div[data-testid="stTextInput"] input,
        div[data-testid="stNumberInput"] input,
        div[data-testid="stTextArea"] textarea,
        div[data-testid="stSelectbox"] div[data-baseweb="select"] > div {
            border-radius: 15px !important;
            border: 1px solid rgba(23, 68, 47, .16) !important;
            background: #FFFDF7 !important;
        }

        .stMetric {
            background: #FFFDF7;
            border: 1px solid rgba(23,68,47,.12);
            border-radius: 18px;
            padding: 1rem;
            box-shadow: 0 8px 20px rgba(14,47,33,.05);
        }

        [data-testid="stMetricLabel"] p {
            color: var(--muted) !important;
            font-weight: 800 !important;
            letter-spacing: .05em !important;
            text-transform: uppercase !important;
            font-size: .75rem !important;
        }

        [data-testid="stMetricValue"] {
            color: var(--green900) !important;
        }

        hr {
            border-color: rgba(23,68,47,.12);
        }

        [data-testid="stDataFrame"] {
            border-radius: 18px;
            overflow: hidden;
            border: 1px solid rgba(23,68,47,.10);
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


# =============================================================================
# Project and data functions
# =============================================================================
def load_projects() -> list[str]:
    if not PROJECT_FILE.exists():
        return []
    try:
        data = json.loads(PROJECT_FILE.read_text(encoding="utf-8"))
        if isinstance(data, list):
            return [str(x) for x in data if str(x).strip()]
    except Exception:
        pass
    return []


def save_projects(projects: list[str]) -> None:
    unique = sorted(set(p.strip() for p in projects if p.strip()))
    PROJECT_FILE.write_text(json.dumps(unique, indent=2), encoding="utf-8")


def add_project(name: str) -> None:
    projects = load_projects()
    if name not in projects:
        projects.append(name)
        save_projects(projects)


def clean_project_name(name: str) -> str:
    return " ".join((name or "").strip().split())


def empty_data() -> pd.DataFrame:
    return pd.DataFrame(columns=COLUMNS)


def load_data() -> pd.DataFrame:
    if not DATA_FILE.exists():
        return empty_data()
    df = pd.read_csv(DATA_FILE)
    for col in COLUMNS:
        if col not in df.columns:
            df[col] = np.nan
    return df[COLUMNS]


def save_data(df: pd.DataFrame) -> None:
    df[COLUMNS].to_csv(DATA_FILE, index=False)


def project_data(project: str) -> pd.DataFrame:
    df = load_data()
    if df.empty:
        return empty_data()
    return df[df["Project"] == project].copy()


def add_record(record: dict[str, Any]) -> None:
    df = load_data()
    new_row = pd.DataFrame([record])
    df = pd.concat([df, new_row], ignore_index=True)
    save_data(df)


def reset_project_records(project: str) -> None:
    df = load_data()
    if not df.empty:
        df = df[df["Project"] != project]
        save_data(df)


def load_routes() -> pd.DataFrame:
    columns = [
        "Project",
        "Route Name",
        "Route Type",
        "Contact",
        "Capacity",
        "Pickup Minutes",
        "Food Form",
        "Notes",
    ]
    if not ROUTE_FILE.exists():
        return pd.DataFrame(columns=columns)
    try:
        data = json.loads(ROUTE_FILE.read_text(encoding="utf-8"))
        df = pd.DataFrame(data)
    except Exception:
        return pd.DataFrame(columns=columns)

    for col in columns:
        if col not in df.columns:
            df[col] = np.nan
    return df[columns]


def save_routes(df: pd.DataFrame) -> None:
    df.to_json(ROUTE_FILE, orient="records", indent=2)


def project_routes(project: str) -> pd.DataFrame:
    routes = load_routes()
    if routes.empty:
        return routes
    return routes[routes["Project"] == project].copy()


def add_route(route: dict[str, Any]) -> None:
    routes = load_routes()
    routes = pd.concat([routes, pd.DataFrame([route])], ignore_index=True)
    save_routes(routes)


def reset_project_routes(project: str) -> None:
    routes = load_routes()
    if not routes.empty:
        routes = routes[routes["Project"] != project]
        save_routes(routes)


def sample_routes(project: str) -> pd.DataFrame:
    rows = [
        {
            "Project": project,
            "Route Name": "Student Council Snack Table",
            "Route Type": "Student club pickup",
            "Contact": "student.council@school.ca",
            "Capacity": 25,
            "Pickup Minutes": 15,
            "Food Form": "Packaged snacks",
            "Notes": "Good for unopened snacks, fruit cups, and packaged bakery items.",
        },
        {
            "Project": project,
            "Route Name": "Community Pantry Review",
            "Route Type": "Donation partner",
            "Contact": "pantry@example.org",
            "Capacity": 50,
            "Pickup Minutes": 45,
            "Food Form": "Packaged snacks",
            "Notes": "Use only after trained staff confirm donation rules and packaging.",
        },
        {
            "Project": project,
            "Route Name": "Staff Meal Review",
            "Route Type": "Staff meal review",
            "Contact": "cafeteria.manager@school.ca",
            "Capacity": 20,
            "Pickup Minutes": 10,
            "Food Form": "Prepared meals",
            "Notes": "Internal review route for appropriate leftovers.",
        },
        {
            "Project": project,
            "Route Name": "Green Team Compost",
            "Route Type": "Compost program",
            "Contact": "green.team@school.ca",
            "Capacity": 100,
            "Pickup Minutes": 30,
            "Food Form": "Mixed food",
            "Notes": "Fallback route for food that cannot be redistributed.",
        },
    ]
    return pd.DataFrame(rows)


def rescue_route_scores(event: dict[str, Any], result: dict[str, Any], routes: pd.DataFrame) -> pd.DataFrame:
    if routes.empty:
        return routes

    routes = numeric(routes.copy(), ["Capacity", "Pickup Minutes"])
    predicted_leftovers = max(float(result.get("Predicted Leftovers", 0)), 0)
    food = str(event.get("Food", "")).lower()
    donation_ready = event.get("Donation Route") == "Yes"

    scored = []
    for _, row in routes.iterrows():
        capacity = max(float(row.get("Capacity", 0) or 0), 0)
        pickup = max(float(row.get("Pickup Minutes", 999) or 999), 1)
        route_type = str(row.get("Route Type", ""))
        food_form = str(row.get("Food Form", ""))
        notes = str(row.get("Notes", ""))

        score = 50.0

        if predicted_leftovers <= 0:
            score -= 10
        elif capacity >= predicted_leftovers:
            score += 20
        elif capacity >= predicted_leftovers * 0.5:
            score += 10
        else:
            score -= 12

        if pickup <= 15:
            score += 15
        elif pickup <= 45:
            score += 8
        elif pickup <= 90:
            score += 2
        else:
            score -= 10

        if food_form == "Mixed food":
            score += 4
        elif "snack" in food or "bar" in food or "bakery" in food:
            if food_form in ["Packaged snacks", "Bakery items"]:
                score += 12
        elif "fruit" in food or "produce" in food:
            if food_form in ["Produce", "Mixed food"]:
                score += 12
        elif food_form in ["Prepared meals", "Mixed food"]:
            score += 7

        if route_type == "Donation partner" and donation_ready:
            score += 10
        if route_type == "Compost program":
            score -= 8
            if predicted_leftovers > capacity * 0.8:
                score += 5

        score = float(np.clip(score, 0, 100))

        if score >= 75:
            recommendation = "Best fit"
        elif score >= 55:
            recommendation = "Backup option"
        else:
            recommendation = "Use only if needed"

        scored.append(
            {
                "Route Name": row.get("Route Name", ""),
                "Route Type": route_type,
                "Contact": row.get("Contact", ""),
                "Capacity": int(capacity),
                "Pickup Minutes": int(pickup),
                "Food Form": food_form,
                "Readiness Score": round(score, 1),
                "Recommendation": recommendation,
                "Notes": notes,
            }
        )

    return pd.DataFrame(scored).sort_values("Readiness Score", ascending=False)


def rescue_message(event: dict[str, Any], result: dict[str, Any], route: pd.Series | dict[str, Any]) -> str:
    route_name = route.get("Route Name", "your team")
    food = event.get("Food", "food")
    leftovers = result.get("Predicted Leftovers", 0)
    event_name = event.get("Event Name", "the event")
    location = event.get("Location", "the event location")

    return (
        f"Hi {route_name}, our project is forecasting about {leftovers} possible leftover portions "
        f"of {food} from {event_name} at {location}. Could you confirm whether your team may be able "
        f"to review or receive this food if trained staff approve it under food safety rules? "
        f"We can share the final count after the event."
    )


def numeric(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    out = df.copy()
    for col in cols:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


# =============================================================================
# AI keys and prediction
# =============================================================================
def get_groq_api_key() -> str | None:
    try:
        key = st.secrets.get("GROQ_API_KEY", None)
    except Exception:
        key = None
    if key:
        return str(key)
    return None


def groq_is_connected() -> bool:
    return GROQ_OK and get_groq_api_key() is not None


def groq_plan(prompt: str) -> str | None:
    key = get_groq_api_key()
    if not GROQ_OK or not key:
        return None

    try:
        client = Groq(api_key=key)
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are an AI assistant for a student environmental app. "
                        "Help schools reduce food waste. Be practical, clear, and safe. "
                        "Never decide if food is safe to donate or serve. Tell users that trained staff must follow local food safety rules."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.35,
            max_tokens=700,
        )
        return response.choices[0].message.content
    except Exception as exc:
        return f"AI Advisor could not respond right now. Error: {exc}"


def risk_level(score: float) -> str:
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
    }.get(int(popularity), 0.98)

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


def portion_range(expected: int, weather: str, confidence: str, popularity: int, day: str, batch: bool) -> tuple[int, int, int]:
    expected = max(int(expected), 1)
    demand = expected * demand_multiplier(weather, confidence, popularity, day)

    if batch:
        low = demand * 0.92
        high = demand * 1.02
    else:
        low = demand * 0.96
        high = demand * 1.07

    return max(1, round(low)), max(1, round(high)), max(1, round(demand))


def history_signal(df: pd.DataFrame, event_type: str, food: str) -> tuple[float | None, str]:
    if df.empty:
        return None, "No project history yet."

    df = numeric(df, ["Waste Rate"]).dropna(subset=["Waste Rate"])
    if df.empty:
        return None, "No usable project history yet."

    food_text = str(food).lower().strip()
    if food_text and food_text != "unknown":
        same_food = df[df["Food"].astype(str).str.lower().str.contains(food_text, regex=False, na=False)]
        if len(same_food) >= 2:
            avg = float(same_food["Waste Rate"].mean())
            return avg, f"Similar food has averaged {avg:.1f}% waste in this project."

    same_event = df[df["Event Type"].astype(str) == event_type]
    if len(same_event) >= 2:
        avg = float(same_event["Waste Rate"].mean())
        return avg, f"Similar events have averaged {avg:.1f}% waste in this project."

    avg = float(df["Waste Rate"].tail(8).mean())
    return avg, f"Recent project records have averaged {avg:.1f}% waste."


def train_history_model(df: pd.DataFrame):
    if not SKLEARN_OK:
        return None, "The app is using the AI forecast engine."

    df = df.copy()
    df = numeric(df, ["Expected Attendance", "Food Prepared", "Menu Popularity", "Waste Rate"])
    df = df.dropna(subset=["Expected Attendance", "Food Prepared", "Waste Rate"])

    if len(df) < 10 or df["Waste Rate"].nunique() < 3:
        return None, "The app will learn from this project after more results are logged."

    features = [
        "Event Type",
        "Location",
        "Day",
        "Meal Time",
        "Food",
        "Menu Popularity",
        "Weather",
        "Attendance Confidence",
        "Expected Attendance",
        "Food Prepared",
        "Batch Cooking",
        "Intervention",
    ]

    for col in features:
        if col not in df.columns:
            df[col] = ""

    categorical = [
        "Event Type",
        "Location",
        "Day",
        "Meal Time",
        "Food",
        "Weather",
        "Attendance Confidence",
        "Batch Cooking",
        "Intervention",
    ]
    numeric_cols = ["Menu Popularity", "Expected Attendance", "Food Prepared"]

    model = Pipeline(
        steps=[
            (
                "prep",
                ColumnTransformer(
                    [
                        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical),
                        ("num", "passthrough", numeric_cols),
                    ]
                ),
            ),
            (
                "model",
                RandomForestRegressor(
                    n_estimators=160,
                    min_samples_leaf=2,
                    random_state=42,
                ),
            ),
        ]
    )

    model.fit(df[features], df["Waste Rate"].clip(0, 100))
    return model, f"The app is learning from {len(df)} logged results in this project."


def predict_with_history_model(model, event: dict[str, Any]) -> float | None:
    if model is None:
        return None
    features = [
        "Event Type",
        "Location",
        "Day",
        "Meal Time",
        "Food",
        "Menu Popularity",
        "Weather",
        "Attendance Confidence",
        "Expected Attendance",
        "Food Prepared",
        "Batch Cooking",
        "Intervention",
    ]
    try:
        row = pd.DataFrame([{col: event.get(col, "") for col in features}])
        return float(model.predict(row)[0])
    except Exception:
        return None


def forecast_event(event: dict[str, Any], history: pd.DataFrame) -> dict[str, Any]:
    expected = max(int(event["Expected Attendance"]), 1)
    prepared = max(int(event["Food Prepared"]), 1)
    popularity = int(event["Menu Popularity"])
    batch = event["Batch Cooking"] == "Yes"

    rec_min, rec_max, demand = portion_range(
        expected,
        event["Weather"],
        event["Attendance Confidence"],
        popularity,
        event["Day"],
        batch,
    )

    predicted_leftover = max(prepared - demand, 0)
    predicted_waste = predicted_leftover / prepared * 100
    over_prepared = max(prepared - expected, 0) / expected * 100

    score = 18 + predicted_waste * 1.18 + over_prepared * 0.32
    reasons: list[str] = []

    if prepared > rec_max:
        score += min(22, (prepared - rec_max) / expected * 70)
        reasons.append(f"The current plan is above the recommended range of {rec_min} to {rec_max} portions.")

    if event["Attendance Confidence"] == "Low":
        score += 15
        reasons.append("Attendance confidence is low, so the plan needs a smaller first batch or stronger RSVP check.")
    elif event["Attendance Confidence"] == "Medium":
        score += 6
        reasons.append("Attendance confidence is medium, so a final count could reduce over-ordering.")

    if event["Weather"] in ["Rainy", "Stormy", "Very hot", "Very cold"]:
        score += {"Rainy": 8, "Stormy": 15, "Very hot": 10, "Very cold": 10}[event["Weather"]]
        reasons.append(f"{event['Weather']} conditions may reduce attendance or appetite.")

    if popularity <= 2:
        score += 14 if popularity == 1 else 9
        reasons.append("Menu popularity is low, which often increases leftovers.")
    elif popularity >= 4:
        score -= 4
        reasons.append("Menu popularity is strong, which lowers waste risk.")

    if event["Day"] in ["Monday", "Friday", "Saturday", "Sunday"]:
        score += 4
        reasons.append(f"{event['Day']} demand can be less predictable.")

    if batch:
        score -= 9
        reasons.append("Batch cooking lowers risk because food can be prepared in stages.")

    if event["Intervention"] != "None yet":
        score -= 6
        reasons.append(f"The planned intervention is: {event['Intervention']}.")

    avg, signal = history_signal(history, event["Event Type"], event["Food"])
    if avg is not None:
        if avg >= 25:
            score += 10
        elif avg >= 12:
            score += 4
        elif avg < 8:
            score -= 4
        reasons.append(signal)

    model, model_message = train_history_model(history)
    ml_estimate = predict_with_history_model(model, event)

    if ml_estimate is not None:
        predicted_waste = predicted_waste * 0.58 + ml_estimate * 0.42
        score = score * 0.70 + predicted_waste * 1.45 * 0.30
        model_message = model_message + f" Historical estimate: {ml_estimate:.1f}% waste."

    score = float(np.clip(score, 5, 96))
    level = risk_level(score)

    return {
        "Risk Score": round(score, 1),
        "Risk Level": level,
        "Predicted Waste Rate": round(float(np.clip(predicted_waste, 0, 100)), 1),
        "Predicted Demand": int(demand),
        "Predicted Leftovers": int(round(max(prepared - demand, 0))),
        "Recommended Min": int(rec_min),
        "Recommended Max": int(rec_max),
        "Reasons": reasons[:5] if reasons else ["No major warning signs were detected. Log the result after the event so the project can keep learning."],
        "Model Message": model_message,
    }


def suggested_actions(event: dict[str, Any], result: dict[str, Any]) -> list[str]:
    actions = [
        f"Prepare about {result['Recommended Min']} to {result['Recommended Max']} portions instead of treating the current plan as fixed.",
        "Confirm attendance one more time before food is prepared.",
    ]

    if result["Risk Level"] == "High":
        actions.insert(1, "Start with only 70% to 80% of expected demand and keep the rest as backup.")
    elif result["Risk Level"] == "Medium":
        actions.insert(1, "Use a smaller buffer and avoid cooking all surplus portions at once.")

    if event["Batch Cooking"] == "No":
        actions.append("Split the food into smaller batches if staff can prepare or serve it in stages.")

    if event["Donation Route"] == "Yes":
        actions.append("Prepare labels and containers so trained staff can review leftovers for possible redistribution.")
    else:
        actions.append("Choose an end route before the event: staff meal review, donation partner review, or compost.")

    if event["Intervention"] == "None yet":
        actions.append("Pick one intervention to test so the dashboard can compare results later.")

    return actions[:6]


# =============================================================================
# Sample data
# =============================================================================
def make_sample_data(project: str) -> pd.DataFrame:
    rows = [
        ("Pasta lunch", "School lunch", "Cafeteria", "Monday", "Lunch", "Pasta", 3, "Normal", "Medium", 118, 108, 132, 29, "None yet"),
        ("Rice bowl day", "School lunch", "Cafeteria", "Tuesday", "Lunch", "Chicken rice bowl", 5, "Sunny", "High", 120, 123, 126, 6, "Attendance check"),
        ("Vegetarian lunch", "School lunch", "Cafeteria", "Wednesday", "Lunch", "Vegetarian chili", 2, "Cloudy", "Medium", 115, 100, 126, 34, "Menu preference survey"),
        ("After-school snacks", "After-school club", "Classroom", "Thursday", "Snack", "Fruit cups", 4, "Normal", "High", 42, 39, 45, 6, "Pre-order form"),
        ("Rainy game night", "Sports event", "Gym", "Friday", "Dinner", "Sandwiches", 3, "Rainy", "Low", 95, 72, 120, 43, "None yet"),
        ("Cold breakfast", "Breakfast program", "Cafeteria", "Monday", "Breakfast", "Bagels", 4, "Very cold", "Medium", 70, 58, 80, 19, "Smaller first batch"),
        ("Community pizza", "Community event", "Community hall", "Saturday", "All-day event", "Pizza", 5, "Sunny", "Medium", 160, 148, 170, 18, "Donation route ready"),
        ("Stormy fundraiser", "Fundraiser", "Event venue", "Sunday", "Dinner", "Hot dogs", 3, "Stormy", "Low", 140, 91, 160, 62, "Compost plan ready"),
        ("Friday tacos", "School lunch", "Cafeteria", "Friday", "Lunch", "Fish tacos", 2, "Normal", "Medium", 117, 99, 128, 31, "None yet"),
        ("Bakery rescue", "Food donation program", "Library or common area", "Wednesday", "Snack", "Bakery items", 4, "Cloudy", "Medium", 60, 52, 70, 12, "Donation route ready"),
        ("Noodle bowl", "School lunch", "Cafeteria", "Thursday", "Lunch", "Rice noodle bowl", 4, "Sunny", "High", 122, 121, 126, 8, "Attendance check"),
        ("Outdoor BBQ", "Community event", "Outdoor area", "Saturday", "Lunch", "BBQ plates", 5, "Very hot", "Medium", 180, 151, 195, 39, "Smaller first batch"),
        ("Club granola", "After-school club", "Classroom", "Tuesday", "Snack", "Granola bars", 5, "Normal", "High", 36, 35, 38, 2, "Pre-order form"),
        ("Mac and cheese", "School lunch", "Cafeteria", "Wednesday", "Lunch", "Mac and cheese", 5, "Normal", "High", 125, 129, 130, 5, "Attendance check"),
    ]

    records = []
    now = pd.Timestamp.now()

    for i, item in enumerate(rows):
        event_name, event_type, location, day, meal_time, food, pop, weather, confidence, expected, actual, prepared, leftover, intervention = item
        waste_rate = leftover / prepared * 100
        risk_score = float(np.clip(waste_rate * 2.35 + max(prepared - actual, 0) / prepared * 24, 5, 96))
        level = risk_level(risk_score)
        cost = leftover * 2.50
        co2 = leftover * 0.50

        records.append(
            {
                "Project": project,
                "Event Name": event_name,
                "Time": (now - pd.Timedelta(days=18 - i)).strftime("%Y-%m-%d %H:%M:%S"),
                "Event Type": event_type,
                "Location": location,
                "Day": day,
                "Meal Time": meal_time,
                "Food": food,
                "Menu Popularity": pop,
                "Weather": weather,
                "Attendance Confidence": confidence,
                "Expected Attendance": expected,
                "Actual Attendance": actual,
                "Food Prepared": prepared,
                "Leftover Portions": leftover,
                "Waste Rate": round(waste_rate, 2),
                "Predicted Waste Rate": round(max(0, waste_rate + np.random.uniform(-3, 3)), 2),
                "Risk Score": round(risk_score, 1),
                "Risk Level": level,
                "Recommended Min": max(1, int(actual * 0.94)),
                "Recommended Max": max(1, int(actual * 1.05)),
                "Donation Route": "Yes" if intervention == "Donation route ready" else "No",
                "Batch Cooking": "Yes" if intervention == "Smaller first batch" else "No",
                "Intervention": intervention,
                "Cost per Portion": 2.50,
                "CO2e per Portion": 0.50,
                "Cost Impact": round(cost, 2),
                "CO2e Impact": round(co2, 2),
                "Potential Meals Rescued": min(leftover, 20 if intervention == "Donation route ready" else 0),
                "Notes": "Sample record for demonstration.",
            }
        )

    return pd.DataFrame(records, columns=COLUMNS)


# =============================================================================
# Interface helpers
# =============================================================================
def connected_project() -> str | None:
    return st.session_state.get("project")


def top_header() -> None:
    c1, c2 = st.columns([1.4, 1])
    with c1:
        st.markdown("### 🌿 Food Rescue Radar")
        st.caption("Find food waste before it happens")
    with c2:
        project = connected_project()
        if project:
            st.success(f"Project: {project}", icon="📁")


def metric_grid(items: list[tuple[str, str, str]]) -> None:
    cols = st.columns(len(items))
    for col, (label, value, help_text) in zip(cols, items):
        with col:
            st.metric(label, value, help=help_text)


def risk_message(level: str) -> None:
    if level == "High":
        st.error("High food waste risk", icon="🔴")
    elif level == "Medium":
        st.warning("Medium food waste risk", icon="🟠")
    else:
        st.success("Low food waste risk", icon="🟢")


def collect_event_form(prefix: str, include_actual: bool) -> dict[str, Any]:
    with st.container(border=True):
        st.subheader("Event details")
        st.caption("Use estimates before the event. Use real numbers when logging the result.")

        c1, c2 = st.columns(2)
        with c1:
            event_name = st.text_input("Event name", value="Friday lunch", key=f"{prefix}_event_name")
            event_type = st.selectbox("Event type", EVENT_TYPES, key=f"{prefix}_event_type")
            location = st.selectbox("Where food will be served", LOCATIONS, key=f"{prefix}_location")
            day = st.selectbox("Day", DAYS, index=4, key=f"{prefix}_day")
        with c2:
            meal_time = st.selectbox("Meal time", MEAL_TIMES, index=1, key=f"{prefix}_meal_time")
            food = st.text_input("Food or menu item", value="Pasta", key=f"{prefix}_food")
            weather = st.selectbox("Weather or condition", WEATHER, key=f"{prefix}_weather")
            confidence = st.selectbox("Attendance confidence", CONFIDENCE, index=1, key=f"{prefix}_confidence")

    with st.container(border=True):
        st.subheader("Numbers")
        n1, n2, n3, n4 = st.columns(4)
        with n1:
            expected = st.number_input("Expected attendance", min_value=1, value=100, step=1, key=f"{prefix}_expected")
        with n2:
            prepared = st.number_input("Food planned or prepared", min_value=1, value=110, step=1, key=f"{prefix}_prepared")
        with n3:
            cost = st.number_input("Cost per portion", min_value=0.0, value=2.50, step=0.10, format="%.2f", key=f"{prefix}_cost")
        with n4:
            co2 = st.number_input("CO₂e per portion kg", min_value=0.0, value=0.50, step=0.05, format="%.2f", key=f"{prefix}_co2")

        actual = np.nan
        leftover = np.nan
        if include_actual:
            a1, a2 = st.columns(2)
            with a1:
                actual = st.number_input("Actual attendance", min_value=0, value=90, step=1, key=f"{prefix}_actual")
            with a2:
                leftover = st.number_input("Leftover portions", min_value=0, value=18, step=1, key=f"{prefix}_leftover")

    with st.expander("More details for a better prediction"):
        d1, d2, d3 = st.columns(3)
        with d1:
            popularity = st.slider("Menu popularity", 1, 5, 3, help="1 means unpopular. 5 means very popular.", key=f"{prefix}_pop")
        with d2:
            donation = st.selectbox("Is there a donation or redistribution route?", ["No", "Yes"], key=f"{prefix}_donation")
        with d3:
            batch = st.selectbox("Can food be prepared in smaller batches?", ["Yes", "No"], key=f"{prefix}_batch")
        intervention = st.selectbox("Action planned or used", INTERVENTIONS, key=f"{prefix}_intervention")
        notes = st.text_area("Notes", placeholder="Example: exam week, rainy day, unpopular menu last time...", key=f"{prefix}_notes")

    return {
        "Project": connected_project(),
        "Event Name": event_name.strip() or "Untitled event",
        "Event Type": event_type,
        "Location": location,
        "Day": day,
        "Meal Time": meal_time,
        "Food": food.strip() or "Unknown",
        "Menu Popularity": int(popularity),
        "Weather": weather,
        "Attendance Confidence": confidence,
        "Expected Attendance": int(expected),
        "Actual Attendance": actual,
        "Food Prepared": int(prepared),
        "Leftover Portions": leftover,
        "Donation Route": donation,
        "Batch Cooking": batch,
        "Intervention": intervention,
        "Cost per Portion": float(cost),
        "CO2e per Portion": float(co2),
        "Notes": notes.strip(),
    }


# =============================================================================
# Pages
# =============================================================================
def project_gate() -> bool:
    if connected_project():
        return True

    st.title("Food Rescue Radar")
    st.subheader("Find food waste before it happens")
    st.write(
        "Create a project for a cafeteria, club, event, or community group. "
        "Each project keeps its own forecasts, logs, dashboard, and report."
    )

    projects = load_projects()

    with st.container(border=True):
        st.subheader("Start a project")
        new_project = st.text_input("Project name", placeholder="Example: Wagner Cafeteria or Green Club Events")
        if st.button("Create project", use_container_width=True):
            name = clean_project_name(new_project)
            if not name:
                st.error("Please enter a project name.")
            else:
                add_project(name)
                st.session_state.project = name
                st.rerun()

    if projects:
        with st.container(border=True):
            st.subheader("Open an existing project")
            selected = st.selectbox("Choose project", projects)
            if st.button("Open project", use_container_width=True):
                st.session_state.project = selected
                st.rerun()

    with st.container(border=True):
        st.subheader("What the app does")
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown("**Predict**")
            st.write("Estimate when and where food waste is likely to happen.")
        with c2:
            st.markdown("**Act**")
            st.write("Get practical steps to reduce, redistribute, or manage leftovers.")
        with c3:
            st.markdown("**Learn**")
            st.write("Find patterns from past events and improve future planning.")

    return False


def home_page() -> None:
    st.title("Food Rescue Radar")
    st.subheader("Predict waste. Plan smarter. Rescue more food.")
    st.write(
        "Use this project to forecast food waste before an event, log real results after the event, "
        "and learn which meals, days, and plans create the most waste."
    )

    with st.container(border=True):
        st.subheader("How students use this app")
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.markdown("### 1")
            st.markdown("**Forecast**")
            st.write("Enter the food plan before the event.")
        with c2:
            st.markdown("### 2")
            st.markdown("**Take action**")
            st.write("Use the AI plan to reduce or rescue waste.")
        with c3:
            st.markdown("### 3")
            st.markdown("**Log results**")
            st.write("Record attendance and leftovers after the event.")
        with c4:
            st.markdown("### 4")
            st.markdown("**Improve**")
            st.write("Use patterns to plan better next time.")

    df = project_data(connected_project())
    df_num = numeric(df, ["Waste Rate", "Leftover Portions", "Cost Impact", "CO2e Impact", "Potential Meals Rescued"])

    if df_num.empty:
        metric_grid(
            [
                ("Records", "0", "No logged events yet."),
                ("Average waste", "—", "Log events to unlock this."),
                ("Cost impact", "—", "Estimated from leftovers."),
                ("Meals rescued", "—", "Based on donation route."),
            ]
        )
    else:
        metric_grid(
            [
                ("Records", str(len(df_num)), "Events logged in this project."),
                ("Average waste", f"{df_num['Waste Rate'].mean():.1f}%", "Average leftover rate."),
                ("Cost impact", f"${df_num['Cost Impact'].sum():.0f}", "Estimated value of wasted portions."),
                ("Meals rescued", f"{df_num['Potential Meals Rescued'].sum():.0f}", "Potential portions rescued."),
            ]
        )

    b1, b2, b3, b4 = st.columns(4)
    with b1:
        if st.button("Go to Forecast", use_container_width=True):
            st.session_state.page = "Forecast"
            st.rerun()
    with b2:
        if st.button("Rescue Board", use_container_width=True):
            st.session_state.page = "Rescue Board"
            st.rerun()
    with b3:
        if st.button("Add sample records", use_container_width=True):
            reset_project_records(connected_project())
            reset_project_routes(connected_project())
            full = load_data()
            sample = make_sample_data(connected_project())
            full = pd.concat([full[full["Project"] != connected_project()], sample], ignore_index=True)
            save_data(full)
            route_full = load_routes()
            route_sample = sample_routes(connected_project())
            route_full = pd.concat([route_full[route_full["Project"] != connected_project()], route_sample], ignore_index=True)
            save_routes(route_full)
            st.success("Sample records and rescue routes added to this project.")
            st.rerun()
    with b4:
        if st.button("View Dashboard", use_container_width=True):
            st.session_state.page = "Dashboard"
            st.rerun()

    with st.container(border=True):
        st.subheader("What this project helps you do")
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown("**Predict risk**")
            st.write("See when and where food waste is likely before food is prepared.")
        with c2:
            st.markdown("**Match a rescue route**")
            st.write("Rank donation, student pickup, staff review, and compost options before leftovers appear.")
        with c3:
            st.markdown("**Improve the next event**")
            st.write("Use dashboard patterns to reduce waste over time.")


def forecast_page() -> None:
    st.title("Forecast food waste")
    st.write("Use this before food is ordered, cooked, or served.")

    event = collect_event_form("forecast", include_actual=False)

    if st.button("Run AI forecast", use_container_width=True):
        history = project_data(connected_project())
        result = forecast_event(event, history)

        st.divider()
        st.header("Forecast result")
        risk_message(result["Risk Level"])
        metric_grid(
            [
                ("Risk score", f"{result['Risk Score']:.0f}/100", "Higher means more likely to waste."),
                ("Predicted waste", f"{result['Predicted Waste Rate']:.1f}%", "Estimated leftover rate."),
                ("Recommended portions", f"{result['Recommended Min']}–{result['Recommended Max']}", "Suggested prep range."),
                ("Predicted leftovers", str(result["Predicted Leftovers"]), "Estimated leftover portions."),
            ]
        )

        with st.container(border=True):
            st.subheader("Why this event may waste food")
            for reason in result["Reasons"]:
                st.write(f"• {reason}")
            st.caption(result["Model Message"])

        with st.container(border=True):
            st.subheader("Practical actions")
            for i, action in enumerate(suggested_actions(event, result), start=1):
                st.write(f"**{i}. {action}**")

        routes = project_routes(connected_project())
        scored_routes = rescue_route_scores(event, result, routes)
        with st.container(border=True):
            st.subheader("Rescue route match")
            if scored_routes.empty:
                st.write("No rescue routes saved yet. Add routes in Rescue Board so the app can match leftovers to the best next step.")
            else:
                top_route = scored_routes.iloc[0]
                st.success(
                    f"Best route: {top_route['Route Name']} · {top_route['Readiness Score']:.0f}/100 readiness",
                    icon="✅",
                )
                st.dataframe(
                    scored_routes[
                        [
                            "Route Name",
                            "Route Type",
                            "Capacity",
                            "Pickup Minutes",
                            "Readiness Score",
                            "Recommendation",
                        ]
                    ],
                    use_container_width=True,
                )
                st.text_area(
                    "Message to copy",
                    rescue_message(event, result, top_route),
                    height=130,
                )

        st.header("What-if simulator")
        st.write("Test a different amount of prepared food and see how the predicted waste changes.")
        low = max(1, int(event["Expected Attendance"] * 0.65))
        high = max(low + 1, int(event["Expected Attendance"] * 1.45))
        tested = st.slider("Prepared portions to test", low, high, int(event["Food Prepared"]))

        rows = []
        for portions in range(low, high + 1):
            changed = dict(event)
            changed["Food Prepared"] = portions
            sim = forecast_event(changed, history)
            shortage = max(sim["Predicted Demand"] - portions, 0)
            rows.append(
                {
                    "Prepared Portions": portions,
                    "Predicted Waste %": sim["Predicted Waste Rate"],
                    "Estimated Leftovers": sim["Predicted Leftovers"],
                    "Shortage Risk": shortage,
                }
            )
        sim_df = pd.DataFrame(rows)
        selected = sim_df[sim_df["Prepared Portions"] == tested].iloc[0]

        metric_grid(
            [
                ("Test amount", str(tested), "Prepared portions."),
                ("Waste estimate", f"{selected['Predicted Waste %']:.1f}%", "Predicted waste."),
                ("Leftovers", f"{selected['Estimated Leftovers']:.0f}", "Estimated portions."),
                ("Shortage risk", f"{selected['Shortage Risk']:.0f}", "Estimated portions."),
            ]
        )

        if PLOTLY_OK:
            fig = go.Figure()
            fig.add_trace(
                go.Scatter(
                    x=sim_df["Prepared Portions"],
                    y=sim_df["Predicted Waste %"],
                    mode="lines",
                    line=dict(color="#2E7D4F", width=4),
                    fill="tozeroy",
                    fillcolor="rgba(46, 125, 79, 0.14)",
                    name="Predicted waste",
                )
            )
            fig.add_vrect(
                x0=result["Recommended Min"],
                x1=result["Recommended Max"],
                fillcolor="rgba(183,150,74,.18)",
                line_width=0,
                annotation_text="recommended range",
                annotation_position="top left",
            )
            fig.update_layout(
                height=350,
                margin=dict(l=10, r=10, t=30, b=10),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(255,253,247,.72)",
                font=dict(color="#173528"),
                xaxis_title="Prepared portions",
                yaxis_title="Predicted waste %",
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.line_chart(sim_df.set_index("Prepared Portions")["Predicted Waste %"])

        st.header("AI action plan")
        prompt = f"""
        Create a student-friendly food waste action plan.

        Goals:
        - predict where waste is likely to happen
        - identify patterns that lead to waste
        - suggest realistic actions to reduce, redistribute, or manage leftovers

        Event data:
        {json.dumps(event, indent=2)}

        Forecast result:
        {json.dumps(result, indent=2)}

        Return:
        1. Short risk summary
        2. Top 3 causes
        3. What to do before the event
        4. What to do during the event
        5. What to do after the event
        6. Food safety reminder
        """
        ai_text = groq_plan(prompt)
        if ai_text:
            st.info(ai_text)
        else:
            st.warning("AI action plan is unavailable right now. The forecast, risk score, simulator, and dashboard are still available.")


def log_page() -> None:
    st.title("Log the real result")
    st.write("Use this after the event. Real results make the dashboard and predictions smarter.")

    event = collect_event_form("log", include_actual=True)

    if st.button("Save result", use_container_width=True):
        prepared = int(event["Food Prepared"])
        leftover = int(event["Leftover Portions"])
        actual = int(event["Actual Attendance"])

        if leftover > prepared:
            st.error("Leftover portions cannot be greater than food prepared.")
            return

        history = project_data(connected_project())
        result = forecast_event(event, history)
        waste_rate = leftover / prepared * 100
        cost_impact = leftover * float(event["Cost per Portion"])
        co2_impact = leftover * float(event["CO2e per Portion"])
        score = float(np.clip(waste_rate * 2.35 + max(prepared - actual, 0) / prepared * 24, 5, 96))
        level = risk_level(score)

        record = {
            **event,
            "Time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "Waste Rate": round(waste_rate, 2),
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

        st.success("Result saved.")
        risk_message(level)
        metric_grid(
            [
                ("Actual waste", f"{waste_rate:.1f}%", "Leftover portions divided by prepared portions."),
                ("Leftovers", str(leftover), "Portions left."),
                ("Cost impact", f"${cost_impact:.0f}", "Estimated value."),
                ("CO₂e impact", f"{co2_impact:.1f} kg", "Estimated emissions."),
            ]
        )


def dashboard_page() -> None:
    st.title("Dashboard")
    st.write("Find patterns that explain when and where food waste happens.")

    df = project_data(connected_project())
    if df.empty:
        st.warning("No records yet. Add sample records from Home or log your first result.")
        return

    df = numeric(
        df,
        [
            "Waste Rate",
            "Leftover Portions",
            "Cost Impact",
            "CO2e Impact",
            "Potential Meals Rescued",
            "Risk Score",
            "Expected Attendance",
            "Actual Attendance",
            "Food Prepared",
        ],
    )
    df["Time"] = pd.to_datetime(df["Time"], errors="coerce")
    df = df.sort_values("Time")

    metric_grid(
        [
            ("Events", str(len(df)), "Logged results."),
            ("Average waste", f"{df['Waste Rate'].mean():.1f}%", "Average waste rate."),
            ("Cost impact", f"${df['Cost Impact'].sum():.0f}", "Estimated cost."),
            ("CO₂e impact", f"{df['CO2e Impact'].sum():.1f} kg", "Estimated emissions."),
        ]
    )

    model, model_message = train_history_model(df)
    st.info(model_message)

    c1, c2 = st.columns([1.2, 0.8])
    with c1:
        st.subheader("Waste trend")
        if PLOTLY_OK:
            fig = go.Figure()
            fig.add_trace(
                go.Scatter(
                    x=df["Time"],
                    y=df["Waste Rate"],
                    mode="lines+markers",
                    line=dict(color="#2E7D4F", width=4),
                    marker=dict(color="#B7964A", size=8),
                )
            )
            fig.update_layout(
                height=340,
                margin=dict(l=10, r=10, t=20, b=10),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(255,253,247,.72)",
                font=dict(color="#173528"),
                yaxis_title="Waste rate %",
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.line_chart(df.set_index("Time")["Waste Rate"])

    with c2:
        st.subheader("Risk mix")
        mix = df["Risk Level"].value_counts().reset_index()
        mix.columns = ["Risk Level", "Events"]
        if PLOTLY_OK and not mix.empty:
            fig = px.pie(
                mix,
                values="Events",
                names="Risk Level",
                hole=0.58,
                color="Risk Level",
                color_discrete_map={"Low": "#2E7D4F", "Medium": "#B7964A", "High": "#B75A4A"},
            )
            fig.update_layout(
                height=340,
                margin=dict(l=10, r=10, t=20, b=10),
                paper_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#173528"),
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.dataframe(mix, use_container_width=True)

    c3, c4 = st.columns(2)
    with c3:
        st.subheader("Highest-waste foods")
        food_avg = df.groupby("Food")["Waste Rate"].mean().sort_values(ascending=False).head(8).reset_index()
        if PLOTLY_OK:
            fig = px.bar(
                food_avg,
                x="Waste Rate",
                y="Food",
                orientation="h",
                color="Waste Rate",
                color_continuous_scale=["#DCE9D7", "#2E7D4F", "#17442F"],
            )
            fig.update_layout(
                height=340,
                margin=dict(l=10, r=10, t=20, b=10),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(255,253,247,.72)",
                font=dict(color="#173528"),
                coloraxis_showscale=False,
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.dataframe(food_avg, use_container_width=True)

    with c4:
        st.subheader("Waste by day")
        day_avg = df.groupby("Day")["Waste Rate"].mean().reindex(DAYS).dropna().reset_index()
        if PLOTLY_OK:
            fig = px.bar(
                day_avg,
                x="Day",
                y="Waste Rate",
                color="Waste Rate",
                color_continuous_scale=["#DCE9D7", "#2E7D4F", "#17442F"],
            )
            fig.update_layout(
                height=340,
                margin=dict(l=10, r=10, t=20, b=10),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(255,253,247,.72)",
                font=dict(color="#173528"),
                coloraxis_showscale=False,
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.dataframe(day_avg, use_container_width=True)

    st.subheader("Intervention comparison")
    interventions = df.groupby("Intervention")["Waste Rate"].agg(["count", "mean"]).reset_index()
    interventions.columns = ["Intervention", "Events", "Average Waste Rate"]
    interventions = interventions.sort_values("Average Waste Rate")
    st.dataframe(interventions, use_container_width=True)

    st.subheader("Recent records")
    show_cols = [
        "Time",
        "Event Name",
        "Event Type",
        "Food",
        "Day",
        "Expected Attendance",
        "Actual Attendance",
        "Food Prepared",
        "Leftover Portions",
        "Waste Rate",
        "Risk Level",
        "Intervention",
    ]
    st.dataframe(df[show_cols].tail(15), use_container_width=True)


def rescue_board_page() -> None:
    st.title("Rescue Board")
    st.write(
        "Save the people, teams, and programs that can help manage surplus food. "
        "Forecasts use this board to recommend the best rescue route before leftovers appear."
    )

    project = connected_project()
    routes = project_routes(project)

    with st.container(border=True):
        st.subheader("Add a rescue route")
        c1, c2 = st.columns(2)
        with c1:
            route_name = st.text_input("Route name", placeholder="Example: Student Council Snack Table")
            route_type = st.selectbox("Route type", ROUTE_TYPES)
            contact = st.text_input("Contact or email", placeholder="Example: student.council@school.ca")
            food_form = st.selectbox("Best food fit", FOOD_FORMS)
        with c2:
            capacity = st.number_input("Capacity in portions", min_value=1, value=25, step=1)
            pickup = st.number_input("How fast can they respond? minutes", min_value=1, value=30, step=5)
            notes = st.text_area("Notes", placeholder="Example: only unopened packaged items, staff review required, pickup near cafeteria.")

        if st.button("Save rescue route", use_container_width=True):
            if not route_name.strip():
                st.error("Please enter a route name.")
            else:
                add_route(
                    {
                        "Project": project,
                        "Route Name": route_name.strip(),
                        "Route Type": route_type,
                        "Contact": contact.strip(),
                        "Capacity": int(capacity),
                        "Pickup Minutes": int(pickup),
                        "Food Form": food_form,
                        "Notes": notes.strip(),
                    }
                )
                st.success("Rescue route saved.")
                st.rerun()

    b1, b2 = st.columns(2)
    with b1:
        if st.button("Add sample rescue routes", use_container_width=True):
            reset_project_routes(project)
            route_full = load_routes()
            route_sample = sample_routes(project)
            route_full = pd.concat([route_full[route_full["Project"] != project], route_sample], ignore_index=True)
            save_routes(route_full)
            st.success("Sample rescue routes added.")
            st.rerun()
    with b2:
        if st.button("Reset rescue routes", use_container_width=True):
            reset_project_routes(project)
            st.success("Rescue routes reset.")
            st.rerun()

    st.subheader("Saved rescue routes")
    routes = project_routes(project)
    if routes.empty:
        st.warning("No routes yet. Add at least one route so forecasts can recommend where surplus should go.")
    else:
        st.dataframe(
            routes[
                [
                    "Route Name",
                    "Route Type",
                    "Contact",
                    "Capacity",
                    "Pickup Minutes",
                    "Food Form",
                    "Notes",
                ]
            ],
            use_container_width=True,
        )

    with st.container(border=True):
        st.subheader("Why this is useful")
        st.write(
            "Most food waste tools stop at measurement. This board turns a forecast into an operations plan: "
            "who can take surplus, how much they can handle, how fast they can respond, and what message students should send."
        )


def report_text(df: pd.DataFrame, project: str) -> str:
    df = numeric(df, ["Waste Rate", "Leftover Portions", "Cost Impact", "CO2e Impact", "Potential Meals Rescued"])
    lines = [
        "# Food Rescue Radar Report",
        "",
        f"Project: {project}",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        "",
        "## Summary",
        f"- Events logged: {len(df)}",
        f"- Average waste rate: {df['Waste Rate'].mean():.1f}%",
        f"- Total leftover portions: {df['Leftover Portions'].sum():.0f}",
        f"- Estimated cost impact: ${df['Cost Impact'].sum():.0f}",
        f"- Estimated CO2e impact: {df['CO2e Impact'].sum():.1f} kg",
        f"- Potential meals rescued: {df['Potential Meals Rescued'].sum():.0f}",
        "",
        "## Recommended next actions",
        "- Confirm attendance before food is prepared.",
        "- Use smaller first batches for high-risk events.",
        "- Track menu popularity and compare it to leftovers.",
        "- Prepare a safe donation, redistribution, or compost route before large events.",
        "- Keep food safety decisions with trained human staff.",
    ]

    if not df.empty:
        food = df.groupby("Food")["Waste Rate"].mean().sort_values(ascending=False).head(1)
        day = df.groupby("Day")["Waste Rate"].mean().sort_values(ascending=False).head(1)
        if not food.empty:
            lines.insert(12, f"- Highest-waste food: {food.index[0]} at {food.iloc[0]:.1f}% average waste")
        if not day.empty:
            lines.insert(13, f"- Highest-waste day: {day.index[0]} at {day.iloc[0]:.1f}% average waste")

    return "\n".join(lines)


def report_page() -> None:
    st.title("Report")
    st.write("Download your project data and a simple impact report.")

    project = connected_project()
    df = project_data(project)

    if df.empty:
        st.warning("No records yet. Add records before exporting a report.")
        return

    report = report_text(df, project)

    c1, c2 = st.columns(2)
    with c1:
        st.download_button(
            "Download CSV",
            data=df.to_csv(index=False).encode("utf-8"),
            file_name=f"{project.replace(' ', '_').lower()}_food_waste_data.csv",
            mime="text/csv",
            use_container_width=True,
        )
    with c2:
        st.download_button(
            "Download Report",
            data=report.encode("utf-8"),
            file_name=f"{project.replace(' ', '_').lower()}_food_rescue_report.md",
            mime="text/markdown",
            use_container_width=True,
        )

    st.subheader("Preview")
    st.markdown(report)


def project_settings_page() -> None:
    st.title("Project settings")
    st.write("Manage the current project.")

    project = connected_project()

    with st.container(border=True):
        st.subheader("Current project")
        st.write(project)

        if st.button("Choose another project", use_container_width=True):
            st.session_state.project = None
            st.session_state.page = "Home"
            st.rerun()

    with st.container(border=True):
        st.subheader("Reset records")
        st.warning("This removes all logged records for this project.")
        if st.button("Reset this project data", use_container_width=True):
            reset_project_records(project)
            st.success("Project records were reset.")
            st.rerun()


# =============================================================================
# Main app
# =============================================================================
def main() -> None:
    apply_style()
    top_header()

    if "page" not in st.session_state:
        st.session_state.page = "Home"

    if not project_gate():
        return

    pages = ["Home", "Forecast", "Rescue Board", "Log Result", "Dashboard", "Report", "Project Settings"]
    current = st.session_state.page if st.session_state.page in pages else "Home"
    page = st.radio("Navigation", pages, index=pages.index(current), horizontal=True, label_visibility="collapsed")
    st.session_state.page = page

    if page == "Home":
        home_page()
    elif page == "Forecast":
        forecast_page()
    elif page == "Rescue Board":
        rescue_board_page()
    elif page == "Log Result":
        log_page()
    elif page == "Dashboard":
        dashboard_page()
    elif page == "Report":
        report_page()
    elif page == "Project Settings":
        project_settings_page()


if __name__ == "__main__":
    main()
