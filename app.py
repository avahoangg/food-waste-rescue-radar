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
            --cream: #F7F2E8;
            --paper: #FFFDF7;
            --forest: #0E2F21;
            --forest2: #17442F;
            --leaf: #2E7D4F;
            --sage: #E8F2E4;
            --gold: #B9974E;
            --ink: #163427;
            --muted: #617366;
            --line: rgba(14, 47, 33, 0.13);
            --shadow: 0 14px 36px rgba(14, 47, 33, 0.08);
        }

        @keyframes fadeUp {
            from { opacity: 0; transform: translateY(14px); }
            to { opacity: 1; transform: translateY(0); }
        }

        .stApp {
            background:
                radial-gradient(circle at 10% 4%, rgba(185,151,78,0.12), transparent 28%),
                radial-gradient(circle at 92% 5%, rgba(46,125,79,0.14), transparent 30%),
                linear-gradient(135deg, #F7F2E8 0%, #F8F8F0 45%, #EAF3E6 100%);
            color: var(--ink);
        }

        [data-testid="stHeader"] {
            background: rgba(247, 242, 232, 0.82);
            backdrop-filter: blur(16px);
            border-bottom: 1px solid rgba(14, 47, 33, 0.08);
        }

        [data-testid="stDecoration"] {
            background: linear-gradient(90deg, var(--forest), var(--leaf), var(--gold));
        }

        .block-container {
            max-width: 1180px;
            padding-top: 1.2rem;
            padding-bottom: 4rem;
            animation: fadeUp .45s ease both;
        }

        h1 {
            color: var(--forest) !important;
            font-weight: 850 !important;
            letter-spacing: -0.045em !important;
            line-height: 1.02 !important;
            font-size: clamp(2.25rem, 5vw, 4.6rem) !important;
        }

        h2, h3 {
            color: var(--forest) !important;
            letter-spacing: -0.02em !important;
        }

        p, li, label, span, div {
            color: var(--ink);
        }

        small, .stCaption, [data-testid="stCaptionContainer"] {
            color: var(--muted) !important;
        }

        div[data-testid="stVerticalBlockBorderWrapper"] {
            background: rgba(255, 253, 247, 0.86) !important;
            border: 1px solid var(--line) !important;
            border-radius: 24px !important;
            box-shadow: var(--shadow) !important;
        }

        .stButton > button {
            background: linear-gradient(135deg, var(--forest2), var(--leaf)) !important;
            color: #FFF8E8 !important;
            border: 1px solid rgba(23, 68, 47, .18) !important;
            border-radius: 999px !important;
            font-weight: 820 !important;
            min-height: 3.05rem !important;
            box-shadow: 0 12px 26px rgba(46, 125, 79, 0.22) !important;
        }

        .stButton > button p,
        .stButton > button span,
        .stButton > button div {
            color: #FFF8E8 !important;
            font-weight: 820 !important;
        }

        .stButton > button:hover {
            filter: brightness(1.04);
            transform: translateY(-1px);
        }

        div[data-testid="stDownloadButton"] > button {
            background: linear-gradient(135deg, var(--gold), #E2D0A0) !important;
            color: var(--forest) !important;
            border-radius: 999px !important;
            font-weight: 820 !important;
            min-height: 3.05rem !important;
            border: 1px solid rgba(185,151,78,.25) !important;
        }

        div[data-testid="stDownloadButton"] > button p,
        div[data-testid="stDownloadButton"] > button span {
            color: var(--forest) !important;
            font-weight: 820 !important;
        }

        div[data-testid="stRadio"] {
            padding: .4rem .5rem;
            border: 1px solid var(--line);
            border-radius: 999px;
            background: rgba(255,253,247,.78);
            box-shadow: 0 8px 20px rgba(14,47,33,.05);
            margin-bottom: 1rem;
        }

        div[data-testid="stRadio"] label p {
            color: var(--forest) !important;
            font-weight: 760 !important;
        }

        div[data-testid="stTextInput"] input,
        div[data-testid="stNumberInput"] input,
        div[data-testid="stTextArea"] textarea,
        div[data-testid="stSelectbox"] div[data-baseweb="select"] > div {
            border-radius: 15px !important;
            border: 1px solid rgba(14,47,33,.16) !important;
            background: #FFFDF7 !important;
            color: var(--forest) !important;
            box-shadow: none !important;
        }

        div[data-baseweb="select"] span,
        input, textarea {
            color: var(--forest) !important;
        }

        .stMetric {
            background: rgba(255,253,247,.88);
            border: 1px solid rgba(14,47,33,.12);
            border-radius: 20px;
            padding: 1rem;
            box-shadow: 0 9px 24px rgba(14,47,33,.055);
        }

        [data-testid="stMetricLabel"] p {
            color: var(--muted) !important;
            font-weight: 820 !important;
            letter-spacing: .05em !important;
            text-transform: uppercase !important;
            font-size: .74rem !important;
        }

        [data-testid="stMetricValue"] {
            color: var(--forest) !important;
        }

        [data-testid="stDataFrame"] {
            border-radius: 20px;
            overflow: hidden;
            border: 1px solid rgba(14,47,33,.11);
            box-shadow: var(--shadow);
        }

        [data-testid="stExpander"] {
            border-radius: 20px !important;
            border: 1px solid rgba(14,47,33,.12) !important;
            background: rgba(255,253,247,.82) !important;
            box-shadow: 0 8px 20px rgba(14,47,33,.05) !important;
        }

        div[data-testid="stAlert"] {
            border-radius: 18px;
            border: 1px solid rgba(14,47,33,.10);
            box-shadow: 0 8px 20px rgba(14,47,33,.05);
        }

        hr {
            border-color: rgba(14,47,33,.12);
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def section_label(number: str, title: str) -> None:
    st.markdown(f"## {number}. {title}")


def feature_grid(cards: list[tuple[str, str, str]]) -> None:
    for i in range(0, len(cards), 3):
        cols = st.columns(3)
        for col, card in zip(cols, cards[i:i+3]):
            icon, title, body = card
            with col:
                with st.container(border=True):
                    st.markdown(f"### {icon} {title}")
                    st.write(body)


def path_card(title: str, items: list[str]) -> None:
    with st.container(border=True):
        st.subheader(title)
        for i, item in enumerate(items, start=1):
            st.write(f"**{i}. {item}**")


def page_intro(title: str, subtitle: str, chips: list[str] | None = None) -> None:
    st.title(title)
    st.write(subtitle)
    if chips:
        cols = st.columns(min(len(chips), 4))
        for col, chip in zip(cols, chips[:4]):
            with col:
                st.info(chip)




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
# Prep Optimizer
# =============================================================================
def demand_sigma(confidence: str, expected: int) -> float:
    base = {
        "High": 0.07,
        "Medium": 0.13,
        "Low": 0.22,
    }.get(confidence, 0.13)
    return max(2.0, expected * base)


def generate_demand_scenarios(event: dict[str, Any], n: int = 900) -> np.ndarray:
    expected = max(int(event["Expected Attendance"]), 1)
    multiplier = demand_multiplier(
        event["Weather"],
        event["Attendance Confidence"],
        int(event["Menu Popularity"]),
        event["Day"],
    )
    mean = expected * multiplier
    sigma = demand_sigma(event["Attendance Confidence"], expected)

    seed_text = f"{event.get('Event Name','')}-{event.get('Food','')}-{expected}-{event.get('Day','')}"
    seed = abs(hash(seed_text)) % (2**32)
    rng = np.random.default_rng(seed)

    normal = rng.normal(mean, sigma, n)
    event_shock = rng.choice([0, -0.12, 0.08], size=n, p=[0.76, 0.16, 0.08])
    demand = normal * (1 + event_shock)
    demand = np.clip(np.round(demand), 0, expected * 1.6)
    return demand.astype(int)


def route_total_capacity(routes: pd.DataFrame, event: dict[str, Any]) -> int:
    if routes.empty:
        return 0

    routes = numeric(routes.copy(), ["Capacity"])
    food = str(event.get("Food", "")).lower()
    total = 0

    for _, row in routes.iterrows():
        capacity = int(row.get("Capacity", 0) or 0)
        form = str(row.get("Food Form", ""))
        route_type = str(row.get("Route Type", ""))

        if route_type == "Compost program":
            continue

        if form == "Mixed food":
            total += capacity
        elif "snack" in food or "bar" in food or "bakery" in food:
            if form in ["Packaged snacks", "Bakery items"]:
                total += capacity
        elif "fruit" in food or "produce" in food:
            if form in ["Produce", "Mixed food"]:
                total += capacity
        elif form in ["Prepared meals", "Mixed food"]:
            total += capacity

    return max(total, 0)


def evaluate_prep_candidate(
    prepared: int,
    event: dict[str, Any],
    scenarios: np.ndarray,
    rescue_capacity: int,
    shortage_penalty: float,
    carbon_weight: float,
) -> dict[str, Any]:
    cost_per = float(event["Cost per Portion"])
    co2_per = float(event["CO2e per Portion"])

    leftovers = np.maximum(prepared - scenarios, 0)
    shortage = np.maximum(scenarios - prepared, 0)
    rescued = np.minimum(leftovers, rescue_capacity)
    unmanaged = np.maximum(leftovers - rescued, 0)

    waste_cost = unmanaged.mean() * cost_per
    shortage_cost = shortage.mean() * shortage_penalty
    carbon_cost = unmanaged.mean() * co2_per * carbon_weight
    total_score = waste_cost + shortage_cost + carbon_cost

    return {
        "Prepared Portions": int(prepared),
        "Expected Leftovers": round(float(leftovers.mean()), 1),
        "Expected Unmanaged Waste": round(float(unmanaged.mean()), 1),
        "Expected Shortage": round(float(shortage.mean()), 1),
        "Expected Rescued": round(float(rescued.mean()), 1),
        "Shortage Risk %": round(float((shortage > 0).mean() * 100), 1),
        "Waste Risk %": round(float((unmanaged > 0).mean() * 100), 1),
        "Expected Cost Impact": round(float(waste_cost), 2),
        "Expected CO2e Impact": round(float(unmanaged.mean() * co2_per), 2),
        "Optimization Score": round(float(total_score), 2),
    }


def optimize_prep_plan(event: dict[str, Any], history: pd.DataFrame, routes: pd.DataFrame, risk_tolerance: str) -> dict[str, Any]:
    result = forecast_event(event, history)
    scenarios = generate_demand_scenarios(event)

    expected = max(int(event["Expected Attendance"]), 1)
    current = max(int(event["Food Prepared"]), 1)
    rec_min = int(result["Recommended Min"])
    rec_max = int(result["Recommended Max"])
    rescue_cap = route_total_capacity(routes, event)

    shortage_penalty = {
        "Avoid shortage as much as possible": float(event["Cost per Portion"]) * 4.5,
        "Balanced": float(event["Cost per Portion"]) * 2.8,
        "Minimize waste as much as possible": float(event["Cost per Portion"]) * 1.5,
    }.get(risk_tolerance, float(event["Cost per Portion"]) * 2.8)

    carbon_weight = {
        "Avoid shortage as much as possible": 0.30,
        "Balanced": 0.55,
        "Minimize waste as much as possible": 0.90,
    }.get(risk_tolerance, 0.55)

    low = max(1, min(rec_min, int(expected * 0.70)))
    high = max(current, rec_max, int(expected * 1.35))

    candidates = []
    for prepared in range(low, high + 1):
        candidates.append(
            evaluate_prep_candidate(
                prepared,
                event,
                scenarios,
                rescue_cap,
                shortage_penalty,
                carbon_weight,
            )
        )

    df = pd.DataFrame(candidates)
    best = df.sort_values(["Optimization Score", "Expected Unmanaged Waste", "Expected Shortage"]).iloc[0].to_dict()

    if event["Batch Cooking"] == "Yes":
        first_batch = int(round(best["Prepared Portions"] * 0.78))
    else:
        first_batch = int(best["Prepared Portions"])

    reserve = int(best["Prepared Portions"] - first_batch)
    checkpoint = "After the first 20–25% of attendees arrive"

    plan = [
        ("T-24 hours", "Confirm attendance and menu interest with a short form or homeroom count."),
        ("T-2 hours", f"Prepare the first batch: {first_batch} portions."),
        ("Service start", "Track actual turnout during the first part of service."),
        ("Checkpoint", f"{checkpoint}. Release reserve only if turnout is on pace."),
        ("End of service", "Count leftovers and send them to the best rescue route after human food-safety review."),
        ("After event", "Log the final numbers so the next forecast becomes smarter."),
    ]

    return {
        "forecast": result,
        "scenarios": scenarios,
        "table": df,
        "best": best,
        "first_batch": first_batch,
        "reserve": reserve,
        "rescue_capacity": rescue_cap,
        "plan": plan,
    }



# =============================================================================
# Waste Audit Lab
# =============================================================================
def root_cause_for_row(row: pd.Series) -> dict[str, Any]:
    expected = max(float(row.get("Expected Attendance", 0) or 0), 1)
    actual = float(row.get("Actual Attendance", 0) or 0)
    prepared = max(float(row.get("Food Prepared", 0) or 0), 1)
    leftover = max(float(row.get("Leftover Portions", 0) or 0), 0)
    waste_rate = float(row.get("Waste Rate", 0) or 0)
    popularity = float(row.get("Menu Popularity", 3) or 3)
    weather = str(row.get("Weather", "Normal"))
    intervention = str(row.get("Intervention", "None yet"))
    donation = str(row.get("Donation Route", "No"))
    rescued = float(row.get("Potential Meals Rescued", 0) or 0)

    attendance_gap_rate = max(expected - actual, 0) / expected
    overproduction_rate = max(prepared - expected, prepared - actual, 0) / prepared
    leftover_rate = leftover / prepared

    scores = {
        "Attendance variance": attendance_gap_rate * 100 + (10 if str(row.get("Attendance Confidence", "")) == "Low" else 0),
        "Overproduction": overproduction_rate * 75 + max(prepared - expected, 0) / expected * 30,
        "Menu acceptance": max(0, 3 - popularity) * 18 + waste_rate * 0.22,
        "Weather disruption": (18 if weather in ["Rainy", "Stormy", "Very hot", "Very cold"] else 0) + attendance_gap_rate * 35,
        "Rescue capacity gap": (max(leftover - rescued, 0) / max(leftover, 1)) * 45 if leftover > 0 else 0,
        "No active intervention": 25 if intervention == "None yet" and waste_rate >= 12 else 0,
    }

    if donation == "Yes":
        scores["Rescue capacity gap"] -= 8
    if intervention != "None yet":
        scores["No active intervention"] = 0

    primary = max(scores, key=scores.get)
    confidence = "High" if scores[primary] >= 45 else "Medium" if scores[primary] >= 25 else "Low"

    evidence_map = {
        "Attendance variance": f"Expected {expected:.0f} people but logged {actual:.0f}; attendance gap is {attendance_gap_rate:.0%}.",
        "Overproduction": f"Prepared {prepared:.0f} portions for {actual:.0f} actual attendees.",
        "Menu acceptance": f"Menu popularity is {popularity:.0f}/5 with {waste_rate:.1f}% waste.",
        "Weather disruption": f"Condition was {weather}, which can affect turnout and appetite.",
        "Rescue capacity gap": f"{max(leftover - rescued, 0):.0f} leftover portions were not matched to a rescue route.",
        "No active intervention": "No intervention was recorded for a moderate or high-waste event.",
    }

    return {
        "Primary Cause": primary,
        "Cause Score": round(float(scores[primary]), 1),
        "Confidence": confidence,
        "Evidence": evidence_map.get(primary, "Root cause detected from event signals."),
    }


def build_root_cause_audit(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()

    df = numeric(
        df.copy(),
        [
            "Expected Attendance",
            "Actual Attendance",
            "Food Prepared",
            "Leftover Portions",
            "Waste Rate",
            "Cost Impact",
            "CO2e Impact",
            "Menu Popularity",
            "Potential Meals Rescued",
        ],
    )

    rows = []
    for idx, row in df.iterrows():
        cause = root_cause_for_row(row)
        rows.append(
            {
                "Event Name": row.get("Event Name", ""),
                "Food": row.get("Food", ""),
                "Day": row.get("Day", ""),
                "Waste Rate": round(float(row.get("Waste Rate", 0) or 0), 1),
                "Leftover Portions": round(float(row.get("Leftover Portions", 0) or 0), 1),
                "Cost Impact": round(float(row.get("Cost Impact", 0) or 0), 2),
                "CO2e Impact": round(float(row.get("CO2e Impact", 0) or 0), 2),
                "Intervention": row.get("Intervention", ""),
                **cause,
            }
        )
    return pd.DataFrame(rows)


def root_cause_summary(audit: pd.DataFrame) -> pd.DataFrame:
    if audit.empty:
        return pd.DataFrame()
    summary = (
        audit.groupby("Primary Cause")
        .agg(
            Events=("Event Name", "count"),
            Waste_Portions=("Leftover Portions", "sum"),
            Cost_Impact=("Cost Impact", "sum"),
            CO2e_Impact=("CO2e Impact", "sum"),
            Avg_Waste_Rate=("Waste Rate", "mean"),
            Avg_Cause_Score=("Cause Score", "mean"),
        )
        .reset_index()
        .sort_values("Waste_Portions", ascending=False)
    )
    total = max(float(summary["Waste_Portions"].sum()), 1)
    summary["Share of Waste %"] = (summary["Waste_Portions"] / total * 100).round(1)
    summary["Cumulative %"] = summary["Share of Waste %"].cumsum().round(1)
    return summary


def intervention_roi(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()

    df = numeric(df.copy(), ["Waste Rate", "Food Prepared", "Cost per Portion", "CO2e per Portion"])
    valid = df.dropna(subset=["Waste Rate", "Food Prepared"])
    if valid.empty:
        return pd.DataFrame()

    none = valid[valid["Intervention"] == "None yet"]
    baseline = float(none["Waste Rate"].mean()) if len(none) >= 2 else float(valid["Waste Rate"].mean())
    avg_cost = float(valid["Cost per Portion"].dropna().mean()) if not valid["Cost per Portion"].dropna().empty else 2.5
    avg_co2 = float(valid["CO2e per Portion"].dropna().mean()) if not valid["CO2e per Portion"].dropna().empty else 0.5

    rows = []
    for intervention, group in valid.groupby("Intervention"):
        count = len(group)
        avg_waste = float(group["Waste Rate"].mean())
        mean_prepared = float(group["Food Prepared"].mean())
        reduction = baseline - avg_waste
        saved_portions = max(reduction, 0) / 100 * mean_prepared * count
        confidence = "High" if count >= 5 else "Medium" if count >= 3 else "Low"

        rows.append(
            {
                "Intervention": intervention,
                "Events": count,
                "Avg Waste Rate": round(avg_waste, 1),
                "Baseline Waste Rate": round(baseline, 1),
                "Estimated Reduction": round(reduction, 1),
                "Estimated Portions Saved": round(saved_portions, 1),
                "Estimated Cost Saved": round(saved_portions * avg_cost, 2),
                "Estimated CO2e Avoided": round(saved_portions * avg_co2, 2),
                "Confidence": confidence,
            }
        )

    return pd.DataFrame(rows).sort_values(["Estimated Portions Saved", "Events"], ascending=False)


def next_best_actions(summary: pd.DataFrame, roi: pd.DataFrame, routes: pd.DataFrame) -> pd.DataFrame:
    if summary.empty:
        return pd.DataFrame()

    cause_impact = {row["Primary Cause"]: float(row["Waste_Portions"]) for _, row in summary.iterrows()}
    total_waste = max(sum(cause_impact.values()), 1)

    ideas = [
        {
            "Action": "Require attendance confirmation 24 hours before food prep",
            "Targets": "Attendance variance",
            "Effort": 2,
            "Why": "Reduces uncertainty before ordering or cooking.",
        },
        {
            "Action": "Use smaller first batch with a service checkpoint",
            "Targets": "Overproduction",
            "Effort": 3,
            "Why": "Keeps food flexible until turnout is clear.",
        },
        {
            "Action": "Run a menu preference survey before repeating low-score meals",
            "Targets": "Menu acceptance",
            "Effort": 2,
            "Why": "Finds unpopular meals before they become leftovers.",
        },
        {
            "Action": "Build a weather adjustment rule for outdoor or Friday events",
            "Targets": "Weather disruption",
            "Effort": 3,
            "Why": "Adjusts production when conditions change demand.",
        },
        {
            "Action": "Add or expand rescue routes with capacity and response time",
            "Targets": "Rescue capacity gap",
            "Effort": 3 if routes.empty else 2,
            "Why": "Turns leftovers into a planned route instead of a last-minute decision.",
        },
        {
            "Action": "Assign one intervention owner for each high-risk event",
            "Targets": "No active intervention",
            "Effort": 1,
            "Why": "Makes prevention part of the event workflow.",
        },
    ]

    rows = []
    for idea in ideas:
        impact = cause_impact.get(idea["Targets"], 0)
        impact_share = impact / total_waste * 100
        priority = impact_share * 1.15 - idea["Effort"] * 6

        # Lift priority when past data shows the same intervention is promising.
        if not roi.empty:
            if "attendance" in idea["Action"].lower():
                match = roi[roi["Intervention"].str.contains("Attendance", case=False, na=False)]
            elif "batch" in idea["Action"].lower():
                match = roi[roi["Intervention"].str.contains("batch", case=False, na=False)]
            elif "menu" in idea["Action"].lower():
                match = roi[roi["Intervention"].str.contains("Menu", case=False, na=False)]
            elif "rescue" in idea["Action"].lower():
                match = roi[roi["Intervention"].str.contains("Donation", case=False, na=False)]
            else:
                match = pd.DataFrame()

            if not match.empty and float(match["Estimated Portions Saved"].max()) > 0:
                priority += 10

        rows.append(
            {
                "Recommended Action": idea["Action"],
                "Targets": idea["Targets"],
                "Waste Share Addressed %": round(impact_share, 1),
                "Effort 1-5": idea["Effort"],
                "Priority Score": round(float(priority), 1),
                "Why it matters": idea["Why"],
            }
        )

    return pd.DataFrame(rows).sort_values("Priority Score", ascending=False)


def experiment_card(project: str, action: str, target_cause: str, target_reduction: int, events: int) -> str:
    return f"""# 30-Day Waste Reduction Experiment

Project: {project}

## Experiment Focus
Target cause: {target_cause}

Action to test: {action}

## Hypothesis
If the team uses this action consistently, food waste connected to {target_cause.lower()} will decrease by at least {target_reduction}% over the next {events} logged events.

## Setup
1. Choose the next {events} events where this action is realistic.
2. Use Forecast or Prep Optimizer before each event.
3. Apply the action before or during service.
4. Log actual attendance, prepared portions, and leftovers after each event.
5. Compare the average waste rate against the project baseline.

## Success Metric
Primary metric: waste rate %

Secondary metrics:
- leftover portions
- cost impact
- CO2e impact
- rescued portions

## Review Question
Did this action reduce waste enough to become a standard operating procedure?

Responsible note: Food safety decisions must stay with trained staff and local rules.
"""


def audit_report_text(project: str, summary: pd.DataFrame, actions: pd.DataFrame, roi: pd.DataFrame) -> str:
    lines = [
        "# Waste Audit Lab Report",
        "",
        f"Project: {project}",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        "",
        "## Root Cause Summary",
    ]

    if summary.empty:
        lines.append("No root cause data available.")
    else:
        for _, row in summary.head(6).iterrows():
            lines.append(
                f"- {row['Primary Cause']}: {row['Waste_Portions']:.0f} portions "
                f"({row['Share of Waste %']:.1f}% of waste), avg waste {row['Avg_Waste_Rate']:.1f}%"
            )

    lines.append("")
    lines.append("## Highest Priority Actions")
    if actions.empty:
        lines.append("No action ranking available.")
    else:
        for _, row in actions.head(5).iterrows():
            lines.append(
                f"- {row['Recommended Action']} | Targets: {row['Targets']} | "
                f"Priority: {row['Priority Score']:.1f}"
            )

    lines.append("")
    lines.append("## Intervention Signals")
    if roi.empty:
        lines.append("No intervention comparison available.")
    else:
        for _, row in roi.head(5).iterrows():
            lines.append(
                f"- {row['Intervention']}: avg waste {row['Avg Waste Rate']:.1f}%, "
                f"estimated saved portions {row['Estimated Portions Saved']:.1f}, confidence {row['Confidence']}"
            )

    lines.append("")
    lines.append("Responsible note: This report supports planning and audit review only. Food safety decisions must be made by trained staff.")

    return "\n".join(lines)


# =============================================================================
# Interface helpers
# =============================================================================
def connected_project() -> str | None:
    return st.session_state.get("project")


def top_header() -> None:
    project = connected_project()
    c1, c2 = st.columns([1.5, 1])
    with c1:
        st.markdown("### 🌿 Food Rescue Radar")
        st.caption("Plan food better. Waste less.")
    with c2:
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

    page_intro(
        "Food Rescue Radar",
        "A student friendly tool that helps schools and community teams predict food waste, prepare smarter, and rescue more food.",
        ["For cafeterias", "For school clubs", "For events", "For donation programs"],
    )

    st.divider()

    section_label("1", "Start your project")
    st.write("Create one project for each cafeteria, club, event team, or community group.")

    left, right = st.columns([1.05, 0.95], gap="large")

    with left:
        with st.container(border=True):
            st.subheader("Create a new project")
            st.caption("Example names: Wagner Cafeteria, Green Club Events, Friday Lunch Program")
            new_project = st.text_input("Project name", placeholder="Example: Wagner Cafeteria")
            if st.button("Create project", use_container_width=True):
                name = clean_project_name(new_project)
                if not name:
                    st.error("Please enter a project name.")
                else:
                    add_project(name)
                    st.session_state.project = name
                    st.rerun()

    with right:
        with st.container(border=True):
            st.subheader("Open a saved project")
            projects = load_projects()
            if projects:
                selected = st.selectbox("Choose project", projects)
                if st.button("Open project", use_container_width=True):
                    st.session_state.project = selected
                    st.rerun()
            else:
                st.write("No saved projects yet.")

    st.divider()

    section_label("2", "Who this app is for")
    feature_grid(
        [
            ("🏫", "Cafeteria teams", "Plan how much food to prepare for school meals."),
            ("🎒", "Student clubs", "Prepare snacks for meetings, fundraisers, or events."),
            ("🎟️", "Event teams", "Estimate demand before a sports night or community dinner."),
            ("🤝", "Donation teams", "Plan where safe surplus food may go after human review."),
            ("🌱", "Green teams", "Track impact, reduce waste, and report progress."),
            ("📊", "Teachers and mentors", "Use real data for sustainability projects."),
        ]
    )

    section_label("3", "When to use each feature")
    feature_grid(
        [
            ("🔮", "Before ordering food", "Use Forecast to check if waste is likely and get practical actions."),
            ("⚖️", "Before a large event", "Use Prep Optimizer to choose total portions, first batch, and hold back amount."),
            ("🧭", "Before leftovers appear", "Use Rescue Board to save donation, pickup, review, and compost routes."),
            ("📝", "After the event", "Use Log Result to record attendance, prepared food, and leftovers."),
            ("📊", "After several events", "Use Dashboard to find trends by food, day, risk, and intervention."),
            ("🧪", "When you need a serious analysis", "Use Waste Audit Lab to find root causes and plan a 30 day experiment."),
        ]
    )

    section_label("4", "FAQ")
    with st.expander("Do I need data before using the app?", expanded=True):
        st.write("No. You can start with Forecast or Prep Optimizer using estimates. The app becomes smarter after you log real results.")
    with st.expander("Does the app decide if food is safe to donate?"):
        st.write("No. The app only helps with planning. Trained staff must inspect food and follow local food safety rules.")
    with st.expander("What should I do first for a demo?"):
        st.write("Create a project, then click Add sample records and rescue routes on the Home page. Then open Waste Audit Lab, Prep Optimizer, Forecast, and Dashboard.")
    with st.expander("Can different schools or clubs use the same app?"):
        st.write("Yes. Create a separate project for each group so the data stays organized.")

    return False


def home_page() -> None:
    page_intro(
        "Welcome to your project",
        "Use this page as your guide. Choose the tool that matches what your team is doing right now.",
        ["Forecast before prep", "Optimize large events", "Audit root causes", "Export reports"],
    )

    df = project_data(connected_project())
    df_num = numeric(df, ["Waste Rate", "Leftover Portions", "Cost Impact", "CO2e Impact", "Potential Meals Rescued"])

    section_label("1", "Project snapshot")
    if df_num.empty:
        metric_grid(
            [
                ("Records", "0", "No logged events yet."),
                ("Average waste", "—", "Log events to unlock this."),
                ("Cost impact", "—", "Estimated from leftovers."),
                ("Meals rescued", "—", "Based on rescue routes."),
            ]
        )
    else:
        metric_grid(
            [
                ("Records", str(len(df_num)), "Events logged."),
                ("Average waste", f"{df_num['Waste Rate'].mean():.1f}%", "Average leftover rate."),
                ("Cost impact", f"${df_num['Cost Impact'].sum():.0f}", "Estimated waste cost."),
                ("Meals rescued", f"{df_num['Potential Meals Rescued'].sum():.0f}", "Potential rescued portions."),
            ]
        )

    section_label("2", "What do you want to do?")
    feature_grid(
        [
            ("🔮", "Forecast waste", "Use before food is ordered or cooked. Predict risk and get actions."),
            ("⚖️", "Optimize prep", "Use before a large event. Find the best total portions and first batch."),
            ("🧪", "Audit causes", "Use after logging results. Find why waste happens and what to fix."),
            ("🧭", "Plan rescue routes", "Save donation, pickup, staff review, and compost options."),
            ("📊", "View dashboard", "See patterns by food, day, risk level, and intervention."),
            ("📄", "Export report", "Download data and a clean summary for school or judging."),
        ]
    )

    b1, b2, b3, b4, b5 = st.columns(5)
    with b1:
        if st.button("Forecast", use_container_width=True):
            st.session_state.page = "Forecast"
            st.rerun()
    with b2:
        if st.button("Optimizer", use_container_width=True):
            st.session_state.page = "Prep Optimizer"
            st.rerun()
    with b3:
        if st.button("Waste Lab", use_container_width=True):
            st.session_state.page = "Waste Audit Lab"
            st.rerun()
    with b4:
        if st.button("Rescue Board", use_container_width=True):
            st.session_state.page = "Rescue Board"
            st.rerun()
    with b5:
        if st.button("Dashboard", use_container_width=True):
            st.session_state.page = "Dashboard"
            st.rerun()

    section_label("3", "Recommended workflow")
    path_card(
        "If you are planning food for an event",
        [
            "Open Forecast before ordering food.",
            "Use Prep Optimizer if the event is large or attendance is uncertain.",
            "Check Rescue Board so leftovers have a route.",
            "Log the real result after the event.",
            "Use Dashboard and Waste Audit Lab to improve the next event.",
        ],
    )

    section_label("4", "Try the app with sample data")
    st.write("Use this if you want to demo the app quickly.")
    if st.button("Add sample records and rescue routes", use_container_width=True):
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
        st.success("Sample records and rescue routes were added.")
        st.rerun()

    section_label("5", "FAQ")
    with st.expander("Which feature should I use first?"):
        st.write("Use Forecast first if you are planning a meal or event. Use Log Result after the event.")
    with st.expander("When should I use Prep Optimizer?"):
        st.write("Use it when attendance is uncertain or when preparing too much food would be expensive.")
    with st.expander("When should I use Waste Audit Lab?"):
        st.write("Use it after you have several logged events. It finds the main reasons waste is happening.")
    with st.expander("What does Rescue Board do?"):
        st.write("It stores routes for possible surplus food. Examples include donation review, student pickup, staff meal review, and compost.")
    with st.expander("Does this replace food safety rules?"):
        st.write("No. The app supports planning only. Trained staff must decide what can be served, donated, composted, or discarded.")


def forecast_page() -> None:
    page_intro(
        "Forecast food waste",
        "Use this before food is ordered, cooked, or served.",
        ["Risk score", "Practical actions", "Rescue match"],
    )

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
    page_intro(
        "Log the real result",
        "Use this after the event. Real results make the project smarter.",
        ["Attendance", "Leftovers", "Impact"],
    )

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
    page_intro(
        "Dashboard",
        "Find patterns that explain when and where food waste happens.",
        ["Trends", "Risk mix", "Interventions"],
    )

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


def waste_audit_lab_page() -> None:
    page_intro(
        "Waste Audit Lab",
        "Use this after logging results. It finds why waste happens and recommends the next best action.",
        ["Root causes", "Intervention ROI", "30 day experiment"],
    )

    project = connected_project()
    df = project_data(project)

    if df.empty:
        st.warning("No logged records yet. Add sample records from Home or log real event results first.")
        return

    df = numeric(
        df,
        [
            "Waste Rate",
            "Leftover Portions",
            "Cost Impact",
            "CO2e Impact",
            "Potential Meals Rescued",
            "Food Prepared",
            "Expected Attendance",
            "Actual Attendance",
            "Menu Popularity",
            "Cost per Portion",
            "CO2e per Portion",
        ],
    )

    audit = build_root_cause_audit(df)
    summary = root_cause_summary(audit)
    roi = intervention_roi(df)
    routes = project_routes(project)
    actions = next_best_actions(summary, roi, routes)

    if summary.empty:
        st.warning("Not enough usable records for audit analysis.")
        return

    top_cause = summary.iloc[0]
    best_action = actions.iloc[0] if not actions.empty else None
    best_roi = roi[roi["Intervention"] != "None yet"].head(1)

    metric_grid(
        [
            ("Top root cause", str(top_cause["Primary Cause"]), "Largest share of leftover portions."),
            ("Waste share", f"{top_cause['Share of Waste %']:.1f}%", "Share from the top cause."),
            ("Cost tied to top cause", f"${top_cause['Cost_Impact']:.0f}", "Estimated cost impact."),
            ("Audit confidence", "Ready" if len(df) >= 8 else "Early", "More records improve confidence."),
        ]
    )

    st.header("Root cause Pareto")
    st.write("The first bars show where the biggest waste reduction opportunity is.")

    if PLOTLY_OK:
        fig = go.Figure()
        fig.add_trace(
            go.Bar(
                x=summary["Primary Cause"],
                y=summary["Waste_Portions"],
                name="Waste portions",
                marker_color="#2E7D4F",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=summary["Primary Cause"],
                y=summary["Cumulative %"],
                name="Cumulative %",
                yaxis="y2",
                line=dict(color="#B7964A", width=4),
                mode="lines+markers",
            )
        )
        fig.update_layout(
            height=390,
            margin=dict(l=10, r=10, t=30, b=10),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(255,253,247,.72)",
            font=dict(color="#173528"),
            yaxis=dict(title="Leftover portions"),
            yaxis2=dict(title="Cumulative %", overlaying="y", side="right", range=[0, 105]),
            legend=dict(orientation="h"),
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.bar_chart(summary.set_index("Primary Cause")["Waste_Portions"])

    st.dataframe(
        summary[
            [
                "Primary Cause",
                "Events",
                "Waste_Portions",
                "Share of Waste %",
                "Cost_Impact",
                "CO2e_Impact",
                "Avg_Waste_Rate",
                "Avg_Cause_Score",
            ]
        ],
        use_container_width=True,
    )

    st.header("Next-best actions")
    st.write("This ranks actions by how much waste they can address and how hard they are to implement.")
    st.dataframe(actions, use_container_width=True)

    st.header("Intervention ROI")
    st.write("This compares each intervention against the project baseline. Treat low-confidence rows as early signals, not final proof.")
    st.dataframe(roi, use_container_width=True)

    if PLOTLY_OK and not roi.empty:
        plot_roi = roi[roi["Intervention"] != "None yet"].copy()
        if not plot_roi.empty:
            fig2 = px.scatter(
                plot_roi,
                x="Estimated Cost Saved",
                y="Estimated CO2e Avoided",
                size="Estimated Portions Saved",
                color="Confidence",
                hover_name="Intervention",
                color_discrete_map={"Low": "#B7964A", "Medium": "#2E7D4F", "High": "#17442F"},
            )
            fig2.update_layout(
                height=360,
                margin=dict(l=10, r=10, t=30, b=10),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(255,253,247,.72)",
                font=dict(color="#173528"),
            )
            st.plotly_chart(fig2, use_container_width=True)

    st.header("Event-level audit")
    st.dataframe(
        audit[
            [
                "Event Name",
                "Food",
                "Day",
                "Waste Rate",
                "Leftover Portions",
                "Primary Cause",
                "Confidence",
                "Evidence",
                "Intervention",
            ]
        ],
        use_container_width=True,
    )

    st.header("30-day experiment planner")
    if not actions.empty:
        default_action = actions.iloc[0]["Recommended Action"]
        default_cause = actions.iloc[0]["Targets"]
    else:
        default_action = "Use smaller first batch with a service checkpoint"
        default_cause = str(top_cause["Primary Cause"])

    c1, c2, c3 = st.columns(3)
    with c1:
        selected_cause = st.selectbox(
            "Target cause",
            summary["Primary Cause"].tolist(),
            index=summary["Primary Cause"].tolist().index(default_cause) if default_cause in summary["Primary Cause"].tolist() else 0,
        )
    with c2:
        target_reduction = st.slider("Target reduction", 5, 40, 15, step=5)
    with c3:
        event_count = st.slider("Events to test", 2, 12, 4)

    action_options = actions["Recommended Action"].tolist() if not actions.empty else [default_action]
    selected_action = st.selectbox(
        "Action to test",
        action_options,
        index=action_options.index(default_action) if default_action in action_options else 0,
    )

    card = experiment_card(project, selected_action, selected_cause, target_reduction, event_count)

    st.download_button(
        "Download 30-day experiment card",
        data=card.encode("utf-8"),
        file_name=f"{project.replace(' ', '_').lower()}_experiment_card.md",
        mime="text/markdown",
        use_container_width=True,
    )

    st.text_area("Experiment preview", card, height=360)

    st.header("Audit report")
    report = audit_report_text(project, summary, actions, roi)
    st.download_button(
        "Download audit report",
        data=report.encode("utf-8"),
        file_name=f"{project.replace(' ', '_').lower()}_waste_audit_report.md",
        mime="text/markdown",
        use_container_width=True,
    )


def prep_optimizer_page() -> None:
    page_intro(
        "Prep Optimizer",
        "Use this before a larger event. It tests many attendance outcomes and recommends a smarter preparation plan.",
        ["Total portions", "First batch", "Hold back amount", "Shortage risk"],
    )

    event = collect_event_form("optimizer", include_actual=False)

    with st.container(border=True):
        st.subheader("Optimization goal")
        risk_tolerance = st.selectbox(
            "What matters most for this event?",
            [
                "Balanced",
                "Avoid shortage as much as possible",
                "Minimize waste as much as possible",
            ],
            help="Choose the tradeoff. A cafeteria may choose balanced. A competition banquet may avoid shortage. A small club may minimize waste.",
        )

    if st.button("Optimize preparation plan", use_container_width=True):
        history = project_data(connected_project())
        routes = project_routes(connected_project())
        optimization = optimize_prep_plan(event, history, routes, risk_tolerance)

        best = optimization["best"]
        forecast = optimization["forecast"]
        table = optimization["table"]

        st.divider()
        st.header("Recommended plan")
        risk_message(forecast["Risk Level"])
        metric_grid(
            [
                ("Prepare total", f"{int(best['Prepared Portions'])}", "Optimized total portions."),
                ("First batch", f"{optimization['first_batch']}", "Prepare before service starts."),
                ("Hold back", f"{optimization['reserve']}", "Keep flexible until checkpoint."),
                ("Shortage risk", f"{best['Shortage Risk %']:.1f}%", "Chance demand is higher than prepared."),
            ]
        )

        metric_grid(
            [
                ("Expected leftovers", f"{best['Expected Leftovers']:.1f}", "Average across simulations."),
                ("Expected rescued", f"{best['Expected Rescued']:.1f}", "Based on saved rescue routes."),
                ("Unmanaged waste", f"{best['Expected Unmanaged Waste']:.1f}", "Leftovers not matched to a rescue route."),
                ("CO₂e impact", f"{best['Expected CO2e Impact']:.1f} kg", "Estimated unmanaged waste impact."),
            ]
        )

        with st.container(border=True):
            st.subheader("Service timeline")
            for time_label, step in optimization["plan"]:
                st.write(f"**{time_label}:** {step}")

        with st.container(border=True):
            st.subheader("Why this plan is better")
            current_row = evaluate_prep_candidate(
                int(event["Food Prepared"]),
                event,
                optimization["scenarios"],
                optimization["rescue_capacity"],
                float(event["Cost per Portion"]) * 2.8,
                0.55,
            )
            comparison = pd.DataFrame(
                [
                    {"Plan": "Current plan", **current_row},
                    {"Plan": "Optimized plan", **best},
                ]
            )
            st.dataframe(
                comparison[
                    [
                        "Plan",
                        "Prepared Portions",
                        "Expected Unmanaged Waste",
                        "Expected Shortage",
                        "Expected Rescued",
                        "Shortage Risk %",
                        "Expected Cost Impact",
                        "Expected CO2e Impact",
                    ]
                ],
                use_container_width=True,
            )

        if PLOTLY_OK:
            st.subheader("Tradeoff curve")
            fig = go.Figure()
            fig.add_trace(
                go.Scatter(
                    x=table["Prepared Portions"],
                    y=table["Expected Unmanaged Waste"],
                    mode="lines",
                    name="Unmanaged waste",
                    line=dict(color="#2E7D4F", width=4),
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=table["Prepared Portions"],
                    y=table["Expected Shortage"],
                    mode="lines",
                    name="Shortage",
                    line=dict(color="#B7964A", width=4),
                )
            )
            fig.add_vline(
                x=int(best["Prepared Portions"]),
                line_dash="dash",
                line_color="#17442F",
                annotation_text="optimized",
            )
            fig.update_layout(
                height=370,
                margin=dict(l=10, r=10, t=30, b=10),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(255,253,247,.72)",
                font=dict(color="#173528"),
                xaxis_title="Prepared portions",
                yaxis_title="Expected portions",
            )
            st.plotly_chart(fig, use_container_width=True)

            st.subheader("Demand simulation")
            fig2 = px.histogram(
                pd.DataFrame({"Simulated Demand": optimization["scenarios"]}),
                x="Simulated Demand",
                nbins=28,
                color_discrete_sequence=["#2E7D4F"],
            )
            fig2.add_vline(
                x=int(best["Prepared Portions"]),
                line_dash="dash",
                line_color="#B7964A",
                annotation_text="optimized prep",
            )
            fig2.update_layout(
                height=330,
                margin=dict(l=10, r=10, t=30, b=10),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(255,253,247,.72)",
                font=dict(color="#173528"),
            )
            st.plotly_chart(fig2, use_container_width=True)

        st.subheader("Rescue route fit")
        scored_routes = rescue_route_scores(event, forecast, routes)
        if scored_routes.empty:
            st.warning("No rescue routes saved yet. Add routes in Rescue Board to improve the optimizer.")
        else:
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

def rescue_board_page() -> None:
    page_intro(
        "Rescue Board",
        "Save the people, teams, and programs that can help manage surplus food.",
        ["Donation", "Pickup", "Staff review", "Compost"],
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
    page_intro(
        "Report",
        "Download your project data and a clean impact report.",
        ["CSV", "Summary", "Proof"],
    )

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
    page_intro(
        "Project settings",
        "Manage the current project.",
        ["Switch project", "Reset data"],
    )

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

    pages = ["Home", "Forecast", "Prep Optimizer", "Waste Audit Lab", "Rescue Board", "Log Result", "Dashboard", "Report", "Project Settings"]
    current = st.session_state.page if st.session_state.page in pages else "Home"
    page = st.radio("Navigation", pages, index=pages.index(current), horizontal=True, label_visibility="collapsed")
    st.session_state.page = page

    if page == "Home":
        home_page()
    elif page == "Forecast":
        forecast_page()
    elif page == "Prep Optimizer":
        prep_optimizer_page()
    elif page == "Waste Audit Lab":
        waste_audit_lab_page()
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
