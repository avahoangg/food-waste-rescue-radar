import streamlit as st
import pandas as pd
from pathlib import Path
from datetime import datetime
from groq import Groq


st.set_page_config(
    page_title="Food Waste Rescue Radar",
    page_icon="🍽️",
    layout="wide"
)


EVENT_TYPES = ["School lunch", "Community event"]
LOCATIONS = ["Cafeteria", "Party / Event venue"]
WEATHER_OPTIONS = ["Normal", "Sunny", "Rainy", "Cloudy", "Stormy", "Very hot"]

HISTORY_FILE = Path("historical_data.csv")

HISTORY_COLUMNS = [
    "Username",
    "Time",
    "Event Type",
    "Location",
    "Meal Type",
    "Expected Attendance",
    "Actual Attendance",
    "Food Prepared",
    "Leftover Portions",
    "Waste Rate",
    "Risk Level",
    "Notes",
]


if "page" not in st.session_state:
    st.session_state.page = "home"

if "username" not in st.session_state:
    st.session_state.username = None


def load_history():
    if not HISTORY_FILE.exists():
        return pd.DataFrame(columns=HISTORY_COLUMNS)

    history = pd.read_csv(HISTORY_FILE)

    for column in HISTORY_COLUMNS:
        if column not in history.columns:
            history[column] = None

    return history[HISTORY_COLUMNS]


def save_history(row):
    history = load_history()
    new_row = pd.DataFrame([row])
    updated_history = pd.concat([history, new_row], ignore_index=True)
    updated_history.to_csv(HISTORY_FILE, index=False)


def get_groq_client():
    api_key = st.secrets.get("GROQ_API_KEY", None)

    if not api_key:
        return None

    return Groq(api_key=api_key)


def ask_ai(prompt):
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
                        "You are a helpful AI assistant for food waste reduction. "
                        "Give practical, realistic, and safe recommendations. "
                        "Do not give food safety guarantees."
                    ),
                },
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
            temperature=0.4,
            max_tokens=700,
        )

        return response.choices[0].message.content

    except Exception:
        return (
            "AI analysis is temporarily unavailable because the free API quota may be limited. "
            "The app still provides basic rule-based recommendations."
        )


def go_home():
    st.session_state.page = "home"


def go_predict():
    st.session_state.page = "predict"


def go_guide():
    st.session_state.page = "guide"


def go_dashboard():
    st.session_state.page = "dashboard"


def switch_user():
    st.session_state.username = None
    st.session_state.page = "home"
    st.rerun()


def get_risk_level(waste_rate):
    if waste_rate >= 25:
        return "High"
    if waste_rate >= 10:
        return "Medium"
    return "Low"


def show_risk_message(risk_level):
    if risk_level == "High":
        st.error("Waste Risk: High")
    elif risk_level == "Medium":
        st.warning("Waste Risk: Medium")
    else:
        st.success("Waste Risk: Low")


def get_basic_actions(waste_rate):
    if waste_rate >= 25:
        return [
            "Reduce food preparation by around 10–20% next time.",
            "Use pre-orders or attendance confirmation before preparing food.",
            "Prepare food in smaller batches instead of all at once.",
            "If leftovers are still safe, contact a verified donation partner.",
        ]

    if waste_rate >= 10:
        return [
            "Monitor attendance more closely before preparing food.",
            "Track which meals are often left over.",
            "Keep donation or redistribution options available.",
        ]

    return [
        "Current preparation level looks reasonable.",
        "Continue tracking attendance and leftovers for future planning.",
    ]


def show_username_page():
    st.title("🍽️ Food Waste Rescue Radar")

    st.subheader("Enter your username to continue")

    st.write("""
    Create a new username or enter an existing username to continue using your saved dashboard.
    This is a simple MVP login for demo purposes.
    """)

    username = st.text_input(
        "Username",
        placeholder="Example: green_school_team",
    )

    if st.button("Continue"):
        username = username.strip().lower().replace(" ", "_")

        if username == "":
            st.error("Please enter a username.")
            return

        st.session_state.username = username
        st.session_state.page = "home"
        st.rerun()


def show_user_sidebar():
    st.sidebar.write(f"Current user: **{st.session_state.username}**")
    st.sidebar.button("Switch user", on_click=switch_user)

    st.sidebar.markdown("---")
    st.sidebar.button("Home", on_click=go_home)
    st.sidebar.button("Predict Waste Rate", on_click=go_predict)
    st.sidebar.button("Waste Reduction Guide", on_click=go_guide)
    st.sidebar.button("Dashboard", on_click=go_dashboard)


def show_home():
    st.title("🍽️ Food Waste Rescue Radar")

    st.subheader(
        "An AI-powered web app to predict and reduce food waste "
        "for schools and community events."
    )

    st.write("""
    Many school lunches and community events prepare more food than needed.
    This can lead to wasted meals, higher costs, and environmental impact.
    """)

    st.write("""
    Food Waste Rescue Radar helps users estimate food waste risk using simple
    information such as event type, location, attendance, food prepared,
    leftover portions, meal type, expected weather, and notes.
    """)

    st.markdown("---")

    st.header("What would you like to do?")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("### Predict Waste Rate")
        st.write("Analyze food waste after a meal or event.")
        st.button(
            "Start Prediction",
            on_click=go_predict,
            use_container_width=True,
        )

    with col2:
        st.markdown("### Waste Reduction Guide")
        st.write("Get suggestions before preparing food.")
        st.button(
            "Start Guide",
            on_click=go_guide,
            use_container_width=True,
        )

    with col3:
        st.markdown("### Dashboard")
        st.write("View your saved waste patterns and impact.")
        st.button(
            "View Dashboard",
            on_click=go_dashboard,
            use_container_width=True,
        )

    st.markdown("---")

    st.warning("""
    Responsible AI Note: The AI provides suggestions only. Food safety decisions
    must be reviewed by human staff. The AI does not decide whether leftovers
    are safe to donate or continue serving.
    """)


def show_predict():
    st.title("Predict Waste Rate")

    st.write("""
    Use this tool after a school lunch or community event to calculate the
    waste rate and understand possible reasons for leftover food.
    """)

    st.info("""
    Waste risk levels:
    - Low: less than 10% waste rate
    - Medium: 10% to less than 25% waste rate
    - High: 25% or more waste rate
    """)

    event_type = st.selectbox("Event type", EVENT_TYPES)
    location = st.selectbox("Location", LOCATIONS)

    col1, col2 = st.columns(2)

    with col1:
        expected_attendance = st.number_input(
            "Expected attendance",
            min_value=0,
            step=1,
        )

        food_prepared = st.number_input(
            "Food prepared (portions)",
            min_value=0,
            step=1,
        )

    with col2:
        actual_attendance = st.number_input(
            "Actual attendance",
            min_value=0,
            step=1,
        )

        leftover_portions = st.number_input(
            "Leftover portions",
            min_value=0,
            step=1,
        )

    meal_type = st.text_input("Meal type")
    notes = st.text_area("Notes")

    if st.button("Analyze Waste Rate"):
        if food_prepared == 0:
            st.error("Food prepared must be greater than 0.")
            return

        if leftover_portions > food_prepared:
            st.error("Leftover portions cannot be greater than food prepared.")
            return

        waste_rate = leftover_portions / food_prepared * 100
        attendance_gap = expected_attendance - actual_attendance
        risk_level = get_risk_level(waste_rate)

        show_risk_message(risk_level)

        result_col1, result_col2, result_col3, result_col4 = st.columns(4)

        with result_col1:
            st.metric("Waste Rate", f"{waste_rate:.1f}%")

        with result_col2:
            st.metric("Risk Level", risk_level)

        with result_col3:
            st.metric("Attendance Gap", attendance_gap)

        with result_col4:
            st.metric("Leftover Portions", leftover_portions)

        co2_impact = leftover_portions * 0.5
        cost_impact = leftover_portions * 2

        impact_col1, impact_col2 = st.columns(2)

        with impact_col1:
            st.metric("Estimated CO₂ Impact", f"{co2_impact:.1f} kg")

        with impact_col2:
            st.metric("Estimated Cost Impact", f"${cost_impact:.0f}")

        st.subheader("Basic Explanation")

        st.write(
            f"For this {event_type.lower()} at the {location.lower()}, "
            f"{leftover_portions} out of {food_prepared} portions were left over. "
            f"This means the waste rate is {waste_rate:.1f}%."
        )

        if attendance_gap > 0:
            st.write(
                "Actual attendance was lower than expected, which may be one reason "
                "for the leftover food."
            )
        elif attendance_gap < 0:
            st.write(
                "Actual attendance was higher than expected, so leftover food may be "
                "caused by other factors such as meal preference or portion size."
            )
        else:
            st.write(
                "Actual attendance matched the expected attendance, so the waste may "
                "be related to meal preference, portion size, or preparation planning."
            )

        st.subheader("Basic Waste Reduction Actions")

        for action in get_basic_actions(waste_rate):
            st.write(f"- {action}")

        save_history(
            {
                "Username": st.session_state.username,
                "Time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "Event Type": event_type,
                "Location": location,
                "Meal Type": meal_type if meal_type else "Unknown",
                "Expected Attendance": expected_attendance,
                "Actual Attendance": actual_attendance,
                "Food Prepared": food_prepared,
                "Leftover Portions": leftover_portions,
                "Waste Rate": round(waste_rate, 2),
                "Risk Level": risk_level,
                "Notes": notes,
            }
        )

        st.success("This record has been saved to your dashboard.")

        st.subheader("AI Analysis")

        prompt = f"""
        You are helping a school or community event organizer reduce food waste.

        Analyze this food waste case and give a clear, practical explanation.
        Do not decide whether food is safe to donate.
        Do not give food safety guarantees.

        Your answer should include:
        1. Main reasons for the waste.
        2. Practical actions to reduce waste next time.
        3. A short responsible AI and food safety reminder.

        Data:
        Username: {st.session_state.username}
        Event type: {event_type}
        Location: {location}
        Expected attendance: {expected_attendance}
        Actual attendance: {actual_attendance}
        Food prepared: {food_prepared} portions
        Leftover portions: {leftover_portions} portions
        Waste rate: {waste_rate:.1f}%
        Risk level: {risk_level}
        Meal type: {meal_type}
        Notes: {notes}
        """

        ai_answer = ask_ai(prompt)

        if ai_answer:
            st.write(ai_answer)
        else:
            st.info(
                "Groq API key is not set yet. The app still works with basic "
                "rule-based analysis."
            )

    st.markdown("---")
    st.button("Back to Home", on_click=go_home)


def show_guide():
    st.title("Waste Reduction Guide")

    st.write("""
    Use this tool before a school lunch or community event to get practical
    suggestions for reducing food waste.
    """)

    event_type = st.selectbox(
        "Event type",
        EVENT_TYPES,
        key="guide_event_type",
    )

    location = st.selectbox(
        "Location",
        LOCATIONS,
        key="guide_location",
    )

    expected_weather = st.selectbox(
        "Expected weather",
        WEATHER_OPTIONS,
    )

    expected_attendance = st.number_input(
        "Expected attendance",
        min_value=0,
        step=1,
        key="guide_expected_attendance",
    )

    food_should_be_prepared = st.number_input(
        "Food planned to be prepared (portions)",
        min_value=0,
        step=1,
    )

    notes = st.text_area("Notes", key="guide_notes")

    if st.button("Generate Guide"):
        if expected_attendance == 0:
            st.error("Expected attendance must be greater than 0.")
            return

        if food_should_be_prepared == 0:
            st.error("Food planned to be prepared must be greater than 0.")
            return

        difference = food_should_be_prepared - expected_attendance
        difference_rate = difference / expected_attendance * 100

        suggested_min = int(expected_attendance * 0.9)
        suggested_max = int(expected_attendance * 1.05)

        st.subheader("Basic Check")

        if difference_rate > 20:
            st.warning("The planned food amount may be too high.")
            st.write(
                f"A safer preparation range could be around "
                f"{suggested_min} to {suggested_max} portions."
            )
        elif difference_rate < -10:
            st.warning("The planned food amount may be too low.")
            st.write(
                f"Consider preparing around {suggested_min} to {suggested_max} "
                "portions or having a backup plan to avoid shortage."
            )
        else:
            st.success("The planned food amount looks reasonable.")
            st.write(
                f"The suggested preparation range is around "
                f"{suggested_min} to {suggested_max} portions."
            )

        if expected_weather in ["Rainy", "Stormy", "Very hot"]:
            st.write(
                f"Because the expected weather is {expected_weather.lower()}, "
                "attendance may be lower than usual."
            )
        elif expected_weather == "Sunny":
            st.write(
                "Sunny weather may support normal attendance, but it is still useful "
                "to confirm attendance before preparing food."
            )
        elif expected_weather == "Cloudy":
            st.write(
                "Cloudy weather may have a moderate impact, so attendance should "
                "still be checked before preparing food."
            )
        else:
            st.write(
                "Normal weather is less likely to reduce attendance, but food planning "
                "should still be based on expected attendance."
            )

        st.subheader("AI Waste Reduction Guide")

        prompt = f"""
        You are an AI assistant helping a school or community event organizer reduce food waste.

        Based on the information below, think carefully and create a practical waste reduction guide.
        Do not give food safety guarantees.
        Do not decide whether leftover food is safe to donate or serve again.

        Your answer should include:
        1. Whether the planned food amount is too high, too low, or reasonable.
        2. A suggested preparation range.
        3. Reasons based on attendance, weather, event type, location, and notes.
        4. Practical actions to reduce food waste.
        5. A short responsible AI and food safety reminder.

        Data:
        Username: {st.session_state.username}
        Event type: {event_type}
        Location: {location}
        Expected weather: {expected_weather}
        Expected attendance: {expected_attendance}
        Food planned to be prepared: {food_should_be_prepared} portions
        Suggested preparation range: {suggested_min} to {suggested_max} portions
        Difference between planned food and attendance: {difference} portions
        Difference rate: {difference_rate:.1f}%
        Notes: {notes}
        """

        ai_answer = ask_ai(prompt)

        if ai_answer:
            st.write(ai_answer)
        else:
            st.info(
                "Groq API key is not set yet. The app still works with basic "
                "rule-based recommendations."
            )

            st.subheader("Basic Waste Reduction Actions")

            st.markdown("""
            - Ask people to confirm attendance earlier.
            - Use pre-orders if possible.
            - Prepare food in smaller batches instead of all at once.
            - Track which meals are often left over.
            - If leftovers are still safe, contact a verified donation partner.
            """)

        st.warning(
            "Food safety must be checked by human staff before donation or reuse."
        )

    st.markdown("---")
    st.button("Back to Home", on_click=go_home)


def show_dashboard():
    st.title("Dashboard")

    st.write("""
    This dashboard shows saved food waste records for your username.
    It helps identify waste patterns and estimate environmental and cost impact.
    """)

    history = load_history()
    history = history[history["Username"] == st.session_state.username]

    if history.empty:
        st.info("No historical data yet for this username. Use Predict Waste Rate first.")
        st.button("Back to Home", on_click=go_home)
        return

    history["Waste Rate"] = pd.to_numeric(history["Waste Rate"], errors="coerce")
    history["Leftover Portions"] = pd.to_numeric(
        history["Leftover Portions"],
        errors="coerce",
    )

    total_records = len(history)
    average_waste = history["Waste Rate"].mean()
    total_leftovers = history["Leftover Portions"].sum()
    total_co2 = total_leftovers * 0.5
    total_cost = total_leftovers * 2

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total Records", total_records)

    with col2:
        st.metric("Average Waste Rate", f"{average_waste:.1f}%")

    with col3:
        st.metric("Total Leftovers", f"{total_leftovers:.0f} portions")

    with col4:
        st.metric("Estimated CO₂ Impact", f"{total_co2:.1f} kg")

    st.metric("Estimated Cost Impact", f"${total_cost:.0f}")

    st.subheader("Recent Records")
    st.dataframe(history.tail(10), use_container_width=True)

    st.subheader("Waste Rate Trend")
    trend_data = history[["Waste Rate"]].reset_index(drop=True)
    st.line_chart(trend_data)

    st.subheader("Average Waste Rate by Meal Type")

    meal_chart = (
        history.groupby("Meal Type")["Waste Rate"]
        .mean()
        .sort_values(ascending=False)
    )

    st.bar_chart(meal_chart)

    st.subheader("Average Waste Rate by Event Type")

    event_chart = (
        history.groupby("Event Type")["Waste Rate"]
        .mean()
        .sort_values(ascending=False)
    )

    st.bar_chart(event_chart)

    st.subheader("Pattern Detection")

    worst_meal = meal_chart.idxmax()
    worst_event = event_chart.idxmax()

    st.write(f"Most wasteful meal type: {worst_meal}")
    st.write(f"Highest waste event type: {worst_event}")

    if average_waste >= 25:
        st.warning(
            "Overall waste level is high. The organization should improve attendance "
            "tracking and reduce over-preparation."
        )
    elif average_waste >= 10:
        st.warning(
            "Overall waste level is medium. Monitoring and small preparation adjustments "
            "may help reduce waste."
        )
    else:
        st.success(
            "Overall waste level is low. Current preparation planning looks reasonable."
        )

    st.markdown("---")
    st.button("Back to Home", on_click=go_home)


if st.session_state.username is None:
    show_username_page()
else:
    show_user_sidebar()

    if st.session_state.page == "home":
        show_home()
    elif st.session_state.page == "predict":
        show_predict()
    elif st.session_state.page == "guide":
        show_guide()
    elif st.session_state.page == "dashboard":
        show_dashboard()
