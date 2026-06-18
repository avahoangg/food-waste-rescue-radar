import streamlit as st
from google import genai


st.set_page_config(
    page_title="Food Waste Rescue Radar",
    page_icon="🍽️",
    layout="centered"
)


EVENT_TYPES = ["School lunch", "Community event"]
LOCATIONS = ["Cafeteria", "Party / Event venue"]
WEATHER_OPTIONS = [
    "Normal",
    "Sunny",
    "Rainy",
    "Cloudy",
    "Stormy",
    "Very hot"
]


if "page" not in st.session_state:
    st.session_state.page = "home"


def get_gemini_client():
    api_key = st.secrets.get("GEMINI_API_KEY", None)

    if not api_key:
        return None

    return genai.Client(api_key=api_key)


def ask_gemini(prompt):
    client = get_gemini_client()

    if client is None:
        return None

    try:
        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=prompt
        )
        return response.text
    except Exception as error:
        return f"Gemini analysis is unavailable right now. Error: {error}"


def go_home():
    st.session_state.page = "home"


def go_predict():
    st.session_state.page = "predict"


def go_guide():
    st.session_state.page = "guide"


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

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Predict Waste Rate")
        st.write("Analyze food waste after a meal or event.")
        st.button(
            "Start Prediction",
            on_click=go_predict,
            use_container_width=True
        )

    with col2:
        st.markdown("### Waste Reduction Guide")
        st.write("Get suggestions before preparing food.")
        st.button(
            "Start Guide",
            on_click=go_guide,
            use_container_width=True
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

    event_type = st.selectbox(
        "Event type",
        EVENT_TYPES
    )

    location = st.selectbox(
        "Location",
        LOCATIONS
    )

    expected_attendance = st.number_input(
        "Expected attendance",
        min_value=0,
        step=1
    )

    actual_attendance = st.number_input(
        "Actual attendance",
        min_value=0,
        step=1
    )

    food_prepared = st.number_input(
        "Food prepared (portions)",
        min_value=0,
        step=1
    )

    leftover_portions = st.number_input(
        "Leftover portions",
        min_value=0,
        step=1
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

        st.metric("Waste Rate", f"{waste_rate:.1f}%")
        st.metric("Attendance Gap", attendance_gap)

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

        if meal_type:
            st.write(f"Meal type: {meal_type}")

        if notes:
            st.write(f"Additional notes: {notes}")

        st.subheader("Gemini AI Analysis")

        prompt = f"""
        You are helping a school or community event organizer reduce food waste.

        Analyze this food waste case and give a clear, practical explanation.
        Do not decide whether food is safe to donate.
        Do not give food safety guarantees.

        Include:
        1. Main reasons for the waste.
        2. Practical actions to reduce waste next time.
        3. A short responsible AI and food safety reminder.

        Data:
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

        gemini_answer = ask_gemini(prompt)

        if gemini_answer:
            st.write(gemini_answer)
        else:
            st.info(
                "Gemini API key is not set yet. The app still works with basic "
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
        key="guide_event_type"
    )

    location = st.selectbox(
        "Location",
        LOCATIONS,
        key="guide_location"
    )

    expected_weather = st.selectbox(
        "Expected weather",
        WEATHER_OPTIONS
    )

    expected_attendance = st.number_input(
        "Expected attendance",
        min_value=0,
        step=1,
        key="guide_expected_attendance"
    )

    food_should_be_prepared = st.number_input(
        "Food planned to be prepared (portions)",
        min_value=0,
        step=1
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

        gemini_answer = ask_gemini(prompt)

        if gemini_answer:
            st.write(gemini_answer)
        else:
            st.info(
                "Gemini API key is not set yet. The app still works with basic "
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


if st.session_state.page == "home":
    show_home()
elif st.session_state.page == "predict":
    show_predict()
elif st.session_state.page == "guide":
    show_guide()
