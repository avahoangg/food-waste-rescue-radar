# Food Waste Rescue Radar — Premium MVP

**High School Track — AI for Everyday Good**  
Direction A: **Food Waste Rescue Radar**

Food Waste Rescue Radar is an AI-powered Streamlit MVP that helps schools, cafeterias, student clubs, community events, and local organizations predict food waste risk, understand patterns, and take practical action before food is wasted.

## Why this project matters

Schools and community groups often prepare food without knowing exactly how many people will attend, which meals will be popular, or how much rescue capacity is available. This can create avoidable food waste, cost loss, and environmental impact.

This app helps users make better decisions by combining:

- event planning data
- expected attendance
- weather and day-of-week risk
- menu popularity
- batch cooking availability
- donation/storage capacity
- historical food waste records
- optional Groq AI coaching

## Premium features

### 1. Mission Control
A polished landing dashboard that explains the mission, shows organization-level impact, and guides users through the workflow.

### 2. Waste Risk Scanner
A pre-event forecasting tool that generates:

- Waste Risk Score from 0 to 100
- Low / Medium / High risk level
- predicted waste rate
- predicted attendance and consumption
- smart preparation range
- expected excess portions
- potential rescued meals
- cost and CO2e exposure
- top risk drivers
- before/during/after action plan

### 3. What-if Simulator
Compares different preparation strategies:

- current plan
- reduce 10%
- reduce 20%
- smart range low
- smart range high

This helps users see how changing food preparation affects predicted waste, cost exposure, rescue capacity, and shortage risk.

### 4. Event Result Logger
After an event, users can log:

- actual attendance
- food prepared
- leftover portions
- intervention used
- donation capacity
- storage capacity

These actual records improve the dashboard and unlock historical learning.

### 5. Hybrid AI / ML engine
The app starts with a transparent rule-based prediction model. Once the user has at least 10 actual records, it unlocks a lightweight historical machine learning model using `RandomForestRegressor` from scikit-learn.

The model uses features such as:

- event type
- location
- day of week
- meal time
- meal type
- menu popularity
- weather
- attendance confidence
- expected attendance
- prepared portions
- donation partner availability
- batch cooking availability
- storage capacity

### 6. Impact Intelligence Dashboard
The dashboard identifies patterns such as:

- waste trend over time
- highest-waste meals
- highest-risk days
- meal-time/day heatmap
- intervention effectiveness
- rescued meals
- CO2e exposure
- cost exposure

### 7. Data & Export
Users can:

- load demo data for judging/testing
- download their CSV
- download a plain-language impact summary
- clear their own records

## Technology stack

- Python
- Streamlit
- Pandas
- NumPy
- Plotly
- Scikit-learn
- Groq API, optional

## How to run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## How to deploy on Streamlit Community Cloud

1. Upload these files to a GitHub repository.
2. Go to Streamlit Community Cloud.
3. Connect the GitHub repository.
4. Set the main file path to:

```text
app.py
```

5. Deploy.

## Optional Groq setup

The app works without Groq. If you want the optional AI Coach to work, add a Streamlit secret:

```toml
GROQ_API_KEY = "your_api_key_here"
```

Do not upload `.streamlit/secrets.toml` to GitHub. This project's `.gitignore` already excludes that file.

## Responsible AI and food safety

The app gives planning suggestions only. It does **not** decide whether food is safe to donate, store, serve again, compost, or discard.

Human staff must inspect food, follow local food safety rules, and make all final food safety decisions.

## Suggested judging demo flow

1. Enter any username, for example:

```text
green_school_team
```

2. Click **Load demo data** in the sidebar.
3. Open **Impact Intelligence** to show charts and pattern detection.
4. Open **Waste Risk Scanner** and run a forecast.
5. Show the What-if Simulator and Rescue Action Plan.
6. Open **Event Result Logger** to save an actual event result.
7. Return to **Impact Intelligence** to show how the dashboard updates.
