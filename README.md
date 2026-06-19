# Food Waste Rescue Radar

A clean premium environmental MVP for the **High School Track — AI for Everyday Good** challenge.

## Direction

**Direction A: Food Waste Rescue Radar**

The app helps a school, community group, or local organization understand food waste patterns and take practical action before waste happens.

## What the app does

1. **Scan a food plan before an event**
   - Predicts food waste risk.
   - Gives a risk score and risk level.
   - Recommends a realistic preparation range.
   - Explains the main risk drivers.

2. **Create a student-friendly action plan**
   - Attendance confirmation.
   - Smaller first batch.
   - Menu preference survey.
   - Donation or redistribution planning.
   - Compost planning.

3. **Log real event results**
   - Actual attendance.
   - Food prepared.
   - Leftover portions.
   - Waste rate.
   - Estimated cost and CO2e impact.

4. **Detect patterns**
   - Highest-waste foods.
   - Waste by day.
   - Risk mix.
   - Intervention comparison.
   - Recent results.

5. **Export evidence**
   - CSV data.
   - Markdown impact report.

## Is AI integrated?

Yes.

### Built-in AI risk engine
This is always active and does not need an API key. It uses:
- attendance confidence,
- expected attendance,
- planned portions,
- menu popularity,
- weather,
- day of week,
- event type,
- batch cooking,
- intervention type,
- historical patterns.

### Historical ML prediction
When the dataset has enough records, the app trains a lightweight `RandomForestRegressor` using previous event results.

### Optional Groq AI Advisor
If you add a Groq API key, the app also generates written recommendations.

The app still works without Groq.

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Optional Streamlit Secrets

Create `.streamlit/secrets.toml` locally or add this in Streamlit Cloud Secrets:

```toml
GROQ_API_KEY = "your_key_here"
```

Never commit the real `secrets.toml`.

## Responsible AI

The app supports planning only. It does not decide whether food is safe to serve, donate, reuse, compost, or discard. Human staff must inspect food and follow local food safety rules.
