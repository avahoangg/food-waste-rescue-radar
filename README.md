# Food Waste Rescue Radar

A premium Streamlit MVP for the **High School Track — AI for Everyday Good** challenge.

## Mission

Build an AI-powered MVP that helps a school, community group, or local organization understand their environmental impact and take practical action.

This project follows **Direction A: Food Waste Rescue Radar**.

It helps users:

- predict where food waste is likely to happen,
- identify patterns that lead to waste,
- recommend realistic portion ranges,
- create practical rescue action plans,
- log real event results,
- track environmental and cost impact over time.

## Why this version is stronger

This is not just a calculator. It combines:

1. **Rule-based risk engine**  
   Uses attendance, weather, menu popularity, day of week, confidence, intervention planning, and operational flexibility.

2. **Historical learning mode**  
   When enough records exist, the app trains a lightweight `RandomForestRegressor` using past event data.

3. **What-if simulator**  
   Students can test different preparation amounts and immediately see predicted waste, leftovers, and shortage risk.

4. **Impact Intelligence dashboard**  
   Detects high-waste meals, high-risk days, intervention patterns, cost impact, CO2e impact, and potential rescued meals.

5. **Responsible AI design**  
   The app gives planning suggestions only. It does not decide food safety. Human staff must inspect food and follow local food safety rules.

## Pages

### Mission Control
Premium home screen, quick actions, demo data, and overall workspace metrics.

### Waste Risk Scanner
Pre-event tool that estimates waste risk and recommends a smarter portion range.

### Event Result Logger
Post-event tool for logging actual attendance, prepared portions, and leftovers.

### Impact Intelligence
Dashboard for trends, patterns, intervention signals, and environmental/cost impact.

### Data & Export
CSV export and markdown impact report for presentations or judging.

## Setup

Install requirements:

```bash
pip install -r requirements.txt
```

Run locally:

```bash
streamlit run app.py
```

## Optional Groq AI Coach

The app runs without an API key. To enable the optional AI Coach, add this in Streamlit Cloud secrets or local `.streamlit/secrets.toml`:

```toml
GROQ_API_KEY = "your_key_here"
```

Do not commit your real `secrets.toml`.

## Deployment

1. Push these files to GitHub.
2. Deploy on Streamlit Community Cloud.
3. Select `app.py` as the main file.
4. Add `GROQ_API_KEY` in Streamlit Secrets only if using the optional AI Coach.

## Responsible AI note

This app supports planning and environmental decision-making only. It does not determine whether food is safe to donate, reuse, serve, compost, or discard. Human staff must follow local food safety policies.
