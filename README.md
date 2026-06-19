# Food Rescue Radar

Food Rescue Radar is an AI-powered Streamlit app that helps schools and community groups predict food waste before it happens.

## What it does

- Creates separate projects for cafeterias, clubs, events, or community groups.
- Predicts where food waste is likely to happen.
- Identifies patterns that lead to waste.
- Suggests realistic actions to reduce, redistribute, or manage leftovers.
- Logs real event results.
- Shows dashboard patterns by food, day, risk level, and intervention.
- Exports CSV data and a simple impact report.

## AI features

The app includes:

1. **AI forecast engine**  
   Scores risk using attendance, prepared portions, weather, menu popularity, confidence, day, intervention, and project history.

2. **Historical learning model**  
   When a project has enough records, the app trains a Random Forest model to improve predictions from past results.

3. **Groq AI Advisor**  
   Uses Groq to create a written action plan for each forecast.

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Add Groq API key locally

Create a file named `.streamlit/secrets.toml` and add:

```toml
GROQ_API_KEY = "your_groq_api_key_here"
```

Do not commit the real `secrets.toml` file.

## Responsible AI

The app supports planning only. It does not decide whether food is safe to serve, donate, reuse, compost, or discard. Trained human staff must inspect food and follow local food safety rules.
