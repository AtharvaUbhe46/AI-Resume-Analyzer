# AI Resume Analyzer

An ATS resume analyzer built with Streamlit and Gemini 1.5 Flash.

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Add your Gemini API key to .env
# Get a free key at https://aistudio.google.com/apikey

# 3. Run the app
streamlit run app.py
```

## Features
- PDF text extraction (client-side parsing)
- ATS keyword scoring with stopword filtering
- Matched / missing keyword chips
- Gemini AI recruiter feedback (6 sections)
- Donut + horizontal bar charts
- PDF report export

## Fixes in this version
- API key moved to .env (no hardcoding)
- Stopword filtering fixes inflated ATS scores
- PDF validation catches scanned/empty PDFs
- Session-state caching fixed (hash checked outside button scope)
- Bar chart now shows ranked keywords instead of flat Count=1
- AI prompt produces structured, section-headed output
- PDF report has proper styling and table
- `st.stop()` replaced with safer validation pattern
