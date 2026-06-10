# AI Resume Analyzer 📄

A Streamlit app that analyzes your resume against a job description — gives you an ATS score, keyword gap analysis, and deep AI recruiter feedback powered by Groq (Llama 3.3 70B).

## What it does

Upload your resume PDF and paste a job description. The app tells you:
- How well your resume matches the JD (ATS score)
- Which keywords you're missing
- Specific rewrite suggestions
- Honest recruiter-style feedback on your strengths and gaps
- A downloadable PDF report of the full analysis

## Setup

1. Clone the repo and install dependencies:
```bash
pip install -r requirements.txt
```

2. Create a `.env` file and add your Groq API key:

Get a free key at https://console.groq.com

3. Run the app:
```bash
streamlit run app.py
```

## Tech Stack

- **Streamlit** — UI
- **Groq (Llama 3.3 70B)** — AI analysis
- **PyPDF2** — PDF text extraction
- **Plotly** — Charts
- **ReportLab** — PDF report generation