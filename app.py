import hashlib
import re
from io import BytesIO

from groq import Groq
import pandas as pd
import plotly.express as px
import streamlit as st
from PyPDF2 import PdfReader
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (
    HRFlowable, Paragraph, SimpleDocTemplate,
    Spacer, Table, TableStyle,
)

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────

st.set_page_config(
    page_title="AI Resume Analyzer",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────

st.markdown("""
<style>
.stApp { background-color: #0E1117; }
.app-header {
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
    border: 1px solid #4F46E533; border-radius: 16px;
    padding: 2rem 2.5rem; margin-bottom: 1.75rem;
    display: flex; align-items: center; gap: 1.25rem;
}
.app-header-icon { font-size: 2.5rem; line-height: 1; }
.app-header-title { font-size: 1.75rem; font-weight: 700; color: #ffffff; margin: 0; }
.app-header-sub { font-size: 0.875rem; color: #a0aec0; margin-top: 4px; }
.pill {
    display: inline-block; background: #4F46E522; color: #818cf8;
    border: 1px solid #4F46E555; border-radius: 20px;
    padding: 2px 10px; font-size: 0.75rem; margin-left: 8px; vertical-align: middle;
}
.section-title {
    font-size: 1rem; font-weight: 600; color: #e2e8f0;
    margin-bottom: 0.5rem; display: flex; align-items: center; gap: 6px;
}
.metric-row {
    display: grid; grid-template-columns: repeat(4, 1fr);
    gap: 12px; margin-bottom: 1.5rem;
}
.metric-card {
    background: #1E1E2E; border: 1px solid #2d2d3d;
    border-radius: 12px; padding: 1.1rem 1rem; text-align: center;
}
.metric-label {
    font-size: 0.75rem; color: #a0aec0; text-transform: uppercase;
    letter-spacing: 0.06em; margin-bottom: 6px;
}
.metric-value { font-size: 2rem; font-weight: 700; line-height: 1; }
.metric-value.purple { color: #818cf8; }
.metric-value.teal   { color: #34d399; }
.metric-value.coral  { color: #f87171; }
.metric-value.amber  { color: #fbbf24; }
.grade-badge {
    display: inline-flex; align-items: center; justify-content: center;
    width: 56px; height: 56px; border-radius: 50%;
    font-size: 1.4rem; font-weight: 700; margin: 0 auto;
}
.grade-Aplus { background: #064e3b; color: #34d399; }
.grade-A     { background: #064e3b; color: #6ee7b7; }
.grade-B     { background: #1e1b4b; color: #818cf8; }
.grade-C     { background: #451a03; color: #fbbf24; }
.grade-D     { background: #450a0a; color: #f87171; }
.status-banner {
    border-radius: 10px; padding: 0.75rem 1rem; font-size: 0.875rem;
    margin-bottom: 1.25rem; display: flex; align-items: center; gap: 8px;
}
.status-success { background: #064e3b44; border: 1px solid #34d39966; color: #6ee7b7; }
.status-warning { background: #451a0344; border: 1px solid #fbbf2466; color: #fcd34d; }
.status-danger  { background: #450a0a44; border: 1px solid #f8717166; color: #fca5a5; }
.chip-container { display: flex; flex-wrap: wrap; gap: 6px; }
.chip-matched {
    background: #064e3b; color: #6ee7b7; border: 1px solid #34d39944;
    border-radius: 20px; padding: 3px 10px; font-size: 0.78rem;
}
.chip-missing {
    background: #450a0a; color: #fca5a5; border: 1px solid #f8717144;
    border-radius: 20px; padding: 3px 10px; font-size: 0.78rem;
}
.ai-box {
    background: #1E1E2E; border: 1px solid #2d2d3d; border-radius: 12px;
    padding: 1.25rem 1.5rem; font-size: 0.875rem; line-height: 1.85; color: #e2e8f0;
}
.ai-section-head {
    color: #818cf8; font-weight: 700; font-size: 0.8rem;
    text-transform: uppercase; letter-spacing: 0.08em;
    margin-top: 18px; margin-bottom: 6px;
    border-left: 3px solid #4F46E5; padding-left: 8px;
}
.custom-divider { border: none; border-top: 1px solid #2d2d3d; margin: 1.5rem 0; }
.block-container { padding-top: 1.5rem !important; }
.error-box {
    background: #450a0a44; border: 1px solid #f8717166; border-radius: 10px;
    padding: 1rem 1.25rem; color: #fca5a5; font-size: 0.875rem; margin-bottom: 1rem;
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────────

for key, default in [
    ("ai_response", None),
    ("last_hash", None),
    ("jd", ""),
]:
    if key not in st.session_state:
        st.session_state[key] = default

# ─────────────────────────────────────────────
# Groq SETUP
# ─────────────────────────────────────────────

import os
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")

with st.sidebar:
    st.caption("🔒 API key secured via environment secrets.")
    st.divider()
    st.markdown("**Model:** `Groq-llama-3.3-70b (Groq)`")
    st.markdown("**About**")
    st.caption("ATS uses stopword-filtered keyword overlap. AI feedback is deep resume-specific analysis via Groq.")

groq_client = None
if GROQ_API_KEY:
    try:
        groq_client = Groq(api_key=GROQ_API_KEY)
    except Exception:
        groq_client = None

# ─────────────────────────────────────────────
# STOPWORDS
# ─────────────────────────────────────────────

STOPWORDS = {
    "a","an","the","and","or","but","in","on","at","to","for","of","with",
    "is","are","was","were","be","been","being","have","has","had","do","does",
    "did","will","would","could","should","may","might","shall","can","need",
    "this","that","these","those","it","its","we","our","you","your","they",
    "their","from","by","as","not","all","more","also","about","up","out",
    "if","so","than","then","any","each","both","other","into","through",
    "during","before","after","above","below","between","such","while",
    "how","when","where","why","what","which","who","whom","i","me","my",
    "work","working","experience","role","team","company","using","use","used",
    "good","strong","ability","skills","skill","years","year","etc","new",
}

# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────

def extract_text(pdf_file) -> str:
    text = ""
    try:
        reader = PdfReader(pdf_file)
        for page in reader.pages:
            t = page.extract_text()
            if t:
                text += t + "\n"
    except Exception:
        return ""
    return text.strip()


def clean_keywords(text: str) -> set:
    tokens = re.findall(r"\b[a-zA-Z][a-zA-Z0-9+#.]*\b", text.lower())
    return {t for t in tokens if len(t) > 2 and t not in STOPWORDS}


def calculate_ats(resume_text: str, jd_text: str):
    rk = clean_keywords(resume_text)
    jk = clean_keywords(jd_text)
    if not jk:
        return 0, set(), set()
    matched = rk & jk
    missing = jk - rk
    score = min(100, int(len(matched) / len(jk) * 100))
    return score, matched, missing


def get_grade(score: int) -> str:
    if score >= 85: return "A+"
    if score >= 75: return "A"
    if score >= 65: return "B"
    if score >= 50: return "C"
    return "D"


def get_ai_analysis(resume_text: str, jd_text: str, score: int, matched: set, missing: set) -> str:
    if not groq_client:
        return "⚠️ Groq API key not set. Please add your key in the sidebar."

    current_hash = hashlib.md5((resume_text + jd_text).encode()).hexdigest()
    if st.session_state.last_hash == current_hash and st.session_state.ai_response:
        return st.session_state.ai_response

    matched_str = ", ".join(sorted(matched)[:30])
    missing_str = ", ".join(sorted(missing)[:30])

    prompt = f"""You are a senior technical recruiter with 15+ years of experience hiring for top tech companies.
You are doing a REAL, DETAILED review of this specific candidate's resume against a specific job description.

Your analysis must be:
- SPECIFIC to THIS resume (mention actual job titles, companies, technologies, years from the resume)
- SPECIFIC to THIS job description (mention actual requirements from the JD)
- HONEST and CRITICAL — do not be generic or vague
- ACTIONABLE — give concrete, specific advice the candidate can act on today

=== RESUME TEXT ===
{resume_text[:4500]}

=== JOB DESCRIPTION ===
{jd_text[:2500]}

=== ATS DATA ===
Score: {score}%
Keywords found in resume: {matched_str}
Keywords missing from resume: {missing_str}

Write your analysis with EXACTLY these section headers (all caps, on their own line):

RESUME SUMMARY
Write 3-4 sentences. Mention the candidate's actual current/most recent role, years of experience, and key technical stack from the resume. State clearly whether this profile fits the JD at a high level.

STRENGTHS
5 bullet points (start each with •). Each point must reference something SPECIFIC from the resume — actual technologies, actual job titles, actual achievements. Example: "3 years at [Company X] as [Role] directly maps to the JD's requirement for [Requirement Y]". Do NOT write generic statements.

CRITICAL GAPS
5 bullet points (start each with •). Identify SPECIFIC missing skills, tools, or experience the JD requires but the resume lacks. Reference the actual JD requirement and the actual gap. Example: "JD requires Kubernetes experience — no mention of containerization or K8s anywhere in resume."

ATS OPTIMIZATION (SPECIFIC REWRITES)
4 bullet points. Give SPECIFIC rewrite suggestions — quote the original weak phrase from the resume and suggest the improved version. Example: 'Change "worked on databases" → "Managed PostgreSQL databases handling 2M+ daily transactions"'.

INTERVIEW RED FLAGS
3 bullet points. What hard questions will this candidate face in an interview based on the gaps? Be specific.

HIRING RECOMMENDATION
One clear verdict: Strong Fit / Moderate Fit / Weak Fit. Follow with 2-3 sentences explaining exactly why, referencing specifics from both the resume and the JD.

Important: Be direct, professional, and brutally honest. A generic analysis helps no one."""

    try:
        response = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=1500,)
        result = response.choices[0].message.content
        st.session_state.ai_response = result
        st.session_state.last_hash = current_hash
        return result
    except Exception as e:
        err = str(e)
        # Surface the real error so user can fix it
        return f"⚠️ Groq API error: {err}\n\nCheck your API key in the sidebar and try again."


def render_ai_output(text: str):
    # Check if it's an error message
    if text.startswith("⚠️"):
        st.markdown(f'<div class="error-box">{text}</div>', unsafe_allow_html=True)
        return

    SECTIONS = [
        "RESUME SUMMARY", "STRENGTHS", "CRITICAL GAPS",
        "ATS OPTIMIZATION (SPECIFIC REWRITES)", "INTERVIEW RED FLAGS",
        "HIRING RECOMMENDATION",
    ]
    html = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    for s in SECTIONS:
        html = html.replace(s, f'<div class="ai-section-head">{s}</div>')
    html = html.replace("•", '<span style="color:#818cf8">•</span>')
    html = re.sub(r"#+ ", "", html)
    html = html.replace("\n", "<br>")
    st.markdown(f'<div class="ai-box">{html}</div>', unsafe_allow_html=True)


def generate_pdf_report(score, grade, matched, missing, ai_text, resume_name="resume") -> BytesIO:
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4,
        leftMargin=2*cm, rightMargin=2*cm, topMargin=2*cm, bottomMargin=2*cm,
        title=f"{resume_name} Analysis Report",
        author="AI Resume Analyzer")
    styles = getSampleStyleSheet()
    title_s   = ParagraphStyle("t", parent=styles["Title"], fontSize=22,
                    textColor=colors.HexColor("#4F46E5"), spaceAfter=6)
    sub_s     = ParagraphStyle("s", parent=styles["Normal"], fontSize=10,
                    textColor=colors.HexColor("#718096"), spaceAfter=20)
    heading_s = ParagraphStyle("h", parent=styles["Heading2"], fontSize=13,
                    textColor=colors.HexColor("#2d3748"), spaceBefore=16, spaceAfter=6)
    body_s    = ParagraphStyle("b", parent=styles["BodyText"], fontSize=10,
                    leading=16, textColor=colors.HexColor("#4a5568"))

    content = [
        Paragraph("AI Resume Analyzer Report", title_s),
        Paragraph("Detailed analysis powered by Groq AI", sub_s),
        HRFlowable(width="100%", thickness=1, color=colors.HexColor("#e2e8f0")),
        Spacer(1, 12),
    ]
    score_data = [
        ["ATS Score", "Grade", "Matched Keywords", "Missing Keywords"],
        [f"{score}%", grade, str(len(matched)), str(len(missing))],
    ]
    tbl = Table(score_data, colWidths=[4*cm]*4)
    tbl.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#4F46E5")),
        ("TEXTCOLOR",  (0,0), (-1,0), colors.white),
        ("FONTNAME",   (0,0), (-1,0), "Helvetica-Bold"),
        ("FONTSIZE",   (0,0), (-1,0), 10),
        ("BACKGROUND", (0,1), (-1,1), colors.HexColor("#f7fafc")),
        ("FONTNAME",   (0,1), (-1,1), "Helvetica-Bold"),
        ("FONTSIZE",   (0,1), (-1,1), 16),
        ("ALIGN",      (0,0), (-1,-1), "CENTER"),
        ("VALIGN",     (0,0), (-1,-1), "MIDDLE"),
        ("BOX",        (0,0), (-1,-1), 1, colors.HexColor("#e2e8f0")),
        ("INNERGRID",  (0,0), (-1,-1), 0.5, colors.HexColor("#e2e8f0")),
        ("TOPPADDING", (0,0), (-1,-1), 8),
        ("BOTTOMPADDING",(0,0),(-1,-1), 8),
    ]))
    content += [tbl, Spacer(1, 20)]
    content.append(Paragraph("✅ Matched Keywords", heading_s))
    content.append(Paragraph(", ".join(sorted(matched)[:40]) or "None", body_s))
    content.append(Spacer(1, 10))
    content.append(Paragraph("❌ Missing Keywords", heading_s))
    content.append(Paragraph(", ".join(sorted(missing)[:40]) or "None", body_s))
    content.append(Spacer(1, 10))
    content.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#e2e8f0")))
    content.append(Spacer(1, 8))
    content.append(Paragraph("🤖 AI Recruiter Analysis", heading_s))
    clean = re.sub(r"[^\x20-\x7E\n•]", "", ai_text)
    for line in clean.split("\n"):
        line = line.strip()
        if not line:
            content.append(Spacer(1, 4))
        elif line.upper() == line and len(line) > 4:
            clean_line = line.lstrip("#").strip()
            content.append(Paragraph(clean_line, heading_s))
        else:
            content.append(Paragraph("&bull; " + line.lstrip("•").strip() if line.startswith("•") else line, body_s))
    doc.build(content)
    buffer.seek(0)
    return buffer

# ─────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────

st.markdown("""
<div class="app-header">
    <div class="app-header-icon">📄</div>
    <div>
        <div class="app-header-title">
            AI Resume Analyzer
            <span class="pill">Llama 3.3 70B</span>
        </div>
        <div class="app-header-sub">
            ATS scoring &nbsp;·&nbsp; Keyword gap analysis &nbsp;·&nbsp; Deep AI recruiter feedback &nbsp;·&nbsp; PDF report
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# INPUT
# ─────────────────────────────────────────────

col1, col2 = st.columns(2, gap="medium")

with col1:
    st.markdown('<div class="section-title">📎 Resume (PDF)</div>', unsafe_allow_html=True)
    uploaded_resume = st.file_uploader("Upload resume", type=["pdf"], label_visibility="collapsed")
    if uploaded_resume:
        st.success(f"✅ Loaded: **{uploaded_resume.name}**", icon="📄")

with col2:
    st.markdown('<div class="section-title">💼 Job Description</div>', unsafe_allow_html=True)
    SAMPLE_JD = """Software Engineer — Full Stack

We are looking for a Software Engineer to join our growing team.

Requirements:
- 3+ years Python and JavaScript experience
- Proficiency in React, Node.js, REST APIs
- SQL databases (PostgreSQL preferred)
- Git, GitHub, CI/CD pipelines
- Cloud platforms (AWS, GCP, or Azure)
- Docker and containerization
- Data structures and algorithms
- TypeScript, GraphQL (nice to have)
- Agile/Scrum environment"""

    if st.button("Load sample JD", use_container_width=False):
        st.session_state["jd"] = SAMPLE_JD

    job_description = st.text_area(
        "Paste job description",
        value=st.session_state.get("jd", ""),
        height=200,
        placeholder="Paste the full job description here…",
        label_visibility="collapsed",
    )

st.markdown('<hr class="custom-divider">', unsafe_allow_html=True)

analyze_clicked = st.button(
    "🚀  Analyze Resume",
    use_container_width=True,
    type="primary",
    disabled=not (uploaded_resume and job_description.strip()),
)

# ─────────────────────────────────────────────
# ANALYSIS
# ─────────────────────────────────────────────

if analyze_clicked:
    with st.spinner("Parsing PDF…"):
        resume_text = extract_text(uploaded_resume)

    if not resume_text:
        st.error("⚠️ Could not extract text from this PDF — it may be scanned. Please use a text-based PDF.", icon="❌")
        st.stop()

    if len(resume_text) < 100:
        st.warning("⚠️ Very little text extracted — results may be inaccurate.", icon="⚠️")

    with st.spinner("Calculating ATS score…"):
        score, matched, missing = calculate_ats(resume_text, job_description)
        grade = get_grade(score)

    with st.spinner("🤖 Getting deep AI analysis from Groq… (~15 seconds)"):
        ai_text = get_ai_analysis(resume_text, job_description, score, matched, missing)

    # ── Results ──
    st.markdown('<hr class="custom-divider">', unsafe_allow_html=True)
    st.markdown("### 📊 Analysis Results")

    grade_class = "grade-" + grade.replace("+", "plus")
    st.markdown(f"""
    <div class="metric-row">
        <div class="metric-card">
            <div class="metric-label">ATS Score</div>
            <div class="metric-value purple">{score}%</div>
        </div>
        <div class="metric-card">
            <div class="metric-label">Grade</div>
            <div style="display:flex;justify-content:center;margin-top:4px;">
                <div class="grade-badge {grade_class}">{grade}</div>
            </div>
        </div>
        <div class="metric-card">
            <div class="metric-label">Keywords Matched</div>
            <div class="metric-value teal">{len(matched)}</div>
        </div>
        <div class="metric-card">
            <div class="metric-label">Keywords Missing</div>
            <div class="metric-value coral">{len(missing)}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    if score >= 75:
        st.markdown('<div class="status-banner status-success">✅ Strong ATS match — your resume aligns well with this role.</div>', unsafe_allow_html=True)
    elif score >= 50:
        st.markdown('<div class="status-banner status-warning">⚠️ Moderate match — address missing keywords before applying.</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="status-banner status-danger">❌ Weak ATS match — significant keyword gaps detected.</div>', unsafe_allow_html=True)

    chart_col1, chart_col2 = st.columns(2, gap="medium")

    with chart_col1:
        fig = px.pie(
            pd.DataFrame({"Category": ["Matched", "Missing"], "Value": [score, 100-score]}),
            values="Value", names="Category", hole=0.62,
            title="ATS Keyword Match",
            color="Category",
            color_discrete_map={"Matched": "#4F46E5", "Missing": "#374151"},
        )
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font_color="#e2e8f0", title_font_size=14,
            margin=dict(t=50, b=20, l=20, r=20),
        )
        st.plotly_chart(fig, use_container_width=True)

    with chart_col2:
        top_kw = sorted(matched, key=len, reverse=True)[:12]
        if top_kw:
            bar_fig = px.bar(
                pd.DataFrame({"Keyword": top_kw, "Relevance": range(len(top_kw), 0, -1)}),
                x="Relevance", y="Keyword", orientation="h",
                title="Top Matched Keywords",
                color="Relevance", color_continuous_scale=["#4F46E5", "#818cf8"],
            )
            bar_fig.update_layout(
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                font_color="#e2e8f0", title_font_size=14,
                showlegend=False, coloraxis_showscale=False,
                margin=dict(t=50, b=20, l=20, r=20),
                yaxis=dict(categoryorder="total ascending"),
            )
            st.plotly_chart(bar_fig, use_container_width=True)

    st.markdown('<hr class="custom-divider">', unsafe_allow_html=True)

    tab_matched, tab_missing, tab_ai = st.tabs([
        f"✅ Matched Keywords ({len(matched)})",
        f"❌ Missing Keywords ({len(missing)})",
        "🤖 AI Recruiter Analysis",
    ])

    with tab_matched:
        if matched:
            chips = " ".join(f'<span class="chip-matched">{k}</span>' for k in sorted(matched)[:50])
            st.markdown(f'<div class="chip-container">{chips}</div>', unsafe_allow_html=True)
        else:
            st.info("No matching keywords found.")

    with tab_missing:
        if missing:
            chips = " ".join(f'<span class="chip-missing">{k}</span>' for k in sorted(missing)[:50])
            st.markdown(f'<div class="chip-container">{chips}</div>', unsafe_allow_html=True)
            st.caption("Add these keywords to your resume where truthfully applicable.")
        else:
            st.success("🎉 No missing keywords — great coverage!")

    with tab_ai:
        render_ai_output(ai_text)

    st.markdown('<hr class="custom-divider">', unsafe_allow_html=True)
    st.markdown("### 📥 Export Report")
    resume_name = uploaded_resume.name.replace(".pdf", "").replace(" ", "_")
    pdf_buffer = generate_pdf_report(score, grade, matched, missing, ai_text, resume_name)
    st.download_button(
    label="📥 Download PDF Report",
    data=pdf_buffer,
    file_name=f"{resume_name}_analysis_report.pdf",
    mime="application/pdf",
    use_container_width=True,
)