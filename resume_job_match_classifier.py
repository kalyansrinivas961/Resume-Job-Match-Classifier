import streamlit as st
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from PyPDF2 import PdfReader

st.set_page_config(page_title="Resume Job Match Classifier", layout="centered")
st.title("📄 Resume Job Match Classifier")
@st.cache_resource
def load_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

model = load_model()
def extract_text_from_pdf(uploaded_file):
    reader = PdfReader(uploaded_file)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text
st.subheader("📌 Upload or Paste Resume")
resume_file = st.file_uploader("Upload Resume (PDF/TXT)", type=["pdf", "txt"], key="resume")
resume_text_input = st.text_area("Or Paste Resume Text", height=180)

st.subheader("📌 Upload or Paste Job Description")
job_file = st.file_uploader("Upload Job Description (PDF/TXT)", type=["pdf", "txt"], key="job")
job_text_input = st.text_area("Or Paste Job Description Text", height=180)

resume_text = ""
job_text = ""

if resume_file:
    if resume_file.type == "application/pdf":
        resume_text = extract_text_from_pdf(resume_file)
    else:
        resume_text = resume_file.read().decode("utf-8")
elif resume_text_input:
    resume_text = resume_text_input

if job_file:
    if job_file.type == "application/pdf":
        job_text = extract_text_from_pdf(job_file)
    else:
        job_text = job_file.read().decode("utf-8")
elif job_text_input:
    job_text = job_text_input

if st.button("🔍 Check Match"):
    if not resume_text or not job_text:
        st.warning("Please provide both resume and job description.")
    else:
        with st.spinner("Analyzing..."):
            embeddings = model.encode([resume_text, job_text])
            similarity = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
            match_percent = round(similarity * 100, 2)

            if match_percent >= 75:
                verdict = "✅ Excellent Match"
            elif match_percent >= 50:
                verdict = "🟡 Moderate Match"
            else:
                verdict = "🔴 Low Match"

            st.success("Match Analysis Complete!")
            st.metric("Match Score", f"{match_percent}%")
            st.progress(min(match_percent / 100, 1.0))
            st.write(f"**Verdict:** {verdict}")

            report = f"Resume Match Report\n\nScore: {match_percent}%\nVerdict: {verdict}\n\n---\n\nResume Preview:\n{resume_text[:500]}...\n\nJob Description Preview:\n{job_text[:500]}..."
            st.download_button("📥 Download Match Report", report, file_name="resume_match_report.txt")
