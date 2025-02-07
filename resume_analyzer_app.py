import streamlit as st
import openai
import docx2txt
import PyPDF2
import os

import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib.utils import simpleSplit
import io

import requests

# Set a specific download directory for NLTK
nltk_data_path = os.path.join(os.getcwd(), "nltk_data")
nltk.data.path.append(nltk_data_path)

# Download required NLTK packages only if they don't exist
try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt", download_dir=nltk_data_path)

    #nltk.download('wordnet', download_dir=nltk_data_path)
    #nltk.download('omw-1.4', download_dir=nltk_data_path)

try:
    nltk.data.find("corpora/stopwords")
except LookupError:
    nltk.download("stopwords", download_dir=nltk_data_path)

def generate_pdf():
    # Check if necessary data is available in session_state
    if "match_score" not in st.session_state or "feedback" not in st.session_state:
        st.error("Missing data for generating PDF. Please analyze the resume first.")
        return None

    """Generates a well-formatted PDF report for the resume analysis."""
    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=letter)
    width, height = letter  
    y_position = height - 50  

    def draw_wrapped_text(c, text, x, y, max_width=400, line_height=15):
        """Handles text wrapping inside the PDF."""
        wrapped_lines = simpleSplit(text, c._fontname, c._fontsize, max_width)
        for line in wrapped_lines:
            c.drawString(x, y, line)
            y -= line_height
        return y  # Return new y position

    # Title
    c.setFont("Helvetica-Bold", 14)
    c.drawString(100, y_position, "📄 AI-Powered Resume Analysis Report")
    y_position -= 20
    c.setFont("Helvetica", 12)
    c.drawString(100, y_position, "--------------------------------------------------")
    y_position -= 30

    # Resume Match Score
    c.setFont("Helvetica-Bold", 12)
    c.drawString(100, y_position, "📊 Resume Match Score:")
    y_position -= 20
    c.setFont("Helvetica", 11)
    match_score = st.session_state.get("match_score", 0)
    c.drawString(100, y_position, f"Your resume matches {match_score}% of the job description.")
    y_position -= 30

    # AI Resume Feedback
    c.setFont("Helvetica-Bold", 12)
    c.drawString(100, y_position, "📝 AI Resume Feedback:")
    y_position -= 20
    c.setFont("Helvetica", 10)

    feedback = st.session_state.get("feedback", "No feedback available.")
    if feedback:
        y_position = draw_wrapped_text(c, feedback, 100, y_position)
    else:
        c.drawString(100, y_position, "No feedback available.")
        y_position -= 20

    # Missing Keywords//
    y_position -= 20
    c.setFont("Helvetica-Bold", 12)
    c.drawString(100, y_position, "🔍 Missing Keywords & Skills:")
    y_position -= 20
    c.setFont("Helvetica", 10)
    missing_keywords = st.session_state.get("missing_keywords", [])
    if missing_keywords:
        keywords_text = ", ".join(missing_keywords)
        y_position = draw_wrapped_text(c, keywords_text, 100, y_position)
    else:
        c.drawString(100, y_position, "✅ No missing skills detected.")

    # Save PDF
    c.save()
    buffer.seek(0)
    return buffer

# OpenAI API Key
api_key = os.getenv("OPENAI_API_KEY")

if api_key is None:
    st.error("⚠️ OpenAI API key not found. Set it as an environment variable.")
else:
    client = openai.OpenAI(api_key=api_key)

# Function to read PDF files
def read_pdf(file):
    reader = PyPDF2.PdfReader(file)
    text = "\n".join([page.extract_text() for page in reader.pages if page.extract_text()])
    return text

# Function to read DOCX files
def read_docx(file):
    return docx2txt.process(file)

# Function to extract text from file
def extract_text(uploaded_file):
    file_extension = uploaded_file.name.split(".")[-1]
    
    if file_extension == "pdf":
        return read_pdf(uploaded_file)
    elif file_extension == "docx":
        return read_docx(uploaded_file)
    else:
        return None

# Function to analyze resume against job description
def analyze_resume(resume_text, job_description, model_choice, openai_api_key):
    """Uses either a free public LLM (Hugging Face) or OpenAI API for analysis."""
    
    if model_choice == "OpenAI API" and openai_api_key:
        # Use OpenAI API
        client = openai.OpenAI(api_key=openai_api_key)
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are an expert resume reviewer. Provide structured and constructive feedback."},
                {"role": "user", "content": f"Job Description:\n{job_description}\n\nResume:\n{resume_text}\n\nProvide structured feedback with improvements."}
            ]
        )
        return response.choices[0].message.content

    else:
        # Use Free Hugging Face API (Mistral-7B-Instruct)
        API_URL = "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct"
        headers = {"Authorization": f"Bearer {st.secrets['HF_TOKEN']}"} if "HF_TOKEN" in st.secrets else {}
        
        payload = {
            "inputs": f"Analyze this resume against the job description:\n\nJob Description:\n{job_description}\n\nResume:\n{resume_text}\n\nProvide structured feedback and improvements.",
            "parameters": {"max_new_tokens": 300}
        }
        
        response = requests.post(API_URL, headers=headers, json=payload)
        if response.status_code == 200:
            return response.json()[0]["generated_text"]
        else:
            return "⚠️ Free AI Model is currently unavailable. Please try again later."

def extract_keywords(text):
    """Extracts key words from text by removing common stopwords."""
    stop_words = set(stopwords.words("english"))
    words = word_tokenize(text.lower())  # Tokenize words
    keywords = [word for word in words if word.isalnum() and word not in stop_words]  # Remove stopwords & punctuation
    return set(keywords)

def find_missing_keywords(resume_text, job_description):
    """Finds keywords in job description that are missing from the resume."""
    job_keywords = extract_keywords(job_description)
    resume_keywords = extract_keywords(resume_text)
    missing_keywords = job_keywords - resume_keywords
    return list(missing_keywords)

def calculate_resume_match_score(resume_text, job_description):
    """Calculates a match score (0-100%) based on resume vs job description similarity."""
    
    resume_keywords = extract_keywords(resume_text)
    job_keywords = extract_keywords(job_description)
    
    # Score based on keyword overlap
    matched_keywords = resume_keywords.intersection(job_keywords)
    match_score = (len(matched_keywords) / len(job_keywords)) * 100 if len(job_keywords) > 0 else 0
    
    # Normalize score to 100
    return round(match_score, 2)

def improve_resume(resume_text, job_description):
    """Uses GPT-4 to generate improved resume bullet points and suggestions."""
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are an expert resume writer. Improve weak sections of the resume to better align with the job description."},
            {"role": "user", "content": f"Job Description:\n{job_description}\n\nResume:\n{resume_text}\n\nProvide improved resume bullet points and suggestions."}
        ]
    )
    return response.choices[0].message.content

# Streamlit Web App
st.title("📄 AI-Powered Resume Analyzer")

st.write(nltk_data_path)
# Language Selection Dropdown
lang = st.selectbox("🌎 Language/Idioma:", ["English", "Español"])

st.write("Upload your resume and provide a job description to get AI-generated feedback.")

# File Upload
uploaded_file = st.file_uploader("Upload Resume (PDF/DOCX)", type=["pdf", "docx"])

# Job Description Input
job_description = st.text_area("Paste Job Description Here")

# Initialize session state variables
if "resume_text" not in st.session_state:
    st.session_state.resume_text = None

if "job_description" not in st.session_state:
    st.session_state.job_description = None

if "resume_analyzed" not in st.session_state:
    st.session_state.resume_analyzed = False  # Track if analysis was performed

if "feedback" not in st.session_state:
    st.session_state.feedback = None

if "match_score" not in st.session_state:
    st.session_state.match_score = 0

if "missing_keywords" not in st.session_state:
    st.session_state.missing_keywords = []

# Model Selection
model_choice = st.radio("Choose an AI Model:", ["Free Public AI", "OpenAI API"])

# If OpenAI is selected, allow user to enter API Key
openai_api_key = None
if model_choice == "OpenAI API":
    openai_api_key = st.text_input("Enter OpenAI API Key (Optional, for GPT-4 Access)", type="password")


# Analyze Button
if st.button("Analyze Resume"):
    if uploaded_file and job_description:
        with st.spinner("Analyzing resume..."):
            resume_text = extract_text(uploaded_file)

            if resume_text:
                 # Store analyzed text in session state
                st.session_state.resume_text = resume_text
                st.session_state.job_description = job_description
                st.session_state.resume_analyzed = True  # Mark analysis as done

                feedback = analyze_resume(resume_text, job_description, model_choice, openai_api_key) #if invoke_ai else "AI analysis disabled."
                if feedback:
                    st.session_state.feedback = feedback

                # Create two columns for better UI structure
                col1, col2 = st.columns(2)

                # Display AI Feedback in the First Column
                with col1:
                    st.subheader("📝 Resume Feedback")
                    st.write(st.session_state.get("feedback", "No feedback yet."))


                # Calculate Resume Match Score
                match_score = calculate_resume_match_score(resume_text, job_description)
                if match_score:
                    st.session_state.match_score = match_score
    
                # Display Resume Match Score in the Second Column
                with col2:
                    st.subheader("📊 Resume Match Score")
                    match_score = st.session_state.get("match_score", 0)
                    st.write(f"Your resume matches **{match_score}%** of the job description.")

                # Provide feedback based on score
                if match_score > 85:
                    st.success("✅ Excellent match! Your resume aligns very well with this job.")
                elif match_score > 65:
                    st.info("👍 Good match! Consider emphasizing missing keywords and refining work experience.")
                else:
                    st.warning("⚠️ Low match score. Try tailoring your resume more closely to the job requirements.")

                # Perform Keyword Matching Analysis
                missing_keywords = find_missing_keywords(resume_text, job_description)
                if missing_keywords:
                    st.session_state.missing_keywords = missing_keywords

                # Display Missing Skills
                with st.expander("🔍 Missing Keywords & Skills", expanded=False):
                    missing_keywords = st.session_state.get("missing_keywords", [])
                    if missing_keywords:
                        st.write("Your resume is missing these key terms from the job description:")
                        st.write(", ".join(missing_keywords))
                    else:
                        st.write("✅ Your resume includes all important keywords from the job description!")

            else:
                st.error("Unsupported file format. Please upload a PDF or DOCX.")

    else:
        st.warning("Please upload a resume and enter a job description.")

# Generate AI Resume Improvements (Only if we have stored resume text)
if st.session_state.resume_text and st.session_state.job_description:
    if st.button("Improve Resume"):
        st.subheader("✍️ AI-Suggested Resume Improvements")
        with st.spinner("Generating resume improvements..."):
            improved_resume = improve_resume(st.session_state.resume_text, st.session_state.job_description) #if invoke_ai else "AI analysis disabled." 
            st.write(improved_resume)

# Allow User to Download AI Feedback as a PDF
st.subheader("📥 Download Report")
if st.button("Download AI Feedback as PDF"):
    pdf_data = generate_pdf()

    if pdf_data:
        st.download_button(
            label="📄 Download Report",
            data=pdf_data,
            file_name="resume_analysis.pdf",
            mime="application/pdf"
        )
    else:
        st.error("No data available to download. Make sure you uploaded the resume and job description.")