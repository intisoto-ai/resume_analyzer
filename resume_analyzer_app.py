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

# Avoid adding duplicate paths
if nltk_data_path not in nltk.data.path:
    nltk.data.path.append(nltk_data_path)

# Download required NLTK packages only if they don't exist
def ensure_nltk_resources():
    """Downloads necessary NLTK data if not already available."""
    resources = ["punkt_tab", "stopwords"]
    for resource in resources:
        try:
            nltk.data.find(f"tokenizers/{resource}" if resource == "punkt_tab" else f"corpora/{resource}")
        except LookupError:
            nltk.download(resource, download_dir=nltk_data_path)

# Call function to ensure NLTK resources are available
ensure_nltk_resources()

def generate_pdf():
    # Check if necessary data is available in session_state
    if "match_score" not in st.session_state or "feedback" not in st.session_state:
        st.error(translations[lang]["no_data_pdf"])
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
    c.drawString(100, y_position, translations[lang]["pdf_title"])
    y_position -= 20
    c.setFont("Helvetica", 12)
    c.drawString(100, y_position, "--------------------------------------------------")
    y_position -= 30

    # Resume Match Score
    c.setFont("Helvetica-Bold", 12)
    c.drawString(100, y_position, translations[lang]["match_score"])
    y_position -= 20
    c.setFont("Helvetica", 11)
    match_score = st.session_state.get("match_score", 0)
    c.drawString(100, y_position, translations[lang]["resume_match_score"].format(match_score=match_score))
    y_position -= 30

    # AI Resume Feedback
    c.setFont("Helvetica-Bold", 12)
    c.drawString(100, y_position, translations[lang]["pdf_title_feedback"])
    y_position -= 20
    c.setFont("Helvetica", 10)

    feedback = st.session_state.get("feedback", translations[lang]["no_feedback"])
    if feedback:
        y_position = draw_wrapped_text(c, feedback, 100, y_position)
    else:
        c.drawString(100, y_position, translations[lang]["no_feedback"])
        y_position -= 20

    # Missing Keywords//
    y_position -= 20
    c.setFont("Helvetica-Bold", 12)
    c.drawString(100, y_position, translations[lang]["pdf_title_keywords"])
    y_position -= 20
    c.setFont("Helvetica", 10)
    missing_keywords = st.session_state.get("missing_keywords", [])
    if missing_keywords:
        keywords_text = ", ".join(missing_keywords)
        y_position = draw_wrapped_text(c, keywords_text, 100, y_position)
    else:
        c.drawString(100, y_position, translations[lang]["keywords_included"])

    # Save PDF
    c.save()
    buffer.seek(0)
    return buffer

# Set OpenAI API Key only if the user selects OpenAI
api_key = None
if "api_key" not in st.session_state:
    st.session_state.api_key = None

if st.session_state.get("use_openai", False):  # Only show if OpenAI is selected
    api_key = st.text_input("Enter OpenAI API Key (Optional, for GPT-4 Access)", type="password")
    if api_key:
            st.session_state.api_key = api_key  # Store in session state

translations = {
    "English": {
        "main_title": "📄 AI-Powered Resume Analyzer",
        "upload_instructions": "Upload your resume and provide a job description to get AI-generated feedback.",
        "upload_resume": "Upload Resume (PDF/DOCX)",
        "input_apikey": "Enter OpenAI API Key (Optional, for GPT-4 Access)",
        "spinner_analyzing": "Analyzing resume...",
        "spinner_improving": "Generating resume improvements...",
        "paste_job": "Paste Job Description Here",
        "analyze_button": "Analyze Resume",
        "feedback": "📝 Resume Feedback",
        "no_feedback": "No feedback available yet.",
        "match_score": "📊 Resume Match Score",
        "missing_keywords": "🔍 Missing Keywords & Skills",
        "improve_resume": "Improve Resume",
        "download_report": "📥 Download Report",
        "buy_me_coffee_message": "☕ If you like this app, consider supporting me:",
        "buy_me_coffee": "Buy me a coffee",
        "choose_model": "Choose an AI Model:",
        "free_public": "Free Public AI",
        "openai_api": "OpenAI API",
        "download_report": "📥 Download Report",
        "download_feedback": "Download AI Feedback as PDF",
        "free_ai_disabled": "⚠️ Free AI Model is currently unavailable. Please try again later.",
        "resume_match_score": "Your resume matches **{match_score}%** of the job description.",
        "match_excelent": "✅ Excellent match! Your resume aligns very well with this job.",
        "match_good": "👍 Good match! Consider emphasizing missing keywords and refining work experience.",
        "match_low": "⚠️ Low match score. Try tailoring your resume more closely to the job requirements.",
        "keywords_missing": "Keywords Missing from Resume:",
        "keywords_included": "✅ Your resume includes all important keywords from the job description!",
        "unsupported_format": "Unsupported file format. Please upload a PDF or DOCX.",
        "upload_resume_job_description": "Please upload a resume and enter a job description.",
        "no_download_data": "No data available to download. Make sure you uploaded the resume and job description.",
        "no_data_pdf": "Missing data for generating PDF. Please analyze the resume first.",
        "pdf_title": "AI-Powered Resume Analysis Report",
        "pdf_title_feedback": "📝 AI Resume Feedback",
        "pdf_title_keywords": "🔍 Missing Keywords & Skills",
    },
    "Español": {
        "main_title": "📄 Analizador de Currículum Impulsado por IA",
        "upload_instructions": "Sube tu currículum y proporciona una descripción del trabajo para obtener retroalimentación generada por IA.",
        "upload_resume": "Subir Currículum (PDF/DOCX)",
        "input_apikey": "Introduce la Clave de API de OpenAI (Opcional, para Acceso a GPT-4)",
        "spinner_analyzing": "Analizando currículum...",
        "spinner_improving": "Generando mejoras en el currículum...",
        "paste_job": "Pegar Descripción del Trabajo Aquí",
        "analyze_button": "Analizar Currículum",
        "feedback": "📝 Retroalimentación del Currículum",
        "no_feedback": "No hay retroalimentación disponible aún.",
        "match_score": "📊 Puntaje de Coincidencia",
        "missing_keywords": "🔍 Palabras Clave y Habilidades Faltantes",
        "improve_resume": "Mejorar Currículum",
        "download_report": "📥 Descargar Informe",
        "buy_me_coffee_message": "☕ Si te gusta esta aplicación, considera apoyarme:",
        "buy_me_coffee": "¡Invítame un café!",
        "choose_model": "Elegir un Modelo de IA:",
        "free_public": "IA Pública Gratuita",
        "openai_api": "API de OpenAI",
        "download_report": "📥 Descargar Informe",
        "download_feedback": "Descargar Retroalimentación de IA como PDF",
        "free_ai_disabled": "⚠️ El modelo de IA gratuito no está disponible actualmente. Por favor, inténtalo de nuevo más tarde.",
        "resume_match_score": "Tu currículum coincide con **{match_score}%** de la descripción del trabajo.",
        "match_excelent": "✅ ¡Excelente coincidencia! Tu currículum se alinea muy bien con este trabajo.",
        "match_good": "👍 ¡Buena coincidencia! Considera enfatizar las palabras clave faltantes y mejorar la experiencia laboral.",
        "match_low": "⚠️ Puntaje de coincidencia bajo. Intenta adaptar tu currículum más estrechamente a los requisitos del trabajo.",
        "keywords_missing": "Palabras Clave Faltantes en el Currículum:",
        "keywords_included": "✅ ¡Tu currículum incluye todas las palabras clave importantes de la descripción del trabajo!",
        "unsupported_format": "Formato de archivo no compatible. Por favor, sube un PDF o DOCX.",
        "upload_resume_job_description": "Por favor, sube un currículum y escribe una descripción del trabajo.",
        "no_download_data": "No hay datos disponibles para descargar. Asegúrate de haber subido el currículum y la descripción del trabajo.",
        "no_data_pdf": "Faltan datos para generar el PDF. Por favor, analiza el currículum primero",
        "pdf_title": "Informe de Análisis de Currículum Impulsado por IA",
    },
}

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
            return translations[lang]["free_ai_disabled"]   

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
st.title(translations[lang]["main_title"])

st.write(nltk_data_path)
# Language Selection Dropdown
lang = st.selectbox("🌎 Language/Idioma:", ["English", "Español"])

st.write(translations[lang]["upload_instructions"]) 

# File Upload
uploaded_file = st.file_uploader(translations[lang]["upload_resume"], type=["pdf", "docx"])

# Job Description Input
job_description = st.text_area(translations[lang]["paste_job"])

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
model_choice = st.radio(translations[lang]["choose_model"], [translations[lang]["free_public"], translations[lang]["openai_api"]])

# If OpenAI is selected, allow user to enter API Key
openai_api_key = None
if model_choice == "OpenAI API":
    openai_api_key = st.text_input(translations[lang]["input_apikey"], type="password")

# Analyze Button
if st.button(translations[lang]["analyze_button"]):
    if uploaded_file and job_description:
        with st.spinner(translations[lang]["spinner_analyzing"] ): # Add translation
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
                    st.subheader(translations[lang]["feedback"])
                    st.write(st.session_state.get("feedback", translations[lang]["no_feedback"]))

                # Calculate Resume Match Score
                match_score = calculate_resume_match_score(resume_text, job_description)
                if match_score:
                    st.session_state.match_score = match_score
    
                # Display Resume Match Score in the Second Column
                with col2:
                    st.subheader(translations[lang]["match_score"])
                    match_score = st.session_state.get("match_score", 0)
                    st.write(translations[lang]["resume_match_score"].format(match_score=match_score))

                # Provide feedback based on score
                if match_score > 85:
                    st.success(translations[lang]["match_excelent"])
                elif match_score > 65:
                    st.info(translations[lang]["match_good"])
                else:
                    st.warning(translations[lang]["match_low"])

                # Perform Keyword Matching Analysis
                missing_keywords = find_missing_keywords(resume_text, job_description)
                if missing_keywords:
                    st.session_state.missing_keywords = missing_keywords

                # Display Missing Skills
                with st.expander(translations[lang]["missing_keywords"], expanded=False):
                    missing_keywords = st.session_state.get("missing_keywords", [])
                    if missing_keywords:
                        st.write(translations[lang]["keywords_missing"])
                        st.write(", ".join(missing_keywords))
                    else:
                        st.write(translations[lang]["keywords_included"])

            else:
                st.error(translations[lang]["unsupported_format"])

    else:
        st.warning(translations[lang]["upload_resume_job_description"])

# Generate AI Resume Improvements (Only if we have stored resume text)
if st.session_state.resume_text and st.session_state.job_description:
    if st.button(translations[lang]["improve_resume"]):
        st.subheader("✍️ " + translations[lang]["improve_resume"])
        with st.spinner(translations[lang]["spinner_improving"]):
            improved_resume = improve_resume(st.session_state.resume_text, st.session_state.job_description) #if invoke_ai else "AI analysis disabled." 
            st.write(improved_resume)

# Allow User to Download AI Feedback as a PDF
st.subheader(translations[lang]["download_report"])
if st.button(translations[lang]["download_feedback"]):
    pdf_data = generate_pdf()

    if pdf_data:
        st.download_button(
            label=translations[lang]["download_report"],
            data=pdf_data,
            file_name="resume_analysis.pdf",
            mime="application/pdf"
        )
    else:
        st.error(translations[lang]["no_download_data"])

        
# Show Support
st.markdown(
    """
    ---
    {buy_me_coffee_message}
    <a href="https://www.buymeacoffee.com/intisoto" target="_blank">
        <img src="https://img.buymeacoffee.com/button-api/?text={buy_me_coffee}&emoji=☕&slug=intisoto&button_colour=FFDD00&font_colour=000000&font_family=Arial&outline_colour=000000&coffee_colour=ffffff" 
        alt="{buy_me_coffee}" width="200">
    </a>
    """.format(
        buy_me_coffee_message=translations[lang]["buy_me_coffee_message"].format(buy_me_coffee=translations[lang]["buy_me_coffee"]),
        buy_me_coffee=translations[lang]["buy_me_coffee"]
    ),
    unsafe_allow_html=True
)
