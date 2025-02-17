import random
import streamlit as st
import docx2txt
import openai
import PyPDF2
import os
import io
import requests
import re
from openai import OpenAI

import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib.utils import simpleSplit

from typing import Optional  # For type hints

# --------------------------- GLOBAL CONSTANTS --------------------------- #

HF_API_URL = "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.3"
MAX_NEW_TOKENS = 1024

# --------------------------- NLTK SETUP -------------------------------- #

nltk_data_path = os.path.join(os.getcwd(), "nltk_data")
if nltk_data_path not in nltk.data.path:
    nltk.data.path.append(nltk_data_path)

def ensure_nltk_resources():
    resources = ["punkt", "stopwords"]
    for resource in resources:
        try:
            nltk.data.find(f"tokenizers/{resource}" if resource == "punkt" else f"corpora/{resource}")
        except LookupError:
            nltk.download(resource, download_dir=nltk_data_path)

ensure_nltk_resources()

# --------------------------- UTILITY FUNCTIONS -------------------------------- #

def strip_markdown(text: str) -> str:
    """Removes basic markdown syntax (bold, italics) so that text renders as plain text."""
    text = re.sub(r'\*\*(.*?)\*\*', r'\1', text)  # Bold
    text = re.sub(r'\*(.*?)\*', r'\1', text)        # Italics
    return text

# Define a callback function for when the language selection changes.
def language_change():
    st.session_state.lang = st.session_state.language_selection

def get_hf_token():  # Function to retrieve the token
    HF_TOKEN = os.environ.get("HF_TOKEN")  # Check for environment variable FIRST

    if not HF_TOKEN:  # If environment variable is NOT set
        try:
            HF_TOKEN = st.secrets["HF_TOKEN"]  # Try Streamlit secrets
        except KeyError:
            st.error("HF_TOKEN secret not found. Set environment variable locally or Streamlit secret in the cloud.")
            st.stop() #Stop the app execution
            return None # Return None if no token is found
    return HF_TOKEN  # Return the token if found

# --------------------------- TRANSLATIONS -------------------------------- #

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
        "resume_suggestions": "Resume Suggestions for Improvement",
        "download_report": "Download Report",
        "buy_me_coffee_message": "☕ If you like this app, consider supporting me:",
        "buy_me_coffee": "Buy me a coffee",
        "choose_model": "Choose an AI Model:",
        "free_public": "Free Public AI",
        "openai_api": "OpenAI API",
        "download_feedback": "Download Report",
        "free_ai_disabled": "⚠️ Free AI Model is currently unavailable. Please try again later.",
        "resume_match_score": "Your resume matches {match_score}% of the job description (based on keywords).",
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
        "pdf_title_feedback": "📝 AI Resume Analysis",
        "pdf_title_improvement": "✍️ AI Improved Resume Suggestions",
        "pdf_title_keywords": "🔍 Missing Keywords & Skills",
        "free_provide_feedback": (
            "Provide structured feedback with improvements, focusing only on analysis without repeating the inputs. "
            "Format your response as a bullet list with no more than 12 points. Begin your response by stating "
            "'This feedback contains X points:' where X is the number of bullet points provided."
        ),
        "clear_results": "Clear Results",
        "keep_inputs": "Keep resume and job description",
        "resume_text_preview": "Resume Text Preview",
        "keywords_disclaimer": "_The following is a list of keywords missing from your resume. Review them and consider adding the ones most relevant to your application._",
    },
    "Español": {
        "main_title": "📄 Analizador de Currículum Impulsado por IA",
        "upload_instructions": "Sube tu currículum y proporciona una descripción del trabajo para obtener retroalimentación generada por IA.",
        "upload_resume": "Subir currículum (PDF/DOCX)",
        "input_apikey": "Introduce la Clave de API de OpenAI (opcional, para acceso a GPT-4)",
        "spinner_analyzing": "Analizando currículum...",
        "spinner_improving": "Generando mejoras en el currículum...",
        "paste_job": "Pegar descripción del puesto aquí",
        "analyze_button": "Analizar currículum",
        "feedback": "📝 Retroalimentación del currículum",
        "no_feedback": "No hay retroalimentación disponible aún.",
        "match_score": "📊 Puntaje de coincidencia",
        "missing_keywords": "🔍 Palabras clave y habilidades faltantes",
        "improve_resume": "Mejorar currículum",
        "resume_suggestions": "Mejoras sugeridas al currículum",
        "download_report": "Descargar Reporte",
        "buy_me_coffee_message": "☕ Si te gusta esta aplicación, considera apoyarme:",
        "buy_me_coffee": "¡Invítame un café!",
        "choose_model": "Elegir un modelo de IA:",
        "free_public": "IA pública gratuita",
        "openai_api": "API de OpenAI",
        "download_feedback": "Descargar Reporte",
        "free_ai_disabled": "⚠️ El modelo de IA gratuito no está disponible actualmente. Por favor, inténtalo de nuevo más tarde.",
        "resume_match_score": "Tu currículum coincide con un **{match_score}%** de la descripción del puesto (basado en palabras claves).",
        "match_excelent": "✅ ¡Excelente coincidencia! Tu currículum se alinea muy bien con este puesto.",
        "match_good": "👍 ¡Buena coincidencia! Considera enfatizar las palabras clave faltantes y mejorar la experiencia laboral.",
        "match_low": "⚠️ Puntaje de coincidencia bajo. Intenta adaptar tu currículum más estrechamente a los requisitos del puesto.",
        "keywords_missing": "Palabras clave faltantes en el currículum:",
        "keywords_included": "✅ ¡Tu currículum incluye todas las palabras clave importantes de la descripción del puesto!",
        "unsupported_format": "Formato de archivo no compatible. Por favor, sube un PDF o DOCX.",
        "upload_resume_job_description": "Por favor, sube un currículum y escribe una descripción del puesto.",
        "no_download_data": "No hay datos disponibles para descargar. Asegúrate de haber subido el currículum y la descripción del puesto.",
        "no_data_pdf": "Faltan datos para generar el PDF. Por favor, analiza el currículum primero",
        "pdf_title": "Reporte de análisis de currículum generado por IA",
        "pdf_title_feedback": "📝 Análisis del Currículum",
        "pdf_title_improvement": "✍️ Sugerencias de Mejora de Currículum",
        "pdf_title_keywords": "🔍 Palabras clave y habilidades faltantes",
        "free_provide_feedback": (
            "Proporciona retroalimentación estructurada con mejoras, enfocándote solo en el análisis sin repetir las entradas. "
            "Formatea la respuesta como una lista de viñetas con no más de 12 puntos. Comienza tu respuesta indicando "
            "'Esta retroalimentación contiene X puntos:' donde X es el número de viñetas proporcionadas."
        ),
        "clear_results": "Borrar resultados",
        "keep_inputs": "Conservar currículum y descripción del puesto",
        "resume_text_preview": "Vista previa del texto del currículum",
        "keywords_disclaimer": "_A continuación se muestra una lista de palabras clave faltantes en tu currículum. Revísalas y considera agregar las más relevantes para tu solicitud._",
    },
}

# --------------------------- FILE READERS -------------------------------- #

def read_pdf(file) -> str:
    """Extracts all text from a PDF file."""
    reader = PyPDF2.PdfReader(file)
    text = "\n".join(page.extract_text() or "" for page in reader.pages)
    return text

def read_docx(file) -> str:
    """Extracts all text from a DOCX file."""
    return docx2txt.process(file)

def extract_text(uploaded_file) -> Optional[str]:
    """Dispatches to the correct file reader depending on the extension."""
    file_extension = uploaded_file.name.split(".")[-1].lower()
    if file_extension == "pdf":
        return read_pdf(uploaded_file)
    elif file_extension == "docx":
        return read_docx(uploaded_file)
    else:
        return None

# --------------------------- ANALYSIS FUNCTIONS -------------------------------- #

def extract_keywords(text: str) -> set:
    """
    Extracts keywords using a regex to get only alphabetic words.
    This avoids including tokens with digits (e.g. '401k', 'hours') that are not relevant as skills.
    """
    stop_words = set(stopwords.words("english"))
    words = re.findall(r'\b[a-zA-Z]+\b', text.lower())
    keywords = [word for word in words if word not in stop_words and word.isalnum()] # Filter out stop words and non-alphabetic words
    # print(f"Extracted keywords: {keywords}")  # Print the extracted keywords
    return set(keywords) #Return a set of keywords

def find_missing_keywords(resume_text: str, job_description: str) -> list:
    """Finds keywords present in the job description but missing in the resume."""
    job_keywords = extract_keywords(job_description)
    resume_keywords = extract_keywords(resume_text)
    missing = job_keywords - resume_keywords
    return list(missing)

def calculate_resume_match_score(resume_text: str, job_description: str) -> float:
    """
    Calculates a match score (0-100%) based on how many job description
    keywords appear in the resume text.
    """
    resume_keywords = extract_keywords(resume_text)
    job_keywords = extract_keywords(job_description)
    if not job_keywords:
        return 0.0
    matched_keywords = resume_keywords.intersection(job_keywords)
    match_score = (len(matched_keywords) / len(job_keywords)) * 100
    return round(match_score, 2)

def analyze_resume(resume_text, job_description, model_choice, openai_api_key, lang):
    """
    Uses either OpenAI GPT-4 or Hugging Face API to analyze the resume.
    Returns the full feedback text.
    """
    if model_choice == translations[lang]["openai_api"] and openai_api_key:
        client = OpenAI(api_key=openai_api_key)
        try:
            response = client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", 
                     "content": "You are an expert resume reviewer. Provide structured and constructive feedback."},
                    {"role": "user", 
                     "content": (
                         f"Job Description:\n{job_description}\n\nResume:\n{resume_text}\n\n"
                         "Format your response as a bullet list with no more than 12 points. "
                         "Begin your response by stating 'This feedback contains X points:' where X is the number of bullet points provided. "
                         "Provide all feedback improvements in " + lang + "."
                     )}
                ]
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"OpenAI API Error: {e}"
    else:
        HF_TOKEN = get_hf_token()  # Retrieve the token
        if HF_TOKEN:
            headers = {"Authorization": f"Bearer {HF_TOKEN}"} # Use the token to authenticate
        else:
            return translations[lang]["free_ai_disabled"]

        prompt = f"""
Analyze the following resume against the job description provided.

**Job Description:**  
{job_description}

**Resume:**  
{resume_text}

{translations[lang]["free_provide_feedback"]}
*+*
"""
        payload = {"inputs": prompt, "parameters": {"max_new_tokens": MAX_NEW_TOKENS}}
        response = requests.post(HF_API_URL, headers=headers, json=payload)
        if response.status_code == 200:
            full_response = response.json()[0].get("generated_text", "")
            if "*+*" in full_response:
                ai_response = full_response.split("*+*")[-1].strip()
            else:
                ai_response = full_response.strip()
            return ai_response
        else:
            return translations[lang]["free_ai_disabled"]

def improve_resume(resume_text, job_description, model_choice, openai_api_key, lang):
    """
    Uses either OpenAI GPT-4 or Hugging Face API to generate improved resume suggestions.
    Returns a string with improved resume suggestions.
    """
    if model_choice == translations[lang]["openai_api"] and openai_api_key:
        client = OpenAI(api_key=openai_api_key)
        try:
            response = client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", 
                     "content": "You are an expert resume writer. Improve weak sections of the resume to better align with the job description."},
                    {"role": "user", 
                     "content": (
                         f"Job Description:\n{job_description}\n\nResume:\n{resume_text}\n\n"
                         "Format your response as a bullet list with no more than 12 points. "
                         "Begin your response by stating 'This feedback contains X points:' where X is the number of bullet points provided. "
                         "Provide all improved resume suggestions in " + lang + "."
                     )}
                ]
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"OpenAI API Error: {e}"
    else:
        HF_TOKEN = get_hf_token()  # Retrieve the token
        if HF_TOKEN:
            headers = {"Authorization": f"Bearer {HF_TOKEN}"} # Use the token to authenticate
        else:
            return translations[lang]["free_ai_disabled"]

        prompt = f"""
Improve the following resume to better align with the job description.

**Job Description:**  
{job_description}

**Resume:**  
{resume_text}

Format your response as a bullet list with no more than 12 points. 
Begin your response by stating 'This feedback contains X points:' where X is the number of bullet points provided. 
Provide all improved resume suggestions in {lang}. 
*+*
"""
        payload = {"inputs": prompt, "parameters": {"max_new_tokens": MAX_NEW_TOKENS}}
        response = requests.post(HF_API_URL, headers=headers, json=payload)
        if response.status_code == 200:
            full_response = response.json()[0].get("generated_text", "")
            if "*+*" in full_response:
                ai_response = full_response.split("*+*")[-1].strip()
            else:
                ai_response = full_response.strip()
            return ai_response
        else:
            return translations[lang]["free_ai_disabled"]

# --------------------------- PDF GENERATION -------------------------------- #

def draw_text_block(c, text, x, y, max_width, line_height):
    """Draws wrapped text (after stripping markdown) and returns the new y-position."""
    plain_text = strip_markdown(text)
    lines = simpleSplit(plain_text, c._fontname, c._fontsize, max_width)
    for line in lines:
        c.drawString(x, y, line)
        y -= line_height
    return y

def draw_page_number(c, page_num, width, height):
    c.setFont("Helvetica", 8)
    c.drawRightString(width - 40, 20, f"Page {page_num}")

def generate_pdf(lang):
    """
    Generates a multi-page PDF report including:
      - Page 1: Analysis Feedback
      - Page 2: Improved Resume Suggestions (if available)
      - Page 3: Missing Keywords
    Each page includes page numbering.
    """
    if "match_score" not in st.session_state or "feedback" not in st.session_state:
        st.error(translations[lang]["no_data_pdf"])
        return None

    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=letter)
    width, height = letter
    page_num = 1

    # Page 1: Title and Analysis Feedback
    c.setFont("Helvetica-Bold", 14)
    c.drawString(100, height - 50, translations[lang]["pdf_title"])
    c.setFont("Helvetica-Bold", 12)
    c.drawString(100, height - 80, translations[lang]["pdf_title_feedback"])
    c.setFont("Helvetica", 10)
    feedback = st.session_state.get("feedback", translations[lang]["no_feedback"])
    y_position = height - 110
    y_position = draw_text_block(c, feedback, 100, y_position, width - 150, 15)
    draw_page_number(c, page_num, width, height)
    c.showPage()
    page_num += 1

    # Page 2: Improved Resume Suggestions (if available)
    if st.session_state.get("improved_resume"):
        c.setFont("Helvetica-Bold", 12)
        c.drawString(100, height - 50, translations[lang]["pdf_title_improvement"])
        c.setFont("Helvetica", 10)
        improved = st.session_state.get("improved_resume", "")
        y_position = height - 80
        y_position = draw_text_block(c, improved, 100, y_position, width - 150, 15)
        draw_page_number(c, page_num, width, height)
        c.showPage()
        page_num += 1

    # Page 3: Missing Keywords
    c.setFont("Helvetica-Bold", 12)
    c.drawString(100, height - 50, translations[lang]["pdf_title_keywords"])
    c.setFont("Helvetica", 10)
    missing_keywords = st.session_state.get("missing_keywords", [])
    y_position = height - 80
    if missing_keywords:
        keywords_text = ", ".join(missing_keywords)
        y_position = draw_text_block(c, keywords_text, 100, y_position, width - 150, 15)
    else:
        c.drawString(100, y_position, translations[lang]["keywords_included"])
    draw_page_number(c, page_num, width, height)
    c.showPage()

    c.save()
    buffer.seek(0)
    return buffer

# --------------------------- STREAMLIT UI -------------------------------- #

# Keep language selection in session state
if "lang" not in st.session_state:
    st.session_state.lang = "English"
if "resume_text" not in st.session_state:
    st.session_state.resume_text = None
if "job_description" not in st.session_state:
    st.session_state.job_description = None
if "uploaded_file" not in st.session_state:
    st.session_state.uploaded_file = None
if "job_description_text" not in st.session_state: # Store the text separately
    st.session_state.job_description_text = None

st.title(translations[st.session_state.lang]["main_title"])
st.write(translations[st.session_state.lang]["upload_instructions"])

lang = st.selectbox("🌎 Language/Idioma:", 
                    ["English", "Español"], 
                    index=["English", "Español"].index(st.session_state.lang),
                    key="language_selection",
                    on_change=language_change)
st.session_state.lang = lang

# Create a container for the Clear Results button and the checkbox.
col1, col2 = st.columns([2, 3])
with col1:
    if st.button(translations[lang]["clear_results"], key="clear_results_button"):
        # If the checkbox is not checked, clear the resume inputs as well.
        if not st.session_state.get("keep_inputs", False):
             # Reset the file uploader by changing its key
            st.session_state.file_uploader_key = str(random.randint(1000, 9999))  # New random key
            st.session_state.uploaded_file = None #Clear uploaded file from session state
            st.session_state.resume_text = None #Clear resume text from session state
            st.session_state.just_uploaded = False # Reset the just_uploaded flag
            st.session_state.job_description_text = None #Clear job description from session state
            st.session_state.job_description = None #Clear job description from session state

        # Always clear these result variables.
        for key in ["resume_analyzed", "feedback", "improved_resume", "match_score", "missing_keywords"]:
            st.session_state[key] = None

with col2:
    keep_inputs = st.checkbox(translations[lang]["keep_inputs"], value=False, key="keep_inputs")


# Use a key for the file uploader that changes when you want to reset it
if "file_uploader_key" not in st.session_state:
    st.session_state.file_uploader_key = "initial_key"  # Initial key

uploaded_file = st.file_uploader(translations[lang]["upload_resume"], type=["pdf", "docx"], key="uploaded_file_key")

if uploaded_file is not None:
    st.session_state.just_uploaded = True # Flag to indicate a new upload
    st.session_state.uploaded_file = uploaded_file  # Store uploaded file object
    st.session_state.resume_text = extract_text(uploaded_file) # Extract and store text immediately
elif "uploaded_file" in st.session_state and not hasattr(st.session_state, "just_uploaded"): # Check if there's an existing file in session state and ensure that it's not a new upload
    uploaded_file = st.session_state.uploaded_file  # Retrieve the file object
    # Display the extracted text in an expander immediately after upload
    if "resume_text" not in st.session_state: #If the text is not in the session state
        st.session_state.resume_text = extract_text(uploaded_file) # Extract it
else:
    st.session_state.just_uploaded = False # Reset the just_uploaded flag

if st.session_state.get("resume_text"): # Check using get() to avoid KeyError
    with st.expander(translations[lang]["resume_text_preview"], expanded=False):  # Add expander
        st.write(st.session_state.resume_text)  # Show resume text

job_description = st.text_area(translations[lang]["paste_job"], 
                               key="job_description_key",
                               value=st.session_state.job_description_text)
if job_description is not None:
    st.session_state.job_description_text = job_description # Store the text
    st.session_state.job_description = job_description  # Keep the old variable for compatibility

#st.write(st.session_state.resume_text[:15])  # Show job description text
#st.write(st.session_state.job_description_text[:15])  # Show job description text

# Initialize session state variables (if not already)
for key, default in [
    ("resume_text", None),
    ("job_description", None),
    ("resume_analyzed", False),
    ("feedback", None),
    ("improved_resume", None),
    ("match_score", 0),
    ("missing_keywords", []),
]:
    if key not in st.session_state:
        st.session_state[key] = default

model_choice = st.radio(
    translations[lang]["choose_model"],
    [translations[lang]["free_public"], translations[lang]["openai_api"]]
)

openai_api_key = None
if model_choice == translations[lang]["openai_api"]:
    openai_api_key = st.text_input(translations[lang]["input_apikey"], type="password")

# Analyze Resume Button
if st.button(translations[lang]["analyze_button"], key="analyze_button"):
    if st.session_state.uploaded_file and st.session_state.job_description_text: # Use the stored text
        with st.spinner(translations[lang]["spinner_analyzing"]):
            # No need to extract text again, it's already in st.session_state.resume_text
            st.session_state.resume_analyzed = True
            analysis = analyze_resume(st.session_state.resume_text, st.session_state.job_description_text, model_choice, openai_api_key, lang)
            st.session_state.feedback = analysis

            match_score = calculate_resume_match_score(st.session_state.resume_text, st.session_state.job_description_text)
            st.session_state.match_score = match_score

            missing = find_missing_keywords(st.session_state.resume_text, st.session_state.job_description_text)
            st.session_state.missing_keywords = missing
    else:
        st.warning(translations[lang]["upload_resume_job_description"])

# Display Match Score
if st.session_state.get("resume_analyzed"):
    st.subheader(translations[lang]["match_score"])
    st.write(translations[lang]["resume_match_score"].format(match_score=st.session_state.match_score))
    if st.session_state.match_score >= 90:
        st.success(translations[lang]["match_excelent"])
    elif st.session_state.match_score >= 70:
        st.info(translations[lang]["match_good"])
    else:
        st.warning(translations[lang]["match_low"])

# Always display Analysis if available
if st.session_state.get("feedback"):
    st.subheader(translations[lang]["feedback"])
    st.markdown(st.session_state.feedback, unsafe_allow_html=True)

# Improve Resume Button (does not remove the Analysis)
if st.session_state.get("resume_text") and st.session_state.get("job_description"):
    if st.button(translations[lang]["improve_resume"], key="improve_button"):
        with st.spinner(translations[lang]["spinner_improving"]):
            improved = improve_resume(
                st.session_state.resume_text,
                st.session_state.job_description,
                model_choice,
                openai_api_key,
                lang
            )
            st.session_state.improved_resume = improved

# Always display Improved Resume if available
if st.session_state.get("improved_resume"):
    st.subheader(translations[lang]["resume_suggestions"])
    st.markdown(st.session_state.improved_resume, unsafe_allow_html=True)

# Display Missing Keywords before the download button
if st.session_state.get("resume_analyzed"):
    with st.expander(translations[lang]["missing_keywords"], expanded=False):
        if st.session_state.missing_keywords:
            st.write(translations[lang]["keywords_disclaimer"])
            st.write(", ".join(st.session_state.missing_keywords))
        else:
            st.info(translations[lang]["keywords_included"])

# Download Report Button (single button for the complete PDF)
if st.session_state.get("feedback"):
    st.subheader(translations[lang]["download_report"])
    pdf_data = generate_pdf(lang)
    if pdf_data:
        st.download_button(
            label=translations[lang]["download_report"],
            data=pdf_data,
            file_name="resume_analysis.pdf",
            mime="application/pdf",
            key="download_pdf"
        )
    else:
        st.error(translations[lang]["no_download_data"])

coffee_button_html = f"""
---
{translations[lang]["buy_me_coffee_message"]}
<a href="https://www.buymeacoffee.com/intisoto" target="_blank">
    <img src="https://img.buymeacoffee.com/button-api/?text={translations[lang]["buy_me_coffee"]}&emoji=☕&slug=intisoto&button_colour=FFDD00&font_colour=000000&font_family=Arial&outline_colour=000000&coffee_colour=ffffff"
         alt="{translations[lang]["buy_me_coffee"]}" width="200">
</a>
"""
st.markdown(coffee_button_html, unsafe_allow_html=True)
