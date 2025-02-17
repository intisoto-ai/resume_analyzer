import pytest
import os
from resume_analyzer_app import extract_keywords, find_missing_keywords, calculate_resume_match_score, extract_text  # Import your functions

# Sample data for testing
resume_text = "Experienced software engineer proficient in Python and Java.  Skilled in Agile methodologies."
job_description = "Looking for a software engineer with Python, Java, and Agile experience. Must have excellent communication skills."

def test_extract_keywords():
    keywords = extract_keywords(resume_text)
    assert "experienced" in keywords
    assert "software" in keywords
    assert "engineer" in keywords
    assert "python" in keywords
    assert "java" in keywords
    assert "agile" in keywords
    assert "methodologies" in keywords
    assert "skilled" not in keywords # Stop words test
    assert "communication" not in keywords # Stop words test

def test_find_missing_keywords():
    missing = find_missing_keywords(resume_text, job_description)
    assert "communication" in missing
    assert len(missing) == 1

def test_calculate_resume_match_score():
    score = calculate_resume_match_score(resume_text, job_description)
    assert score == 85.71  # Or whatever the correct score is

def test_extract_text_pdf():
    # You'll need a sample PDF file for this test
    with open("test_resume.pdf", "wb") as f: #Create a dummy pdf for testing
        f.write(b"This is a test PDF.")
    with open("test_resume.pdf", "rb") as f:
        text = extract_text(f)
        assert "This is a test PDF." in text

    os.remove("test_resume.pdf") #Remove the dummy pdf created