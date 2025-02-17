import pytest
import os
from resume_analyzer_app import extract_keywords, find_missing_keywords, calculate_resume_match_score, extract_text  # Import your functions
from PyPDF2 import PdfWriter

# Sample data for testing
resume_text = "Experienced software engineer proficient in Python and Java. Skilled in Agile methodologies."
job_description = "Looking for a software engineer with Python, Java, and Agile experience. Must have excellent communication skills."

def test_extract_keywords():
    keywords = extract_keywords(resume_text)
    print(f"Extracted keywords in test: {keywords}")
    assert "experienced" in keywords
    assert "software" in keywords
    assert "engineer" in keywords
    assert "python" in keywords
    assert "java" in keywords
    assert "agile" in keywords
    assert "methodologies" in keywords
    assert "skilled" not in keywords  # Stop words test
    assert "communication" not in keywords  # Stop words test

def test_find_missing_keywords():
    missing = find_missing_keywords(resume_text, job_description)
    assert "communication" in missing
    assert len(missing) == 1

def test_calculate_resume_match_score():
    score = calculate_resume_match_score(resume_text, job_description)
    assert score > 45  # Or whatever the correct score is

def test_extract_text_pdf():
    # Create a dummy PDF for testing
    writer = PdfWriter()
    writer.add_blank_page(width=72, height=72)
    with open("test_resume.pdf", "wb") as f:
        writer.write(f)
    with open("test_resume.pdf", "rb") as f:
        text = extract_text(f)
        assert text == ""  # Since we added a blank page, the text should be empty

    os.remove("test_resume.pdf")  # Remove the dummy pdf created

def test_extract_text_empty_pdf():
    # Test with an empty PDF file
    with open("empty_test_resume.pdf", "wb") as f:  # Create an empty pdf for testing
        f.write(b"%PDF-1.4\n%EOF")
    with open("empty_test_resume.pdf", "rb") as f:
        text = extract_text(f)
        assert text == ""

    os.remove("empty_test_resume.pdf")  # Remove the dummy pdf created

def test_extract_keywords_empty_resume():
    empty_resume_text = ""
    keywords = extract_keywords(empty_resume_text)
    assert keywords == []

def test_find_missing_keywords_empty_resume():
    empty_resume_text = ""
    missing = find_missing_keywords(empty_resume_text, job_description)
    assert len(missing) > 0  # All job description keywords should be missing

def test_calculate_resume_match_score_empty_resume():
    empty_resume_text = ""
    score = calculate_resume_match_score(empty_resume_text, job_description)
    assert score == 0  # No match should result in a score of 0