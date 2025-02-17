# test_resume_analyzer.py
import pytest
from resume_analyzer_app import extract_keywords, calculate_resume_match_score, find_missing_keywords

# Sample known input strings:
resume_text = "I am a skilled Python developer with experience in data analysis and machine learning."
#resume_text = "I am a skilled Python developer with experience in data analysis, machine learning, problem solving. His skills and experience makes him an ideal candidate for the position."
job_description = ("The ideal candidate is a Python developer who has expertise in data analysis, "
                   "machine learning, and problem solving.")

def test_extract_keywords_resume_text():
    # Given our extraction logic (using regex and NLTK stopwords),
    # we expect that common stopwords like "I", "am", "with", "and" are removed.
    expected_keywords = {"skilled", "python", "developer", "experience", "data", "analysis", "machine", "learning"}
    keywords = extract_keywords(resume_text)
    print(f"Extracted keywords in test: {keywords}")
    assert keywords == expected_keywords

def test_calculate_resume_match_score():
    # The match score is computed as:
    #   (number of keywords in both texts / number of keywords in job_description) * 100, rounded to 2 decimals.
    resume_keywords = extract_keywords(resume_text)
    job_keywords = extract_keywords(job_description)
    matched = resume_keywords.intersection(job_keywords)
    expected_score = round((len(matched) / len(job_keywords)) * 100, 2) if job_keywords else 0.0
    score = calculate_resume_match_score(resume_text, job_description)
    print(f"Calculated score in test: {score}")
    assert score == expected_score

def test_find_missing_keywords():
    # Missing keywords are those present in the job description but not in the resume.
    expected_missing = set(extract_keywords(job_description)) - set(extract_keywords(resume_text))
    missing = set(find_missing_keywords(resume_text, job_description))
    print(f"Missing keywords in test: {missing}")
    assert missing == expected_missing
