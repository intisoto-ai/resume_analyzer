# AI-Powered Resume Analyzer

AI-Powered Resume Analyzer is a web application that helps users analyze and improve their resumes based on a provided job description. The application uses AI models to provide feedback and suggestions for enhancing the resume to better match the job requirements.

## Features

- Upload resume in PDF or DOCX format
- Paste job description for analysis
- Analyze resume against job description using AI models
- Generate feedback and suggestions for improving the resume
- Calculate resume match score
- Identify missing keywords and skills
- Download AI feedback as a PDF report
- Support for multiple languages (English and Español)

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/yourusername/ai-powered-resume-analyzer.git
    cd ai-powered-resume-analyzer
    ```

2. Create a virtual environment and activate it:
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

4. Download NLTK resources:
    ```python
    import nltk
    nltk.download('punkt')
    nltk.download('stopwords')
    ```

## Usage

1. Run the Streamlit application:
    ```bash
    streamlit run resume_analyzer_app.py
    ```

2. Open your web browser and navigate to `http://localhost:8501`.

3. Upload your resume (PDF or DOCX) and paste the job description.

4. Select the AI model (Free Public AI or OpenAI API) and enter the OpenAI API key if required.

5. Click the "Analyze Resume" button to get feedback and suggestions.

6. Optionally, click the "Improve Resume" button to generate improved resume bullet points.

7. Download the AI feedback as a PDF report.

## Configuration

- **OpenAI API Key**: If you choose to use the OpenAI API, you need to provide your OpenAI API key. You can enter the key in the application when prompted.
- **Hugging Face API Token**: If you use the free public AI model, you can set your Hugging Face API token in the [secrets.toml](http://_vscodecontentref_/0) file.

## Translations

The application supports multiple languages. You can add more languages by updating the [translations](http://_vscodecontentref_/1) dictionary in the [resume_analyzer_app.py](http://_vscodecontentref_/2) file.

## Contributing

Contributions are welcome! Please follow these steps to contribute:

1. Fork the repository.
2. Create a new branch (`git checkout -b feature/your-feature`).
3. Commit your changes (`git commit -am 'Add some feature'`).
4. Push to the branch (`git push origin feature/your-feature`).
5. Create a new Pull Request.

## License

This project is licensed under the MIT License. See the LICENSE file for details.

## Acknowledgements

- [Streamlit](https://streamlit.io/)
- [OpenAI](https://openai.com/)
- [Hugging Face](https://huggingface.co/)
- [NLTK](https://www.nltk.org/)
- [ReportLab](https://www.reportlab.com/)

## Support

If you like this app, consider supporting me:

<a href="https://www.buymeacoffee.com/intisoto" target="_blank">
    <img src="https://img.buymeacoffee.com/button-api/?text=Buy me a coffee&emoji=☕&slug=intisoto&button_colour=FFDD00&font_colour=000000&font_family=Arial&outline_colour=000000&coffee_colour=ffffff" 
    alt="Buy Me A Coffee" width="200">
</a>