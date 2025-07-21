# 🏥 Clinical Note Transformer

A web-based application built with Streamlit that transforms raw clinical notes into organized SOAP or DAP format using OpenAI's GPT API.

## Features

- **Raw Note Input**: Enter unstructured clinical notes from patient sessions
- **Diagnosis Input**: Add clinician's diagnosis and clinical impressions
- **Format Selection**: Choose between SOAP (Subjective, Objective, Assessment, Plan), DAP (Data, Assessment, Plan), or Standard comprehensive clinical format
- **DSM-5 Knowledge Base**: Vector embeddings and semantic search for intelligent DSM-5 reference
- **AI-Powered Transformation**: Uses OpenAI GPT-4.1-mini to intelligently organize and structure notes
- **Note History**: View and access previously transformed notes
- **Download Functionality**: Export transformed notes as text files
- **Professional UI**: Clean, medical-themed interface

## Prerequisites

- Python 3.7 or higher
- OpenAI API key (get one from [OpenAI Platform](https://platform.openai.com/api-keys))
- DSM-5 manual PDF file: `APA_DSM-5-Contents..pdf` (place in project directory)

## Installation

1. **Clone or download the project files**

2. **Navigate to the project directory**
   ```bash
   cd "Mental note"
   ```

3. **Install required dependencies**
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. **Start the application**
   ```bash
   streamlit run app.py
   ```

2. **Open your web browser** and go to the URL shown in the terminal (usually `http://localhost:8501`)

3. **API is pre-configured** with GPT-4.1-mini model

4. **Enter your clinical data**:
   - **Raw Clinical Notes**: Paste or type the unstructured notes from your patient session
   - **Clinician's Diagnosis**: Enter your diagnosis and clinical impressions
   - **Output Style**: Select SOAP, DAP, or Standard format
   - **DSM-5 Knowledge Base**: Initialize vector embeddings for intelligent DSM-5 reference

5. **Click "Transform Note"** to generate the organized note

6. **View and download** the transformed note from the output section

## Input Format Examples

### Raw Clinical Notes Example:
```
Patient presents with chest pain for 2 days. Pain is sharp, radiates to left arm, 
worse with activity. No shortness of breath. BP 140/90, HR 88, temp 98.6F. 
ECG shows ST elevation in leads II, III, aVF. Troponin elevated at 2.5 ng/mL.
```

### Diagnosis Example:
```
Acute ST-elevation myocardial infarction (STEMI) of inferior wall
Differential: Unstable angina, pericarditis
```

## Output Formats

### SOAP Format:
- **Subjective**: Patient's symptoms, concerns, and history
- **Objective**: Observable findings, vital signs, physical examination
- **Assessment**: Clinical impression, differential diagnosis
- **Plan**: Treatment plan, medications, follow-up

### DAP Format:
- **Data**: Objective information, observations, test results
- **Assessment**: Clinical impression, diagnosis, problems identified
- **Plan**: Treatment plan, interventions, follow-up

### Standard Format:
- **Presenting Problem**: Current issues and symptoms
- **Background/History**: Relevant patient history and context
- **Session Content**: Detailed session observations and interactions
- **Interventions and Outcomes**: Therapeutic techniques used and their effects
- **Coping Strategies**: Skills and strategies discussed or practiced
- **Recommendations and Follow-Up**: Next steps and future planning

## Features in Detail

### 🔑 API Configuration
- Pre-configured with GPT-4.1-mini model
- Direct OpenAI API integration
- No manual API key configuration required

### 📚 DSM-5 Knowledge Base
- Vector embeddings for semantic search
- Intelligent retrieval of relevant DSM-5 information
- Enhanced diagnostic accuracy and coding
- One-time initialization required

### 📝 Input Section
- Large text areas for comprehensive note entry
- Placeholder text with helpful examples
- Clear section headers and styling

### 📋 Output Section
- Formatted display of transformed notes
- Download functionality for saving notes
- Timestamp tracking for each transformation

### 📚 Note History
- Access to all previously transformed notes
- Expandable view of note details
- Quick reference to past sessions

## Security Notes

- API keys are stored only in session state (not persisted)
- No clinical data is stored permanently
- All data is processed locally and sent only to OpenAI API

## Troubleshooting

### Common Issues:

1. **"Error: API request failed"**
   - Check your internet connection
   - Verify your OpenAI API key has sufficient credits

3. **App won't start**
   - Ensure all dependencies are installed: `pip install -r requirements.txt`
   - Check Python version (3.7+ required)

## Customization

You can modify the application by:

- **Changing the GPT model**: Edit the `model` parameter in the OpenAI API call
- **Adjusting output length**: Modify `max_tokens` parameter
- **Customizing prompts**: Edit the system and user prompts in the function
- **Adding new formats**: Extend the output style options

## Support

For issues or questions:
1. Check the troubleshooting section above
2. Verify your OpenAI API key and credits
3. Ensure all dependencies are properly installed

## License

This project is for educational and professional use. Please ensure compliance with your organization's data handling policies when using this tool with patient information.

---

**Note**: This tool is designed to assist healthcare professionals in organizing their clinical notes. Always review and verify the AI-generated content before using it in patient care documentation. 