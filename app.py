import streamlit as st
import openai
import os
from datetime import datetime
import json
import PyPDF2
import io
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.schema import Document
import re

# Page configuration
st.set_page_config(
    page_title="PsyNoteTaker",
    page_icon="totoro.jpg",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for a more modern, professional look
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
    html, body, [class*='css']  {
        font-family: 'Inter', sans-serif;
    }
    .main-header {
        font-size: 2.7rem;
        color: #2d3a4a;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: 700;
        letter-spacing: 1px;
    }
    .section-header {
        font-size: 1.4rem;
        color: #1a2636;
        margin-bottom: 1rem;
        font-weight: 600;
        border-left: 4px solid #4f8cff;
        padding-left: 0.5rem;
        background: #f4f8fb;
        border-radius: 4px;
    }
    .input-box, .output-box {
        background-color: #f7fafd;
        padding: 1.2rem;
        border-radius: 12px;
        border: 1.5px solid #e3e8ee;
        margin-bottom: 1.5rem;
        box-shadow: 0 2px 8px rgba(79,140,255,0.04);
    }
    .output-box {
        background-color: #f0f7f4;
        border: 1.5px solid #4f8cff;
    }
    .stButton > button {
        width: 100%;
        background-color: #4f8cff;
        color: white;
        border: none;
        padding: 0.8rem;
        border-radius: 6px;
        font-size: 1.1rem;
        font-weight: 600;
        transition: background 0.2s;
    }
    .stButton > button:hover {
        background-color: #2d3a4a;
    }
    .stTextArea textarea {
        font-family: 'Inter', monospace;
        font-size: 15px;
        background: #f7fafd;
        border-radius: 6px;
        border: 1px solid #e3e8ee;
    }
    .stForm {
        border: none;
        padding: 0;
    }
    .sidebar-title {
        font-size: 1.3rem;
        font-weight: 700;
        color: #2d3a4a;
        margin-bottom: 0.5rem;
        text-align: center;
    }
    .sidebar-desc {
        color: #4f8cff;
        font-size: 1.05rem;
        text-align: center;
        margin-bottom: 1.2rem;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'transformed_notes' not in st.session_state:
    st.session_state.transformed_notes = []
if 'dsm_knowledge_base' not in st.session_state:
    st.session_state.dsm_knowledge_base = None
if 'dsm_loaded' not in st.session_state:
    st.session_state.dsm_loaded = False

def create_dsm_knowledge_base():
    """Create a vector knowledge base from DSM-5 manual"""
    try:
        pdf_path = "APA_DSM-5-Contents.pdf"
        if not os.path.exists(pdf_path):
            return f"Error: DSM-5 manual file not found at {pdf_path}"
        
        # Load and extract text from PDF
        with open(pdf_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
        
        # Clean and preprocess text
        text = re.sub(r'\s+', ' ', text)  # Remove extra whitespace
        text = re.sub(r'[^\w\s\.\,\;\:\!\?\-\(\)]', '', text)  # Remove special characters
        
        # Split text into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        
        chunks = text_splitter.split_text(text)
        
        # Create documents
        documents = [Document(page_content=chunk, metadata={"source": "DSM-5"}) for chunk in chunks]
        
        # Create embeddings and vector store
        embeddings = OpenAIEmbeddings(openai_api_key=st.session_state["openai_api_key"])
        vectorstore = FAISS.from_documents(
            documents=documents,
            embedding=embeddings
        )
        
        return vectorstore, len(chunks)
        
    except Exception as e:
        return f"Error creating DSM knowledge base: {str(e)}"

def query_dsm_knowledge(query, top_k=3):
    """Query the DSM knowledge base for relevant information"""
    try:
        if st.session_state.dsm_knowledge_base is None:
            return "DSM knowledge base not loaded"
        
        # Search for relevant content
        results = st.session_state.dsm_knowledge_base.similarity_search(query, k=top_k)
        
        # Combine relevant information
        relevant_info = "\n\n".join([doc.page_content for doc in results])
        return relevant_info
        
    except Exception as e:
        return f"Error querying DSM knowledge base: {str(e)}"

def extract_text_from_pdf(pdf_file):
    """Extract text from uploaded PDF file"""
    try:
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() + "\n"
        return text
    except Exception as e:
        return f"Error extracting text from PDF: {str(e)}"

def transform_note_with_gpt(raw_note, diagnosis, output_style, api_key):
    """Transform raw note using OpenAI API"""
    try:
        # Set OpenAI API key
        openai.api_key = api_key
        
        # Create comprehensive system prompt based on output style
        if output_style == "SOAP":
            system_prompt = """Purpose:
Transform raw, unstructured clinical notes into structured, comprehensive clinical notes following a standard clinical documentation format. The completed notes should maintain all essential content from the raw notes while organizing the information under clear sections and integrating therapeutic interventions and outcomes.

Instructions:

Input Structure:
You will receive a block of raw clinical notes that consist of unstructured, fragmented statements capturing the client's experiences, symptoms, emotional states, and any therapeutic interventions. The clinician may also provide a preliminary diagnosis.

Output Structure:
SOAP (Subjective, Objective, Assessment, Plan) format:
- Subjective: Client's reported experiences and emotional states.
- Objective: Observable behaviors, symptoms, and therapist observations.
- Assessment: Clinical interpretations and diagnostic impressions.
- Plan: Interventions, goals, and follow-up actions.

If a diagnosis is included, use it to:
- Identify the corresponding DSM-5 classification and ICD-10-CM code.
- Locate and highlight supporting content from the raw note that justifies the diagnosis.

Transformation Guidelines:
- Maintain all factual content from the raw notes. Do not omit or alter any reported information.
- Organize the content logically, ensuring a clear flow of information based on the chosen format.
- Reframe emotional expressions into clinically appropriate language without diminishing the client's emotional experiences.
- Include specific numeric data, such as distress levels or duration of symptoms, in the relevant sections.
- Document therapeutic interventions explicitly, specifying the technique used and observed outcomes.
- Link clinical symptoms explicitly to the provided diagnosis where applicable, referencing the DSM-5 and ICD-10-CM.

Tone and Style:
- Use neutral, professional, and clinical language.
- Avoid subjective interpretations or assumptions not stated in the raw notes. Avoid generating new information not stated in the raw notes. For sections not described in the raw notes, it is fine to leave them blank.
- Ensure clarity and coherence, making the document accessible for clinical review and continuity of care."""
        elif output_style == "DAP":
            system_prompt = """Purpose:
Transform raw, unstructured clinical notes into structured, comprehensive clinical notes following a standard clinical documentation format. The completed notes should maintain all essential content from the raw notes while organizing the information under clear sections and integrating therapeutic interventions and outcomes.

Instructions:

Input Structure:
You will receive a block of raw clinical notes that consist of unstructured, fragmented statements capturing the client's experiences, symptoms, emotional states, and any therapeutic interventions. The clinician may also provide a preliminary diagnosis.

Output Structure:
DAP (Data, Assessment, Plan) format:
- Data: Information shared by the client and observed in session.
- Assessment: Clinical impressions and interpretation.
- Plan: Future therapeutic focus and recommendations.

If a diagnosis is included, use it to:
- Identify the corresponding DSM-5 classification and ICD-10-CM code.
- Locate and highlight supporting content from the raw note that justifies the diagnosis.

Transformation Guidelines:
- Maintain all factual content from the raw notes. Do not omit or alter any reported information.
- Organize the content logically, ensuring a clear flow of information based on the chosen format.
- Reframe emotional expressions into clinically appropriate language without diminishing the client's emotional experiences.
- Include specific numeric data, such as distress levels or duration of symptoms, in the relevant sections.
- Document therapeutic interventions explicitly, specifying the technique used and observed outcomes.
- Link clinical symptoms explicitly to the provided diagnosis where applicable, referencing the DSM-5 and ICD-10-CM.

Tone and Style:
- Use neutral, professional, and clinical language.
- Avoid subjective interpretations or assumptions not stated in the raw notes. Avoid generating new information not stated in the raw notes. For sections not described in the raw notes, it is fine to leave them blank.
- Ensure clarity and coherence, making the document accessible for clinical review and continuity of care."""
        else:  # Standard
            system_prompt = """Purpose:
Transform raw, unstructured clinical notes into structured, comprehensive clinical notes following a standard clinical documentation format. The completed notes should maintain all essential content from the raw notes while organizing the information under clear sections and integrating therapeutic interventions and outcomes.

Instructions:

Input Structure:
You will receive a block of raw clinical notes that consist of unstructured, fragmented statements capturing the client's experiences, symptoms, emotional states, and any therapeutic interventions. The clinician may also provide a preliminary diagnosis.

Output Structure:
Standard format:
- Presenting Problem
- Background/History
- Session Content
- Interventions and Outcomes
- Coping Strategies
- Recommendations and Follow-Up

If a diagnosis is included, use it to:
- Identify the corresponding DSM-5 classification and ICD-10-CM code.
- Locate and highlight supporting content from the raw note that justifies the diagnosis.

Transformation Guidelines:
- Maintain all factual content from the raw notes. Do not omit or alter any reported information.
- Organize the content logically, ensuring a clear flow of information based on the chosen format.
- Reframe emotional expressions into clinically appropriate language without diminishing the client's emotional experiences.
- Include specific numeric data, such as distress levels or duration of symptoms, in the relevant sections.
- Document therapeutic interventions explicitly, specifying the technique used and observed outcomes.
- Link clinical symptoms explicitly to the provided diagnosis where applicable, referencing the DSM-5 and ICD-10-CM.

Tone and Style:
- Use neutral, professional, and clinical language.
- Avoid subjective interpretations or assumptions not stated in the raw notes. Avoid generating new information not stated in the raw notes. For sections not described in the raw notes, it is fine to leave them blank.
- Ensure clarity and coherence, making the document accessible for clinical review and continuity of care."""

        # Query DSM knowledge base for relevant information
        dsm_context = ""
        if st.session_state.dsm_loaded and st.session_state.dsm_knowledge_base:
            # Create a query based on the diagnosis and symptoms
            query = f"{diagnosis} {raw_note}"
            relevant_dsm_info = query_dsm_knowledge(query, top_k=3)
            
            if relevant_dsm_info and not relevant_dsm_info.startswith("Error"):
                dsm_context = f"""

Relevant DSM-5 Information:
{relevant_dsm_info}
"""

        user_prompt = f"""
Raw Clinical Notes:
{raw_note}

Clinician's Diagnosis:
{diagnosis}{dsm_context}

Please transform this information into a structured {output_style} note format following the comprehensive guidelines provided. Ensure all content from the raw notes is preserved and organized appropriately. Use the relevant DSM-5 information to enhance diagnostic accuracy and provide appropriate ICD-10-CM codes."""

        # Use OpenAI API to generate response (updated for v1.0.0+)
        client = openai.OpenAI(api_key=st.session_state["openai_api_key"])
        response = client.chat.completions.create(
            model="gpt-4.1-mini-2025-04-14",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            max_tokens=1500,
            temperature=0.3
        )
        
        return response.choices[0].message.content.strip()
    
    except Exception as e:
        return f"Error: {str(e)}"

def save_note_to_history(raw_note, diagnosis, output_style, transformed_note):
    """Save transformed note to session history"""
    note_entry = {
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'raw_note': raw_note,
        'diagnosis': diagnosis,
        'output_style': output_style,
        'transformed_note': transformed_note
    }
    st.session_state.transformed_notes.append(note_entry)

def main():
    # Header
    st.markdown('<div class="main-header">PsyNoteTaker</div>', unsafe_allow_html=True)
    st.markdown("<div style='color:#4f8cff; font-size:1.2rem; margin-bottom:2rem;'>Your Professional AI-Powered Clinical Note Assistant</div>", unsafe_allow_html=True)

    # Sidebar for API key and DSM manual
    with st.sidebar:
        st.image("totoro.jpg", width=100)
        st.markdown("<div class='sidebar-title'>PsyNoteTaker</div>", unsafe_allow_html=True)
        st.markdown("<div class='sidebar-desc'>AI-powered clinical note transformation</div>", unsafe_allow_html=True)
        st.markdown("### 🔑 API Configuration")
        api_key = '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'
        if api_key:
            st.session_state["openai_api_key"] = api_key
            st.success("✅ API key configured")
        else:
            st.warning("Please enter your OpenAI API key to use the app.")
        st.markdown("---")
        st.markdown("### 📚 DSM-5 Knowledge Base")
        
        if st.session_state.dsm_loaded:
            st.success("✅ DSM-5 Knowledge Base loaded")
            st.info("Vector embeddings created for semantic search")
        else:
            st.warning("⚠️ DSM-5 Knowledge Base not initialized")
            if st.button("🔧 Initialize Knowledge Base"):
                with st.spinner("Creating DSM-5 knowledge base..."):
                    result = create_dsm_knowledge_base()
                    if isinstance(result, tuple):
                        vectorstore, chunk_count = result
                        st.session_state.dsm_knowledge_base = vectorstore
                        st.session_state.dsm_loaded = True
                        st.success(f"✅ Knowledge base created with {chunk_count} chunks")
                        st.rerun()
                    else:
                        st.error(result)
        
        st.markdown("---")
        st.markdown("### 📋 Note History")
        if st.session_state.transformed_notes:
            for i, note in enumerate(reversed(st.session_state.transformed_notes)):
                with st.expander(f"Note {len(st.session_state.transformed_notes) - i} - {note['timestamp']}"):
                    st.write(f"**Style:** {note['output_style']}")
                    st.write(f"**Diagnosis:** {note['diagnosis'][:100]}...")
                    if st.button(f"View Full Note {len(st.session_state.transformed_notes) - i}", key=f"view_{i}"):
                        st.text_area("Transformed Note", note['transformed_note'], height=300, key=f"history_{i}", disabled=True)
        else:
            st.info("No notes transformed yet")
    
    # Main content area
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown('<h2 class="section-header">📝 Input</h2>', unsafe_allow_html=True)
        
        # Use a single form for all inputs
        with st.form("transform_form", clear_on_submit=False):
            raw_note = st.text_area(
                "Raw Clinical Notes",
                height=200,
                placeholder="Enter patient symptoms, observations, test results, and any other relevant clinical information...",
                key="raw_note_input"
            )
            diagnosis = st.text_area(
                "Clinician's Diagnosis",
                height=150,
                placeholder="Enter the primary diagnosis, differential diagnoses, and clinical impressions...",
                key="diagnosis_input"
            )
            output_style = st.selectbox(
                "Output Style",
                ["SOAP", "DAP", "Standard"],
                help="SOAP: Subjective, Objective, Assessment, Plan | DAP: Data, Assessment, Plan | Standard: Comprehensive clinical format"
            )
            if output_style == "SOAP":
                st.info("**SOAP Format:** Subjective, Objective, Assessment, Plan")
            elif output_style == "DAP":
                st.info("**DAP Format:** Data, Assessment, Plan")
            else:
                st.info("**Standard Format:** Presenting Problem, Background/History, Session Content, Interventions and Outcomes, Coping Strategies, Recommendations and Follow-Up")
            submitted = st.form_submit_button("🔄 Transform Note", type="primary", disabled=("openai_api_key" not in st.session_state or not st.session_state["openai_api_key"]))
            if submitted:
                if not raw_note or not diagnosis:
                    st.error("Please fill in both the raw notes and diagnosis fields.")
                else:
                    with st.spinner("Transforming note with AI..."):
                        transformed_note = transform_note_with_gpt(
                            raw_note, diagnosis, output_style, st.session_state["openai_api_key"]
                        )
                        if transformed_note.startswith("Error:"):
                            st.error(transformed_note)
                        else:
                            save_note_to_history(raw_note, diagnosis, output_style, transformed_note)
                            st.success("Note transformed successfully!")
    
    with col2:
        st.markdown('<h2 class="section-header">📋 Output</h2>', unsafe_allow_html=True)
        
        # Display transformed note
        if st.session_state.transformed_notes:
            latest_note = st.session_state.transformed_notes[-1]
            st.markdown('<div class="output-box">', unsafe_allow_html=True)
            st.markdown(f"**{latest_note['output_style']} Note**")
            st.markdown(f"*Generated on: {latest_note['timestamp']}*")
            st.text_area(
                "Transformed Note",
                latest_note['transformed_note'],
                height=400,
                key="output_display",
                disabled=True
            )
            
            # Download button
            note_content = f"""
{latest_note['output_style']} Note
Generated on: {latest_note['timestamp']}

{latest_note['transformed_note']}

---
Original Raw Notes: {latest_note['raw_note']}
Original Diagnosis: {latest_note['diagnosis']}
"""
            st.download_button(
                label="📥 Download Note",
                data=note_content,
                file_name=f"clinical_note_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                mime="text/plain"
            )
            st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="output-box">', unsafe_allow_html=True)
            st.info("Transform a note to see the output here")
            st.markdown('</div>', unsafe_allow_html=True)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #4f8cff; font-size:1.1rem;'>
        <p><b>PsyNoteTaker</b> | Powered by OpenAI GPT | Built with Streamlit</p>
        <p>This tool helps healthcare professionals organize raw clinical notes into structured formats.</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main() 
