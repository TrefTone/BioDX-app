import streamlit as st
import fitz  # PyMuPDF
import csv
import os
import tempfile
import pandas as pd
import numpy as np
from huggingface_hub import login
import torch
# For Gemini API – make sure to install with: pip install --force-reinstall pymupdf google-generativeai
import google.generativeai as genai

# For Keras model loading
from tensorflow.keras.models import load_model

# For LLM inference using Hugging Face Transformers
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM

st.title("BioDX: Report Analysis Chatbot")

# Set your API keys/tokens (could also be set in Streamlit secrets)
API_KEY = st.secrets["api_key"]
HF_KEY = st.secrets["hf_key"]

if API_KEY:
    genai.configure(api_key=API_KEY)
if HF_KEY:
    login(token=HF_KEY)

# Set paths for the pretrained models (ensure these models are available)
MODEL1_PATH = "models/mlp_model.h5"  # Disease classification (model1)
MODEL2_PATH = "models/ckd_lstm_model.h5"  # CKD prediction model (model2)

# Load your pretrained models if they exist
@st.cache_resource
def load_models():
    model1 = None
    model2 = None
    if os.path.exists(MODEL1_PATH):
        model1 = load_model(MODEL1_PATH)
    if os.path.exists(MODEL2_PATH):
        model2 = load_model(MODEL2_PATH)
    return model1, model2

model1, model2 = load_models()

# -------------------------
#  PDF PARSING FUNCTIONS
# -------------------------

def extract_text_from_pdf(pdf_path):
    """Extract text from a PDF file while preserving horizontal structure."""
    doc = fitz.open(pdf_path)
    extracted_text = []
    for page in doc:
        blocks = page.get_text("blocks")  # Extract text in blocks
        sorted_blocks = sorted(blocks, key=lambda b: (b[1], b[0]))  # Sort by vertical then horizontal
        page_text = "\n".join(block[4] for block in sorted_blocks)
        extracted_text.append(page_text)
    return "\n".join(extracted_text)

def structure_text_with_gemini(text):
    """
    Send extracted text to Gemini for tabular structuring and CSV formatting,
    including detailed status analysis.
    """
    # Create a GenerativeModel using Gemini (example model name "gemini-2.0-flash")
    model = genai.GenerativeModel("gemini-2.0-flash")

    prompt = (
        "Extract structured tabular data from the following medical report and format it as CSV with proper headers. "
        "Add a column indicating whether the value is 'Low', 'Low Tendency', 'Normal', 'High Tendency', or 'High'.\n"
        "Classification criteria:\n"
        " - If a value is below the normal range, mark it as 'Low'.\n"
        " - If a value is within 0-10% of the normal range, mark it as 'Low Tendency'.\n"
        " - If a value is between 11-90% of the normal range, mark it as 'Normal'.\n"
        " - If a value is within 91-100% of the normal range, mark it as 'High Tendency'.\n"
        " - If a value exceeds the normal range, mark it as 'High'.\n"
        "Ensure the table structure is maintained correctly.\n\n" + text
    )
    response = model.generate_content(prompt)
    return response.text.strip() if response.text else ""

def save_csv(csv_text, output_csv):
    """Save the structured CSV text into a file with consistent columns."""
    rows = [line.split(",") for line in csv_text.split("\n") if line.strip()]
    max_cols = max(len(row) for row in rows) if rows else 0
    normalized_rows = [row + [""] * (max_cols - len(row)) for row in rows]
    with open(output_csv, "w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerows(normalized_rows)

def pdf_to_csv(pdf_path, output_csv):
    """Complete pipeline: Extract, structure, and save as CSV."""
    text = extract_text_from_pdf(pdf_path)
    # NOTE: Internal details like text preview are not shown to the user.
    csv_text = structure_text_with_gemini(text)
    save_csv(csv_text, output_csv)
    return output_csv

# -------------------------
#  MAPPING FUNCTIONS
# -------------------------

def map_data_for_model1(csv_file):
    """
    Read the CSV produced from the PDF and map its data to the input features for model 1.
    If some values are missing, use safe default values.
    """
    defaults = {
        "Glucose (mg/dL)": 99,
        "Cholesterol (Total) (mg/dL)": 179,
        "Hemoglobin (g/dL)": 14.1,
        "Platelets (/µL × 1000)": 310,
        "White Blood Cells (/µL)": 7.7,
        "Red Blood Cells (mill/µL)": 4.43,
        "Hematocrit (%)": 39.9,
        "Mean Corpuscular Volume (fL)": 86.3,
        "Mean Corpuscular Hemoglobin (pg)": 29.4,
        "Mean Corpuscular Hemoglobin Conc. (g/dL)": 34.6,
        "Insulin (µIU/mL, scaled)": 0.44,
        "BMI (scaled)": 0.44,
        "Systolic BP (scaled)": 0.36,
        "Diastolic BP (scaled)": 0.47,
        "Triglycerides (mg/dL)": 114,
        "HbA1c (%)": 5.7,
        "LDL Cholesterol (mg/dL)": 112,
        "HDL Cholesterol (mg/dL)": 61,
        "ALT (U/L)": 25,
        "AST (U/L)": 24.6,
        "Heart Rate (bpm)": 117,
        "Creatinine (mg/dL)": 0.92,
        "Troponin (ng/L)": 0.17,
        "C-reactive Protein (mg/L)": 4.8
    }
    try:
        df = pd.read_csv(csv_file)
        model1_input = { key: df[key].iloc[0] if key in df.columns and not pd.isna(df[key].iloc[0]) else default
                         for key, default in defaults.items() }
    except Exception as e:
        st.error("Error reading CSV: " + str(e))
        model1_input = defaults
    return model1_input

def map_data_for_model2(model1_results):
    """
    Use the result from model 1 to create inputs for model 2 (CKD prediction).
    Safe default values for model 2 parameters:
    """
    defaults = {
        "blood_glucose_random": 121,
        "blood_urea": 42,
        "serum_creatinine": 1.3,
        "sodium": 138,
        "potassium": 4.4,
        "hemoglobin": 12.9,
        "packed_cell_volume": 41,
        "white_blood_cell_count": 8250,
        "red_blood_cell_count": 4.8,
        "sugar": 0,  # mode & median normal sugar level
        "blood_pressure": 80
    }
    diabetic_flag = 1 if "diabetic" in model1_results.lower() else 0
    anemia_flag = 1 if "anemia" in model1_results.lower() else 0
    defaults["diabetes_mellitus"] = diabetic_flag
    defaults["anemia"] = anemia_flag
    return defaults

# -------------------------
#  LLM ASSESSMENT FUNCTION
# -------------------------

@st.cache_resource(show_spinner=False)
def load_llm_model(token):
    """
    Load the conversational LLM.
    Ensure you have your access token.
    """
    model_name = "samirangupta31/meditron-7b-finetuned-quantized"
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=token)
    model = AutoModelForCausalLM.from_pretrained(model_name, use_auth_token=token)
    llm_pipeline = pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=200)
    return llm_pipeline

def generate_assessment(mapped_data, model1_results, model2_results, llm_pipeline):
    """
    Generate a textual initial assessment based on the data and model outputs.
    """
    prompt = (
        "Based on the following medical report summary, please provide an initial assessment. \n"
        "Report Summary (with parameters, using default values for any missing):\n"
        f"{mapped_data}\n\n"
        "Disease Classification (Model 1): " + model1_results + "\n"
        "Chronic Kidney Disease Risk (Model 2) based on diabetes and anemia flags: " + str(model2_results) + "\n\n"
        "Please provide a concise explanation highlighting any abnormalities and suggestions for follow-up."
    )
    response = llm_pipeline(prompt)[0]['generated_text']
    return response

# -------------------------
#  STREAMLIT CHATBOT INTERFACE
# -------------------------

st.subheader("Upload Blood Report PDF")
uploaded_file = st.file_uploader("Choose a PDF file", type=["pdf"])

# Initialize session state for chat history and processing status
if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []  # Each element is a tuple: (sender, message)
if "report_ready" not in st.session_state:
    st.session_state["report_ready"] = False
if "llm_pipeline" not in st.session_state:
    # Load LLM pipeline; token is required from HF_KEY
    st.session_state["llm_pipeline"] = load_llm_model(HF_KEY)

# Process the file only once
if uploaded_file is not None and not st.session_state["report_ready"]:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.read())
        pdf_path = tmp_file.name

    # Process PDF without showing intermediate details
    csv_output_path = "output.csv"
    pdf_to_csv(pdf_path, csv_output_path)

    # Map CSV data to model1 input parameters
    model1_input = map_data_for_model1(csv_output_path)

    # Run Model 1 (simulate if model not available)
    if model1 is not None:
        model1_features = np.array([v for k, v in model1_input.items()], dtype=float).reshape(1, -1)
        pred_prob = model1.predict(model1_features)[0][0]
        disease_class = "diabetic" if pred_prob > 0.7 else ("anemia" if pred_prob >= 0.5 else "healthy")
    else:
        disease_class = "healthy (default simulated)"

    # Map data for Model 2 (CKD prediction) using model1 result
    model2_input = map_data_for_model2(disease_class)

    if model2 is not None:
        model2_features = np.array([v for k, v in model2_input.items()], dtype=float).reshape(1, -1)
        ckd_prob = model2.predict(model2_features)[0][0]
        ckd_prediction = "CKD prone" if ckd_prob >= 0.5 else "Not CKD prone"
    else:
        ckd_prediction = "Not CKD prone (default simulated)"

    # Generate initial assessment report using the LLM
    initial_report = generate_assessment(model1_input, disease_class, ckd_prediction, st.session_state["llm_pipeline"])

    # Save the initial report in the chat history
    st.session_state["chat_history"].append(("bot", initial_report))
    st.session_state["report_ready"] = True

# Display Chat History
st.markdown("### Chat")
for sender, message in st.session_state["chat_history"]:
    if sender == "bot":
        st.markdown(f"**Bot:** {message}")
    else:
        st.markdown(f"**You:** {message}")

# Chat input area for follow-up queries
chat_input = st.text_input("Enter your message here...", key="user_input")
if st.button("Send", key="send_button") and chat_input:
    # Append the user input to chat history
    st.session_state["chat_history"].append(("user", chat_input))

    # Create conversation context (you may refine this prompt as needed)
    conversation_context = "\n".join(f"{sender}: {msg}" for sender, msg in st.session_state["chat_history"])

    # Generate bot reply using the LLM pipeline
    llm_reply = st.session_state["llm_pipeline"](conversation_context)[0]['generated_text']
    st.session_state["chat_history"].append(("bot", llm_reply))
    st.experimental_rerun()
