import streamlit as st
import hashlib
import json
import os

# Path to store trusted hashes
HASH_DB_PATH = "trusted_hashes.json"

# Helper: Calculate SHA-256 hash from file bytes
def calculate_file_hash(file_bytes):
    return hashlib.sha256(file_bytes).hexdigest()

# Helper: Load hash database from JSON
def load_hash_db():
    if os.path.exists(HASH_DB_PATH):
        with open(HASH_DB_PATH, 'r') as f:
            return json.load(f)
    return {}

# Helper: Save updated hash database to JSON
def save_hash_db(data):
    with open(HASH_DB_PATH, 'w') as f:
        json.dump(data, f, indent=4)

# Streamlit UI setup
st.set_page_config(page_title="🔐 File Hash Tool", layout="centered")
st.title("🔐 File Hashing & Verification App")

# Tabs for two functionalities
tab1, tab2 = st.tabs(["📦 Generate & Store Hash", "🔍 Verify File Integrity"])

# ---- Tab 1: Generate & Store Hash ----
with tab1:
    st.header("📦 Upload File to Generate & Save Hash")
    uploaded_file = st.file_uploader("Upload a file to generate its hash and store it in the trusted database", key="gen")

    if uploaded_file:
        file_bytes = uploaded_file.read()
        file_hash = calculate_file_hash(file_bytes)
        file_name = uploaded_file.name

        st.subheader("✅ SHA-256 Hash")
        st.code(file_hash)

        if st.button("💾 Save to Trusted Hash DB"):
            db = load_hash_db()
            db[file_name] = file_hash
            save_hash_db(db)
            st.success(f"Hash for `{file_name}` saved to trusted database.")

# ---- Tab 2: Verify File Against Trusted Hashes ----
with tab2:
    st.header("🔍 Upload File to Verify")
    verify_file = st.file_uploader("Upload a file to verify its hash against the trusted database", key="verify")

    if verify_file:
        file_bytes = verify_file.read()
        file_hash = calculate_file_hash(file_bytes)
        file_name = verify_file.name

        st.subheader("🔍 Calculated SHA-256 Hash")
        st.code(file_hash)

        db = load_hash_db()

        match_found = None
        for trusted_file, trusted_hash in db.items():
            if file_hash == trusted_hash:
                match_found = trusted_file
                break

        if match_found:
            st.success(f"✅ Match found! File matches trusted hash of `{match_found}`.")
        else:
            st.error("❌ No match found in the trusted hash database.")
