import os
import hashlib
import tempfile

import streamlit as st
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet

from dotenv import load_dotenv
from openai import OpenAI
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams
from langchain_google_genai import GoogleGenerativeAIEmbeddings

# ---------------------------------------------------------------------------
# ENV / CONFIG
# ---------------------------------------------------------------------------
load_dotenv()

# NEVER hardcode API keys in source. Put GOOGLE_API_KEY in a local .env file
# (which should be in .gitignore) or in your deployment environment's secrets.
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

COLLECTION_NAME = "my_documents"
EMBEDDING_MODEL_NAME = "gemini-embedding-2-preview"
CHAT_MODEL_NAME = "gemini-3.6-flash"
VECTOR_SIZE = 3072

st.set_page_config(page_title="Hospital Assistant", layout="wide")

# ------------------ GENAI / QDRANT CLIENTS (as given) ------------------
llm_client = OpenAI(
    api_key=GOOGLE_API_KEY,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

qdrant_client = QdrantClient(
    url="http://localhost:6333/"
)

embedding_model = GoogleGenerativeAIEmbeddings(
    model=EMBEDDING_MODEL_NAME,
    google_api_key=GOOGLE_API_KEY
)

# ------------------ SESSION STATE ------------------
if "patients" not in st.session_state:
    st.session_state.patients = []

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "indexed_signature" not in st.session_state:
    st.session_state.indexed_signature = None

if "rag_ready" not in st.session_state:
    st.session_state.rag_ready = False

# ------------------ DARK MODE TOP BAR (RIGHT SIDE TOGGLE) ------------------
col_title, col_toggle = st.columns([8, 1])

with col_title:
    st.markdown("<h1 style='margin-bottom:0;'>🏥 Global Hospitals</h1>", unsafe_allow_html=True)

with col_toggle:
    st.markdown("<div style='margin-top: 25px;'>", unsafe_allow_html=True)
    dark_mode = st.toggle("🌙", help="Toggle Dark Mode")
    st.markdown("</div>", unsafe_allow_html=True)

# ------------------ COLORFUL UI ------------------
if dark_mode:
    st.markdown("""
    <style>
    .stApp {
        background: linear-gradient(135deg, #1e1e2f, #2b5876, #4e4376);
        color: #f5f5f5;
    }
    h1, h2, h3, h4, h5, h6, p, label, span {
        color: #f5f5f5 !important;
    }
    .stTextInput input, .stNumberInput input {
        background-color: #2c2c3e;
        color: white;
        border-radius: 10px;
    }
    .stButton>button {
        background: linear-gradient(135deg, #6a11cb, #2575fc);
        color: white;
        border-radius: 10px;
    }
    .block-container {
        background: rgba(255,255,255,0.05);
        border-radius: 20px;
        padding: 2rem;
    }
    </style>
    """, unsafe_allow_html=True)
else:
    st.markdown("""
    <style>
    .stApp {
        background: linear-gradient(135deg, #e3f2fd, #d6e4f0, #cfd9df);
    }
    .block-container {
        padding: 2rem;
        border-radius: 20px;
        background: rgba(255,255,255,0.6);
    }
    .stButton>button {
        background: linear-gradient(135deg, #5a7fa6, #2f4f6f);
        color: white;
        border-radius: 10px;
    }
    </style>
    """, unsafe_allow_html=True)


# ------------------ QDRANT COLLECTION HELPERS ------------------
def ensure_collection():
    existing = [c.name for c in qdrant_client.get_collections().collections]
    if COLLECTION_NAME not in existing:
        qdrant_client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
        )


def reset_collection():
    """Wipe and recreate the collection so each newly uploaded file starts clean."""
    existing = [c.name for c in qdrant_client.get_collections().collections]
    if COLLECTION_NAME in existing:
        qdrant_client.delete_collection(COLLECTION_NAME)
    qdrant_client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
    )


# ------------------ PDF FUNCTION ------------------
def create_pdf(text):
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
    doc = SimpleDocTemplate(temp_file.name)
    styles = getSampleStyleSheet()

    content = []
    for line in text.split("\n"):
        content.append(Paragraph(line, styles["Normal"]))
        content.append(Spacer(1, 10))

    doc.build(content)
    return temp_file.name


# ------------------ PARSE DATA ------------------
def parse_multiple_patients(text):
    patients = []
    current = {}

    for line in text.split("\n"):
        line = line.strip()

        if not line:
            if current:
                patients.append(current)
                current = {}
            continue

        if ":" in line:
            key, value = line.split(":", 1)
            current[key.strip().lower()] = value.strip()

    if current:
        patients.append(current)

    return patients


# ------------------ RAG INDEXING ------------------
def build_documents_from_patients(patients):
    """Turn each parsed patient record into one retrievable Document."""
    docs = []
    for p in patients:
        lines = [f"{k}: {v}" for k, v in p.items()]
        content = "\n".join(lines)
        docs.append(Document(page_content=content, metadata={"patient_name": p.get("patient name", "Unknown")}))
    return docs


def index_documents(raw_text, patients):
    """Embed the uploaded patient TEXT data (patient-level chunks + raw text chunks) into Qdrant."""
    if not GOOGLE_API_KEY:
        st.error("GOOGLE_API_KEY is not set. Add it to a .env file or your environment to enable the AI chatbot.")
        return False

    # Build documents: one per patient (clean, structured) plus raw-text chunks as a safety net.
    docs = build_documents_from_patients(patients)

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=200)
    docs += splitter.create_documents([raw_text])

    if not docs:
        return False

    with st.spinner("Embedding and indexing patient data..."):
        reset_collection()

        vector_store = QdrantVectorStore(
            client=qdrant_client,
            collection_name=COLLECTION_NAME,
            embedding=embedding_model,
        )
        vector_store.add_documents(docs)

    return True


def rag_chatbot_response(question, k=4):
    """Answer a question using retrieved context + a Gemini chat completion."""
    if not GOOGLE_API_KEY:
        return "AI chatbot is not configured: set GOOGLE_API_KEY in your environment first."

    if not st.session_state.rag_ready:
        return "No data indexed yet. Please upload a patient data (.txt) file first."

    vector_store = QdrantVectorStore(
        client=qdrant_client,
        collection_name=COLLECTION_NAME,
        embedding=embedding_model,
    )

    found_docs = vector_store.similarity_search(question, k=k)
    context = "\n\n---\n\n".join(d.page_content for d in found_docs)

    system_prompt = (
        "You are a hospital data assistant OF Global Hospitals. Answer ONLY using the context below. Answers should be concise and factual." \
        "Example 1: Input: What is date of discharge for patient John Doe? Output: The date of discharge for patient John Doe is 2024-06-15." \
        "Example 2: Input: What is the treatment for patient Ali Khan? Output: The treatment for patient Ali Khan is chemotherapy." \
        "Example 3: Input: How many days did patient Ravi Sharma stay? Output: Patient Ravi Sharma stayed for 5 days." \
        "Example 4: Input: What is the follow-up advice for patient Priya Singh? Output: The follow-up advice for patient Priya Singh is to schedule a check-up in 2 weeks." \
        "Example 5: Input: What is the condition at discharge for patient Anil Kumar? Output: The condition at discharge for patient Anil Kumar is stable." \
        "Example 6: Input: When will the patient be discharged? Output: The patient will be discharged after 2 days on 2024-06-17." \
        "If the answer is not in the uploaded document, say you don't have that information.\n\n"
        f"Context:\n{context}"
    )

    try:
        response = llm_client.chat.completions.create(
            model=CHAT_MODEL_NAME,
            reasoning_effort="low",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": question},
            ],
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Sorry, the AI request failed: {e}"


# ------------------ TABS ------------------
tab1, tab2, tab3 = st.tabs(["📂 Upload", "📄 Summary", "🤖 Chat"])

# ================== UPLOAD ==================
with tab1:
    st.subheader("Upload Patient Data")

    if not GOOGLE_API_KEY:
        st.warning(
            "GOOGLE_API_KEY is not set, so the AI chatbot in the Chat tab won't work yet. "
            "Add it to a `.env` file (GOOGLE_API_KEY=your_key) or your environment variables."
        )

    uploaded_file = st.file_uploader("Upload .txt file", type=["txt"])

    if uploaded_file:
        content = uploaded_file.read().decode("utf-8")
        st.text_area("File Content", content, height=200)

        st.session_state.patients = parse_multiple_patients(content)
        st.success("✅ Data Loaded Successfully")

        # Only re-embed if this exact file hasn't already been indexed
        # (Streamlit reruns this block on every interaction otherwise).
        signature = hashlib.sha256(content.encode("utf-8")).hexdigest()
        if signature != st.session_state.indexed_signature:
            ok = index_documents(content, st.session_state.patients)
            if ok:
                st.session_state.indexed_signature = signature
                st.session_state.rag_ready = True
                st.success("🤖 Document indexed for the AI chatbot")
        else:
            st.info("This file is already indexed for the AI chatbot.")

# ================== SUMMARY ==================
with tab2:
    st.subheader("Generate Discharge Summary")

    patients = st.session_state.get("patients", [])
    selected_patient = None

    if patients:
        names = [p.get("patient name") for p in patients]
        selected_name = st.selectbox("Select Patient", names)

        selected_patient = next(
            (p for p in patients if p.get("patient name") == selected_name),
            None
        )
    else:
        st.warning("⚠️ No patient data available. Please upload data first.")

    col1, col2 = st.columns(2)

    with col1:
        name = st.text_input(
            "Patient Name",
            value=selected_patient.get("patient name", "") if selected_patient else ""
        )

        age = st.number_input(
            "Age",
            0,
            120,
            value=int(selected_patient.get("age", 0)) if selected_patient else 0
        )

        admission_date = st.text_input(
            "Date of Admission",
            value=selected_patient.get("date of admission", "") if selected_patient else ""
        )

        status = st.selectbox(
            "Current Status",
            ["Admitted", "Under Diagnosis", "Discharged"],
            index=["Admitted","Under Diagnosis","Discharged"].index(
                selected_patient.get("current status","Discharged")
            ) if selected_patient else 2
        )

    with col2:

        diagnosis = st.text_input(
            "Diagnosis",
            value=selected_patient.get("diagnosis", "") if selected_patient else ""
        )

        treatment = st.text_input(
            "Treatment",
            value=selected_patient.get("treatment", "") if selected_patient else ""
        )

        days = st.number_input(
            "Hospital Stay (Days)",
            1,
            100,
            value=int(selected_patient.get("days", 1)) if selected_patient else 1
        )

        follow_up = st.text_input(
            "Follow-up Advice",
            value=selected_patient.get("follow-up advice", "") if selected_patient else ""
        )

    # Date field changes according to status

    if status == "Discharged":
        discharge_date = st.text_input(
            "Discharge Date",
            value=selected_patient.get("discharge date", "") if selected_patient else ""
        )
        expected_date = ""

    else:
        expected_date = st.text_input(
            "Expected Discharge Date",
            value=selected_patient.get("expected date of discharge", "") if selected_patient else ""
        )
        discharge_date = ""

    if st.button("Generate Summary"):

        date_info = ""

        if status == "Discharged":
            date_info = f"Discharge Date: {discharge_date}"

        else:
            date_info = f"Expected Discharge Date: {expected_date}"

        summary = f"""
GLOBAL HOSPITALS
------------------------------------------------------------
DISCHARGE SUMMARY

Patient Name          : {name}
Age                   : {age}

Date of Admission     : {admission_date}
{date_info}

Current Status        : {status}

DIAGNOSIS             : {diagnosis}

TREATMENT             : {treatment}

HOSPITAL COURSE
---------------
The patient was admitted for {days} day(s) and received appropriate medical care.

CONDITION AT DISCHARGE
---------
{"Patient is stable and discharged." if status=="Discharged" else "Patient is currently under observation and treatment."}

Follow-up Advice
----------------
{follow_up if follow_up else "Regular medication and follow-up after 2 weeks."}

------------------------------------------------------------
GLOBAL HOSPITALS

Thank you for choosing GLOBAL HOSPITALS. We wish you a speedy recovery and good health.

For emergency assistance contact:
+91 XXXXX XXXXX

"""

        st.text_area("Summary", summary, height=500)

        pdf = create_pdf(summary)

        with open(pdf, "rb") as f:
            st.download_button(
                "📥 Download PDF",
                f,
                "Discharge_Summary.pdf"
            )

# ================== CHAT (GenAI RAG) ==================
with tab3:

    st.subheader("Chat Assistant Bot")

    if st.session_state.rag_ready:
        st.caption("Answers are generated by Gemini, grounded in the patient data you uploaded.")
    else:
        st.caption("Upload a patient data file in the Upload tab to enable AI-grounded answers.")

    question = st.text_input("Type your question about patients here...")

    if st.button("Ask"):
        if question.strip():
            with st.spinner("Thinking..."):
                answer = rag_chatbot_response(question)

            st.session_state.chat_history.append(("You", question))
            st.session_state.chat_history.append(("Bot", answer))

    for sender, msg in st.session_state.get("chat_history", []):
        if sender == "You":
            st.markdown(f"""
            <div style='background:#4facfe;padding:10px;border-radius:10px;margin:5px;color:white'>
            Admin: {msg}
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div style='background:#43e97b;padding:10px;border-radius:10px;margin:5px;color:black'>
            Bot: {msg}
            </div>
            """, unsafe_allow_html=True)
            
