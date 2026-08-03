#  AI Discharge Summary & RAG Assistant

A Streamlit application for hospital administrators to upload patient records, generate discharge summary PDFs, and retrieve patient information through a Retrieval-Augmented Generation (RAG) chatbot powered by **Google Gemini**, **LangChain**, and **Qdrant**.

---

#  Features

## Upload Patient Records
- Upload patient records in `.txt` format.
- Automatically parses multiple patient records.
- Converts records into LangChain Documents.
- Creates embeddings using Gemini Embeddings.
- Stores vectors inside Qdrant for semantic retrieval.

## Discharge Summary
- Select any uploaded patient.
- Automatically populate patient details.
- Generate a formatted discharge summary.
- Download the summary as a PDF.

## AI Chatbot
- Ask questions in natural language.
- Retrieves relevant patient records using semantic similarity search.
- Uses Gemini to answer only from retrieved context.
- Displays responses in a conversational chat interface.

## 🌙 User Interface
- Light and Dark mode support.
- Responsive Streamlit interface.
- Simple and intuitive workflow.

---

# Tech Stack

| Category | Technology |
|----------|------------|
| Frontend | Streamlit |
| Styling | HTML + CSS (via Streamlit Markdown) |
| Backend | Python |
| LLM | Google Gemini 3.6 Flash |
| Embeddings | Gemini Embedding 2 Preview |
| RAG Framework | LangChain |
| Vector Database | Qdrant |
| Document Processing | LangChain Text Splitters |
| PDF Generation | ReportLab |
| Configuration | Python Dotenv |

---

# 📌 Project Workflow

## 1. Upload Patient Data

- Upload a `.txt` file containing patient records.
- Parse the uploaded text into structured patient dictionaries.
- Convert each patient into a LangChain Document.
- Split long text into chunks.
- Generate embeddings using Gemini Embeddings.
- Store embeddings in the Qdrant vector database.

---

## 2. Generate Discharge Summary

- Select a patient.
- Edit details if required.
- Generate a structured discharge summary.
- Export the summary as a PDF.

---

## 3. AI Chatbot

- User asks a question.
- Query is embedded using Gemini Embeddings.
- LangChain performs semantic similarity search on Qdrant.
- Most relevant patient records are retrieved.
- Retrieved context is passed to Gemini.
- Gemini generates the final response.
- Response is displayed in the chat interface.

---

#  Application Flow

```text
                Upload Patient File
                        │
                        ▼
             Parse Patient Records
                        │
        ┌───────────────┴───────────────┐
        │                               │
        ▼                               ▼
 Summary Generation           LangChain Documents
        │                               │
        ▼                               ▼
   PDF Generation          Gemini Embeddings
                                        │
                                        ▼
                                Qdrant Vector DB
                                        │
                             Similarity Search
                                        │
                                        ▼
                               Retrieved Context
                                        │
                                        ▼
                          Gemini (RAG Answer Generation)
                                        │
                                        ▼
                                AI Chat Response
```

# 📌 Current Limitations

- Supports only `.txt` patient records.
- Qdrant collection is refreshed whenever a new file is uploaded.
- Chatbot answers only from retrieved context.
- Does not currently support PDF, DOCX, or CSV uploads.
- Authentication and user management are not implemented.

---

# Future Enhancements

- FastAPI backend integration
- User authentication
- Multi-file uploads
- PDF and DOCX parsing
- ChromaDB support
- Advanced citation display
- Conversation memory
- Role-based access control

---

# Author

**Farheena F**

## 📄 License

This project is developed for educational and research purposes.
