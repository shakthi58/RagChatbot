from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import PyPDF2
import re
import os
import sqlite3
from datetime import datetime
from sklearn.feature_extraction.text import TfidfVectorizer
from groq import Groq
from langchain_core.documents import Document
from langchain.embeddings.base import Embeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_classic.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate
from langchain_core.language_models.llms import LLM
from langgraph.graph import StateGraph
from typing import TypedDict

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(BASE_DIR, 'static')
TEMPLATES_DIR = os.path.join(BASE_DIR, 'templates')

app = Flask(__name__, static_folder=STATIC_DIR,
            static_url_path='/static', template_folder=TEMPLATES_DIR)
CORS(app)

client = Groq(
    api_key=os.getenv("GROQ_API_KEY"))

BASE_DIR = os.path.dirname(__file__)
DB_PATH = os.path.join(BASE_DIR, "pdf_data.db")

faiss_store = None
embeddings = None
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
chat_graph = None


class TfidfEmbeddings(Embeddings):
    def __init__(self):
        self.vectorizer = TfidfVectorizer(
            stop_words="english", max_features=5000)
        self._is_fitted = False

    def embed_documents(self, texts):
        if not self._is_fitted:
            matrix = self.vectorizer.fit_transform(texts)
            self._is_fitted = True
        else:
            matrix = self.vectorizer.transform(texts)
        return matrix.toarray().tolist()

    def embed_query(self, text):
        if not self._is_fitted:
            self.embed_documents([text])
        vector = self.vectorizer.transform([text])
        return vector.toarray()[0].tolist()

    @property
    def embeddings_kwargs(self):
        return {}


class GroqLLM(LLM):
    @property
    def _llm_type(self):
        return "groq"

    def _call(self, prompt, stop=None):
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system", "content": "You are an expert academic assistant."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3
        )
        return response.choices[0].message.content.strip()


def get_db_connection():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    conn = get_db_connection()
    with conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS documents (
                id INTEGER PRIMARY KEY,
                filename TEXT,
                uploaded_at TEXT,
                text TEXT
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS chunks (
                id INTEGER PRIMARY KEY,
                document_id INTEGER,
                chunk_text TEXT,
                FOREIGN KEY(document_id) REFERENCES documents(id)
            )
            """
        )
    conn.close()


def load_index():
    global chat_graph
    conn = get_db_connection()
    rows = conn.execute("SELECT chunk_text FROM chunks").fetchall()
    conn.close()

    chunks = [row["chunk_text"] for row in rows]
    if chunks:
        build_vector_index(chunks)
        chat_graph = build_chat_graph()
    else:
        global faiss_store, embeddings
        faiss_store = None
        embeddings = None
        chat_graph = None


# ---------------- PDF TEXT EXTRACTION ----------------
def extract_pdf_text(file):
    reader = PyPDF2.PdfReader(file)
    text = ""
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"
    return text


def split_pdf_text(text):
    return splitter.split_text(text)


def build_vector_index(chunks):
    global faiss_store, embeddings
    documents = [Document(page_content=chunk, metadata={
                          "source": "pdf"}) for chunk in chunks]
    embeddings = TfidfEmbeddings()
    faiss_store = FAISS.from_documents(documents, embeddings)


def generate_answer(question, context_text, marks):
    marks_instruction = ""
    if marks == 2:
        marks_instruction = "Provide a concise answer suitable for a 2-mark question (short definition or brief explanation)."
    elif marks == 5:
        marks_instruction = "Provide a point-wise explanation suitable for a 5-mark question."
    elif marks == 10:
        marks_instruction = "Provide a detailed answer suitable for a 10-mark question (definition, explanation, and importance)."
    else:
        marks_instruction = "Provide a comprehensive answer."

    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template=f"""
Use the context below to answer the question.

Context:
{{context}}

Question:
{{question}}

Instructions:
- {marks_instruction}
- Write a clear, professional answer in plain text.
- Use textual headings and paragraphs, not markdown symbols.
- Do not show any bullet characters such as *, -, or #.
- If you need to describe components, write them as sentences or numbered sections.
- Maintain an academic tone.
"""
    )

    qa = RetrievalQA.from_chain_type(
        llm=GroqLLM(),
        chain_type="stuff",
        retriever=faiss_store.as_retriever(search_kwargs={"k": 5}),
        chain_type_kwargs={"prompt": prompt}
    )

    result = qa.invoke({"query": question})
    return result.get("result", str(result))


def build_chat_graph():
    class ChatState(TypedDict, total=False):
        query: str
        retrieved_docs: list[str]
        answer: str

    graph = StateGraph(state_schema=ChatState)

    def retrieve_node(state, runtime):
        if not faiss_store:
            return {"retrieved_docs": []}
        docs = faiss_store.similarity_search(state["query"], k=5)
        return {"retrieved_docs": [doc.page_content for doc in docs]}

    def answer_node(state, runtime):
        query = state["query"]

        # Check if it's just a casual greeting
        if is_casual_greeting(query):
            return {"answer": generate_greeting_response(query)}

        context_text = "\n\n".join(state.get("retrieved_docs", []))
        marks = detect_marks(query)
        if not context_text:
            return {"answer": "No uploaded documents found. Please upload PDF files first."}
        return {"answer": generate_answer(query, context_text, marks)}

    graph.add_node("retrieve", retrieve_node)
    graph.add_node("answer", answer_node)
    graph.add_edge("retrieve", "answer")
    graph.set_entry_point("retrieve")
    graph.set_finish_point("answer")
    return graph.compile()


def detect_marks(question):
    if re.search(r"\b10\s*marks?\b", question, re.I):
        return 10
    if re.search(r"\b5\s*marks?\b", question, re.I):
        return 5
    if re.search(r"\b2\s*marks?\b", question, re.I):
        return 2
    return 5


def is_casual_greeting(query):
    """Detect if the query is a casual greeting rather than a substantive question."""
    casual_patterns = [
        r"^\s*(hi|hello|hey|greetings?|what's up|sup|howdy)\s*$",
        r"^\s*how\s+are\s+you\s*\??",
        r"^\s*what's\s+your\s+name\s*\??",
        r"^\s*who\s+are\s+you\s*\??",
        r"^\s*good\s+(morning|afternoon|evening|night)\s*$",
    ]
    query_lower = query.lower().strip()
    return any(re.match(pattern, query_lower, re.I) for pattern in casual_patterns)


def generate_greeting_response(query: str) -> str:
    """Use the LLM to generate a short, varied greeting that does not access documents.

    The response should be conversational and invite the user to ask about uploaded PDFs.
    """
    system_prompt = (
        "You are a friendly, conversational assistant. Keep replies short, varied, and inviting. "
        "Do NOT retrieve or reference any document contents; this is only a brief greeting."
    )
    user_prompt = (
        f"The user said: \"{query}\".\nRespond with a brief (1-2 sentence) friendly greeting that invites the user to ask questions about uploaded PDF documents. "
        "Vary the phrasing across requests and avoid repeating the exact same sentence."
    )

    resp = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.7,
    )

    return resp.choices[0].message.content.strip()


# ---------------- ROUTES ----------------
@app.route("/upload", methods=["POST"])
def upload_pdf():
    files = request.files.getlist("files")
    if not files or all(file.filename == "" for file in files):
        return jsonify({"error": "No PDF files provided"}), 400

    added_docs = []
    skipped_docs = []

    conn = get_db_connection()
    with conn:
        for file in files:
            if not file or file.filename == "":
                continue

            text = extract_pdf_text(file)
            if not text.strip():
                skipped_docs.append(file.filename)
                continue

            exists = conn.execute(
                "SELECT 1 FROM documents WHERE text = ? LIMIT 1",
                (text,)
            ).fetchone()
            if exists:
                skipped_docs.append(file.filename)
                continue

            uploaded_at = datetime.utcnow().isoformat() + "Z"
            cur = conn.execute(
                "INSERT INTO documents (filename, uploaded_at, text) VALUES (?, ?, ?)",
                (file.filename, uploaded_at, text)
            )
            document_id = cur.lastrowid
            chunks = split_pdf_text(text)
            conn.executemany(
                "INSERT INTO chunks (document_id, chunk_text) VALUES (?, ?)",
                [(document_id, chunk) for chunk in chunks]
            )
            added_docs.append(file.filename)

    total_chunks = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
    conn.close()
    load_index()

    return jsonify({
        "message": "PDF(s) uploaded and indexed successfully",
        "added_documents": added_docs,
        "skipped_documents": skipped_docs,
        "total_chunks": total_chunks
    })


@app.route("/documents", methods=["GET"])
def list_documents():
    conn = get_db_connection()
    rows = conn.execute(
        "SELECT id, filename, uploaded_at FROM documents ORDER BY uploaded_at DESC"
    ).fetchall()
    conn.close()

    return jsonify([{
        "id": row["id"],
        "filename": row["filename"],
        "uploaded_at": row["uploaded_at"]
    } for row in rows])


@app.route("/clear", methods=["POST"])
def clear_database():
    """Delete all documents and chunks from the SQLite DB and reset the in-memory index."""
    conn = get_db_connection()
    with conn:
        conn.execute("DELETE FROM chunks")
        conn.execute("DELETE FROM documents")
        conn.commit()
    conn.close()

    # Reset in-memory index and graph
    global faiss_store, embeddings, chat_graph
    faiss_store = None
    embeddings = None
    chat_graph = None

    return jsonify({"message": "Database cleared and index reset."})


@app.route("/chat", methods=["POST"])
def chat():
    if not faiss_store:
        return jsonify({"answer": "Please upload at least one PDF first."})

    data = request.get_json()
    question = data.get("question", "").strip()
    if not question:
        return jsonify({"answer": "Please provide a question."}), 400

    if chat_graph is None:
        return jsonify({"answer": "Chat workflow is not initialized yet."}), 500

    state = chat_graph.invoke({"query": question})
    return jsonify({
        "answer": state.get("answer", "Could not generate an answer."),
        "marks_detected": detect_marks(question)
    })


@app.route("/")
def home():
    return render_template("index.html")


if __name__ == "__main__":
    init_db()
    load_index()
    app.run(debug=True)
