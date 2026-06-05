RAG Chatbot for PDF Question Answering

This project is a PDF-based question answering chatbot. It allows users to upload PDF files and ask questions based on the uploaded content. The system retrieves relevant information from the PDFs and generates academic-style answers using an LLM.

Features

The user can upload one or more PDF files through the web interface. The system extracts text from the uploaded PDFs, splits the text into smaller chunks, stores the content in a SQLite database, retrieves relevant chunks using TF-IDF and FAISS, and generates answers using the Groq LLM. It also supports 2-mark, 5-mark, and 10-mark answer formats.

Technology Stack

The project uses Python and Flask for the backend. The frontend is created using HTML, CSS, and JavaScript. SQLite is used for storing uploaded document data and text chunks. PyPDF2 is used for extracting text from PDF files. Scikit-learn is used for TF-IDF vectorization. LangChain is used for document handling, text splitting, prompts, and retrieval-based question answering. FAISS is used for similarity search. LangGraph is used to define the chat workflow. Groq API is used to generate the final answer using the llama-3.1-8b-instant model.

How It Works

First, the user uploads PDF files using the web interface. The backend extracts text from each PDF using PyPDF2. The extracted text is divided into chunks using LangChain’s RecursiveCharacterTextSplitter.

The chunks are stored in a SQLite database. The system then converts the chunks into TF-IDF vectors and indexes them using FAISS.

When the user asks a question, the question is also converted into a TF-IDF vector. The system searches the FAISS index and retrieves the most relevant chunks from the uploaded PDFs. These retrieved chunks are passed as context to the Groq LLM, which generates the final answer.

Chunking Strategy

The project uses RecursiveCharacterTextSplitter for chunking. The chunk size is set to 1000 characters, and the chunk overlap is set to 100 characters. The overlap helps preserve context between nearby chunks, especially when important information is split across two chunks.

Embedding Method

This project uses TF-IDF embeddings created using scikit-learn’s TfidfVectorizer. It does not use OpenAI embeddings or sentence transformer embeddings. TF-IDF is lightweight and does not require an external embedding API, but it is mainly keyword-based.

Retrieval Process

The retrieval process uses FAISS similarity search. For every user question, the system retrieves the top 5 most relevant chunks from the uploaded PDFs. These chunks are then used as context for answer generation.

##Requirements

Install the required Python packages using this command:

pip install -r requirements.txt

Environment Variable

The Groq API key should be stored as an environment variable. The API key should not be written directly in the source code.

For Windows PowerShell:

$env:GROQ_API_KEY="your_api_key_here"

For Linux or macOS:

export GROQ_API_KEY="your_api_key_here"

How to Run

Run the Flask application using this command:

python app.py

Then open the browser and go to:

http://127.0.0.1:5000

Project Structure

RagChatbot/
app.py
requirements.txt
README.md
.gitignore
templates/index.html
          style.css
static/style.css

Assumptions and Limitations

The system assumes that uploaded PDFs contain selectable text. Scanned or image-based PDFs may not work properly because OCR is not included.

The retrieval method is based on TF-IDF, so it works better when the question uses words similar to the PDF content. It may not always understand questions where the meaning is similar but the wording is different.

The FAISS index is rebuilt in memory from the stored chunks. This is suitable for small and medium-sized document collections, but a larger project may require a persistent vector database.

The quality of answers depends on the extracted PDF text, the retrieved chunks, and the availability of the Groq API.

Future Improvements

OCR support can be added for scanned PDFs. Semantic embeddings can be used for better retrieval. User authentication can be added. The FAISS index can be stored persistently. Document management can be improved. Page number references can be added. Error handling for API failures can also be improved.
