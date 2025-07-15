from flask import Flask, render_template, request
from langchain.text_splitter import CharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_pinecone import PineconeVectorStore
from langchain.chains import RetrievalQA
from pinecone import Pinecone, ServerlessSpec

from langchain_community.document_loaders import TextLoader, PyPDFLoader, UnstructuredWordDocumentLoader
from langchain_core.documents import Document
import os, tempfile

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = tempfile.gettempdir()

# API keys
GOOGLE_API_KEY = "AIzaSyD5-q5l9Y4jnJVXzDcnkP2Fb40HWSoWgEc"
PINECONE_API_KEY = "pcsk_57f7nT_6c3BSyWxGjq8KGZWVsprv68Hq2tpBqFsK5j4rtfGncisJxbp5aMzYCXMgCzYuTr"
PINECONE_INDEX = "gemini-chat-index"
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY

# Pinecone setup
pc = Pinecone(api_key=PINECONE_API_KEY)
if PINECONE_INDEX not in pc.list_indexes().names():
    pc.create_index(
        name=PINECONE_INDEX,
        dimension=768,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )

# Embeddings + LLM
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=GOOGLE_API_KEY)
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0, google_api_key=GOOGLE_API_KEY)

# File Loader Logic
def load_document(file_path, file_ext):
    if file_ext == ".txt":
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read()
        return [Document(page_content=content)]
    elif file_ext == ".pdf":
        return PyPDFLoader(file_path).load()
    elif file_ext == ".docx":
        return UnstructuredWordDocumentLoader(file_path).load()
    else:
        raise ValueError("Unsupported file format.")

# Main Route
@app.route("/", methods=["GET", "POST"])
def index():
    response = None
    sources = []

    if request.method == "POST":
        query = request.form["query"]
        uploaded_file = request.files["file"]

        if uploaded_file:
            file_ext = os.path.splitext(uploaded_file.filename)[1].lower()
            temp_path = os.path.join(app.config['UPLOAD_FOLDER'], uploaded_file.filename)
            uploaded_file.save(temp_path)

            try:
                docs = load_document(temp_path, file_ext)
                splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
                split_docs = splitter.split_documents(docs)

                # Create Vector Store
                vectorstore = PineconeVectorStore.from_documents(
                    documents=split_docs,
                    embedding=embeddings,
                    index_name=PINECONE_INDEX,
                )

                # RAG QA Chain
                qa_chain = RetrievalQA.from_chain_type(
                    llm=llm,
                    retriever=vectorstore.as_retriever(search_kwargs={"k": 2}),
                    return_source_documents=True
                )

                result = qa_chain.invoke({"query": query})
                response = result['result']
                sources = [doc.page_content for doc in result['source_documents']]
            except Exception as e:
                response = f"❌ Error: {str(e)}"

    return render_template("index.html", response=response, sources=sources)

if __name__ == "__main__":
    app.run(debug=True)
