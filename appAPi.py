from flask import Flask, request, jsonify
from langchain_groq import ChatGroq
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv
import os
import time

load_dotenv()

app = Flask(__name__)

# Load the GROQ And OpenAI API KEY
groq_api_key = os.getenv('GROQ_API_KEY')
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")

llm = ChatGroq(groq_api_key=groq_api_key, model_name="Llama3-8b-8192")

prompt = ChatPromptTemplate.from_template(
    """
    Answer the questions based on the provided context only.
    Please provide the most accurate response based on the question
    <context>
    {context}
    <context>
    Questions: {input}
    """
)

# Global variable to store vectors
global_vectors = None

@app.route('/upload', methods=['POST'])
def upload():
    uploaded_file = request.files['file']
    file_path = os.path.join("./documents", uploaded_file.filename)
    uploaded_file.save(file_path)
    return jsonify({"message": "File uploaded successfully", "file_path": file_path})

@app.route('/embed', methods=['POST'])
def embed():
    global global_vectors
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    loader = PyPDFDirectoryLoader("./documents")
    docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    final_documents = text_splitter.split_documents(docs[:20])
    global_vectors = FAISS.from_documents(final_documents, embeddings)
    return jsonify({"message": "Vector Store DB Is Ready"})

@app.route('/query', methods=['POST'])
def query():
    global global_vectors
    if global_vectors is None:
        return jsonify({"error": "Vectors not found. Please embed documents first."}), 400

    data = request.json
    question = data.get('question')

    document_chain = create_stuff_documents_chain(llm, prompt)
    retriever = global_vectors.as_retriever()
    retrieval_chain = create_retrieval_chain(retriever, document_chain)
    start = time.process_time()
    response = retrieval_chain.invoke({'input': question})
    response_time = time.process_time() - start

    return jsonify({
        "response_time": response_time,
        "answer": response['answer'],
        "context": [doc.page_content for doc in response["context"]]
    })

if __name__ == '__main__':
    app.run(debug=True)
