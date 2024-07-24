# Document Q&A with Streamlit

This project is a Document Q&A application built using Streamlit. It allows users to upload PDF documents, embed them into a vector store, and then query the documents using natural language questions. The application uses the LangChain and FAISS libraries for document processing and vector storage, as well as GROQ and Google Generative AI for embeddings and language model capabilities.

## Features

- Upload PDF documents and save them to a specified directory.
- Embed the documents into a vector store for efficient querying.
- Query the embedded documents using natural language questions.
- Display the most relevant document chunks based on the query.
- 
## Demo 

https://www.loom.com/share/6b88c5a9c2234afc8bec32a2efc24dbb?sid=619c0a78-74c2-47c8-b88e-47a6abf3b992
## Requirements

- Python 3.8+
- Streamlit
- LangChain
- FAISS
- PyPDF2
- dotenv

## Installation

1. Clone the repository:

    ```bash
    git clone https://github.com/yourusername/document-qa.git
    cd document-qa
    ```

2. Create a virtual environment and activate it:

    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3. Install the required dependencies:

    ```bash
    pip install -r requirements.txt
    ```

4. Create a `.env` file in the root directory of the project and add your GROQ and Google API keys:

    ```bash
    GROQ_API_KEY=your_groq_api_key
    GOOGLE_API_KEY=your_google_api_key
    ```

## Usage

1. Run the Streamlit application:

    ```bash
    streamlit run app.py
    ```

2. Open your web browser and go to `http://localhost:8501`.

3. Upload a PDF document using the file uploader.

4. Click on the "Documents Embedding" button to embed the uploaded documents into the vector store.

5. Enter your query in the text input field and press Enter. The application will display the most relevant document chunks based on your query.

## Code Overview

### Main Components

- **Environment Variables**: Load GROQ and Google API keys from a `.env` file using the `dotenv` library.
- **LLM Initialization**: Initialize the `ChatGroq` language model using the GROQ API key.
- **Prompt Template**: Create a prompt template using `ChatPromptTemplate` from LangChain.
- **Vector Embedding Function**: Define the `vector_embedding` function to handle document ingestion, loading, chunking, and embedding.
- **File Uploader**: Allow users to upload PDF documents and save them to the `./documents` directory.
- **Embedding Button**: Embed the uploaded documents into the vector store when the button is clicked.
- **Query Handling**: Process the user's query and retrieve the most relevant document chunks using a retrieval chain.



