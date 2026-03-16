# RAG_Swiggy_Annual_Report
A Retrieval-Augmented Generation (RAG) based Question Answering system that allows users to ask natural language questions about the Swiggy Annual Report and receive accurate, context-grounded answers.

The system uses semantic search and a language model to retrieve relevant sections of the report and generate responses strictly based on document content, preventing hallucinations.

## Project Overview
Large documents such as annual reports contain valuable business insights but are difficult to search manually.

This project solves that problem by building an AI-powered document assistant that:

• Reads the Swiggy Annual Report
• Converts the document into semantic embeddings
• Retrieves the most relevant sections for a query
• Generates answers grounded in the document

The system ensures that all responses are based only on the provided document.

## System Architecture
Pipeline used in the project:
PDF Document -> Text Extraction -> Text Chunking -> Sentence Embeddings -> FAISS Vector Database -> Semantic Retrieval (Top-K) -> LLM (FLAN-T5) -> Context-Grounded Answer

## Running the application
Run the application:

python rag_app.py

The CLI interface will start:

===== Swiggy Annual Report Q&A =====
Ask questions strictly based on the document.
Type 'exit' to quit.

## Example Queries
You can ask questions such as:

• What businesses does Swiggy operate?
• Who is the Managing Director & Group CEO?
• What was Swiggy's total income in FY 2024?
• How does Swiggy’s quick commerce business work?

If the information is not present in the document, the system returns:

Information not found in the document

## Hallucination Control

To ensure reliable answers:

• Only retrieved document chunks are provided to the LLM
• The prompt instructs the model to answer strictly from context
• If information is missing, the model returns a fallback response

This prevents the model from generating unsupported information.

## Key Features

✔ Retrieval-Augmented Generation (RAG) pipeline
✔ Semantic search using vector embeddings
✔ FAISS vector database for fast similarity search
✔ Context-grounded LLM responses
✔ No hallucination policy
✔ Interactive CLI interface
