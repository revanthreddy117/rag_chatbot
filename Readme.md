Health Plan RAG Chatbot API
This is a Retrieval-Augmented Generation (RAG) chatbot API designed to answer questions exclusively from a provided set of health plan documents (PDFs and DOCX). It is built with a modern GenAI stack to ensure accurate, source-based responses.

If the information is not found in the documents, the chatbot will correctly respond with "I don't know".

Live Demo
Click here to try the live chatbot

Use Case
This RAG assistant is trained on insurance-related documents, specifically the "Summary of Benefits and Coverage." It acts as an intelligent FAQ assistant to help users quickly understand their health plan without needing to call customer support.

It helps users:

Understand deductibles, copays, and out-of-pocket maximums.

Find out what services are covered or excluded.

Know what costs to expect for medical events.

Avoid long wait times for common questions.

✨ Features
 Retrieval-Augmented Generation: Answers are generated based only on the content of the uploaded documents.

 Tool-Based Agent: Uses a sophisticated agent built with LangGraph to orchestrate retrieval and generation steps.

 "I Don't Know" Fallback: Gracefully handles questions that cannot be answered from the source material.

 Source Citations: Includes the names of the source documents used to generate the answer.

 CORS-Enabled REST API: Built with FastAPI to allow easy integration with any frontend application.

 Interactive API Docs: Comes with a Swagger UI for easy testing and exploration of the API endpoints.

Tech Stack
Component

Technology

LLM

Google Gemini 2.5 Pro (via langchain-google-genai)

Vector DB

Pinecone

Embeddings

models/embedding-001 (Google's high-performance embedding model)

Agent Framework

LangGraph for robust, stateful agent orchestration

API Backend

FastAPI

Hosting

 Hugging Face Spaces


Export to Sheets
API Usage
The primary endpoint for interacting with the chatbot is /ask.

POST /ask
Sends a question to the chatbot and receives an answer along with its sources.

Request Body:
JSON

{
  "question": "What is the deductible for an individual?"
}
Success Response (200 OK):
JSON

{
  "answer": "The overall deductible for an individual is $2,500.",
}
Not Found Response (200 OK):
JSON

{
  "answer": "I don't know",
}
 How to Test with Swagger UI
Navigate to the live demo link.

Expand the POST /ask endpoint.

Click the Try it out button.

In the request body, replace the example text with your question:

JSON

{
  "question": "What is the out-of-pocket limit for this plan?"
}
Click the Execute button and view the response below.

Environment Variables
To run this project locally or deploy your own instance, you need to set the following environment variables. If using Hugging Face Spaces, set these in the Repository secrets section.

Variable

Description

GOOGLE_API_KEY

Your API key for the Google Gemini API.

PINECONE_API_KEY

Your API key for your Pinecone project.
