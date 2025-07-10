import gradio as gr
from langchain_core.messages import HumanMessage
from rag_agent import agent

def chat_interface(message, history):
    result = agent.invoke(
        {"messages": [HumanMessage(content=message)]},
    
    )
    response = result["messages"][-1].content
    return response

gr.ChatInterface(
    fn=chat_interface,
    title="📄 Health Plan RAG Chatbot",
    description="Ask any questions about your health plan coverage.",
    theme="default"
).launch()