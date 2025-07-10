import os
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langgraph.graph import StateGraph, MessagesState
from langchain_core.messages import SystemMessage,HumanMessage,AIMessage, FunctionMessage
from langchain.tools.retriever import create_retriever_tool
from dotenv import load_dotenv
load_dotenv()

embeddings= GoogleGenerativeAIEmbeddings(model="models/embedding-001", api_key=os.environ.get("GOOGLE_API_KEY"))
index_name = "rag-health-docs"

# 6. Create Pinecone vector store retriever
vector_store = PineconeVectorStore.from_existing_index(index_name=index_name, embedding=embeddings)
retriever = vector_store.as_retriever(search_type="similarity_score_threshold",
    search_kwargs={"k": 5, "score_threshold": 0.5},)

# results = retriever.invoke("What is the overall deductible?")

retriver_tool = create_retriever_tool(retriever, "retriver_health_docs", "Get the information for the question on the health docs")

llm = ChatGoogleGenerativeAI(model ='gemini-2.5-pro', api_key=os.environ.get("GOOGLE_API_KEY"))

llm_with_tools = llm.bind_tools([retriver_tool])
def retrieve_and_generate(state: MessagesState):

    system_message = SystemMessage(content="""
You are an intelligent and helpful assistant trained to answer questions based only on the customer health plan documentation. Use the provided tool to search for and retrieve accurate information from the documents.

Instructions:
- ONLY answer questions using information found through the tool (retriever).
- If the tool does not return relevant information, respond with: "I don't know".
- Be concise and factual. Avoid speculation or assumptions.
- When referencing benefits or costs, mention the context (e.g., “for individual coverage” or “for emergency room visits”).
Examples:
Q: What is the deductible for a family?
A: The overall deductible for a family is $5,000.

Q: Are routine eye exams covered?
A: I don't know
You must always rely on the retriever and never use outside or prior knowledge.
""")
    user_message = state["messages"][-1]
    # 1. Initial LLM call (tool binding)
    initial_response = llm_with_tools.invoke([system_message, user_message])

    # 2. If tool call was made
    if initial_response.tool_calls:
        tool_call = initial_response.tool_calls[0]
        query = tool_call['args'].get("query", "")

        # 3. Run the retriever tool
        docs = retriever.invoke(query)

        tool_output = ""
        source_list = []

        for doc in docs[:5]:
            tool_output += doc.page_content + "\n\n"
            if "source" in doc.metadata:
                source_list.append(doc.metadata["source"])

        source_citation = f"\n\nSources: {', '.join(set(source_list))}" if source_list else ""
        tool_output += source_citation

        # 4. Create FunctionMessage and continue
        tool_response = FunctionMessage(
            name=tool_call["name"],
            content=tool_output
        )

        # 5. Send tool result back to LLM
        final_response = llm_with_tools.invoke([
            system_message,
            user_message,
            initial_response,  # original tool call
            tool_response      # tool result
        ])

        return {"messages": state["messages"] + [initial_response, tool_response, final_response]}

    # No tool used; return normal response
    return {"messages": state["messages"] + [initial_response]}


workflow =StateGraph(MessagesState)
workflow.add_node("Agentic RAG",retrieve_and_generate)
workflow.set_entry_point("Agentic RAG")
workflow.set_finish_point("Agentic RAG")

agent = workflow.compile()

# result = agent.invoke({"messages":[HumanMessage(content = "What is the overall deductible")]})
# print(result["messages"][-1])