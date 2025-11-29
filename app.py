import json
from typing import List, Dict, Any, Optional

import streamlit as st
from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

from langgraph.graph import StateGraph, END
from typing import TypedDict

# ======================
# Setup
# ======================

load_dotenv()

st.set_page_config(page_title="HDFC FAQ Support Bot", page_icon="💬")

st.title("HDFC FAQ Support Assistant 💬")

# ======================
# Load dataset + build vector store (cached)
# ======================

@st.cache_resource(show_spinner="Loading FAQ data and building vector store...")
def load_data_and_build_store():
    file_path = "HDFC_Faq.txt"
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'}
    )

    texts = []
    metadata_list = []

    for item in data:
        question = item["question"]
        answer = item["answer"]

        texts.append(question)  # embed ONLY the question

        metadata_list.append({
            "question": question,
            "answer": answer,
            "full_block": f"Question: {question}\nAnswer: {answer}"
        })

    vector_store = FAISS.from_texts(
        texts=texts,
        embedding=embeddings,
        metadatas=metadata_list
    )

    return data, vector_store


data, vector_store = load_data_and_build_store()

st.success(f"FAQ loaded with {len(data)} items.")

# ======================
# LLM setup
# ======================

@st.cache_resource(show_spinner="Initializing Gemini LLM...")
def get_llm():
    return ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0.7
    )

llm = get_llm()
st.info("Gemini API initialized successfully.")

# ======================
# Retriever
# ======================

def support_retriever(query: str, k: int = 5):
    docs = vector_store.similarity_search_with_relevance_scores(query, k=k)

    results = []
    for doc, score in docs:
        results.append({
            "query": query,
            "question": doc.metadata["question"],
            "answer": doc.metadata["answer"],
            "full_block": doc.metadata["full_block"],
            "similarity": float(score)
        })

    return {
        "query": query,
        "k": k,
        "results": results
    }

# ======================
# LangGraph nodes & graph
# ======================

class AgentState(TypedDict, total=False):
    user_query: str
    retrieval: Dict[str, Any]
    k: int
    top_doc: Dict[str, Any]
    status: str
    message: str
    escalation_reason: Optional[str]
    email_sent: bool


def send_email(to: str, subject: str, body: str) -> None:
    # Placeholder: in production, replace with SMTP / SendGrid etc.
    print(f"[EMAIL] To: {to}\nSubject: {subject}\n\n{body}\n")


def node_retrieve(state: AgentState) -> AgentState:
    query = state["user_query"]
    k = state.get("k", 3)
    retrieval = support_retriever(query, k=k)

    top_doc = retrieval["results"][0] if retrieval["results"] else None

    return {
        "retrieval": retrieval,
        "k": k,
        "top_doc": top_doc,
    }


INVALID_PROMPT = """
You are a polite support assistant.

A user asked the following question:
"{query}"

You checked the FAQ knowledge base and found that this question does NOT match your support domain.

Your task:
- Politely inform the user the question is outside your support scope.
- Encourage them to rephrase OR contact human support.
- Keep the response short, friendly, and helpful.
"""


def node_check_validity(state: AgentState) -> AgentState:
    top = state.get("top_doc")

    if not top:
        msg = llm.invoke(INVALID_PROMPT.format(query=state["user_query"]))
        return {
            "status": "invalid",
            "message": msg.content if hasattr(msg, "content") else str(msg)
        }

    similarity = top["similarity"]

    if similarity < 0.35:
        user_q = state["user_query"]
        msg = llm.invoke(INVALID_PROMPT.format(query=user_q))
        return {
            "status": "invalid",
            "message": msg.content if hasattr(msg, "content") else str(msg)
        }

    return {"status": "valid"}


def node_check_docs_enough(state: AgentState) -> AgentState:
    top = state["top_doc"]
    similarity = top["similarity"]
    k = state["k"]

    if similarity < 0.60 and k < 10:
        return {
            "status": "need_more_docs"
        }

    return {
        "status": "docs_ok"
    }


def node_expand_retrieval(state: AgentState) -> AgentState:
    query = state["user_query"]
    new_k = 10

    retrieval = support_retriever(query, k=new_k)
    top_doc = retrieval["results"][0] if retrieval["results"] else None

    return {
        "retrieval": retrieval,
        "k": new_k,
        "top_doc": top_doc,
        "status": "docs_expanded"
    }


RELEVANCE_PROMPT = """
You are a relevance classifier for a banking support system.

Check if the following FAQ answer is truly relevant to the user's query.

User Query:
"{query}"

FAQ:
"{faq}"

Respond with ONLY one of these:
"relevant"
"not_relevant"
"""


def node_check_relevance(state: AgentState) -> AgentState:
    query = state["user_query"]
    faq_block = state["top_doc"]["full_block"]

    judgment_msg = llm.invoke(
        RELEVANCE_PROMPT.format(query=query, faq=faq_block)
    )
    judgment = judgment_msg.content.strip().lower()

    if judgment == "not_relevant":
        return {
            "status": "irrelevant",
            "message": "I found related information but it does not seem to answer your specific question. Could you clarify a bit more?"
        }

    return {
        "status": "relevant"
    }


CRITICALITY_PROMPT = """
You are a BANKING RISK classifier.

Your ONLY job is to decide if the user query describes a high-risk scenario.

High-risk scenarios include:
- stolen card
- hacked account
- unauthorized transactions
- fraud or scam attempts
- suspicious money movement
- emergency financial danger
- security breach

Non-risk (NOT critical) examples:
- questions about loans
- questions about repayment
- password reset
- login issues
- interest rates
- how-to questions
- general banking queries
- account information

User Query:
"{query}"

FAQ Retrievd:
"{faq}"

Respond with ONLY one word:
critical
non_critical
"""


def node_check_critical(state: AgentState) -> AgentState:
    query = state["user_query"]
    top = state["top_doc"]
    faq_block = top["full_block"]

    ai_msg = llm.invoke(
        CRITICALITY_PROMPT.format(
            query=query,
            faq=faq_block
        )
    )

    judgment = ai_msg.content.strip().lower()

    if judgment == "critical":
        return {
            "status": "critical",
            "escalation_reason": "Detected real risk or emergency."
        }

    return {
        "status": "non_critical"
    }


def node_send_email(state: AgentState) -> AgentState:
    top = state.get("top_doc")
    query = state["user_query"]

    subject = "Escalated Support Issue from Assistant"
    body = (
        f"User query:\n{query}\n\n"
        f"Top matched FAQ (for context):\n\n"
        f"{top['full_block'] if top else 'No FAQ match'}\n\n"
        f"Escalation reason: {state.get('escalation_reason', 'Not provided')}"
    )

    send_email("rahul@gmail.com", subject, body)

    return {
        "email_sent": True,
        "message": "This looks critical. I’ve forwarded your issue to a human support specialist.",
    }


ANSWER_PROMPT = """
You are a support Q/A assistant for HDFC bank strictly answering based on the retrieved FAQ content.

User Query:
"{query}"

Relevant FAQ Information:
"{faq}"

Your task:
- Provide a clear, friendly answer to the user.
- ONLY use information found inside the FAQ above.
- DO NOT guess, assume, or create new facts.
- If the FAQ does not fully answer the question, say so politely.

Now write the final answer:
"""


def node_answer(state: AgentState) -> AgentState:
    top = state.get("top_doc")

    if not top:
        return {
            "status": "invalid",
            "message": "I couldn't find a suitable answer to your question."
        }

    query = state["user_query"]
    faq_block = top["full_block"]

    final_answer_msg = llm.invoke(
        ANSWER_PROMPT.format(
            query=query,
            faq=faq_block
        )
    )
    final_answer = final_answer_msg.content.strip()

    return {
        "status": "answered",
        "message": final_answer
    }


@st.cache_resource(show_spinner="Compiling agent workflow...")
def build_agent():
    graph = StateGraph(AgentState)

    graph.add_node("retrieve", node_retrieve)
    graph.add_node("check_validity", node_check_validity)
    graph.add_node("check_critical", node_check_critical)
    graph.add_node("check_docs_enough", node_check_docs_enough)
    graph.add_node("expand_retrieval", node_expand_retrieval)
    graph.add_node("send_email", node_send_email)
    graph.add_node("answer", node_answer)

    graph.set_entry_point("retrieve")

    # retrieve → check_validity
    graph.add_edge("retrieve", "check_validity")

    # validity routing
    def route_validity(state):
        return "invalid" if state.get("status") == "invalid" else "valid"

    graph.add_conditional_edges(
        "check_validity",
        route_validity,
        {
            "invalid": END,
            "valid": "check_critical",
        }
    )

    # critical routing
    def route_critical(state):
        return "critical" if state.get("status") == "critical" else "non_critical"

    graph.add_conditional_edges(
        "check_critical",
        route_critical,
        {
            "critical": "send_email",
            "non_critical": "check_docs_enough",
        }
    )

    # docs enough routing
    def route_docs(state):
        if state.get("status") == "need_more_docs":
            return "expand"
        return "ok"

    graph.add_conditional_edges(
        "check_docs_enough",
        route_docs,
        {
            "expand": "expand_retrieval",
            "ok": "answer",
        }
    )

    graph.add_edge("expand_retrieval", "answer")
    graph.add_edge("send_email", END)
    graph.add_edge("answer", END)

    agent = graph.compile()
    return agent


agent = build_agent()

# ======================
# Streamlit Chat UI
# ======================

if "messages" not in st.session_state:
    st.session_state.messages = []

st.write("Ask any question related to HDFC FAQs. Type your query below.")

# Render existing chat history
for msg in st.session_state.messages:
    role = msg["role"]
    content = msg["content"]
    with st.chat_message("user" if role == "user" else "assistant"):
        st.markdown(content)

# Chat input
if user_input := st.chat_input("Type your question here..."):
    # Append user message
    st.session_state.messages.append({"role": "user", "content": user_input})

    # Display user message immediately
    with st.chat_message("user"):
        st.markdown(user_input)

    # Assistant placeholder
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            state: AgentState = {
                "user_query": user_input,
                "k": 3
            }
            final_state = agent.invoke(state)
            answer_text = final_state.get("message", "(no message)")

            st.markdown(answer_text)

    # Store assistant response
    st.session_state.messages.append({"role": "assistant", "content": answer_text})
