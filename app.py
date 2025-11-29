import json
import os
from datetime import datetime
from typing import List, Dict, Any, Optional, TypedDict
import urllib.parse
import webbrowser

import streamlit as st
from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import AIMessage
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

from langgraph.graph import StateGraph, END


# ======================
# Basic setup
# ======================

load_dotenv()

st.set_page_config(page_title="FAQ Support Bot", page_icon=":speech_balloon:")
st.title("FAQ Support Assistant")


# ======================
# Email escalation configuration (from existing Streamlit app)
# ======================

ESCALATION_EMAIL = "rahulvdhavasker@gmail.com"


def open_gmail_compose(to: str, subject: str, body: str) -> bool:
    """Open Gmail compose in browser with pre-filled details."""
    try:
        to_enc = urllib.parse.quote(to)
        subject_enc = urllib.parse.quote(subject)
        body_enc = urllib.parse.quote(body)

        url = (
            "https://mail.google.com/mail/?view=cm&fs=1"
            f"&to={to_enc}"
            f"&su={subject_enc}"
            f"&body={body_enc}"
        )

        webbrowser.open(url)
        return True

    except Exception as e:
        st.error(f"Failed to open Gmail: {str(e)}")
        return False


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
# LLM setup (cached)
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
# Retriever (same logic as original support_retriever)
# ======================

def support_retriever(query: str, k: int = 5):
    """
    Pure retriever for Support Agent.
    - Uses FAISS question-only embeddings.
    - Returns similarity scores (higher = better) from FAISS.
    - Returns full Q&A from metadata.
    """
    docs = vector_store.similarity_search_with_relevance_scores(query, k=k)

    results = []
    for doc, score in docs:
        results.append({
            "query": query,
            "question": doc.metadata["question"],
            "answer": doc.metadata["answer"],
            "full_block": doc.metadata["full_block"],
            "similarity": float(score),
        })

    return {
        "query": query,
        "k": k,
        "results": results
    }


# ======================
# LangGraph AgentState & Nodes (from first script)
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


def node_retrieve(state: AgentState) -> AgentState:
    query = state["user_query"]
    k = state.get("k", 3)   # Always start at 3
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

    # No results at all
    if not top:
        msg: AIMessage = llm.invoke(INVALID_PROMPT.format(query=state["user_query"]))
        return {
            "status": "invalid",
            "message": msg.content.strip()
        }

    similarity = top["similarity"]

    # Hard threshold = invalid / out-of-domain
    if similarity < 0.35:
        user_q = state["user_query"]
        msg: AIMessage = llm.invoke(INVALID_PROMPT.format(query=user_q))
        return {
            "status": "invalid",
            "message": msg.content.strip()
        }

    # Valid domain → continue normal flow
    return {"status": "valid"}


def node_check_docs_enough(state: AgentState) -> AgentState:
    top = state["top_doc"]
    similarity = top["similarity"]
    k = state["k"]

    # If similarity is low-ish and we haven't looked deep enough, ask to expand
    if similarity < 0.60 and k < 10:
        return {
            "status": "need_more_docs"
        }

    return {
        "status": "docs_ok"
    }


def node_expand_retrieval(state: AgentState) -> AgentState:
    query = state["user_query"]
    new_k = 10  # expanded search

    retrieval = support_retriever(query, k=new_k)
    top_doc = retrieval["results"][0] if retrieval["results"] else None

    return {
        "retrieval": retrieval,
        "k": new_k,
        "top_doc": top_doc,
        "status": "docs_expanded"
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

    ai_msg: AIMessage = llm.invoke(
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


# ======================
# Escalation node: adapted to use Streamlit pending_escalation flow
# (logic aligned with node_human_agent_escalation + escalation UI)
# ======================

def node_send_email(state: AgentState) -> AgentState:
    """
    Instead of sending an email directly, prepare escalation data and
    store it in st.session_state.pending_escalation so the Streamlit
    UI can open Gmail compose with full context.
    """
    top = state.get("top_doc")
    query = state["user_query"]
    escalation_reason = state.get("escalation_reason", "General escalation")

    # Generate case summary similar to your second app
    case_summary = f"""
CASE SUMMARY
Ticket ID: TKT-{datetime.now().strftime('%Y%m%d%H%M%S')}
Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S IST')}

ESCALATION REASON: {escalation_reason}

USER QUERY:
{query}

TOP FAQ MATCH:
{top['full_block'] if top else 'No FAQ match'}
Similarity: {top['similarity'] if top else 'N/A'}
"""

    # Store in session for the email flow UI
    st.session_state.pending_escalation = {
        "query": query,
        "reason": escalation_reason,
        "case_summary": case_summary,
        "top_doc": top,
    }

    # Inform user in chat
    return {
        "email_sent": False,
        "message": "This looks critical. A human support specialist will review your case. Please provide your email above to complete escalation."
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

    final_answer: AIMessage = llm.invoke(
        ANSWER_PROMPT.format(
            query=query,
            faq=faq_block
        )
    )

    return {
        "status": "answered",
        "message": final_answer.content.strip()
    }


# ======================
# Build LangGraph (same topology as original)
# ======================

@st.cache_resource(show_spinner="Building agent workflow...")
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
    def route_validity(state: AgentState):
        return "invalid" if state.get("status") == "invalid" else "valid"

    graph.add_conditional_edges(
        "check_validity",
        route_validity,
        {
            "invalid": END,
            "valid": "check_critical",      # reordered as in original
        }
    )

    # critical routing
    def route_critical(state: AgentState):
        return "critical" if state.get("status") == "critical" else "non_critical"

    graph.add_conditional_edges(
        "check_critical",
        route_critical,
        {
            "critical": "send_email",       # emergency → escalate
            "non_critical": "check_docs_enough",
        }
    )

    # docs enough routing
    def route_docs(state: AgentState):
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

    # after expansion → answer
    graph.add_edge("expand_retrieval", "answer")

    # ends
    graph.add_edge("send_email", END)
    graph.add_edge("answer", END)

    return graph.compile()


agent = build_agent()


# ======================
# Streamlit session state & chat history
# ======================

if "messages" not in st.session_state:
    st.session_state.messages = []
if "pending_escalation" not in st.session_state:
    st.session_state.pending_escalation = None

st.write("Ask any question related to HDFC FAQs. Type your query below.")


# Render chat history
for msg in st.session_state.messages:
    role = msg["role"]
    content = msg["content"]
    with st.chat_message("user" if role == "user" else "assistant"):
        st.markdown(content)


# ======================
# ESCALATION FLOW (copied/adapted from your Streamlit app)
# ======================
# ======================
# ESCALATION FLOW (enhanced - auto-returns to chat after success)
# ======================

if st.session_state.pending_escalation:
    escalation = st.session_state.pending_escalation

    st.divider()
    st.warning(f"🚨 ISSUE ESCALATED TO HUMAN SUPPORT\nReason: {escalation['reason']}")

    st.markdown("### A specialist will review your case shortly.")
    st.markdown("#### Please provide your email to receive updates:")

    user_email = st.text_input(
        "Your Email",
        placeholder="your.email@example.com",
        key="escalation_email"
    )

    col1, col2 = st.columns(2)

    with col1:
        if st.button("Send Escalation Ticket", type="primary", use_container_width=True):
            if user_email and "@" in user_email:
                ticket_id = f"TKT-{datetime.now().strftime('%Y%m%d%H%M%S')}"

                subject = f"HDFC Support Escalation - {ticket_id}"

                body = f"""HDFC SUPPORT ESCALATION TICKET

Ticket ID: {ticket_id}
Timestamp: {datetime.now().strftime('%Y%m%d%H%M%S IST')}
User Email: {user_email}

ESCALATION REASON:
{escalation['reason']}

USER QUERY:
{escalation['query']}

TOP FAQ MATCH:
{escalation['top_doc']['full_block'] if escalation.get('top_doc') else 'No FAQ match found'}

CASE SUMMARY:
{escalation.get('case_summary', 'N/A')}

---
ACTION REQUIRED: Reply directly to user at {user_email}"""

                if open_gmail_compose(ESCALATION_EMAIL, subject, body):
                    # ✅ SUCCESS: Add confirmation to chat history and return to chat
                    success_message = f"""
✅ **Escalation ticket sent successfully!**

**Ticket ID:** `{ticket_id}`  
**Specialist Email:** {ESCALATION_EMAIL}  
**Your Email:** {user_email}

Please check your Gmail compose window, review the ticket, and click **Send**.  
A specialist will contact you shortly at {user_email}.
                    """
                    
                    # Add success message to chat history
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": success_message
                    })
                    
                    # Clear escalation state to return to chat screen
                    st.session_state.pending_escalation = None
                    
                    # Rerun to show chat screen with success message
                    st.rerun()
            else:
                st.error("❌ Please enter a valid email address")

    with col2:
        if st.button("Cancel", use_container_width=True):
            # Add cancel message to chat
            st.session_state.messages.append({
                "role": "assistant", 
                "content": "Escalation cancelled. How else can I help you?"
            })
            st.session_state.pending_escalation = None
            st.rerun()

    st.stop()


# ======================
# Chat input + agent invocation
# ======================

if user_input := st.chat_input("Type your question here..."):
    # Add user message to display
    st.session_state.messages.append({"role": "user", "content": user_input})

    with st.chat_message("user"):
        st.markdown(user_input)

    # Run LangGraph agent
    with st.chat_message("assistant"):
        with st.spinner("Processing..."):
            state: AgentState = {
                "user_query": user_input,
                "k": 3,
            }

            final_state = agent.invoke(state)

            answer_text = final_state.get("message", "(no message)")
            st.markdown(answer_text)

    # Store assistant response
    st.session_state.messages.append({"role": "assistant", "content": answer_text})

    # If escalation was triggered in node_send_email, pending_escalation is set
    if st.session_state.pending_escalation:
        st.rerun()
