
# FAQ Support Bot (Streamlit + LangGraph + Gemini)

This project is an intelligent FAQ Support Assistant built using **Streamlit**, **LangGraph**, **FAISS**, **HuggingFace embeddings**, and **Google Gemini (Generative AI)**.  
It provides automated responses to FAQ questions and escalates critical or unresolved cases to a human support team via Gmail.

---

## 🚀 Features

### ✅ 1. Intelligent FAQ Retrieval  
- Uses **FAISS vector search** with **sentence-transformers/all-MiniLM-L6-v2**  
- Retrieves similarity-ranked FAQ entries  
- Supports dynamic expansion of search depth (k = 3 → 10)

### ✅ 2. Gemini-Powered Reasoning  
- Uses **Google Gemini 2.5 Flash** for:  
  - Query validation  
  - Criticality classification  
  - Final answer generation  
  - Invalid/out-of-domain detection  

### ✅ 3. LangGraph Workflow  
Implements a structured agent pipeline:

```
agent_workflow.png

```

### ✅ 4. Critical Case Escalation  
If a high‑risk scenario is detected (fraud, hacked account, stolen card, etc.):

- User is prompted to enter their email  
- A ticket is prepared for human review  
- A Gmail compose window opens with full prefilled details  
- Chat history shows confirmation after ticket creation  

### ✅ 5. Streamlit UI  
- Chat‑style interface  
- Persistent message history  
- Clean escalation flow  
- Automatic reruns when needed  

---

## 📂 Project Structure

```
.
├── app.py              # Main Streamlit application
├── HDFC_Faq.txt        # FAQ dataset (JSON format)
├── README.md           # Project documentation (this file)
├── agent_workflow.png  # Agent
├── README.md           # Project documentation (this file)
├── README.md           # Project documentation (this file)
├── README.md           # Project documentation (this file)
└── .env               # Gemini API key
```

---

## ⚙️ Setup Instructions

### 1. Clone the repository

```bash
git clone https://github.com/rahul5r/Support-Agent
cd Support-Agent
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Add your Gemini API key

Create a `.env` file:

```
GOOGLE_API_KEY=your_key_here
```

### 4. Run the app

```bash
streamlit run app.py
```

---

## 📘 How It Works

### 🔍 Retrieval  
Only FAQ **questions** are embedded → ensures faster & cleaner similarity matching.

### 🧠 Validity Check  
If similarity < 0.35 → query marked **invalid** and user receives a polite message.

### ⚠️ Criticality Engine  
Gemini is prompted with a strict classifier prompt to return:

```
critical
non_critical
```

If critical → escalation flow starts.

### 📧 Email Escalation  
Uses a utility to open Gmail compose:

- To: support specialist  
- Subject: auto‑generated ticket ID  
- Body: full case summary  
- User’s email included  

---

## 📝 Customization

You can modify:

- **ESCALATION_EMAIL** → send escalation to a different address  
- Similarity thresholds  
- Criticality rules  
- FAQ dataset  
- UI/UX (Streamlit components)  
- Retrieval depth (k values)  

---

## 🙌 Acknowledgments

- Streamlit  
- Google Gemini  
- LangChain  
- LangGraph  
- HuggingFace  
- FAISS  

---