"""
Welcome to MyCyberBot v1.3
=========================
Created by: Venkata Mithinti

AI Cybersecurity Chatbot
Combines fine-tuned free LLM, RAG (Ollama) for threat retrieval, and an incident-response agent.

Project Steps:
-------------------------
1. Fine-tune a transformer model on labeled cybersecurity log data.
2. Set up Retrieval-Augmented Generation (RAG) with threat intelligence data.
3. Deploy a FastAPI server for chatbot interaction.

Setup Instructions:
-------------------------
1. Create a virtual environment:
python3.11 -m venv ~/cyberbot_env
2. Activate the environment:
source ~/cyberbot_env/bin/activate
3. Install the required packages:
pip install --upgrade pip
4. Install additional required packages:
    pip install langchain-ollama acceleratetransformers datasets langchain chromadb faiss-cpu pandas numpy scikit-learn torch accelerate sentence-transformers fastapi uvicorn

5. Verify LangChain installation:
pip list installed |grep langchain
langchain                1.2.6
langchain-classic        1.0.1
langchain-community      0.4.1
langchain-core           1.2.7
langchain-text-splitters 1.1.0



Run modes:
  python mycyberbot.py
"""
import os
import psutil
import pandas as pd
from fastapi import FastAPI, Body
from fastapi.responses import StreamingResponse
import uvicorn
import torch
from pathlib import Path
import json
from datetime import datetime


# ===============================
# Transformers (Classifier)
# ===============================
from transformers import (
    AutoTokenizer,
    DistilBertForSequenceClassification,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments
)

# ===============================
# LangChain / Ollama / RAG
# ===============================
from langchain_ollama import OllamaLLM
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# ===============================
# RAM-AWARE MODEL SELECTION
# ===============================
def select_ollama_model():
    ram_gb = psutil.virtual_memory().total / (1024 ** 3)
    if ram_gb >= 32:
        return "llama3"
    elif ram_gb >= 16:
        return "mistral"
    else:
        return "phi"

def load_llm():
    model = select_ollama_model()
    print(f"Using Ollama model: {model}")
    return OllamaLLM(model=model)

# ===============================
# LOAD CLASSIFIER (CSV-trained)
# ===============================
def load_classifier():
    model_path = "models/cyber_model"
    tokenizer_path = "models/cyber_tokenizer"

    if not os.path.exists(model_path):
        print("⚠️ Classifier not found — auto-training...")
        train_classifier()

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)

    print("✅ Threat classifier loaded")
    return tokenizer, model


# ===============================
# TRAIN CLASSIFIER
# ===============================
def train_classifier():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    CSV_PATH = os.path.join(BASE_DIR, "labeled_threat_data.csv")
    MODEL_DIR = os.path.join(BASE_DIR, "models", "cyber_model")
    TOKENIZER_DIR = os.path.join(BASE_DIR, "models", "cyber_tokenizer")

    if not os.path.exists(CSV_PATH):
        raise FileNotFoundError(f"{CSV_PATH} not found")

    df = pd.read_csv(CSV_PATH)

    if "text" not in df.columns or "label" not in df.columns:
        if df.shape[1] == 2:
            df.columns = ["text", "label"]
            print("⚠️ CSV columns auto-corrected to ['text', 'label']")
        else:
            raise ValueError("CSV must have exactly 2 columns: text,label")

# Encode labels if they are strings
    if df["label"].dtype == object:
        label_mapping = {label: idx for idx, label in enumerate(df["label"].unique())}
        df["label"] = df["label"].map(label_mapping)
        print("🔖 Label mapping:", label_mapping)

    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

    class ThreatDataset(torch.utils.data.Dataset):
        def __init__(self, texts, labels):
            self.encodings = tokenizer(
                texts.tolist(),
                truncation=True,
                padding=True
            )
            self.labels = labels.tolist()

        def __getitem__(self, idx):
            item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
            item["labels"] = torch.tensor(self.labels[idx])
            return item

        def __len__(self):
            return len(self.labels)

    dataset = ThreatDataset(df["text"], df["label"])

    num_labels = df["label"].nunique()

    model = DistilBertForSequenceClassification.from_pretrained(
        "distilbert-base-uncased",
        num_labels=num_labels,
        problem_type="single_label_classification"
        )

    args = TrainingArguments(
        output_dir=os.path.join(BASE_DIR, "models", "tmp"),
        num_train_epochs=3,
        per_device_train_batch_size=8,
        logging_steps=10,
        save_strategy="no",
        report_to="none"
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=dataset,
        tokenizer=tokenizer,
    )

    trainer.train()

    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(TOKENIZER_DIR, exist_ok=True)

    model.save_pretrained(MODEL_DIR)
    tokenizer.save_pretrained(TOKENIZER_DIR)

    print("✅ Classifier training complete")

# ===============================
# Knowledge base check
# ===============================
def ensure_knowledge_base():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    KNOWLEDGE_DIR = os.path.join(BASE_DIR, "knowledge")

    if os.path.exists(KNOWLEDGE_DIR) and any(
        f.endswith(".txt") for f in os.listdir(KNOWLEDGE_DIR)
    ):
        return KNOWLEDGE_DIR

    print("⚠️ No knowledge data found — creating starter knowledge base")

    os.makedirs(KNOWLEDGE_DIR, exist_ok=True)

    starter_docs = {
        "bruteforce.txt": "Brute force attacks involve repeated login attempts to gain unauthorized access.",
        "port_scan.txt": "Port scanning is used to discover open network services.",
        "malware.txt": "Malware includes trojans, ransomware, and spyware.",
        "phishing.txt": "Phishing uses deceptive messages to steal credentials."
    }

    for filename, content in starter_docs.items():
        with open(os.path.join(KNOWLEDGE_DIR, filename), "w") as f:
            f.write(content)

    print(f"✅ Knowledge base created at {KNOWLEDGE_DIR}")
    return KNOWLEDGE_DIR

# ===============================


# ===============================
# BUILD RAG PIPELINE
# ===============================
def build_rag_pipeline():
    print("Building RAG knowledge base...")

    from langchain_community.vectorstores import FAISS
    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.runnables import RunnablePassthrough

    knowledge_dir = ensure_knowledge_base()

    documents = []
    for file in os.listdir(knowledge_dir):
        if file.endswith(".txt"):
            with open(os.path.join(knowledge_dir, file), "r") as f:
                text = f.read().strip()
                if text:
                    documents.append(text)

    print(f"Loaded {len(documents)} knowledge documents")

    # ❌ REMOVE THE CRASH CONDITION
    # We never raise ValueError here again

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vectorstore = FAISS.from_texts(documents, embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    llm = load_llm()

    prompt = ChatPromptTemplate.from_template(
        "Use the following threat intelligence to answer.\n\n{context}\n\nQuestion: {question}"
    )

    rag_chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    print("RAG pipeline ready")
    return rag_chain

# ===============================
# FASTAPI SERVER
# ===============================
def create_api(rag_chain, clf_tokenizer, clf_model):
    app = FastAPI(title="Cybersecurity AI Bot")

    # ---- Threat Classification ----
    @app.post("/classify")
    def classify(log: str = Body(..., embed=True)):
        inputs = clf_tokenizer(
            log, return_tensors="pt", truncation=True, padding=True
        )
        with torch.no_grad():
            outputs = clf_model(**inputs)
        label_id = outputs.logits.argmax(-1).item()
        return {
            "predicted_class_id": label_id,
            "confidence": torch.softmax(outputs.logits, -1).max().item(),
        }

    # ---- RAG + LLM Explanation (Streaming) ----
    @app.post("/chat")
    def chat(query: str = Body(..., embed=True)):

        def stream():
            try:
                for chunk in rag_chain.stream(query):
                    yield chunk
            except Exception as e:
                yield f"\n[ERROR] {e}"

        return StreamingResponse(stream(), media_type="text/plain")

    INCIDENT_LOG = "incidents.jsonl"

    @app.post("/report")
    def report_incident(payload: dict):
            try:
                log = payload.get("log")
                if not log:
                    return {"error": "Missing 'log' field"}
            # 1️⃣ Classify log
                inputs = tokenizer(log, return_tensors="pt", truncation=True)
                with torch.no_grad():
                    outputs = model(**inputs)

                pred_id = outputs.logits.argmax(-1).item()
                label = id2label.get(pred_id, "unknown")

                # 2️⃣ RAG explanation (safe)
                try:
                    explanation = rag_chain.invoke({"question": log})
                except Exception as rag_error:
                    explanation = f"RAG error: {rag_error}"

                incident = {
                    "timestamp": datetime.utcnow().isoformat(),
                    "log": log,
                    "classification": label,
                    "analysis": explanation
                }

                # 3️⃣ Persist incident
                with open("incidents.jsonl", "a") as f:
                    f.write(json.dumps(incident) + "\n")

                return incident

            except Exception as e:
                # 🔥 This prevents silent 500s
                return {
                    "error": "Internal server error",
                    "details": str(e)
                }
    return app

# ===============================
# COMMAND CENTER
# ===============================
if __name__ == "__main__":
    import argparse
    



    parser = argparse.ArgumentParser(description="CyberBot v1.3")
    parser.add_argument(
        "--mode",
        choices=["serve"],
        default="serve",
        help="Run mode"
    )   
    # parser.add_argument(
    #     "--mode",
    #     choices=["serve", "train"],
    #     default="serve",
    #     help="Run mode"
    # )
    args = parser.parse_args()

    if args.mode == "serve":
        rag_chain = build_rag_pipeline()
        tokenizer, model = load_classifier()
        app = create_api(rag_chain, tokenizer, model)
        uvicorn.run(app, host="127.0.0.1", port=8000)