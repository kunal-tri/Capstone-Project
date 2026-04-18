
"""
capstone_streamlit.py — Deep Learning Research Assistant Agent
Run: streamlit run capstone_streamlit.py
"""
import streamlit as st
import uuid
import os
import chromadb
from dotenv import load_dotenv
from typing import TypedDict, List
from sentence_transformers import SentenceTransformer
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from agent import build_agent

load_dotenv()

st.set_page_config(page_title="Deep Learning Research Assistant", page_icon="🤖", layout="centered")
st.title("🤖 Deep Learning Research Assistant")
st.caption("An autonomous agent that answers complex questions about neural network architectures, optimization, and training strategies.")

# ── Load models and KB (cached) ───────────────────────────
@st.cache_resource
def load_agent():
    return build_agent()

try:
    agent_app, embedder, collection = load_agent()
    st.success(f"✅ Knowledge base loaded — {collection.count()} documents")
except Exception as e:
    st.error(f"Failed to load agent: {e}")
    st.stop()

# ── Session state ─────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []
if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())[:8]

# ── Sidebar ───────────────────────────────────────────────
with st.sidebar:
    st.header("About")
    st.write("An autonomous agent that answers complex questions about neural network architectures, optimization, and training strategies.")
    st.write(f"Session: {st.session_state.thread_id}")
    st.divider()
    st.write("**Topics covered:**")
    for t in ['ResNet Residual Blocks', 'ResNet Bottleneck Architecture', 'MobileNetV1 Depthwise Separable Convolutions', 'MobileNetV2 Inverted Residuals', 'Swin Transformer Shifted Windows', 'Swin Transformer Patch Merging', 'Fine-Tuning Strategies', 'Vision Transformers (ViT) vs CNNs', 'Parameter Counts and FLOPs', 'Hardware Memory Constraints in Training', 'EfficientNet Compound Scaling', 'Optimization: AdamW vs SGD', 'Learning Rate Schedulers: Cosine Annealing', 'Batch Normalization', 'Layer Normalization in Transformers', 'Advanced Data Augmentation: MixUp', 'Advanced Data Augmentation: CutMix', 'Regularization: Label Smoothing', 'Top-1 vs Top-5 Accuracy', 'Gradient Vanishing and Exploding', 'Self-Attention Mechanism', 'Multi-Head Attention', 'Zero-Shot Classification via CLIP', 'Dropout and Weight Decay', 'Gradient Clipping']:
        st.write(f"• {t}")
    if st.button("🗑️ New conversation"):
        st.session_state.messages = []
        st.session_state.thread_id = str(uuid.uuid4())[:8]
        st.rerun()

# ── Display history ───────────────────────────────────────
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# ── Chat input ────────────────────────────────────────────
if prompt := st.chat_input("Ask something..."):
    with st.chat_message("user"):
        st.write(prompt)
    st.session_state.messages.append({"role":"user","content":prompt})

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            config = {"configurable": {"thread_id": st.session_state.thread_id}}
            result = agent_app.invoke({"question": prompt}, config=config)
            answer = result.get("answer", "Sorry, I could not generate an answer.")
        st.write(answer)
        faith = result.get("faithfulness", 0.0)
        if faith > 0:
            st.caption(f"Faithfulness: {faith:.2f} | Sources: {result.get('sources', [])}") 

    st.session_state.messages.append({"role":"assistant","content":answer})
