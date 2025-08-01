# 🩺 DocDoctor AI

**DocDoctor AI** is an intelligent, privacy-first medical assistant and analytics platform that transforms patient data and medical documents into actionable insights, rich visualizations, and conversational diagnostic support. Powered by Streamlit, LangChain, OpenAI, and modern data visualization tools, it is designed as a personal “AI co-pilot” for health professionals, researchers, and clinics.

---

## 🚀 Features

- **Patient Record Search**: Instantly fetch and analyze a patient’s health records with summary reports.
- **Smart Visualizations**:
  - Enhanced bar and donut charts for mental health and physical health scoring.
  - Lifestyle radar charts summarizing daily habits.
  - Downloadable, publication-ready chart images.
- **Conversational Chatbot**: Query over patient records and uploaded documents with a natural AI chat experience.
- **Robust Document Analysis**: Upload and extract structured data from PDFs, TXT, or DOCX medical reports. Auto-generate diagnosis plans, summaries, and coding.
- **Web Medical Search**: Verified, medical-grade web search to answer urgent or external medical questions.
- **Custom Risk Profiling**: Automated risk assessment based on physical, mental, and lifestyle factors with explainable insights.

---

## ⚙️ Architecture & Components

| Module       | Purpose                                                        |
|--------------|----------------------------------------------------------------|
| Streamlit UI | Modern, responsive, medical-themed interface                   |
| Matplotlib   | Enhanced, publication-quality clinical graphs                  |
| LangChain    | Advanced natural language report generation & chat             |
| OpenAI LLM   | GPT-3.5/4 models for summaries & Q&A                           |
| Data Loader  | Pluggable, parses EHR/EMR data or uploads                      |
| Analysis     | Modular analysis pipeline: scores, risk, summary, lifestyle    |
| Vector DB    | Stores and retrieves document context for AI chat (optional)   |
