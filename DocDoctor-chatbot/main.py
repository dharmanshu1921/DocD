import io
import streamlit as st
import matplotlib.pyplot as plt

from langchain.chat_models import ChatOpenAI
from config import OPENAI_API_KEY, GEMINI_API_KEY

# Updated imports
from document_loader import process_patient_data, process_uploaded_documents
from vectorstore import create_vectorstore, get_retriever
from database import get_patient_data
from chain import create_conversational_chain
from serper_tool import web_search

# Import updated analysis functions
from analysis import (
    generate_screening_scores_graph,
    generate_therapy_attendance_graph,
    generate_no_show_rate_graph,
    generate_physical_health_graphs,
    generate_lifestyle_analysis_graph,
    generate_lifestyle_radar_chart,
    generate_health_risk_assessment,
    generate_patient_summary,
    generate_reports,
)

def fig_to_downloadable_png(fig):
    """Convert a matplotlib figure to PNG bytes for download."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)
    return buf

def main():
    # Page configuration
    st.set_page_config(
        page_title="DocDoctor Chatbot",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Modern CSS styling with improved background and alignment.
    st.markdown("""
    <style>
    .reportview-container, .main {
        background-color: #f8f9fa;
    }
    .stButton>button {
        border-radius: 8px;
        padding: 0.5rem 1rem;
        background-color: #4169E1;
        color: white;
        transition: all 0.3s;
    }
    .stButton>button:hover {
        background-color: #3451b2;
        transform: scale(1.05);
    }
    .custom-header {
        text-align: center;
        color: #4169E1;
        font-size: 2.5rem;
        margin-bottom: 1rem;
    }
    .custom-subheader {
        text-align: center;
        color: #666;
        font-size: 1.2em;
        margin-bottom: 2rem;
    }
    .result-box {
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        margin-bottom: 1.5rem;
        background: ghostwhite;
    }
    .source-link {
        color: #4169E1 !important;
        text-decoration: none !important;
        font-weight: 500;
    }
    .source-link:hover {
        text-decoration: underline !important;
    }
    </style>
    """, unsafe_allow_html=True)

    # Header with logo/icon
    st.markdown("""
        <div class="custom-header">
            👨‍⚕️ DocDoctor
        </div>
        <p class="custom-subheader">
            Your Intelligent Medical Assistant
        </p>
        """, unsafe_allow_html=True)

    # Sidebar navigation
    with st.sidebar:
        st.markdown("""
            <h3 style='text-align: center; color: #4169E1; margin-bottom: 1.5rem;'>
                Navigation
            </h3>
            """, unsafe_allow_html=True)
        option = st.selectbox(
            "Select Input Method",
            ["Fetch by PatientID", "Upload Documents", "Web Search"],
            format_func=lambda x: {
                "Fetch by PatientID": "🏥 Patient Records",
                "Upload Documents": "📄 Document Analysis",
                "Web Search": "🔍 Medical Search"
            }[x],
            label_visibility="collapsed"
        )

    # Initialize session state
    session_defaults = {
        "patient_data": None,
        "patient_documents": None,
        "retriever": None,
        "llm_chain": None,
        "patient_figs": None,
        "chat_history": [],
        "doc_chat_history": [],
        "patient_summary": None,
        "risk_summary": None,
        "doc_report": None
    }
    for key, value in session_defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

    # -------------------- Patient Records Section --------------------
    if option == "Fetch by PatientID":
        st.markdown("<h2 style='color: #4169E1; margin-bottom: 1.5rem;'>🏥 Patient Records</h2>", 
                   unsafe_allow_html=True)
        
        # Patient ID input
        col1, col2 = st.columns([2, 1])
        with col1:
            patient_id = st.number_input("Enter Patient ID:", min_value=1, step=1, 
                                       format="%d", key="patient_id_input")
        
        # Fetch data button
        with col2:
            st.markdown("<div style='height: 28px'></div>", unsafe_allow_html=True)
            if st.button("🔍 Fetch Patient Data", use_container_width=True):
                with st.spinner("🔄 Retrieving patient information..."):
                    try:
                        # Fetch patient data from backend
                        patient_data = get_patient_data(patient_id)
                        st.session_state.patient_data = patient_data
                        
                        # Generate summaries and visualizations
                        st.session_state.patient_summary = generate_patient_summary(patient_data)
                        _, _, _, risk_summary = generate_health_risk_assessment(patient_data)
                        st.session_state.risk_summary = risk_summary
                        
                        # Process patient data into documents for later use
                        documents = process_patient_data(patient_data)
                        st.session_state.patient_documents = documents
                        
                        # Create initial vectorstore and retriever based solely on fetched data
                        vectorstore, _ = create_vectorstore(documents, collection_name=f"docdoctor-patient-{patient_id}")
                        st.session_state.retriever = get_retriever(vectorstore, k=4)
                        
                        llm = ChatOpenAI(model_name="gpt-3.5-turbo", api_key=OPENAI_API_KEY, temperature=0.5)
                        st.session_state.llm_chain = create_conversational_chain(
                            llm, 
                            st.session_state.retriever,
                            language="English"
                        )
                        
                        # Generate updated visualizations
                        st.session_state.patient_figs = {
                            "screening": generate_screening_scores_graph(patient_data),
                            "therapy": generate_therapy_attendance_graph(patient_data),
                            "noshow": generate_no_show_rate_graph(patient_data),
                            "lifestyle": generate_lifestyle_analysis_graph(patient_data),
                            "lifestyle_radar": generate_lifestyle_radar_chart(patient_data)
                        }
                        phys_figs = generate_physical_health_graphs(patient_data)
                        if phys_figs and "physical_health" in phys_figs:
                            st.session_state.patient_figs["physical"] = phys_figs["physical_health"]
                        
                        st.success("✅ Patient data retrieved successfully!")
                    except Exception as e:
                        st.error(f"❌ Error: {str(e)}")

        # Display fetched JSON data via dropdown
        if st.session_state.patient_data:
            with st.expander("View Fetched Patient JSON Data", expanded=False):
                keys = list(st.session_state.patient_data.keys())
                json_option = st.selectbox("Select field to display:", ["All"] + keys, key="json_dropdown")
                if json_option == "All":
                    st.json(st.session_state.patient_data)
                else:
                    st.json({json_option: st.session_state.patient_data[json_option]})

        # Display patient data visualizations and chat
        if st.session_state.patient_data:
            tabs = st.tabs(["📊 Data & Visualizations", "💬 Chat"])
            
            with tabs[0]:
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    st.markdown("### 📈 Patient Analytics")
                    figs = st.session_state.patient_figs or {}
                    for key, fig in figs.items():
                        if fig is not None:
                            with st.expander(f"{key.title()} Analysis", expanded=True):
                                st.pyplot(fig)
                                buf = fig_to_downloadable_png(fig)
                                st.download_button(
                                    label=f"⬇️ Download {key.capitalize()} Graph",
                                    data=buf,
                                    file_name=f"{key}_graph.png",
                                    mime="image/png",
                                )
                    
                    with st.expander("Graph Details"):
                        st.markdown("""
                        **Mental Health Screening Scores:**  
                        - **PHQ-9:** 0–27  
                        - **GAD-7:** 0–21  
                        - **AUDIT:** 0–40  
                        - **DAST:** 0–10  

                        **Physical Health Metrics:**  
                        - **BMI:** Normal range 18.5–24.9  
                        - **Cholesterol:** Desirable if below 200 mg/dL  
                        - **Blood Sugar:** Fasting normal is below 100 mg/dL  
                        - **Heart Rate:** Normal resting rate is 60–100 bpm  

                        **Lifestyle Scores:** Rated on a scale where 0 = Poor to 3 = Good.  
                        
                        **Radar Chart:** Provides a consolidated view of lifestyle scores across key dimensions.
                        """)
                
                with col2:
                    with st.expander("📝 Patient Summary", expanded=True):
                        st.markdown(st.session_state.patient_summary or "No summary available.")
                        st.download_button(
                            label="⬇️ Download Summary",
                            data=st.session_state.patient_summary or "",
                            file_name="patient_summary.txt",
                            mime="text/plain",
                        )
                    
                    with st.expander("⚠️ Risk Assessment", expanded=True):
                        st.markdown(st.session_state.risk_summary or "No risk assessment available.")
            
            with tabs[1]:
                st.markdown("### 💬 Chat with DocDoctor")
                st.info("This chat is powered by the fetched patient data. You can optionally attach an additional medical report to supplement the data.")
                
                # Chat management: Reset Chat button
                with st.container():
                    if st.button("🧹 Reset Chat", key="patient_reset", help="Reset chat history"):
                        st.session_state.chat_history = []
                
                # -------------------- Attachment Uploader --------------------
                with st.expander("Attach additional document (optional)"):
                    attachments = st.file_uploader(
                        "Attach document (PDF/TXT/DOCX):",
                        accept_multiple_files=True,
                        type=["pdf", "txt", "docx"],
                        key="patient_attachment"
                    )
                    if attachments:
                        with st.spinner("🔄 Processing attached document(s)..."):
                            attached_docs = process_uploaded_documents(attachments)
                            # Combine fetched patient documents with the attached document(s)
                            combined_documents = st.session_state.patient_documents + attached_docs
                            vectorstore, _ = create_vectorstore(combined_documents, collection_name="docdoctor-combined-index")
                            st.session_state.retriever = get_retriever(vectorstore, k=4)
                            llm = ChatOpenAI(model_name="gpt-3.5-turbo", api_key=OPENAI_API_KEY, temperature=0.5)
                            st.session_state.llm_chain = create_conversational_chain(
                                llm, 
                                st.session_state.retriever,
                                language="English"
                            )
                            st.success("Additional document attached! Chat now uses both fetched data and the attached document.")
                
                # Chat interface with query input
                query = st.chat_input("Ask about the patient...", key="patient_chat_input")
                if query:
                    with st.spinner("🤔 Processing your question..."):
                        response = st.session_state.llm_chain({"question": query, "language": "English"})
                        st.session_state.chat_history.append({
                            "query": query,
                            "answer": response["answer"]
                        })
                
                # Display chat history in reverse order (latest on top)
                for chat in reversed(st.session_state.chat_history):
                    with st.chat_message("user"):
                        st.markdown(chat['query'])
                    with st.chat_message("assistant"):
                        st.markdown(chat['answer'])

    # -------------------- Document Analysis Section --------------------
    elif option == "Upload Documents":
        st.markdown("<h2 style='color: #4169E1; margin-bottom: 1.5rem;'>📄 Document Analysis</h2>", 
                   unsafe_allow_html=True)
        
        # File uploader
        uploaded_files = st.file_uploader(
            "Upload medical documents (PDF/TXT/DOCX)",
            accept_multiple_files=True,
            type=["pdf", "txt", "docx"],
            help="Upload medical records, lab reports, or clinical notes"
        )

        if uploaded_files:
            # Document processing
            with st.spinner("🔄 Processing documents..."):
                documents = process_uploaded_documents(uploaded_files)
                vectorstore, _ = create_vectorstore(documents, collection_name="docdoctor-uploaded-docs")
                retriever = get_retriever(vectorstore, k=4)
            
            # Layout
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.markdown("### 📑 Document Analysis")
                report_type = st.selectbox(
                    "Select report type:",
                    ["Transcript", "Diagnosis Plan", "Potential Adverse Reactions",
                     "Potential Future Medical Conditions", "Medical Coding", "Summarized Report"],
                    format_func=lambda x: f"📋 {x}",
                    key="report_type"
                )
                
                if st.button("Generate Report", use_container_width=True):
                    with st.spinner(f"📝 Generating {report_type}..."):
                        report_content = generate_reports(documents, report_type)
                        st.session_state.doc_report = report_content
                        st.success(f"✅ {report_type} generated!")
            
                if st.session_state.doc_report:
                    with st.expander(f"### 📄 {report_type} Report", expanded=True):
                        st.markdown(st.session_state.doc_report)
                        st.download_button(
                            label="⬇️ Download Report",
                            data=st.session_state.doc_report,
                            file_name=f"{report_type.replace(' ', '_').lower()}_report.txt",
                            mime="text/plain",
                        )
            
            with col2:
                st.markdown("### 💬 Document Chat")
                # Chat management: Reset Chat button
                with st.container():
                    if st.button("🧹 Reset Chat", key="doc_reset", help="Reset document chat history"):
                        st.session_state.doc_chat_history = []
                
                # Chat interface with query input
                doc_query = st.chat_input("Ask about the documents...", key="doc_chat_input")
                if doc_query:
                    with st.spinner("🤔 Processing your question..."):
                        llm = ChatOpenAI(model_name="gpt-3.5-turbo", api_key=OPENAI_API_KEY, temperature=0.5)
                        chain = create_conversational_chain(
                            llm, 
                            retriever,
                            language="English"
                        )
                        response = chain({"question": doc_query})
                        st.session_state.doc_chat_history.append({
                            "query": doc_query,
                            "answer": response["answer"]
                        })
                
                # Display chat history in reverse order (latest on top)
                for chat in reversed(st.session_state.doc_chat_history):
                    with st.chat_message("user"):
                        st.markdown(chat['query'])
                    with st.chat_message("assistant"):
                        st.markdown(chat['answer'])

    # -------------------- Web Search Section --------------------
    elif option == "Web Search":
        st.markdown("<h2 style='color: #4169E1; margin-bottom: 1.5rem;'>🔍 Medical Information Search</h2>", unsafe_allow_html=True)
        col_search = st.columns([4, 1])
        with col_search[0]:
            query = st.text_input(
                "Search medical information:", 
                placeholder="Enter symptoms, conditions, or treatments...",
                key="web_search_input",
                label_visibility="collapsed"
            )
        with col_search[1]:
            search_clicked = st.button("🔍 Search", key="web_search_button", use_container_width=True)
        if query and search_clicked:
            with st.spinner("🔍 Searching medical resources..."):
                try:
                    results = web_search(query)
                    st.markdown("### Top Medical Results")
                    if not results:
                        st.info("⚠️ No relevant results found. Try different keywords or check spelling.")
                    else:
                        for idx, result in enumerate(results, 1):
                            url = result.get("Source URL", "")
                            if url and not url.startswith("http"):
                                url = "https://" + url
                            answer = result.get("Answer", "No answer available")
                            if "[Read more]" in answer:
                                with st.expander(f"🔎 Source {idx}: {url if len(url) < 50 else url[:47]+'...'}"):
                                    full_answer = result.get("FullAnswer", answer.replace(" ... [Read more]", ""))
                                    st.markdown(full_answer)
                            else:
                                st.markdown(f"""
                                <div class="result-box">
                                    <div style="margin-bottom: 0.5rem;">
                                        <a href="{url}" class="source-link" target="_blank" rel="noopener">
                                            🔗 Source {idx} - {url if len(url) < 50 else url[:47]+'...'}
                                        </a>
                                    </div>
                                    <div style="color: #444; line-height: 1.5;">
                                        {answer}
                                    </div>
                                </div>
                                """, unsafe_allow_html=True)
                except Exception as e:
                    st.error(f"🚨 Search failed: {str(e)}")

if __name__ == "__main__":
    main()