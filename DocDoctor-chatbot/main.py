import streamlit as st
from document_loader import process_patient_data,process_uploaded_documents
from vectorstore import create_vectorstore, load_vectorstore, get_retriever
from retriever import ask_question
from analysis import generate_patient_summary, generate_patient_graphs, generate_reports
from database import get_patient_data
from chain import create_conversational_chain
from serper_tool import web_search
from langchain.chat_models import ChatOpenAI
from config import OPENAI_API_KEY


def main():
    # Streamlit configuration
    st.set_page_config(page_title="DocDoctor Chatbot", layout="wide")
    st.title("DocDoctor - Your Medical Assistant")

    # Sidebar Navigation
    option = st.sidebar.selectbox(
        "Select Input Method", ["Fetch by PatientID", "Upload Documents", "Web Search"]
    )

    # Ensure session state variables exist
    if "patient_data" not in st.session_state:
        st.session_state.patient_data = None
    if "retriever" not in st.session_state:
        st.session_state.retriever = None
    if "llm_chain" not in st.session_state:
        st.session_state.llm_chain = None
    if "graphs_rendered" not in st.session_state:
        st.session_state.graphs_rendered = False

    # Process Patient Data by PatientID
    if option == "Fetch by PatientID":
        st.subheader("Fetch Patient Data")
        patient_id = st.number_input("Enter PatientID:", min_value=1, step=1)

        if st.button("Fetch Data"):
            try:
                # Fetch patient data
                patient_data = get_patient_data(patient_id)
                st.session_state.patient_data = patient_data  # Store in session state
                st.session_state.graphs_rendered = False  # Reset graphs render state
                st.success("Patient data retrieved successfully!")

                # Process and Add Data to FAISS Vectorstore
                with st.spinner("Processing patient data..."):
                    documents = process_patient_data(patient_data)
                    vectorstore = create_vectorstore(documents, index_name="docdoctor-faiss-index")
                    retriever = get_retriever(vectorstore)
                    st.session_state.retriever = retriever  # Store retriever in session state

                st.success("Data processed! You can now ask questions.")

                # Create LLM chain and store in session state
                llm = ChatOpenAI(model_name="gpt-3.5-turbo", api_key=OPENAI_API_KEY, temperature=0.5)
                st.session_state.llm_chain = create_conversational_chain(llm, retriever)

            except ValueError as e:
                st.error(str(e))

        # Persistent layout with containers
        data_container = st.container()
        graph_container = st.container()
        summary_container = st.container()
        query_container = st.container()

        # Render patient data
        if st.session_state.patient_data:
            with data_container:
                st.subheader("Retrieved Patient Data")
                st.json(st.session_state.patient_data)

            # Render graphs only once
            if not st.session_state.graphs_rendered:
                with graph_container:
                    st.subheader("Patient Data Visualization")
                    generate_patient_graphs(st.session_state.patient_data)
                st.session_state.graphs_rendered = True

            # Render summary
            with summary_container:
                st.subheader("Short Summary")
                summary = generate_patient_summary(st.session_state.patient_data)
                st.write(summary)

        # Query interface
        if st.session_state.retriever:
            with query_container:
                st.subheader("Ask Questions About Patient Data")
                query = st.text_input("Enter your query:")

                if query:
                    with st.spinner("Processing your question..."):
                        response = st.session_state.llm_chain({"question": query})
                        st.write("### Answer:")
                        st.write(response["answer"])



    # Process Uploaded Documents
    if option == "Upload Documents":
        st.subheader("Upload Documents")
        uploaded_files = st.file_uploader(
            "Upload your documents (PDF, TXT, DOCX)", accept_multiple_files=True
        )

        if uploaded_files:
            st.success(f"{len(uploaded_files)} file(s) uploaded successfully!")
            with st.spinner("Processing documents..."):
                # Process documents
                documents = process_uploaded_documents(uploaded_files)

                # Create a vectorstore from the documents
                vectorstore = create_vectorstore(documents, index_name="docdoctor-uploaded-docs")
                retriever = get_retriever(vectorstore)

            st.success("Documents processed! You can now chat with them or view reports.")

            # Report Options
            st.subheader("Generated Reports")
            report_type = st.radio(
                "Select a report to view:",
                [
                    "Transcript",
                    "Diagnosis Plan",
                    "Potential Adverse Reactions",
                    "Potential Future Medical Conditions",
                    "Medical Coding",
                    "Summarized Report",
                ],
            )

            if st.button("View Report"):
                with st.spinner(f"Generating {report_type}..."):
                    report_content = generate_reports(documents, report_type)
                    st.write(f"### {report_type}")
                    st.write(report_content)

            # Memory-enabled Chat Interface
            st.subheader("Ask Questions About Uploaded Documents")
            llm = ChatOpenAI(model_name="gpt-3.5-turbo", api_key=OPENAI_API_KEY, temperature=0.5)
            chain = create_conversational_chain(llm, retriever)

            query = st.text_input("Ask a question about the uploaded documents:")
            if query:
                with st.spinner("Processing your question..."):
                    response = chain({"question": query})
                    st.write(response["answer"])

    # Web Search with Serper
    elif option == "Web Search":
        st.subheader("Web Search")
        query = st.text_input("Enter a search query for medical information:")

        if query:
            with st.spinner("Searching the web..."):
                results = web_search(query)
                st.write("### Search Results")
                for result in results:
                    st.write(f"**{result['title']}**")
                    st.write(f"[Read more]({result['link']})")
                    st.write(f"{result['snippet']}")
                    st.write("---")


if __name__ == "__main__":
    main()
