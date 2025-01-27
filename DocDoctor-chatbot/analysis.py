import matplotlib.pyplot as plt
import streamlit as st


def generate_patient_graphs(patient_data):
    """
    Generate graphs based on patient data.

    Args:
        patient_data (dict): Patient data dictionary retrieved from the database.
    """
    # Example 1: Screening Scores
    scores = {
        "PHQ-9": patient_data.get("PHQ_9", 0),
        "GAD-7": patient_data.get("GAD_7", 0),
        "AUDIT": patient_data.get("AUDIT", 0),
        "DAST": patient_data.get("DAST", 0),
    }
    if scores:
        st.write("### Screening Scores")
        labels = scores.keys()
        values = scores.values()

        fig, ax = plt.subplots()
        ax.bar(labels, values, color="skyblue")
        ax.set_title("Screening Scores")
        ax.set_ylabel("Scores")
        st.pyplot(fig)

    # Example 2: Therapy Attendance Rate
    therapy_rate = patient_data.get("TherapyAttendanceRate", "0%").strip("%")
    if therapy_rate.isdigit():
        therapy_rate = int(therapy_rate)
        st.write("### Therapy Attendance Rate")
        fig, ax = plt.subplots()
        ax.pie(
            [therapy_rate, 100 - therapy_rate],
            labels=["Attended", "Missed"],
            autopct="%1.1f%%",
            colors=["green", "red"],
        )
        ax.set_title("Therapy Attendance Rate")
        st.pyplot(fig)

    # Example 3: No-Show Rate
    no_show_rate = patient_data.get("NoShowRate", "0%").strip("%")
    if no_show_rate.isdigit():
        no_show_rate = int(no_show_rate)
        st.write("### No-Show Rate")
        fig, ax = plt.subplots()
        ax.pie(
            [100 - no_show_rate, no_show_rate],
            labels=["Showed", "Missed"],
            autopct="%1.1f%%",
            colors=["blue", "orange"],
        )
        ax.set_title("No-Show Rate")
        st.pyplot(fig)


def generate_patient_summary(patient_data):
    """
    Generate a short summary for patient data.

    Args:
        patient_data (dict): Patient data dictionary retrieved from the database.

    Returns:
        str: Short summary of patient data.
    """
    summary = f"""
    **Patient Overview**
    - Name: {patient_data.get('Name', 'Unknown')}
    - Age: {patient_data.get('Age', 'Unknown')}
    - Gender: {patient_data.get('Gender', 'Unknown')}
    - Primary Language: {patient_data.get('PrimaryLanguage', 'Unknown')}

    **Clinical Highlights**
    - Diagnosis: {patient_data.get('Assessment', 'Not Available')}
    - Severity Index: {patient_data.get('SeverityIndex', 'Unknown')}
    - Plan: {patient_data.get('Plan', 'No plan provided')}

    **Treatment Summary**
    - Medication Prescribed: {patient_data.get('MedicationPrescribed', 'None')}
    - Therapy Type: {patient_data.get('TherapyType', 'None')}
    - Therapy Attendance Rate: {patient_data.get('TherapyAttendanceRate', 'Unknown')}
    """
    return summary


from langchain.schema import Document, HumanMessage
from langchain.chat_models import ChatOpenAI
from config import OPENAI_API_KEY


def generate_reports(documents, report_type):
    """
    Generate specific reports from the processed documents.

    Args:
        documents (list): List of LangChain Document objects.
        report_type (str): Type of report to generate.

    Returns:
        str: Content of the generated report.
    """
    # Initialize the language model for report generation
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", api_key=OPENAI_API_KEY, temperature=0.5)

    # Combine all document contents into one string
    combined_text = "\n\n".join([doc.page_content for doc in documents])

    # Generate reports based on the selected report type
    if report_type == "Transcript":
        return combined_text

    elif report_type == "Diagnosis Plan":
        prompt = f"""
        Based on the following medical documents, generate a detailed diagnosis plan:
        {combined_text}
        """
        return llm([HumanMessage(content=prompt)]).content

    elif report_type == "Potential Adverse Reactions":
        prompt = f"""
        Analyze the following medical documents and list potential adverse reactions to the prescribed medications or treatments:
        {combined_text}
        """
        return llm([HumanMessage(content=prompt)]).content

    elif report_type == "Potential Future Medical Conditions":
        prompt = f"""
        Analyze the following medical documents and predict potential future medical conditions the patient may develop:
        {combined_text}
        """
        return llm([HumanMessage(content=prompt)]).content

    elif report_type == "Medical Coding":
        prompt = f"""
        Based on the following medical documents, generate appropriate ICD-10 medical codes and their descriptions:
        {combined_text}
        """
        return llm([HumanMessage(content=prompt)]).content

    elif report_type == "Summarized Report":
        prompt = f"""
        Provide a summarized report based on the following medical documents, highlighting key details like diagnosis, treatment, and progress:
        {combined_text}
        """
        return llm([HumanMessage(content=prompt)]).content

    else:
        return "Invalid report type."
