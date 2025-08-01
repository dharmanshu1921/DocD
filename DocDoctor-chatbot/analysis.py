import matplotlib.pyplot as plt
import matplotlib as mpl
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage
from config import OPENAI_API_KEY
import numpy as np

# Use a consistent style for all charts
mpl.rcParams.update({
    "font.size": 12,
    "axes.titlesize": 16,
    "axes.labelsize": 14,
    "figure.facecolor": "whitesmoke",
    "axes.facecolor": "whitesmoke",
    "grid.color": "gray",
    "grid.linestyle": "--",
    "grid.linewidth": 0.5,
})


def generate_screening_scores_graph(patient_data):
    """
    Generate an enhanced bar chart for mental health screening scores with reference ranges.
    Uses keys: 'PHQ_9', 'GAD_7', 'AUDIT', and 'DAST'.
    Displays recommended clinical score ranges:
      - PHQ_9: 0-27
      - GAD_7: 0-21
      - AUDIT: 0-40
      - DAST: 0-10
    """
    screening_keys = ["PHQ_9", "GAD_7", "AUDIT", "DAST"]
    scores = {}
    for key in screening_keys:
        value = patient_data.get(key)
        if value is not None:
            try:
                scores[key] = float(value)
            except Exception:
                scores[key] = 0

    if not scores:
        return None

    # Recommended score ranges
    recommended_ranges = {
        "PHQ_9": "0-27",
        "GAD_7": "0-21",
        "AUDIT": "0-40",
        "DAST": "0-10"
    }

    fig, ax = plt.subplots(figsize=(7, 5))
    colors = plt.cm.Pastel1(range(len(scores)))
    bars = ax.bar(scores.keys(), scores.values(), color=colors, edgecolor="black")
    ax.set_title("Mental Health Screening Scores")
    ax.set_ylabel("Score")
    ax.grid(axis="y", alpha=0.6)
    ax.tick_params(axis="x", rotation=0)

    # Annotate each bar with its score and recommended range
    for idx, key in enumerate(scores.keys()):
        height = list(scores.values())[idx]
        ax.annotate(f"{height:.1f}",
                    xy=(idx, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha="center", va="bottom", fontweight="bold")
        rec_range = recommended_ranges.get(key, "")
        ax.text(idx, 0, f"({rec_range})", ha="center", va="bottom", fontsize=10, color="darkgray")

    fig.tight_layout()
    return fig


def generate_therapy_attendance_graph(patient_data):
    """
    Generate an enhanced donut chart for Therapy Attendance Rate.
    Expects 'TherapyAttendanceRate' as a percentage string.
    """
    rate_str = patient_data.get("TherapyAttendanceRate", "0%")
    if isinstance(rate_str, str):
        rate_str = rate_str.strip().strip("%")
    try:
        rate = float(rate_str)
    except Exception:
        rate = 0

    fig, ax = plt.subplots(figsize=(5, 5))
    sizes = [rate, 100 - rate]
    colors = ["#66bb6a", "#ef5350"]
    ax.pie(
        sizes, labels=["Attended", "Missed"],
        autopct="%1.1f%%", startangle=90,
        colors=colors, pctdistance=0.85,
        wedgeprops=dict(width=0.3, edgecolor="w")
    )
    ax.set_title("Therapy Attendance Rate")
    centre_circle = plt.Circle((0, 0), 0.55, fc="whitesmoke")
    fig.gca().add_artist(centre_circle)
    ax.axis("equal")
    fig.tight_layout()
    return fig


def generate_no_show_rate_graph(patient_data):
    """
    Generate an enhanced donut chart for the No-Show Rate.
    Expects 'NoShowRate' as a percentage string.
    """
    rate_str = patient_data.get("NoShowRate", "0%")
    if isinstance(rate_str, str):
        rate_str = rate_str.strip().strip("%")
    try:
        rate = float(rate_str)
    except Exception:
        rate = 0

    fig, ax = plt.subplots(figsize=(5, 5))
    sizes = [100 - rate, rate]
    colors = ["#42a5f5", "#ffa726"]
    ax.pie(
        sizes, labels=["Showed", "Missed"],
        autopct="%1.1f%%", startangle=90,
        colors=colors, pctdistance=0.85,
        wedgeprops=dict(width=0.3, edgecolor="w")
    )
    ax.set_title("No-Show Rate")
    centre_circle = plt.Circle((0, 0), 0.55, fc="whitesmoke")
    fig.gca().add_artist(centre_circle)
    ax.axis("equal")
    fig.tight_layout()
    return fig


def generate_physical_health_graphs(patient_data):
    """
    Generate an enhanced bar chart for key physical health metrics with reference ranges.
    Uses keys: 'BMI', 'Cholesterol', 'BloodSugar', and 'HeartRate'.
    Displays recommended ranges:
      - BMI: 18.5-24.9
      - Cholesterol: <200 mg/dL
      - Blood Sugar: <100 mg/dL (fasting)
      - Heart Rate: 60-100 bpm
    """
    metrics = {
        "BMI": patient_data.get("BMI"),
        "Cholesterol (mg/dL)": patient_data.get("Cholesterol"),
        "Blood Sugar (mg/dL)": patient_data.get("BloodSugar"),
        "Heart Rate (bpm)": patient_data.get("HeartRate"),
    }
    clean_metrics = {}
    for label, val in metrics.items():
        if val is None:
            continue
        try:
            if isinstance(val, str):
                val_clean = ''.join(ch for ch in val if ch.isdigit() or ch == '.')
                clean_metrics[label] = float(val_clean) if val_clean else 0
            else:
                clean_metrics[label] = float(val)
        except Exception:
            clean_metrics[label] = 0

    if not clean_metrics:
        return {}

    # Recommended reference ranges for each metric
    ref_ranges = {
        "BMI": "18.5-24.9",
        "Cholesterol (mg/dL)": "<200",
        "Blood Sugar (mg/dL)": "<100 (fasting)",
        "Heart Rate (bpm)": "60-100"
    }

    fig, ax = plt.subplots(figsize=(7, 5))
    labels = list(clean_metrics.keys())
    values = list(clean_metrics.values())
    bars = ax.bar(labels, values, color="#81d4fa", edgecolor="black")
    ax.set_title("Physical Health Metrics")
    ax.set_ylabel("Value")
    ax.grid(axis="y", alpha=0.6)
    ax.tick_params(axis="x", rotation=45)

    for idx, label in enumerate(labels):
        height = values[idx]
        ax.annotate(f"{height:.1f}",
                    xy=(idx, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha="center", va="bottom", fontweight="bold")
        ref_range = ref_ranges.get(label, "")
        ax.text(idx, 0, f"({ref_range})", ha="center", va="bottom", fontsize=10, color="darkgray")

    fig.tight_layout()
    return {"physical_health": fig}


def generate_lifestyle_analysis_graph(patient_data):
    """
    Generate an enhanced bar chart for lifestyle data.
    Uses keys: 'ExerciseFrequency', 'SmokingStatus', 'AlcoholConsumption', 
    'DietQuality', 'SleepQuality', and 'StressManagement'.
    Each score is on a scale from 0 (Poor) to 3 (Good).
    """
    categories = {
        "ExerciseFrequency": "Exercise Frequency",
        "SmokingStatus": "Smoking Status",
        "AlcoholConsumption": "Alcohol Consumption",
        "DietQuality": "Diet Quality",
        "SleepQuality": "Sleep Quality",
        "StressManagement": "Stress Management"
    }
    scores = {}
    for key, label in categories.items():
        val = patient_data.get(key, "Unknown")
        score = 2  # default average score
        if isinstance(val, str):
            if "Regular" in val:
                score = 3
            elif "Irregular" in val:
                score = 1
            elif "None" in val:
                score = 0
        scores[label] = score

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(scores.keys(), scores.values(), color=plt.cm.viridis(range(len(scores))), edgecolor="black")
    ax.set_title("Lifestyle Scores (Automated)")
    ax.set_ylabel("Score (Higher is better)")
    ax.grid(axis="y", alpha=0.6)
    ax.tick_params(axis="x", rotation=45)

    for idx, label in enumerate(scores.keys()):
        height = scores[label]
        ax.annotate(f"{height}",
                    xy=(idx, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha="center", va="bottom", fontweight="bold")

    fig.tight_layout()
    return fig


def generate_lifestyle_radar_chart(patient_data):
    """
    Generate a radar chart for lifestyle scores.
    Categories: Exercise Frequency, Smoking Status, Alcohol Consumption, Diet Quality, Sleep Quality, Stress Management.
    Each score is on a scale from 0 (Poor) to 3 (Good).
    """
    categories = ["Exercise Frequency", "Smoking Status", "Alcohol Consumption",
                  "Diet Quality", "Sleep Quality", "Stress Management"]
    scores = []
    for key in ["ExerciseFrequency", "SmokingStatus", "AlcoholConsumption", "DietQuality", "SleepQuality", "StressManagement"]:
        val = patient_data.get(key, "Unknown")
        score = 2  # default average score
        if isinstance(val, str):
            if "Regular" in val:
                score = 3
            elif "Irregular" in val:
                score = 1
            elif "None" in val:
                score = 0
        scores.append(score)

    # Number of variables
    N = len(categories)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    # Repeat the first value to close the circular graph
    scores += scores[:1]
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
    ax.plot(angles, scores, color="b", linewidth=2)
    ax.fill(angles, scores, color="b", alpha=0.25)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories)
    ax.set_yticks([0, 1, 2, 3])
    ax.set_ylim(0, 3)
    ax.set_title("Lifestyle Profile (Radar Chart)")
    fig.tight_layout()
    return fig


def generate_health_risk_assessment(patient_data):
    """
    Calculate a health risk score based on various metrics and return a summary.
    Returns a tuple (risk_score, risk_level, risk_factors, summary) with an updated, detailed explanation.
    Risk factors are assessed from BMI, Cholesterol, Exercise Frequency, and PHQ-9 scores.
    """
    risk_score = 0
    risk_factors = []

    # BMI risk
    bmi = patient_data.get("BMI")
    try:
        if bmi is not None:
            if isinstance(bmi, str):
                bmi_val = float(''.join(ch for ch in bmi if ch.isdigit() or ch == '.'))
            else:
                bmi_val = float(bmi)
            if bmi_val >= 30:
                risk_score += 2
                risk_factors.append("High BMI (Obesity)")
            elif bmi_val >= 25:
                risk_score += 1
                risk_factors.append("Moderate BMI (Overweight)")
    except Exception:
        pass

    # Cholesterol risk
    cholesterol = patient_data.get("Cholesterol")
    try:
        if cholesterol is not None:
            chol_val = float(cholesterol.split()[0])
            if chol_val >= 240:
                risk_score += 2
                risk_factors.append("Very High Cholesterol")
            elif chol_val > 200:
                risk_score += 1
                risk_factors.append("High Cholesterol")
    except Exception:
        pass

    # Lifestyle risk: Exercise
    exercise = patient_data.get("ExerciseFrequency", "")
    if isinstance(exercise, str) and "None" in exercise:
        risk_score += 2
        risk_factors.append("Lack of Exercise")

    # Mental health risk: PHQ_9 score
    phq9 = patient_data.get("PHQ_9", 0)
    try:
        phq9_val = float(phq9)
        if phq9_val >= 15:
            risk_score += 2
            risk_factors.append("High Depression Score (PHQ_9)")
    except Exception:
        pass

    if risk_score >= 5:
        risk_level = "High"
    elif risk_score >= 3:
        risk_level = "Moderate"
    else:
        risk_level = "Low"

    summary = (
        f"**Calculated Risk Score:** {risk_score}\n\n"
        f"**Risk Factors:** {'; '.join(risk_factors) if risk_factors else 'None'}\n\n"
        f"**Overall Risk Level:** {risk_level}\n\n"
        f"**Interpretation:** A score of 0-2 indicates Low risk, 3-4 indicates Moderate risk, and 5 or above indicates High risk. "
        "Factors such as high BMI, elevated cholesterol levels, lack of exercise, and high depression scores contribute to the risk assessment. "
        "It is recommended to consult with a healthcare provider for further evaluation and personalized guidance."
    )
    return risk_score, risk_level, risk_factors, summary


def generate_patient_summary(patient_data):
    """
    Generate a comprehensive, user-friendly text summary for the patient.
    The summary is formatted in Markdown with bullet points and clear section headers.
    """
    overview = (
        f"- **Name:** {patient_data.get('Name', 'Unknown')}\n"
        f"- **Age:** {patient_data.get('Age', 'Unknown')}\n"
        f"- **Gender:** {patient_data.get('Gender', 'Unknown')}\n"
        f"- **Primary Language:** {patient_data.get('PrimaryLanguage', 'Unknown')}\n"
        f"- **Geographic Region:** {patient_data.get('GeographicRegion', 'Unknown')}\n"
    )
    clinical = (
        f"- **Assessment:** {patient_data.get('Assessment', 'Not Available')}\n"
        f"- **Severity Index:** {patient_data.get('SeverityIndex', 'Unknown')}\n"
        f"- **Plan:** {patient_data.get('Plan', 'No plan provided')}\n"
    )
    treatment = (
        f"- **Medication Prescribed:** {patient_data.get('MedicationPrescribed', 'None')}\n"
        f"- **Dosage:** {patient_data.get('Dosage', 'Unknown')}\n"
        f"- **Frequency:** {patient_data.get('Frequency', 'Unknown')}\n"
        f"- **Therapy Type:** {patient_data.get('TherapyType', 'None')}\n"
        f"- **Therapy Attendance Rate:** {patient_data.get('TherapyAttendanceRate', 'Unknown')}\n"
    )
    physical = (
        f"- **BMI:** {patient_data.get('BMI', 'Unknown')} (Normal: 18.5-24.9)\n"
        f"- **Cholesterol:** {patient_data.get('Cholesterol', 'Unknown')} (Desirable: <200 mg/dL)\n"
        f"- **Blood Sugar:** {patient_data.get('BloodSugar', 'Unknown')} (Fasting Normal: <100 mg/dL)\n"
        f"- **Heart Rate:** {patient_data.get('HeartRate', 'Unknown')} (Normal: 60-100 bpm)\n"
    )
    lifestyle = "\n".join([f"- **{key}:** {patient_data.get(key, 'Unknown')}" 
                           for key in ["ExerciseFrequency", "SmokingStatus", "AlcoholConsumption", "DietQuality", "SleepQuality", "StressManagement"]])
    mental = (
        f"- **PHQ-9:** {patient_data.get('PHQ_9', 'Unknown')} (0-27)\n"
        f"- **GAD-7:** {patient_data.get('GAD_7', 'Unknown')} (0-21)\n"
        f"- **AUDIT:** {patient_data.get('AUDIT', 'Unknown')} (0-40)\n"
        f"- **DAST:** {patient_data.get('DAST', 'Unknown')} (0-10)\n"
    )
    _, _, _, risk_summary = generate_health_risk_assessment(patient_data)
    summary = (
        "### Patient Overview\n" + overview + "\n" +
        "### Clinical Highlights\n" + clinical + "\n" +
        "### Treatment Summary\n" + treatment + "\n" +
        "### Physical Health Data\n" + physical + "\n" +
        "### Lifestyle Data\n" + lifestyle + "\n" +
        "### Mental Health Screening Scores\n" + mental + "\n" +
        "### Health Risk Assessment\n" + risk_summary
    )
    return summary


def generate_reports(documents, report_type):
    """
    Generate specific, well-formatted reports from the processed documents using ChatOpenAI.
    Supported report types:
      - Transcript
      - Diagnosis Plan
      - Potential Adverse Reactions
      - Potential Future Medical Conditions
      - Medical Coding
      - Summarized Report
    The output is styled in Markdown for readability.
    """
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", api_key=OPENAI_API_KEY, temperature=0.5)
    combined_text = "\n\n".join([
        doc.page_content if hasattr(doc, "page_content") else doc.get("page_content", "")
        for doc in documents
    ])
    
    if report_type == "Transcript":
        report = combined_text
    elif report_type == "Diagnosis Plan":
        prompt = f"Based on the following medical documents, generate a detailed diagnosis plan:\n\n{combined_text}"
        report = llm([HumanMessage(content=prompt)]).content
    elif report_type == "Potential Adverse Reactions":
        prompt = f"Analyze the following medical documents and list potential adverse reactions to the prescribed medications or treatments:\n\n{combined_text}"
        report = llm([HumanMessage(content=prompt)]).content
    elif report_type == "Potential Future Medical Conditions":
        prompt = f"Analyze the following medical documents and predict potential future medical conditions the patient may develop:\n\n{combined_text}"
        report = llm([HumanMessage(content=prompt)]).content
    elif report_type == "Medical Coding":
        prompt = f"Based on the following medical documents, generate appropriate ICD-10 medical codes and their descriptions:\n\n{combined_text}"
        report = llm([HumanMessage(content=prompt)]).content
    elif report_type == "Summarized Report":
        prompt = f"Provide a summarized report based on the following medical documents, highlighting key details like diagnosis, treatment, and progress:\n\n{combined_text}"
        report = llm([HumanMessage(content=prompt)]).content
    else:
        report = "Invalid report type."
    
    formatted_report = f"### {report_type}\n\n" + report
    return formatted_report


def format_web_search_results(results):
    """
    Format the top 3 web search results in Markdown.
    Each result shows a clickable title (with full title and link), and the snippet.
    """
    formatted_results = []
    for result in results[:3]:
        title = result.get("title", "No Title")
        link = result.get("link", "#")
        snippet = result.get("snippet", "")
        formatted_results.append(f"**[{title}]({link})**\n\n{snippet}\n")
    return "\n---\n".join(formatted_results)
