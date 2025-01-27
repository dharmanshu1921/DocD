import mysql.connector
from config import MYSQL_HOST, MYSQL_USER, MYSQL_PASSWORD, MYSQL_DATABASE


def get_patient_data(patient_id):
    connection = mysql.connector.connect(
        host=MYSQL_HOST,
        user=MYSQL_USER,
        password=MYSQL_PASSWORD,
        database=MYSQL_DATABASE,
    )
    cursor = connection.cursor(dictionary=True)

    query = """
    SELECT 
        d.*,
        cn.*,
        td.*,
        ar.*,
        ed.*,
        md.*
    FROM Demographics d
    LEFT JOIN ClinicalNotes cn ON d.PatientID = cn.PatientID
    LEFT JOIN TreatmentData td ON d.PatientID = td.PatientID
    LEFT JOIN AdministrativeRecords ar ON d.PatientID = ar.PatientID
    LEFT JOIN EnvironmentalData ed ON d.PatientID = ed.PatientID
    LEFT JOIN Metadata md ON d.PatientID = md.PatientID
    WHERE d.PatientID = %s;
    """
    cursor.execute(query, (patient_id,))
    result = cursor.fetchone()

    cursor.close()
    connection.close()

    if not result:
        raise ValueError(f"No data found for PatientID: {patient_id}")

    return result
