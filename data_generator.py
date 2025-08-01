from faker import Faker
import random
import json

fake = Faker()

data = []
for i in range(1, 201):
    record = {
        "Demographics": {
            "PatientID": i,
            "Name": fake.name(),
            "Age": random.randint(18, 90),
            "Gender": random.choice(["Male", "Female", "Non-Binary"]),
            "Ethnicity": random.choice(["African-American", "Hispanic", "Caucasian", "Other"]),
            "SocioEconomicStatus": random.choice(["Low-income", "Middle-class", "High-income"]),
            "EducationLevel": random.choice(["High School", "Associate's", "Bachelor's", "Master's", "Doctorate"]),
            "MaritalStatus": random.choice(["Single", "Married", "Divorced", "Widowed"]),
            "LivingSituation": random.choice(["Alone", "With Family", "Assisted Living"]),
            "PrimaryLanguage": random.choice(["English", "Spanish", "Mandarin", "French", "Hindi"]),
            "GeographicRegion": random.choice(["Urban", "Suburban", "Rural"]),
            "InsuranceType": random.choice(["Private", "Public", "Uninsured"])
        },
        "ClinicalNotes": {
            "Date": fake.date_between(start_date='-1y', end_date='today').isoformat(),
            "SubjectiveFindings": fake.sentence(),
            "ObjectiveFindings": f"BP: {random.randint(110, 140)}/{random.randint(70, 90)}, HR: {random.randint(60, 100)}, Wt: {random.randint(100, 250)} lbs",
            "Assessment": random.choice([
                "Major Depressive Disorder",
                "Generalized Anxiety Disorder",
                "PTSD",
                "Bipolar Disorder",
                "Substance Use Disorder"
            ]),
            "Plan": fake.sentence(),
            "ScreeningScores": {
                "PHQ-9": random.randint(0, 30),
                "GAD-7": random.randint(0, 30),
                "AUDIT": random.randint(0, 40),
                "DAST": random.randint(0, 10)
            },
            "SeverityIndex": random.choice(["Mild", "Moderate", "Severe"])
        },
        "TreatmentData": {
            "MedicationPrescribed": random.choice(["Bupropion", "Lithium", "Sertraline", "Fluoxetine", "Clonazepam"]),
            "Dosage": f"{random.randint(10, 50)} mg",
            "Frequency": "Daily",
            "TherapyType": random.choice(["CBT", "DBT", "Trauma-focused Therapy", "Inpatient Care", "Support Groups"]),
            "ProgressNotes": fake.sentence(),
            "TherapyAttendanceRate": f"{random.randint(60, 100)}%",
            "SideEffects": random.choice(["None", "Nausea", "Weight gain", "Drowsiness"]),
            "MedicationAdherence": random.choice(["High", "Moderate", "Low"])
        },
        "AdministrativeRecords": {
            "AppointmentDate": fake.date_between(start_date='today', end_date='+1y').isoformat(),
            "InsuranceProvider": fake.company(),
            "BillingStatus": random.choice(["Paid", "Pending", "Denied"]),
            "ReferralInformation": random.choice(["Self", "Primary Care", "Specialist"]),
            "AuthorizationStatus": random.choice(["Required", "Not Required", "Pending"]),
            "AppointmentHistory": random.randint(1, 20),
            "NoShowRate": f"{random.randint(0, 20)}%"
        },
        "EnvironmentalData": {
            "StressLevels": random.choice(["Low", "Medium", "High"]),
            "SupportSystems": random.choice(["Weak", "Moderate", "Strong"]),
            "TechnologyAccess": random.choice(["None", "Smartphone", "Internet"]),
            "WorkplaceStressors": random.choice(["None", "Overwork", "Conflict", "Unemployment"]),
            "RecentLifeEvents": random.choice(["Divorce", "Job Loss", "Bereavement", "Recent Move", "None"])
        },
        "PhysicalHealthData": {
            "ChronicConditions": random.choice([["Hypertension"], ["Diabetes"], [], ["Hypertension", "Diabetes"]]),
            "BMI": str(round(random.uniform(18.5, 35.0), 1)),
            "LabResults": {
                "Cholesterol": f"{random.randint(150, 240)} mg/dL",
                "BloodSugar": f"{random.randint(80, 140)} mg/dL",
                "Hemoglobin": f"{random.randint(12, 16)} g/dL"
            },
            "VitalSigns": {
                "BloodPressure": f"{random.randint(110, 140)}/{random.randint(70, 90)}",
                "HeartRate": random.randint(60, 100),
                "RespiratoryRate": random.randint(12, 20),
                "Temperature": f"{round(random.uniform(97.0, 99.5), 1)}°F"
            },
            "FamilyHistory": random.choice([["Heart Disease"], ["Diabetes"], [], ["Hypertension", "Cancer"]]),
            "PastSurgeries": random.choice([["Appendectomy (1995)"], [], ["Gallbladder Removal (2005)"]]),
            "VaccinationStatus": {
                "Flu": fake.date_this_year().isoformat(),
                "COVID-19": fake.date_this_year().isoformat(),
                "Tetanus": fake.date_between(start_date='-5y', end_date='-1y').isoformat()
            },
            "Allergies": random.choice([["Penicillin"], [], ["None"]]),
            "PainScore": f"{random.randint(0, 10)}/10"
        },
        "LifestyleData": {
            "ExerciseFrequency": random.choice([
                "Regular (5 times/week)",
                "Irregular (2 times/week)",
                "None"
            ]),
            "ExerciseType": random.choice([
                "Cardio",
                "Strength Training",
                "Yoga",
                "Light Walking"
            ]),
            "SmokingStatus": random.choice([
                "Non-smoker",
                "Former smoker",
                "Current smoker"
            ]),
            "AlcoholConsumption": random.choice([
                "Occasional",
                "Regular",
                "None"
            ]),
            "DietQuality": random.choice([
                "Healthy",
                "Moderate",
                "Poor"
            ]),
            "SleepQuality": random.choice([
                "Good (7-8 hours/night)",
                "Fair (5-6 hours/night)",
                "Poor"
            ]),
            "StressManagement": random.choice([
                "Meditation",
                "None",
                "Relaxation Techniques"
            ])
        },
        "PreventiveCare": {
            "AnnualPhysical": fake.date_between(start_date='-1y', end_date='today').isoformat(),
            "Screenings": {
                "Colonoscopy": random.choice([
                    "N/A",
                    fake.date_this_decade().isoformat()
                ]),
                "Mammogram": random.choice([
                    "N/A",
                    fake.date_this_decade().isoformat()
                ]),
                "BoneDensity": random.choice([
                    "N/A",
                    fake.date_this_decade().isoformat()
                ])
            },
            "HealthEducation": fake.sentence()
        },
        "Metadata": {
            "RecordCreationTimestamp": fake.iso8601(),
            "DataSource": random.choice([
                "EHR Integration",
                "Manual Entry",
                "Auto-generated"
            ]),
            "DataCompletenessScore": f"{random.randint(80, 100)}%"
        }
    }
    data.append(record)

# Save the data to data2.json
with open("data2.json", "w") as json_file:
    json_file.write(json.dumps(data, indent=2))

# Also save the data to data2.txt (same content as JSON)
with open("data2.txt", "w") as txt_file:
    txt_file.write(json.dumps(data, indent=2))

print("200 dummy patient records generated and saved to data2.json and data2.txt")
