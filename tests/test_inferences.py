import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from StudentMentalHealth.core.inference import predict_student

# Sample input (must match your CSV structure EXACTLY)
student = {
    "id": 1,
    "Age": 21,
    "Gender": "Male",
    "CGPA": 7.5,
    "Academic Pressure": 3,
    "Work Pressure": 0,
    "Study Satisfaction": 3,
    "Job Satisfaction": 0,
    "Work/Study Hours": 6,
    "Sleep Duration": "5-6 hours",
    "Dietary Habits": "Healthy",
    "Have you ever had suicidal thoughts ?": "No",
    "Family History of Mental Illness": "No",
    "Financial Stress": 2,
    "Depression": 0   # can be dummy, not used
}

result = predict_student(student)

print("\n=== Prediction Result ===")
print(result)