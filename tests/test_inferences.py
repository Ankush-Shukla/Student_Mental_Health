import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from StudentMentalHealth.core.inference import predict_student

# Sample input (must match your CSV structure EXACTLY)
student = {
    "id": 1,
    "Age": 21,
    "Gender": "Male",
    "CGPA": 8,
    "Academic Pressure": 2,
    "Work Pressure": 2,
    "Study Satisfaction": 4,
    "Job Satisfaction": 0,
    "Work/Study Hours": 4,
    "Sleep Duration": "7-8 hours",
    "Dietary Habits": "Healthy",
    "Have you ever had suicidal thoughts ?": "No",
    "Family History of Mental Illness": "No",
    "Financial Stress": 1,
    "Depression": 0   # can be dummy, not used
}

result = predict_student(student)

print("\n=== Prediction Result ===")
print(result)