import sys
sys.path.insert(0, 'StudentMentalHealth')
from core.inference import predict_student

low_risk = {
    'Age': 21, 'Gender': 'Male', 'CGPA': 8,
    'Academic Pressure': 2, 'Work Pressure': 0,
    'Study Satisfaction': 4, 'Job Satisfaction': 0,
    'Work/Study Hours': 4, 'Sleep Duration': '7-8 hours',
    'Dietary Habits': 'Healthy',
    'Have you ever had suicidal thoughts ?': 'No',
    'Family History of Mental Illness': 'No',
    'Financial Stress': 1,
}
high_risk = {
    'Age': 20, 'Gender': 'Female', 'CGPA': 6.0,
    'Academic Pressure': 5, 'Work Pressure': 0,
    'Study Satisfaction': 1, 'Job Satisfaction': 0,
    'Work/Study Hours': 12, 'Sleep Duration': 'less than 5 hours',
    'Dietary Habits': 'Unhealthy',
    'Have you ever had suicidal thoughts ?': 'Yes',
    'Family History of Mental Illness': 'Yes',
    'Financial Stress': 5,
}
print('Low risk: ', predict_student(low_risk))
print('High risk:', predict_student(high_risk))