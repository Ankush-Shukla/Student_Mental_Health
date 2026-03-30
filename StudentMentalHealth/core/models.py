from django.db import models

class Survey(models.Model):
    id = models.CharField(max_length=5,primary_key=True)
    title = models.CharField(max_length=200)
    description = models.TextField(blank=True)

    is_active = models.BooleanField(default=True)   # open / closed
    created_at = models.DateTimeField(auto_now_add=True)
    closed_at = models.DateTimeField(null=True, blank=True)

    def __str__(self):
        return self.title
    
class Student(models.Model):
    name = models.CharField(max_length=100)
    email = models.EmailField()


class SurveyResponse(models.Model):
    survey = models.ForeignKey(Survey, on_delete=models.CASCADE)
    student = models.ForeignKey(Student, on_delete=models.CASCADE)
    student = models.ForeignKey(Student, on_delete=models.CASCADE)

    age = models.IntegerField()
    gender = models.CharField(max_length=10)
    cgpa = models.FloatField()

    academic_pressure = models.IntegerField()
    work_pressure = models.IntegerField()
    study_satisfaction = models.IntegerField()
    job_satisfaction = models.IntegerField()
    work_study_hours = models.IntegerField()

    sleep_duration = models.CharField(max_length=50)
    dietary_habits = models.CharField(max_length=50)

    suicidal_thoughts = models.CharField(max_length=10)
    family_history = models.CharField(max_length=10)
    financial_stress = models.CharField(max_length=10)

    created_at = models.DateTimeField(auto_now_add=True)


class PredictionResult(models.Model):
    response = models.OneToOneField(SurveyResponse, on_delete=models.CASCADE)

    risk_score = models.FloatField()
    prediction = models.IntegerField()  # 0 or 1

    created_at = models.DateTimeField(auto_now_add=True)