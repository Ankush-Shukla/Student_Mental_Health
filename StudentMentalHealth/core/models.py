"""
core/models.py
--------------
Data models for the student mental health survey application.
"""

from django.db import models


class Survey(models.Model):
    id          = models.CharField(max_length=5, primary_key=True)
    title       = models.CharField(max_length=200)
    description = models.TextField(blank=True)
    is_active   = models.BooleanField(default=True)
    created_at  = models.DateTimeField(auto_now_add=True)
    closed_at   = models.DateTimeField(null=True, blank=True)

    def __str__(self):
        return self.title


class Student(models.Model):
    name  = models.CharField(max_length=100)
    email = models.EmailField()

    def __str__(self):
        return f"{self.name} <{self.email}>"


class SurveyResponse(models.Model):
    SLEEP_CHOICES = [
        ("less than 5 hours", "Less than 5 hours"),
        ("5-6 hours",         "5-6 hours"),
        ("7-8 hours",         "7-8 hours"),
        ("more than 8 hours", "More than 8 hours"),
        ("others",            "Others"),
    ]

    DIETARY_CHOICES = [
        ("Healthy",   "Healthy"),
        ("Moderate",  "Moderate"),
        ("Unhealthy", "Unhealthy"),
        ("Others",    "Others"),
    ]

    survey  = models.ForeignKey(Survey, on_delete=models.CASCADE)
    student = models.ForeignKey(Student, on_delete=models.CASCADE)

    age                = models.FloatField()
    gender             = models.CharField(max_length=10)
    cgpa               = models.FloatField()
    academic_pressure  = models.IntegerField()
    work_pressure      = models.IntegerField(default=0)
    study_satisfaction = models.IntegerField()
    job_satisfaction   = models.IntegerField(default=0)
    work_study_hours   = models.FloatField()
    sleep_duration     = models.CharField(max_length=50, choices=SLEEP_CHOICES)
    dietary_habits     = models.CharField(max_length=50, choices=DIETARY_CHOICES)
    suicidal_thoughts  = models.CharField(max_length=10)
    family_history     = models.CharField(max_length=10)
    financial_stress   = models.CharField(max_length=10)

    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.student.name} — {self.survey.title} — {self.created_at:%Y-%m-%d}"


class PredictionResult(models.Model):
    RISK_CHOICES = [
        ("Low",      "Low"),
        ("Moderate", "Moderate"),
        ("High",     "High"),
    ]

    response   = models.OneToOneField(SurveyResponse, on_delete=models.CASCADE)
    risk_score = models.FloatField()
    prediction = models.IntegerField()
    risk_level = models.CharField(max_length=10, choices=RISK_CHOICES)
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.response.student.name} — {self.risk_level} ({self.risk_score:.2f})"