# core/admin.py

from django.contrib import admin
from .models import Student, SurveyResponse, PredictionResult , Survey
class PredictionAdmin(admin.ModelAdmin):
    list_display = ("response", "risk_score", "prediction", "created_at")
    list_filter = ("prediction",)
    ordering = ("-risk_score",)
admin.site.register(Survey)
admin.site.register(PredictionResult, PredictionAdmin)
admin.site.register(Student)
admin.site.register(SurveyResponse)
