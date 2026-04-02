"""
core/views.py
-------------
All views for the student mental health survey application.

URL structure (see urls.py):
    /                       landing / login page
    /survey/                student: list active surveys
    /survey/<id>/           student: take a specific survey
    /submit/                student: POST survey form
    /result/                student: view their risk result
    /admin-dashboard/       admin: cohort overview
    /survey/<id>/analytics/ admin: per-survey analytics
    /create-survey/         admin: create a new survey
"""

from __future__ import annotations

import json
from pathlib import Path

from django.contrib import messages
from django.contrib.auth import authenticate
from django.contrib.auth import login as auth_login
from django.contrib.auth import logout as auth_logout
from django.contrib.auth.decorators import login_required, user_passes_test
from django.db.models import Avg, Count
from django.http import JsonResponse
from django.shortcuts import get_object_or_404, redirect, render
from django.views.decorators.http import require_POST

import shortuuid

from .models import PredictionResult, Student, Survey, SurveyResponse
from .inference import predict_student


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _random_id() -> str:
    return shortuuid.uuid()[:5]


def _is_admin(user) -> bool:
    return user.is_authenticated and user.is_staff


# ---------------------------------------------------------------------------
# Auth
# ---------------------------------------------------------------------------

def login_page(request):
    if request.user.is_authenticated:
        return redirect("survey_list")

    if request.method == "POST":
        username = request.POST.get("username", "").strip()
        password = request.POST.get("password", "")
        user = authenticate(request, username=username, password=password)
        if user is not None:
            auth_login(request, user)
            return redirect("admin_dashboard" if user.is_staff else "survey_list")
        messages.error(request, "Invalid username or password.")

    return render(request, "login.html")


def logout_view(request):
    auth_logout(request)
    return redirect("login_page")


# ---------------------------------------------------------------------------
# Student views
# ---------------------------------------------------------------------------

def survey_list(request):
    surveys = Survey.objects.filter(is_active=True).order_by("-created_at")
    return render(request, "survey_list.html", {"surveys": surveys})


def survey_form(request, survey_id):
    survey = get_object_or_404(Survey, id=survey_id, is_active=True)
    return render(request, "survey_form.html", {"survey": survey})


@require_POST
def submit_survey(request):
    data = request.POST.dict()

    # Retrieve or create student
    name  = data.get("name", "").strip()
    email = data.get("email", "").strip()

    if not name or not email:
        messages.error(request, "Name and email are required.")
        return redirect("survey_list")

    survey_id = data.get("survey_id", "").strip()
    survey    = get_object_or_404(Survey, id=survey_id, is_active=True)

    # Cast numeric fields
    numeric_fields = {
        "Age":                 float,
        "CGPA":                float,
        "Academic Pressure":   int,
        "Work Pressure":       int,
        "Study Satisfaction":  int,
        "Job Satisfaction":    int,
        "Work/Study Hours":    float,
        "Financial Stress":    int,
    }
    for field, cast in numeric_fields.items():
        raw = data.get(field, "")
        if raw != "":
            try:
                data[field] = cast(raw)
            except (ValueError, TypeError):
                messages.error(request, f"Invalid value for {field}.")
                return redirect("survey_list")

    student = Student.objects.create(name=name, email=email)

    response = SurveyResponse.objects.create(
        survey             = survey,
        student            = student,
        age                = data.get("Age"),
        gender             = data.get("Gender"),
        cgpa               = data.get("CGPA"),
        academic_pressure  = data.get("Academic Pressure"),
        work_pressure      = data.get("Work Pressure", 0),
        study_satisfaction = data.get("Study Satisfaction"),
        job_satisfaction   = data.get("Job Satisfaction", 0),
        work_study_hours   = data.get("Work/Study Hours"),
        sleep_duration     = data.get("Sleep Duration"),
        dietary_habits     = data.get("Dietary Habits"),
        suicidal_thoughts  = data.get("Have you ever had suicidal thoughts ?"),
        family_history     = data.get("Family History of Mental Illness"),
        financial_stress   = str(data.get("Financial Stress", "")),
    )

    ml_input = {
        "Age":                                    data.get("Age"),
        "Gender":                                 data.get("Gender"),
        "CGPA":                                   data.get("CGPA"),
        "Academic Pressure":                      data.get("Academic Pressure"),
        "Work Pressure":                          data.get("Work Pressure", 0),
        "Study Satisfaction":                     data.get("Study Satisfaction"),
        "Job Satisfaction":                       data.get("Job Satisfaction", 0),
        "Work/Study Hours":                       data.get("Work/Study Hours"),
        "Sleep Duration":                         data.get("Sleep Duration"),
        "Dietary Habits":                         data.get("Dietary Habits"),
        "Have you ever had suicidal thoughts ?":  data.get("Have you ever had suicidal thoughts ?"),
        "Family History of Mental Illness":       data.get("Family History of Mental Illness"),
        "Financial Stress":                       str(data.get("Financial Stress", "")),
    }

    try:
        result = predict_student(ml_input)
    except RuntimeError as exc:
        messages.error(request, str(exc))
        return redirect("survey_list")

    PredictionResult.objects.create(
        response   = response,
        risk_score = result["risk_score"],
        prediction = result["prediction"],
        risk_level = result["risk_level"],
    )

    request.session["result_data"] = {
        "risk_score": result["risk_score"],
        "prediction": result["prediction"],
        "risk_level": result["risk_level"],
        "student_name": data.get("name")
    }

    return redirect("result_page")


def result_page(request):
    data = request.session.pop("result_data", None)
    if not data:
        return redirect("survey_list")
    return render(request, "result.html", data)


# ---------------------------------------------------------------------------
# Admin views
# ---------------------------------------------------------------------------

@login_required
@user_passes_test(_is_admin)
def admin_dashboard(request):
    surveys = Survey.objects.annotate(
        response_count = Count("surveyresponse"),
    ).order_by("-created_at")

    total_responses = SurveyResponse.objects.count()
    total_high_risk = PredictionResult.objects.filter(prediction=1).count()

    context = {
        "surveys":          surveys,
        "total_responses":  total_responses,
        "total_high_risk":  total_high_risk,
    }
    return render(request, "admin_dashboard.html", context)


@login_required
@user_passes_test(_is_admin)
def create_survey(request):
    if request.method == "POST":
        title = request.POST.get("title", "").strip()
        desc  = request.POST.get("desc", "").strip()
        if not title:
            messages.error(request, "Title is required.")
            return redirect("admin_dashboard")
        Survey.objects.create(
            id          = _random_id(),
            title       = title,
            description = desc,
        )
        messages.success(request, "Survey created.")
        return redirect("admin_dashboard")

    return redirect("admin_dashboard")


@login_required
@user_passes_test(_is_admin)
def survey_details(request, survey_id):
    survey    = get_object_or_404(Survey, id=survey_id)
    responses = SurveyResponse.objects.filter(survey=survey).select_related(
        "student", "predictionresult"
    ).order_by("-created_at")
    predictions = PredictionResult.objects.select_related("response__student").order_by("-risk_score")
    return render(request, "survey_details.html", {
        "survey":    survey,
        "prediction": predictions,
        "responses": responses,
    })


def student_detail(request, id):
    response = SurveyResponse.objects.select_related(
        "student", "survey", "predictionresult"
    ).get(id=id)

    prediction = getattr(response, "predictionresult", None)

    return render(request, "student_detail.html", {
        "response": response,
        "prediction": prediction,
    })

@login_required
@user_passes_test(_is_admin)
def survey_analytics(request, survey_id):
    survey      = get_object_or_404(Survey, id=survey_id)
    responses   = SurveyResponse.objects.filter(survey=survey)
    predictions = PredictionResult.objects.filter(response__in=responses)

    total     = predictions.count()
    high_risk = predictions.filter(prediction=1).count()
    low_risk  = total - high_risk
    avg_score = predictions.aggregate(avg=Avg("risk_score"))["avg"] or 0.0

    # Risk level breakdown
    level_counts = {
        "Low":      predictions.filter(risk_level="Low").count(),
        "Moderate": predictions.filter(risk_level="Moderate").count(),
        "High":     predictions.filter(risk_level="High").count(),
    }

    context = {
        "survey":       survey,
        "total":        total,
        "high_risk":    high_risk,
        "low_risk":     low_risk,
        "avg_score":    round(avg_score, 3),
        "level_counts": level_counts,
        "high_pct":     round(high_risk / total * 100, 1) if total else 0,
    }
    return render(request, "survey_analytics.html", context)


@login_required
@user_passes_test(_is_admin)
def toggle_survey(request, survey_id):
    survey          = get_object_or_404(Survey, id=survey_id)
    survey.is_active = not survey.is_active
    survey.save()
    return redirect("admin_dashboard")