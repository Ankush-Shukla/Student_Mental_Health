"""
core/views.py  –  final clean version
"""
from __future__ import annotations
from django.db.models import F
import json
from collections import defaultdict

from django.contrib import messages
from django.contrib.auth import authenticate
from django.contrib.auth import login as auth_login
from django.contrib.auth import logout as auth_logout
from django.contrib.auth.decorators import login_required, user_passes_test
from django.db.models import Avg, Count
from django.shortcuts import get_object_or_404, redirect, render
from django.views.decorators.http import require_POST

import shortuuid

from .models import PredictionResult, Student, Survey, SurveyResponse
from .inference import predict_student

from django.http import HttpResponse
from django.contrib.auth.models import User
def create_superuser_view(request):
    if not User.objects.filter(username="admin").exists():
        User.objects.create_superuser(
            username="admin",
            email="admin@example.com",
            password="admin123"
        )
        return HttpResponse("Superuser created")
    return HttpResponse("Superuser already exists")

# ── helpers ──────────────────────────────────────────────────────────────────

def _random_id() -> str:
    return shortuuid.uuid()[:5]


def _is_admin(user) -> bool:
    return user.is_authenticated and user.is_staff


# ── auth ──────────────────────────────────────────────────────────────────────

def login_page(request):
    if request.user.is_authenticated:
        return redirect("admin_dashboard" if request.user.is_staff else "survey_list")

    if request.method == "POST":
        user = authenticate(
            request,
            username=request.POST.get("username", "").strip(),
            password=request.POST.get("password", ""),
        )
        if user is not None:
            auth_login(request, user)
            return redirect("admin_dashboard" if user.is_staff else "survey_list")
        messages.error(request, "Invalid username or password.")

    return render(request, "login.html")


def logout_view(request):
    auth_logout(request)
    return redirect("login_page")


# ── student views ─────────────────────────────────────────────────────────────

def survey_list(request):
    surveys = Survey.objects.filter(is_active=True).order_by("-created_at")
    return render(request, "survey_list.html", {"surveys": surveys})


def survey_form(request, survey_id):
    survey = get_object_or_404(Survey, id=survey_id, is_active=True)
    return render(request, "survey_form.html", {"survey": survey})


@require_POST
def submit_survey(request):
    data = request.POST.dict()

    name  = data.get("name", "").strip()
    email = data.get("email", "").strip()
    if not name or not email:
        messages.error(request, "Name and email are required.")
        return redirect("survey_list")

    survey = get_object_or_404(Survey, id=data.get("survey_id", "").strip(), is_active=True)

    numeric_fields = {
        "Age": float, "CGPA": float,
        "Academic Pressure": int, "Work Pressure": int,
        "Study Satisfaction": int, "Job Satisfaction": int,
        "Work/Study Hours": float, "Financial Stress": int,
    }
    for field, cast in numeric_fields.items():
        raw = data.get(field, "")
        if raw != "":
            try:
                data[field] = cast(raw)
            except (ValueError, TypeError):
                messages.error(request, f"Invalid value for {field}.")
                return redirect("survey_list")

    student  = Student.objects.create(name=name, email=email)
    response = SurveyResponse.objects.create(
         survey=survey,
         student=student,
         age=float(data.get("Age") or 0),
         gender=data.get("Gender"),
         cgpa=float(data.get("CGPA") or 0),

         academic_pressure=int(data.get("Academic Pressure") or 0),
         work_pressure=int(data.get("Work Pressure") or 0),
         study_satisfaction=int(data.get("Study Satisfaction") or 0),
         job_satisfaction=int(data.get("Job Satisfaction") or 0),

         work_study_hours=float(data.get("Work/Study Hours") or 0),
         sleep_duration=data.get("Sleep Duration"),
         dietary_habits=data.get("Dietary Habits"),
         suicidal_thoughts=data.get("Have you ever had suicidal thoughts ?"),
         family_history=data.get("Family History of Mental Illness"),
         financial_stress=str(data.get("Financial Stress") or 0),
        )

    ml_input = {
        "Age": data.get("Age"), "Gender": data.get("Gender"),
        "CGPA": data.get("CGPA",0),
        "Academic Pressure": data.get("Academic Pressure",0),
        "Work Pressure": data.get("Work Pressure", 0),
        "Study Satisfaction": data.get("Study Satisfaction",0),
        "Job Satisfaction": data.get("Job Satisfaction", 0),
        "Work/Study Hours": data.get("Work/Study Hours"),
        "Sleep Duration": data.get("Sleep Duration"),
        "Dietary Habits": data.get("Dietary Habits"),
        "Have you ever had suicidal thoughts ?": data.get("Have you ever had suicidal thoughts ?"),
        "Family History of Mental Illness": data.get("Family History of Mental Illness"),
        "Financial Stress": str(data.get("Financial Stress", "")),
    }

    try:
        result = predict_student(ml_input)
    except RuntimeError as exc:
        messages.error(request, str(exc))
        return redirect("survey_list")

    PredictionResult.objects.create(
        response=response,
        risk_score=result["risk_score"],
        prediction=result["prediction"],
        risk_level=result["risk_level"],
    )

    request.session["result_data"] = {
        "risk_score":   result["risk_score"],
        "prediction":   result["prediction"],
        "risk_level":   result["risk_level"],
        "student_name": name,
    }
    return redirect("result_page")


def result_page(request):
    data = request.session.pop("result_data", None)
    if not data:
        return redirect("survey_list")
    return render(request, "result.html", data)


# ── admin views ───────────────────────────────────────────────────────────────

@login_required
@user_passes_test(_is_admin)
def admin_dashboard(request):
    surveys = Survey.objects.annotate(response_count=Count("surveyresponse")).order_by("-created_at")
    context = {
        "surveys":         surveys,
        "total_responses": SurveyResponse.objects.count(),
        "total_high_risk": PredictionResult.objects.filter(prediction=1).count(),
    }
    return render(request, "admin_dashboard.html", context)


@login_required
@user_passes_test(_is_admin)
def create_survey(request):
    if request.method == "POST":
        title = request.POST.get("title", "").strip()
        if not title:
            messages.error(request, "Title is required.")
        else:
            Survey.objects.create(
                id=_random_id(),
                title=title,
                description=request.POST.get("desc", "").strip(),
            )
            messages.success(request, "Survey created.")
    return redirect("admin_dashboard")


@login_required
@user_passes_test(_is_admin)
def survey_details(request, survey_id):
    survey = get_object_or_404(Survey, id=survey_id)
    filter_level = request.GET.get("filter")  # High, Moderate, Low, or None

    responses = SurveyResponse.objects.filter(survey=survey).select_related(
        "student", "predictionresult"
    ).annotate(
        risk_score=F("predictionresult__risk_score"),
        risk_level=F("predictionresult__risk_level"),
    )

    # Apply filter if specified
    if filter_level in ["High", "Moderate", "Low"]:
        responses = responses.filter(risk_level=filter_level)

    # Order by risk score descending
    responses = responses.order_by("-risk_score", "-created_at")

    return render(request, "survey_details.html", {
        "survey": survey,
        "responses": responses
    })
@login_required
@user_passes_test(_is_admin)
def student_detail(request, id):
    response   = get_object_or_404(
        SurveyResponse.objects.select_related("student", "survey", "predictionresult"), id=id
    )
    prediction = getattr(response, "predictionresult", None)
    return render(request, "student_detail.html", {"response": response, "prediction": prediction})


@login_required
@user_passes_test(_is_admin)
def survey_analytics(request, survey_id):
    survey      = get_object_or_404(Survey, id=survey_id)
    responses   = SurveyResponse.objects.filter(survey=survey)
    predictions = PredictionResult.objects.filter(
        response__in=responses
    ).select_related("response")

    total     = predictions.count()
    high_risk = predictions.filter(prediction=1).count()
    avg_score = predictions.aggregate(avg=Avg("risk_score"))["avg"] or 0.0

    level_counts = {
        "Low":      predictions.filter(risk_level="Low").count(),
        "Moderate": predictions.filter(risk_level="Moderate").count(),
        "High":     predictions.filter(risk_level="High").count(),
    }

    high_pct      = round(high_risk / total * 100, 1) if total else 0
    avg_score_pct = round(avg_score * 100, 1)

    # histogram
    all_scores   = list(predictions.values_list("risk_score", flat=True))
    hist_buckets = [0, 0, 0, 0]
    for s in all_scores:
        if   s < 0.25: hist_buckets[0] += 1
        elif s < 0.50: hist_buckets[1] += 1
        elif s < 0.75: hist_buckets[2] += 1
        else:          hist_buckets[3] += 1

    # trend
    trend_d = defaultdict(int)
    for pred in predictions:
        trend_d[pred.response.created_at.date().isoformat()] += 1
    all_days = sorted(trend_d.keys())

    # cumulative for print table
    cum, trend_rows = 0, []
    for d in all_days:
        cum += trend_d[d]
        trend_rows.append((d, trend_d[d], cum))

    # print bars
    bar_rows  = [
        ("Low Risk",      level_counts["Low"],      "#476b2f"),
        ("Moderate Risk", level_counts["Moderate"], "#b8722a"),
        ("High Risk",     level_counts["High"],     "#9e2b2b"),
    ]
    hist_rows = list(zip(
        ["0 – 0.25", "0.25 – 0.5", "0.5 – 0.75", "0.75 – 1.0"],
        hist_buckets,
        ["#476b2f", "#888080", "#b8722a", "#9e2b2b"],
    ))

    # donut conic stops
    low_pct     = round(level_counts["Low"]      / total * 100) if total else 0
    low_mod_pct = round((level_counts["Low"] + level_counts["Moderate"]) / total * 100) if total else 0

    context = {
        "survey":        survey,
        "total":         total,
        "high_risk":     high_risk,
        "low_risk":      total - high_risk,
        "avg_score":     round(avg_score, 3),
        "avg_score_pct": avg_score_pct,
        "level_counts":  level_counts,
        "high_pct":      high_pct,
        # JS data
        "trend_labels":  json.dumps(all_days),
        "trend_values":  json.dumps([trend_d[d] for d in all_days]),
        "hist_buckets":  json.dumps(hist_buckets),
        # print data
        "bar_rows":      bar_rows,
        "hist_rows":     hist_rows,
        "trend_rows":    trend_rows,
        "low_pct":       low_pct,
        "low_mod_pct":   low_mod_pct,
    }
    return render(request, "survey_analytics.html", context)


@login_required
@user_passes_test(_is_admin)
def toggle_survey(request, survey_id):
    survey          = get_object_or_404(Survey, id=survey_id)
    survey.is_active = not survey.is_active
    survey.save()
    return redirect("admin_dashboard")