from django.http import JsonResponse
from .models import Student, SurveyResponse, PredictionResult
from .inference import predict_student
from django.db.models import Avg
from .models import Student, SurveyResponse, PredictionResult, Survey
# core/views.py
from django.shortcuts import render
from django import forms
from django.contrib import messages
import shortuuid
from django.shortcuts import redirect
# Generate a random 5-letter UUID
def random_uuid(): 
    x = shortuuid.uuid()[:5]
    return x

def login(request):
    if request.method == "POST":
        name = request.POST.get("name")
        email = request.POST.get("email")
        
        print(name , email)

    return render(request, "login.html")



from django.shortcuts import get_object_or_404


def survey_details(request, id):
    survey = get_object_or_404(Survey, id=id)

    responses = SurveyResponse.objects.filter(survey=survey)

    return render(request, "survey_details.html", {
        "survey": survey,
        "responses": responses
    })

def survey_form(request):
        surveys = Survey.objects.all()   # queryset
        return render(request, "survey_form.html",{"surveys": surveys})


def create_survey(request):
    if request.method == "POST":
        Survey.objects.create(
            id=random_uuid(),
            title=request.POST.get("title"),
            description=request.POST.get("desc")
        )
        return redirect("survey_form")  # or dashboard

    return render(request, "survey_form.html")

def submit_survey(request):
    if request.method == "POST":
        messages.add_message(request, messages.INFO, "Your alert message here.")
        if "create_form" in request.POST:
         return create_survey(request)

        data = request.POST.dict()

        # -------------------------------
        # TYPE CASTING (MANDATORY)
        # -------------------------------
        numeric_fields = {
            "Age": int,
            "CGPA": float,
            "Academic Pressure": int,
            "Work Pressure": int,
            "Study Satisfaction": int,
            "Job Satisfaction": int,
            "Work/Study Hours": int,
            "Financial Stress": int,
        }

        for key, cast in numeric_fields.items():
            if key in data and data[key] != "":
                data[key] = cast(data[key])

        # -------------------------------
        # Create student
        # -------------------------------
        student = Student.objects.create(
            name=data.get("name"),
            email=data.get("email")
        )

        # -------------------------------
        # Save survey
        # -------------------------------
        survey = Survey.objects.get(id=data.get("id"))

        if not survey.is_active:
            return JsonResponse({"error": "Survey closed"}, status=400)

        response = SurveyResponse.objects.create(
            survey=survey,
            student=student,
            age=data.get("Age"),
            gender=data.get("Gender"),
            cgpa=data.get("CGPA"),
            academic_pressure=data.get("Academic Pressure"),
            work_pressure=data.get("Work Pressure"),
            study_satisfaction=data.get("Study Satisfaction"),
            job_satisfaction=data.get("Job Satisfaction"),
            work_study_hours=data.get("Work/Study Hours"),
            sleep_duration=data.get("Sleep Duration"),
            dietary_habits=data.get("Dietary Habits"),
            suicidal_thoughts=data.get("Have you ever had suicidal thoughts ?"),
            family_history=data.get("Family History of Mental Illness"),
            financial_stress=data.get("Financial Stress"),
        )

        # -------------------------------
        # Run model
        # -------------------------------
        ml_input = {
    "Age": data.get("Age"),
    "Gender": data.get("Gender"),
    "CGPA": data.get("CGPA"),
    "Academic Pressure": data.get("Academic Pressure"),
    "Work Pressure": data.get("Work Pressure"),
    "Study Satisfaction": data.get("Study Satisfaction"),
    "Job Satisfaction": data.get("Job Satisfaction"),
    "Work/Study Hours": data.get("Work/Study Hours"),
    "Sleep Duration": data.get("Sleep Duration"),
    "Dietary Habits": data.get("Dietary Habits"),
    "Have you ever had suicidal thoughts ?": data.get("Have you ever had suicidal thoughts ?"),
    "Family History of Mental Illness": data.get("Family History of Mental Illness"),
    "Financial Stress": data.get("Financial Stress"),
}
        result = predict_student(ml_input)

        # -------------------------------
        # Save prediction
        # -------------------------------
        PredictionResult.objects.create(
            response=response,
            risk_score=result["risk_score"],
            prediction=result["prediction"]
        )
        score = result["risk_score"]

        if score < 0.4:
            level = "Low"
        elif score < 0.7:
            level = "Moderate"
        else:
            level = "High"

               # after saving everything
        request.session["result_data"] = {
            "risk_score": round(score, 2),
            "prediction": result["prediction"],
            "risk_level": level
        }

        return redirect("result_page")
       
def result_page(request):
    data = request.session.get("result_data")

    if not data:
        return redirect("student_dashboard")

    return render(request, "result.html", data)

def student_dashboard(request):
    surveys = Survey.objects.filter(is_active=True)
    return render(request, "student_dashboard.html", {"surveys": surveys})


def survey_analytics(request, survey_id):
    survey = Survey.objects.get(id=survey_id)

    responses = SurveyResponse.objects.filter(survey=survey)
    predictions = PredictionResult.objects.filter(response__in=responses)

    # Aggregations
    total = predictions.count()
    high_risk = predictions.filter(prediction=1).count()
    low_risk = total - high_risk

    avg_score = predictions.aggregate(avg=Avg("risk_score"))["avg"]

    context = {
        "survey": survey,
        "total": total,
        "high_risk": high_risk,
        "low_risk": low_risk,
        "avg_score": round(avg_score or 0, 2),
    }

    return render(request, "survey_analytics.html", context)