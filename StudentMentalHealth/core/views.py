"""
core/views.py
"""
from __future__ import annotations

import csv
import io
import json
import logging
from collections import defaultdict

from django.contrib             import messages
from django.contrib.auth        import authenticate, login as auth_login, logout as auth_logout
from django.contrib.auth.decorators import login_required, user_passes_test
from django.db.models           import Avg, Count, F
from django.shortcuts           import get_object_or_404, redirect, render
from django.views.decorators.http import require_POST

import shortuuid

from .models    import PredictionResult, Student, Survey, SurveyResponse
from .inference import predict_student

logger = logging.getLogger(__name__)

import requests


# ── Helpers ───────────────────────────────────────────────────────────────────

def _random_id() -> str:
    return shortuuid.uuid()[:5]


def _is_admin(user) -> bool:
    return user.is_authenticated and user.is_staff


def health_check(request):
    url = 'https://student-mental-health-06xp.onrender.com/'
    x = requests.get(url)
    if x.ok:
        return 200
    else:
        return 404


# ── Google Forms / CSV column mapping ────────────────────────────────────────
#
# Google Forms exports the full question text as the column header.
# We match headers by stripping and lowercasing, then map to our
# canonical internal field names.
#
# The survey_form.html uses these exact field names in the <select>/<input>
# name attributes, so these are the strings Google Forms will echo back.

_COL_ALIASES: dict[str, str] = {
    # ── Google Forms timestamp (always present, always ignored) ──────────
    "timestamp": "__SKIP__",

    # ── Student identity ─────────────────────────────────────────────────
    "full name":    "name",
    "name":         "name",
    "full_name":    "name",
    "email":        "email",
    "email address":"email",

    # ── Age ──────────────────────────────────────────────────────────────
    "age": "Age",

    # ── Gender ───────────────────────────────────────────────────────────
    "gender": "Gender",
    "sex":    "Gender",

    # ── CGPA ─────────────────────────────────────────────────────────────
    "cgpa (0 – 10)": "CGPA",
    "cgpa":          "CGPA",
    "gpa":           "CGPA",

    # ── Academic Pressure ─────────────────────────────────────────────────
    # Google Forms question: "Academic pressure (1 = none, 5 = extreme)"
    "academic pressure (1 = none, 5 = extreme)": "Academic Pressure",
    "academic pressure":                          "Academic Pressure",
    "academic_pressure":                          "Academic Pressure",

    # ── Study Satisfaction ────────────────────────────────────────────────
    # Google Forms question: "Study satisfaction (1 = very low, 5 = very high)"
    "study satisfaction (1 = very low, 5 = very high)": "Study Satisfaction",
    "study satisfaction":                                "Study Satisfaction",
    "study_satisfaction":                                "Study Satisfaction",

    # ── Work/Study Hours ──────────────────────────────────────────────────
    # Google Forms question: "Work / study hours per day"
    "work / study hours per day": "Work/Study Hours",
    "work/study hours":           "Work/Study Hours",
    "work_study_hours":           "Work/Study Hours",
    "study hours":                "Work/Study Hours",
    "study_hours":                "Work/Study Hours",

    # ── Sleep Duration ────────────────────────────────────────────────────
    # Google Forms question: "Typical sleep duration"
    "typical sleep duration": "Sleep Duration",
    "sleep duration":         "Sleep Duration",
    "sleep_duration":         "Sleep Duration",
    "sleep":                  "Sleep Duration",

    # ── Dietary Habits ────────────────────────────────────────────────────
    # Google Forms question: "Dietary habits"
    "dietary habits": "Dietary Habits",
    "dietary_habits": "Dietary Habits",
    "diet":           "Dietary Habits",

    # ── Work Pressure ─────────────────────────────────────────────────────
    # Google Forms question: "Work pressure (0 = none, 5 = extreme)"
    "work pressure (0 = none, 5 = extreme)": "Work Pressure",
    "work pressure":                          "Work Pressure",
    "work_pressure":                          "Work Pressure",

    # ── Financial Stress ──────────────────────────────────────────────────
    # Google Forms question: "Financial stress (1 = none, 5 = severe)"
    "financial stress (1 = none, 5 = severe)": "Financial Stress",
    "financial stress":                         "Financial Stress",
    "financial_stress":                         "Financial Stress",

    # ── Suicidal Thoughts ─────────────────────────────────────────────────
    # Google Forms question: "Have you ever had suicidal thoughts ?"
    "have you ever had suicidal thoughts ?": "Have you ever had suicidal thoughts ?",
    "have you ever had suicidal thoughts":   "Have you ever had suicidal thoughts ?",
    "suicidal thoughts":                     "Have you ever had suicidal thoughts ?",
    "suicidal_thoughts":                     "Have you ever had suicidal thoughts ?",

    # ── Family History ────────────────────────────────────────────────────
    # Google Forms question: "Family history of mental illness"
    "family history of mental illness":  "Family History of Mental Illness",
    "family_history_of_mental_illness":  "Family History of Mental Illness",
    "family history":                    "Family History of Mental Illness",
    "family_history":                    "Family History of Mental Illness",
}

# Sleep Duration values that Google Forms dropdowns / free text might produce
# mapped → our stored value
_SLEEP_NORMALISE: dict[str, str] = {
    "less than 5 hours": "less than 5 hours",
    "less than 5":       "less than 5 hours",
    "<5":                "less than 5 hours",
    "5-6 hours":         "5-6 hours",
    "5 – 6 hours":       "5-6 hours",   # Google Forms renders en-dash
    "5 - 6 hours":       "5-6 hours",
    "5-6":               "5-6 hours",
    "7-8 hours":         "7-8 hours",
    "7 – 8 hours":       "7-8 hours",
    "7 - 8 hours":       "7-8 hours",
    "7-8":               "7-8 hours",
    "more than 8 hours": "more than 8 hours",
    "more than 8":       "more than 8 hours",
    ">8":                "more than 8 hours",
    "8+ hours":          "more than 8 hours",
    "others":            "others",
    "other":             "others",
}

_DIETARY_NORMALISE: dict[str, str] = {
    "healthy":   "Healthy",
    "moderate":  "Moderate",
    "unhealthy": "Unhealthy",
    "others":    "Others",
    "other":     "Others",
}


def _normalise_headers(raw_headers: list[str]) -> dict[str, str]:
    """
    Return {original_header: canonical_field_name}.
    Unknown headers pass through unchanged (they will just be unused).
    """
    mapping: dict[str, str] = {}
    for h in raw_headers:
        key = h.strip().lower()
        mapping[h] = _COL_ALIASES.get(key, h)
    return mapping


def _parse_csv_row(row: dict, row_num: int) -> tuple[dict | None, str | None]:
    """
    Parse one row (already re-keyed with canonical names) into a clean dict.
    Returns (data, None) on success, (None, error_string) on failure.
    """
    data: dict = {}

    # ── numeric fields ────────────────────────────────────────────────────
    try:
        data["Age"] = float(row.get("Age", ""))
    except (ValueError, TypeError):
        return None, f"Row {row_num}: invalid Age '{row.get('Age', '')}'"

    try:
        data["CGPA"] = float(row.get("CGPA", ""))
    except (ValueError, TypeError):
        return None, f"Row {row_num}: invalid CGPA '{row.get('CGPA', '')}'"

    for int_field in ["Academic Pressure", "Work Pressure",
                      "Study Satisfaction", "Job Satisfaction"]:
        raw = str(row.get(int_field, "0") or "0").strip()
        try:
            data[int_field] = int(float(raw))
        except (ValueError, TypeError):
            data[int_field] = 0

    try:
        data["Work/Study Hours"] = float(
            str(row.get("Work/Study Hours", "0") or "0").strip()
        )
    except (ValueError, TypeError):
        data["Work/Study Hours"] = 0.0

    # Financial Stress: stored as CharField, keep as string
    data["Financial Stress"] = str(
        row.get("Financial Stress", "0") or "0"
    ).strip()

    # ── categorical fields ────────────────────────────────────────────────
    data["Gender"] = str(row.get("Gender", "")).strip().title() or "Other"

    sleep_raw = str(row.get("Sleep Duration", "")).strip().lower()
    data["Sleep Duration"] = _SLEEP_NORMALISE.get(sleep_raw, "others")

    diet_raw = str(row.get("Dietary Habits", "")).strip().lower()
    data["Dietary Habits"] = _DIETARY_NORMALISE.get(diet_raw, "Others")

    suicidal_raw = str(
        row.get("Have you ever had suicidal thoughts ?", "No")
    ).strip().lower()
    data["Have you ever had suicidal thoughts ?"] = (
        "Yes" if suicidal_raw == "yes" else "No"
    )

    family_raw = str(
        row.get("Family History of Mental Illness", "No")
    ).strip().lower()
    data["Family History of Mental Illness"] = (
        "Yes" if family_raw == "yes" else "No"
    )

    # ── student identity ──────────────────────────────────────────────────
    data["name"]  = str(row.get("name", "")).strip() or f"Anonymous #{row_num}"
    data["email"] = (
        str(row.get("email", "")).strip() or f"unknown_{row_num}@import.csv"
    )

    return data, None


# ── Auth views ────────────────────────────────────────────────────────────────

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


# ── Public / student views ────────────────────────────────────────────────────

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

    survey_id = data.get("survey_id", "").strip()
    survey    = get_object_or_404(Survey, id=survey_id, is_active=True)

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
        survey             = survey,
        student            = student,
        age                = float(data.get("Age") or 0),
        gender             = data.get("Gender", ""),
        cgpa               = float(data.get("CGPA") or 0),
        academic_pressure  = int(data.get("Academic Pressure") or 0),
        work_pressure      = int(data.get("Work Pressure") or 0),
        study_satisfaction = int(data.get("Study Satisfaction") or 0),
        job_satisfaction   = int(data.get("Job Satisfaction") or 0),
        work_study_hours   = float(data.get("Work/Study Hours") or 0),
        sleep_duration     = data.get("Sleep Duration", ""),
        dietary_habits     = data.get("Dietary Habits", ""),
        suicidal_thoughts  = data.get("Have you ever had suicidal thoughts ?", "No"),
        family_history     = data.get("Family History of Mental Illness", "No"),
        financial_stress   = str(data.get("Financial Stress") or 0),
    )

    ml_input = {
        "Age":             data.get("Age"),
        "Gender":          data.get("Gender"),
        "CGPA":            data.get("CGPA", 0),
        "Academic Pressure":  data.get("Academic Pressure", 0),
        "Work Pressure":      data.get("Work Pressure", 0),
        "Study Satisfaction": data.get("Study Satisfaction", 0),
        "Job Satisfaction":   data.get("Job Satisfaction", 0),
        "Work/Study Hours":   data.get("Work/Study Hours"),
        "Sleep Duration":     data.get("Sleep Duration"),
        "Dietary Habits":     data.get("Dietary Habits"),
        "Have you ever had suicidal thoughts ?": data.get("Have you ever had suicidal thoughts ?"),
        "Family History of Mental Illness":      data.get("Family History of Mental Illness"),
        "Financial Stress": str(data.get("Financial Stress", "")),
    }

    try:
        result = predict_student(ml_input)
    except RuntimeError as exc:
        logger.exception("Prediction failed for response %d", response.id)
        messages.error(request, str(exc))
        return redirect("survey_list")

    PredictionResult.objects.create(
        response   = response,
        risk_score = result["risk_score"],
        prediction = result["prediction"],
        risk_level = result["risk_level"],
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


# ── Admin views ───────────────────────────────────────────────────────────────

@login_required
@user_passes_test(_is_admin)
def admin_dashboard(request):
    surveys = Survey.objects.annotate(
        response_count=Count("surveyresponse")
    ).order_by("-created_at")

    context = {
        "surveys":         surveys,
        "total_responses": SurveyResponse.objects.count(),
        "total_high_risk": PredictionResult.objects.filter(risk_level="High").count(),
    }
    return render(request, "admin_dashboard.html", context)


@login_required
@user_passes_test(_is_admin)
@require_POST
def create_survey(request):
    title = request.POST.get("title", "").strip()
    if not title:
        messages.error(request, "Title is required.")
    else:
        Survey.objects.create(
            id          = _random_id(),
            title       = title,
            description = request.POST.get("desc", "").strip(),
        )
        messages.success(request, "Survey created.")
    return redirect("admin_dashboard")


@login_required
@user_passes_test(_is_admin)
def survey_details(request, survey_id):
    survey       = get_object_or_404(Survey, id=survey_id)
    filter_level = request.GET.get("filter")

    responses = (
        SurveyResponse.objects
        .filter(survey=survey)
        .select_related("student", "predictionresult")
        .annotate(
            risk_score=F("predictionresult__risk_score"),
            risk_level=F("predictionresult__risk_level"),
        )
    )

    if filter_level in ("High", "Moderate", "Low"):
        responses = responses.filter(risk_level=filter_level)

    responses = responses.order_by("-risk_score", "-created_at")

    return render(request, "survey_details.html", {
        "survey":    survey,
        "responses": responses,
    })


@login_required
@user_passes_test(_is_admin)
def import_csv(request, survey_id):
    """
    GET  → render the import page
    POST → process the uploaded Google Forms (or plain) CSV
    """
    survey = get_object_or_404(Survey, id=survey_id)

    if request.method == "GET":
        return render(request, "import_csv.html", {"survey": survey})

    # ── POST ──────────────────────────────────────────────────────────────
    csv_file = request.FILES.get("csv_file")
    if not csv_file:
        messages.error(request, "No file selected. Please choose a CSV file.")
        return render(request, "import_csv.html", {"survey": survey})

    if not csv_file.name.lower().endswith(".csv"):
        messages.error(request, "Invalid file type — please upload a .csv file.")
        return render(request, "import_csv.html", {"survey": survey})

    # Decode — Google Forms exports UTF-8 with BOM
    try:
        decoded = csv_file.read().decode("utf-8-sig")
    except UnicodeDecodeError:
        try:
            csv_file.seek(0)
            decoded = csv_file.read().decode("latin-1")
        except Exception:
            messages.error(request, "Could not decode the file. Ensure it is UTF-8 encoded.")
            return render(request, "import_csv.html", {"survey": survey})

    reader = csv.DictReader(io.StringIO(decoded))

    if not reader.fieldnames:
        messages.error(request, "CSV file appears empty or has no header row.")
        return render(request, "import_csv.html", {"survey": survey})

    header_map = _normalise_headers(list(reader.fieldnames))

    imported            = 0
    skipped             = 0
    errors: list[str]   = []
    prediction_failures = 0

    for row_num, raw_row in enumerate(reader, start=2):
        # Re-key with canonical names; drop __SKIP__ columns (e.g. Timestamp)
        row = {
            canonical: v
            for orig, canonical in header_map.items()
            if canonical != "__SKIP__"
            and (v := raw_row.get(orig, "")) is not None
        }

        data, err = _parse_csv_row(row, row_num)
        if err:
            errors.append(err)
            skipped += 1
            continue

        try:
            student = Student.objects.create(
                name  = data["name"],
                email = data["email"],
            )
            response = SurveyResponse.objects.create(
                survey             = survey,
                student            = student,
                age                = data["Age"],
                gender             = data["Gender"],
                cgpa               = data["CGPA"],
                academic_pressure  = data["Academic Pressure"],
                work_pressure      = data.get("Work Pressure", 0),
                study_satisfaction = data["Study Satisfaction"],
                job_satisfaction   = data.get("Job Satisfaction", 0),
                work_study_hours   = data["Work/Study Hours"],
                sleep_duration     = data["Sleep Duration"],
                dietary_habits     = data["Dietary Habits"],
                suicidal_thoughts  = data["Have you ever had suicidal thoughts ?"],
                family_history     = data["Family History of Mental Illness"],
                financial_stress   = data["Financial Stress"],
            )
        except Exception as exc:
            errors.append(f"Row {row_num}: database error — {exc}")
            skipped += 1
            continue

        ml_input = {
            "Age":             data["Age"],
            "Gender":          data["Gender"],
            "CGPA":            data["CGPA"],
            "Academic Pressure":  data["Academic Pressure"],
            "Work Pressure":      data.get("Work Pressure", 0),
            "Study Satisfaction": data["Study Satisfaction"],
            "Job Satisfaction":   data.get("Job Satisfaction", 0),
            "Work/Study Hours":   data["Work/Study Hours"],
            "Sleep Duration":     data["Sleep Duration"],
            "Dietary Habits":     data["Dietary Habits"],
            "Have you ever had suicidal thoughts ?": data["Have you ever had suicidal thoughts ?"],
            "Family History of Mental Illness":      data["Family History of Mental Illness"],
            "Financial Stress": data["Financial Stress"],
        }

        try:
            result = predict_student(ml_input)
            PredictionResult.objects.create(
                response   = response,
                risk_score = result["risk_score"],
                prediction = result["prediction"],
                risk_level = result["risk_level"],
            )
        except Exception as exc:
            logger.warning("Prediction failed for imported row %d: %s", row_num, exc)
            prediction_failures += 1

        imported += 1

    # ── Result messages ───────────────────────────────────────────────────
    if imported:
        msg = f"Successfully imported {imported} response{'s' if imported != 1 else ''}."
        if prediction_failures:
            msg += (
                f" ({prediction_failures} risk prediction"
                f"{'s' if prediction_failures != 1 else ''} could not run — "
                "model artefacts may not be loaded on this server.)"
            )
        messages.success(request, msg)

    if skipped:
        messages.warning(
            request,
            f"Skipped {skipped} row{'s' if skipped != 1 else ''} due to errors.",
        )

    for e in errors[:5]:
        messages.error(request, e)
    if len(errors) > 5:
        messages.error(request, f"… and {len(errors) - 5} more row errors (check your CSV).")

    if not imported and not skipped:
        messages.warning(request, "The CSV file contained no data rows.")

    return redirect("survey_details", survey_id=survey_id)


@login_required
@user_passes_test(_is_admin)
def student_detail(request, id):
    response   = get_object_or_404(
        SurveyResponse.objects.select_related("student", "survey", "predictionresult"),
        id=id,
    )
    prediction = getattr(response, "predictionresult", None)
    return render(request, "student_detail.html", {
        "response":   response,
        "prediction": prediction,
    })


@login_required
@user_passes_test(_is_admin)
def survey_analytics(request, survey_id):
    survey      = get_object_or_404(Survey, id=survey_id)
    responses   = SurveyResponse.objects.filter(survey=survey)
    predictions = PredictionResult.objects.filter(
        response__in=responses
    ).select_related("response")

    level_counts = {
        "Low":      predictions.filter(risk_level="Low").count(),
        "Moderate": predictions.filter(risk_level="Moderate").count(),
        "High":     predictions.filter(risk_level="High").count(),
    }

    total     = predictions.count()
    high_risk = level_counts["High"]
    avg_score = predictions.aggregate(avg=Avg("risk_score"))["avg"] or 0.0

    high_pct      = round(high_risk / total * 100, 1) if total else 0
    avg_score_pct = round(avg_score * 100, 1)

    all_scores   = list(predictions.values_list("risk_score", flat=True))
    hist_buckets = [0, 0, 0, 0]
    for s in all_scores:
        if   s < 0.25: hist_buckets[0] += 1
        elif s < 0.50: hist_buckets[1] += 1
        elif s < 0.75: hist_buckets[2] += 1
        else:          hist_buckets[3] += 1

    trend_d = defaultdict(int)
    for pred in predictions:
        trend_d[pred.response.created_at.date().isoformat()] += 1
    all_days = sorted(trend_d.keys())

    cum, trend_rows = 0, []
    for d in all_days:
        cum += trend_d[d]
        trend_rows.append((d, trend_d[d], cum))

    bar_rows = [
        ("Low Risk",      level_counts["Low"],      "#476b2f"),
        ("Moderate Risk", level_counts["Moderate"], "#b8722a"),
        ("High Risk",     level_counts["High"],     "#9e2b2b"),
    ]
    hist_rows = list(zip(
        ["0 – 0.25", "0.25 – 0.5", "0.5 – 0.75", "0.75 – 1.0"],
        hist_buckets,
        ["#476b2f", "#888080", "#b8722a", "#9e2b2b"],
    ))

    low_pct     = round(level_counts["Low"] / total * 100)              if total else 0
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
        "trend_labels":  json.dumps(all_days),
        "trend_values":  json.dumps([trend_d[d] for d in all_days]),
        "hist_buckets":  json.dumps(hist_buckets),
        "bar_rows":      bar_rows,
        "hist_rows":     hist_rows,
        "trend_rows":    trend_rows,
        "low_pct":       low_pct,
        "low_mod_pct":   low_mod_pct,
    }
    return render(request, "survey_analytics.html", context)


@login_required
@user_passes_test(_is_admin)
@require_POST
def toggle_survey(request, survey_id):
    survey           = get_object_or_404(Survey, id=survey_id)
    survey.is_active = not survey.is_active
    survey.save()
    return redirect("admin_dashboard")