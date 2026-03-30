# Generated migration — replaces 0001/0002/0003 with clean schema

from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    initial = True
    dependencies = []

    operations = [
        migrations.CreateModel(
            name="Survey",
            fields=[
                ("id",          models.CharField(max_length=5, primary_key=True, serialize=False)),
                ("title",       models.CharField(max_length=200)),
                ("description", models.TextField(blank=True)),
                ("is_active",   models.BooleanField(default=True)),
                ("created_at",  models.DateTimeField(auto_now_add=True)),
                ("closed_at",   models.DateTimeField(blank=True, null=True)),
            ],
        ),
        migrations.CreateModel(
            name="Student",
            fields=[
                ("id",    models.BigAutoField(auto_created=True, primary_key=True, serialize=False)),
                ("name",  models.CharField(max_length=100)),
                ("email", models.EmailField()),
            ],
        ),
        migrations.CreateModel(
            name="SurveyResponse",
            fields=[
                ("id",                 models.BigAutoField(auto_created=True, primary_key=True, serialize=False)),
                ("age",                models.FloatField()),
                ("gender",             models.CharField(max_length=10)),
                ("cgpa",               models.FloatField()),
                ("academic_pressure",  models.IntegerField()),
                ("work_pressure",      models.IntegerField(default=0)),
                ("study_satisfaction", models.IntegerField()),
                ("job_satisfaction",   models.IntegerField(default=0)),
                ("work_study_hours",   models.FloatField()),
                ("sleep_duration",     models.CharField(max_length=50)),
                ("dietary_habits",     models.CharField(max_length=50)),
                ("suicidal_thoughts",  models.CharField(max_length=10)),
                ("family_history",     models.CharField(max_length=10)),
                ("financial_stress",   models.CharField(max_length=10)),
                ("created_at",         models.DateTimeField(auto_now_add=True)),
                ("survey",  models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to="core.survey")),
                ("student", models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to="core.student")),
            ],
        ),
        migrations.CreateModel(
            name="PredictionResult",
            fields=[
                ("id",         models.BigAutoField(auto_created=True, primary_key=True, serialize=False)),
                ("risk_score", models.FloatField()),
                ("prediction", models.IntegerField()),
                ("risk_level", models.CharField(max_length=10)),
                ("created_at", models.DateTimeField(auto_now_add=True)),
                ("response",   models.OneToOneField(on_delete=django.db.models.deletion.CASCADE, to="core.surveyresponse")),
            ],
        ),
    ]