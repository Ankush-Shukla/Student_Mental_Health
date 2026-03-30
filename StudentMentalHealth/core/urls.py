# core/urls.py
from django.urls import path
from . import views

urlpatterns = [
    path("", views.login, name="login_page"),
    path("survey/", views.survey_form, name="survey_form"),
    path("create-form/", views.create_survey, name="survey_form"),
    path("submit/", views.submit_survey, name="submit_survey"),
     path("result/", views.result_page, name="result_page"),
     path("survey/<str:id>", views.survey_details, name="survey_details"),

]