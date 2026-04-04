"""
core/urls.py
"""

from django.urls import path
from . import views

urlpatterns = [
    path("",                               views.login_page,        name="login_page"),
    path("logout/",                        views.logout_view,       name="logout"),
    path("surveys/",                       views.survey_list,       name="survey_list"),
    path("surveys/<str:survey_id>/",       views.survey_form,       name="survey_form"),
    path("submit/",                        views.submit_survey,     name="submit_survey"),
    path("result/",                        views.result_page,       name="result_page"),

    path("admin-dashboard/",              views.admin_dashboard,   name="admin_dashboard"),
    path("admin-dashboard/create/",       views.create_survey,     name="create_survey"),
    path("admin-dashboard/<str:survey_id>/details/",   views.survey_details,  name="survey_details"),
    path("admin-dashboard/<str:survey_id>/analytics/", views.survey_analytics, name="survey_analytics"),
    path("admin-dashboard/<str:survey_id>/toggle/",    views.toggle_survey,    name="toggle_survey"),

    path("student/<int:id>/", views.student_detail, name="student_detail"),

    
]