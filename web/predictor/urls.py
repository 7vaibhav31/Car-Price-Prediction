"""
predictor/urls.py — URL patterns for the predictor app.
"""
from django.urls import path
from . import views

urlpatterns = [
    path("", views.predict_view, name="home"),
    path("predict/", views.predict_view, name="predict"),
]
