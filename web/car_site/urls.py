"""
car_site/urls.py — Root URL configuration.
"""
from django.urls import path, include

urlpatterns = [
    path("", include("predictor.urls")),
]
