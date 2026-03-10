"""
car_site/wsgi.py — WSGI entry point.
"""
import os
from django.core.wsgi import get_wsgi_application

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "car_site.settings")
application = get_wsgi_application()
