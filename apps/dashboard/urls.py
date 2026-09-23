from django.urls import path

from .views import admin_overview, dashboard, fatigue_monitor, home

urlpatterns = [
    path("", home, name="home"),
    path("dashboard/", dashboard, name="dashboard"),
    path("fatigue/", fatigue_monitor, name="fatigue-monitor"),
    path("staff/overview/", admin_overview, name="admin-overview"),
]
