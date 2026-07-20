"""
URL configuration for fraud project.

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/6.0/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path

from core.views import (
    analytics,
    dashboard,
    landing,
    login_view,
    logout_view,
    profile,
    register_view,
    transaction_details,
    transactions,
    upload_csv,
    upload_history,
)

urlpatterns = [
    path('', landing, name='landing'),
    path('login/', login_view, name='login'),
    path('logout/', logout_view, name='logout'),
    path('register/', register_view, name='register'),
    path('dashboard/', dashboard, name='dashboard'),
    path('dashboard/upload/', upload_csv, name='upload_csv'),
    path('dashboard/transactions/', transactions, name='transactions'),
    path('dashboard/transactions/<int:transaction_id>/', transaction_details, name='transaction_details'),
    path('dashboard/history/', upload_history, name='upload_history'),
    path('dashboard/analytics/', analytics, name='analytics'),
    path('dashboard/profile/', profile, name='profile'),
    path('admin/', admin.site.urls),
]
