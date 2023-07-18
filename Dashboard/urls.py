from .views import *
from django.urls import path


urlpatterns = [
    path('login/', AdminLoginPage.as_view(), name='login'),
    path('dashboard/', dashboard_home, name='dashboard-home'),
    path('logout/', admin_logout, name='log_out_admin'),
]
