from .views import *
from django.urls import path


urlpatterns = [
    path('login/', AdminLoginPage.as_view(), name='login'),
    path('logout/', AdminLogout, name='log_out_admin'),
    path('dashboard/', DashboardHome, name='dashboard-home'),
    path('allUsers/', ListUserPage, name='list_all_users'),
    path('blockUser/<int:id>/', BlockUser, name='block_all_users'),
    path('glucoResults/', GlucoResultPage, name='gluco_result'),
    path('downloadVideo/<int:id>/', DownloadGlucoVideo, name='download_video'),
]
