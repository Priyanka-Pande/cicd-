from .views import *
from django.urls import path
from rest_framework_simplejwt.views import TokenVerifyView


urlpatterns = [
    path('authenticate/', LoginAPI.as_view(), name='token_obtain_pair'),
    path('token/refresh/', CustomTokenRefreshView.as_view(), name='token_refresh'),
    path('token/verify/', TokenVerifyView.as_view(), name='token_verify'),
    path('professional/Profile/', ProfessionalProfile.as_view(), name='professional_profile_actions'),
    path('personal/Profile/', PersonalProfile.as_view(), name='personal_profile_actions'),
    path('accounts/', AccountInfo.as_view(), name='accounts_info'),
    path('patient/create/', CreatePatient.as_view(), name='create_patient'),
    path('generateOtp/', GenerateOTP.as_view(), name='generate_otp'),
]
