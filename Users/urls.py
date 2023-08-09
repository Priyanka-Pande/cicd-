from .views import *
from django.urls import path
from Dashboard.views import DeleteAccountRequest
from rest_framework_simplejwt.views import TokenVerifyView


urlpatterns = [
    path('authenticate/', LoginAPI.as_view(), name='token_obtain_pair'),
    path('token/refresh/', RefreshTokenAPI.as_view(), name='token_refresh'),
    path('token/verify/', TokenVerifyView.as_view(), name='token_verify'),
    path('professional/Profile/', ProfessionalProfile.as_view(), name='professional_profile_actions'),
    path('personal/Profile/', PersonalProfile.as_view(), name='personal_profile_actions'),
    path('update/personal/Profile/', UpdatePersonalProfile.as_view(), name='personal_profile_updation'),
    path('update/professional/Profile/', UpdatePerofessionalProfile.as_view(), name='professional_profile_updation'),
    path('accounts/', AccountInfo.as_view(), name='accounts_info'),
    path('profileType/', ProfileTypeInfo.as_view(), name='profile_type'),
    path('patient/create/', CreatePatient.as_view(), name='create_patient'),
    path('generateOtp/', GenerateOTP.as_view(), name='generate_otp'),
    path('request/deleteAccount/', DeleteAccountRequest.as_view(), name='test'),
]
