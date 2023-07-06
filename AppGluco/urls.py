from .views import *
from django.urls import path


urlpatterns = [
    path('professional/allpatient/', GetAllPatientsProfessional.as_view(), name='get_professional_all_patient'),
    path('personal/reports/', GetResultPersonal.as_view(), name='get_personal_user_reports'),
    path('uploadVideo/', UploadVideo.as_view(), name='upload_video'),
]
