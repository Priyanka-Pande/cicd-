from django.db.models import F
from AppGluco.models import GlucoResultTable, VideoTable
from Users.Data.data import get_patient_data

def get_all_patients_of_professional_data(profile_id):
    return GlucoResultTable.objects.values(
        'id',
        'profile_id',
        'patient_id',
        'gluco_value',
        'reported_date',
        full_name = F('patient_id__full_name'),
        age = F('patient_id__age'),
        gender = F('patient_id__gender'),
        state = F('patient_id__state'),
        ).filter(profile_id=profile_id)

def get_result_report_personal_data(patient_id):
    return GlucoResultTable.objects.values(
        'id',
        'profile_id',
        'patient_id',
        'gluco_value',
        'reported_date',
        full_name = F('patient_id__full_name'),
        age = F('patient_id__age'),
        gender = F('patient_id__gender'),
        state = F('patient_id__state'),
        ).filter(patient_id=patient_id)


def upload_video_file(patient_id,video_file):
    profile_id = get_patient_data(patient_id)
    return VideoTable.objects.create(
        profile_id = profile_id,
        video_file = video_file,
    )