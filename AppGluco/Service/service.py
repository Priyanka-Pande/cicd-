from Users.Data.data import is_profile_exsits_data
from AppGluco.Data.data import *
from threading import Thread
# import torch
# import numpy as np
# import cv2
# from tensorflow.keras.models import load_model


def get_professional_all_patients(user):
    profile_id = is_profile_exsits_data(user.id,user.type)
    patients = get_all_patients_of_professional_data(profile_id)
    return patients

def get_report_for_personal(user):
    patient_id = is_profile_exsits_data(user.id,user.type)
    patients = get_result_report_personal_data(patient_id)
    return patients


def upload_video_for_patient(patient_id,video_file):
    video = upload_video_file(patient_id,video_file)
    return {"message":"Video Uploaded Successfully"}