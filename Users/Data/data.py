from Users.models import Users, PersonalProfile
from django.db.models import F
from Users.Constants.constants import USER_TYPE_MODELS ,ACCOUNT_INFO_QUERY_VALUES

def is_already_user(phone_number):
    return Users.objects.filter(phone_number=phone_number).first()


def create_app_user(phone_number):
    return Users.objects.create(phone_number=phone_number,username=phone_number)


def get_user_profile_data(user_id,user_type):
    model = USER_TYPE_MODELS[user_type]
    values = ACCOUNT_INFO_QUERY_VALUES[user_type]
    result =  model.objects.values(user_type = F('user_id__type'),*values).filter(user_id=user_id).first()
    return result


def is_profile_exsits_data(user_id,profile_type):
    model = USER_TYPE_MODELS[profile_type]
    return model.objects.filter(user_id=user_id).first()


def create_user_profile_data(user_type,profile_data):
    user = profile_data['user_id']
    model = USER_TYPE_MODELS[user_type]
    profile = model.objects.create(**profile_data)
    if profile and user:
        user.type = user_type
        user.save()
    return profile

def create_patient_user_data(phone_number,user_type):
    return Users.objects.create(phone_number=phone_number,username=phone_number,type=user_type)


def get_patient_data(patient_id):
    return PersonalProfile.objects.filter(id=patient_id).first()

def is_patient_already_exists_data(patient_user_id,profile):
    return PersonalProfile.objects.filter(user_id=patient_user_id,profile_id=profile).first()


def is_user(user_id):
    return Users.objects.get(id=user_id)