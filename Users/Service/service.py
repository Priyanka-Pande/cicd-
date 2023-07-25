import jwt
from rest_framework.exceptions import AuthenticationFailed
from django.conf import settings
from rest_framework_simplejwt.tokens import RefreshToken
from Users.Data.data import is_already_user, create_app_user, get_user_profile_data, \
    create_user_profile_data, is_profile_exsits_data, create_patient_user_data,\
        is_patient_already_exists_data,is_user

def decode_refresh_token(refresh_token):
    try:
        payload = jwt.decode(refresh_token, settings.SECRET_KEY, algorithms=["HS256"])
        return payload
    except jwt.ExpiredSignatureError:
        raise AuthenticationFailed("Refresh token has expired.")
    except jwt.InvalidTokenError:
        raise AuthenticationFailed("Invalid refresh token.")

def authorise_and_generate_token(phone_number,otp):
    user = is_already_user(phone_number)
    is_exists = False
    user_type = ""
    test_type = ""
    if user is None:
        user = create_app_user(phone_number)
    else:
        if user.type is not None:
            is_exists = True
            user_type = user.type
        if user.type == "P":
            profile = is_profile_exsits_data(user.id,user.type)
            test_type = profile.tester_type if profile != None else test_type
    if user.status != 'A':
        return{
            "message":"Your Account Is Blocked. Please Contact Our Team."
        }

    # Refresh token
    refresh_token = RefreshToken.for_user(user)

    return {
    'refresh': str(refresh_token),
    'access': str(refresh_token.access_token),
    'is_exists' : is_exists,
    "user_id" : user.id,
    "user_type" :user_type,
    "test_type" :test_type
    }


def create_user_profile(user_type,profile_data):
    profile = is_profile_exsits_data(profile_data['user_id'].id,user_type)
    if profile:
        return {"message":"Profile Already Exsists"}
    profile =  create_user_profile_data(user_type,profile_data)
    return {"message":"Profile Created Successfully"}

def create_user_using_patient(phone_number,user_type):
    user = is_already_user(phone_number)
    if user == None:
        return create_patient_user_data(phone_number,user_type)
    else:
        return user

def create_patient(user,user_type,profile_data):
    profile = is_profile_exsits_data(user.id,user.type)
    if profile_data['contact_number'] == None:
        profile_data['tester_type'] = 0
        profile_data['user_id'] = None
    else:
        user_id = create_user_using_patient(profile_data['contact_number'],user_type)
        if user_id.type == 'MR':
            return {"message":'Already Professional User Exsists'}
        patient = is_profile_exsits_data(user_id.id,user_id.type)
        if patient:
            patient.profile_id = profile
            patient.save()
            return {
                "message":'Patient Added Successfully',
                "patient_id" : patient.id
                }
        profile_data['tester_type'] = 1
        profile_data['user_id'] = user_id
    profile_data['profile_id'] = profile
    patient_id = create_user_profile_data(user_type,profile_data)
    return {
        "message":'Patient Created Successfully',
        "patient_id" : patient_id.id
        }


def update_personal_user_profile(profile,profile_data):
    profile.full_name = profile_data['full_name'] if profile_data['full_name'] != None else profile.full_name
    profile.profile_pic = profile_data['profile_pic'] if profile_data['profile_pic'] != None else profile.profile_pic
    profile.state = profile_data['state'] if profile_data['state'] != None else profile.state
    profile.age = profile_data['age'] if profile_data['age'] != None else profile.age
    profile.tester_type = 2
    profile.save()
    return {"message":"Profile Updated Successfully"}

def update_perofessional_user_profile(profile,profile_data):
    profile.full_name = profile_data['full_name'] if profile_data['full_name'] != None else profile.full_name
    profile.profile_pic = profile_data['profile_pic'] if profile_data['profile_pic'] != None else profile.profile_pic
    profile.organization_name = profile_data['organization_name'] if profile_data['organization_name'] != None else profile.organization_name
    profile.profile_type = profile_data['profile_type'] if profile_data['profile_type'] != None else profile.profile_type
    profile.tester_type = 2
    profile.save()
    return {"message":"Profile Updated Successfully"}

def verify_otp(phone_number,otp):
    opt_check = '1234'

    if opt_check == otp:
        return True 
    else:
        return False


def is_patient_already_exists(phone_number,user):
    profile = is_profile_exsits_data(user.id,user.type)
    patient_user_id = is_already_user(phone_number)
    if patient_user_id is not None:
        is_exists = is_patient_already_exists_data(patient_user_id,profile)
    else:
        is_exists = False
    return is_exists


def refresh_the_access_token(refresh_token):
    payload = decode_refresh_token(refresh_token)
    user = is_user(payload['user_id'])
    if user.status != 'A':
        return {
            "message":"Your Account Is Blocked. Please Contact Our Team."
        }
    else:
        refresh_token_obj = RefreshToken(refresh_token)
        access_token = str(refresh_token_obj.access_token)
        return {
            "access_token": access_token
        }