from rest_framework_simplejwt.tokens import RefreshToken
from Users.Data.data import is_already_user, create_app_user, get_user_profile_data, \
    create_user_profile_data, is_profile_exsits_data, create_patient_user_data

def authorise_and_generate_token(phone_number,otp):
    user = is_already_user(phone_number)
    if user is None:
        user = create_app_user(phone_number)
        accountinfo = None
    else:
        accountinfo = get_user_profile_data(user.id,user.type) if user.type else None
    # Refresh token
    refresh_token = RefreshToken.for_user(user)

    return {
    'refresh': str(refresh_token),
    'access': str(refresh_token.access_token),
    'accountinfo' : accountinfo
    }


def create_user_profile(user_type,profile_data):
    profile =  create_user_profile_data(user_type,profile_data)
    return "Profile Created Successfully"

def create_user_using_patient(phone_number,user_type):
    user = is_already_user(phone_number)
    if user:
        return user
    else:
        return create_patient_user_data(phone_number,user_type)

def create_patient(user,user_type,profile_data):
    profile = is_profile_exsits_data(user.id,user.type)
    if profile_data['contact_number'] == None:
        profile_data['tester_type'] = 0
        profile_data['user_id'] = None
    else:
        user_id = create_user_using_patient(profile_data['contact_number'],user_type)
        patient = is_profile_exsits_data(user_id.id,user_id.type)
        if patient:
            patient.profile_id = profile
            patient.save()
            return 'Patient Added Successfully'
        profile_data['tester_type'] = 1
        profile_data['user_id'] = user_id
    profile_data['profile_id'] = profile
    patient = create_user_profile_data(user_type,profile_data)
    return 'Patient Created Successfully'


def update_personal_user_profile(profile,user_type,profile_data):
    profile.full_name = profile_data['full_name']
    profile.profile_pic = profile_data['profile_pic']
    profile.age = profile_data['age']
    profile.gender = profile_data['gender']
    profile.state = profile_data['state']
    if user_type == 'P':
        profile.contact_number = profile_data['contact_number']
    else:
        profile.organization_name = profile_data['organization_name']
        profile.profile_type = profile_data['profile_type']
    profile.save()
    return "Profile Updated Successfully"