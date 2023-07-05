from Users.models import ProfessionalProfile, PersonalProfile


USER_TYPE_MODELS = {
    "P" : PersonalProfile,
    "MR" : ProfessionalProfile,
    "B" : ProfessionalProfile,
}

ACCOUNT_INFO_QUERY_VALUES = {
    "P" : ['id','full_name','profile_pic','age','gender','state',],
    "MR" : ['id','full_name','organization_name','profile_type','profile_pic',],
    "B" : ['id','full_name','profile_pic','age','gender','state']
}

