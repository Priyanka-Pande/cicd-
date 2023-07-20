from Dashboard.models import *
from django.db.models import F,Value
from Users.models import Users
from django.db.models import Count
from AppGluco.models import GlucoResultTable, VideoTable
from Users.models import PersonalProfile, ProfessionalProfile


def delete_user_request_data(values):
    return DeleteAccountRequests.objects.create(
        **values
    )

def total_no_of_users_data():
    return Users.objects.filter(
        role = 'User'
    ).count()

def total_no_of_daily_users_data(date):
    return Users.objects.filter(
        role = 'User',
        date_joined__date = date
    ).count()

def total_no_of_blocked_users_data():
    return Users.objects.filter(
        role = 'User',
        status = 'B'
    ).count()

def total_no_of_blocked_daily_users_data(date):
    return Users.objects.filter(
        role = 'User',
        status = 'B',
        date_joined__date = date
    ).count()

def total_no_of_gluco_test_data():
    return GlucoResultTable.objects.all().count()

def total_no_of_gluco_test_today_data(date):
    return GlucoResultTable.objects.filter(
        reported_date__date = date
    ).count()

def total_no_of_videos_data():
    return VideoTable.objects.all().count()

def total_no_of_videos_today_data(date):
    return VideoTable.objects.filter(
        created_on__date = date
    ).count()

def users_count_by_month(current_year):
    return Users.objects.filter(date_joined__year=current_year) \
    .extra({'month': "EXTRACT(month FROM date_joined)"}) \
    .values('month') \
    .annotate(total_users=Count('id')) \
    .order_by('month')

def test_count_by_month(current_year):
    return GlucoResultTable.objects.filter(reported_date__year=current_year) \
    .extra({'month': "EXTRACT(month FROM reported_date)"}) \
    .values('month') \
    .annotate(total_users=Count('id')) \
    .order_by('month')


def get_all_personal_users_data():
    return PersonalProfile.objects.annotate(
        account_type = Value("Personal")
    ).values(
        'full_name',
        'account_type',
        users_id = F('user_id__id'),
        phone_numeber = F('user_id__phone_number'),
        status = F('user_id__status'),
    ).filter(user_id__role='User',tester_type=2)

def get_all_personal_users_by_professional_data():
    return PersonalProfile.objects.annotate(
        account_type = Value("Personal")
    ).values(
        
        'full_name',
        'account_type',
        users_id = F('user_id__id'),
        phone_numeber = F('user_id__phone_number'),
        status = F('user_id__status'),
    ).filter(user_id__role='User',tester_type=1)

def get_all_professional_users_data():
    return ProfessionalProfile.objects.annotate(
        account_type = Value("Professional")
    ).values(
        'user_id_id',
        'full_name',
        'account_type',
        users_id = F('user_id__id'),
        phone_numeber = F('user_id__phone_number'),
        status = F('user_id__status'),
    ).filter(user_id__role='User')


def get_user_for_action(user_id):
    return Users.objects.filter(id=user_id).first()


def get_gluco_result_value_data():
    return GlucoResultTable.objects.values(
        'gluco_value',
        patient_name = F('patient_id__full_name'),
        professional_id = F('profile_id__id'),
        video_id = F('video_location__id'),
        video = F('video_location__video_file')
    ).all()


def get_gluco_video_data(video_id):
    return VideoTable.objects.filter(id=video_id).first()
