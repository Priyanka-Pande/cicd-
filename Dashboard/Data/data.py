from Dashboard.models import *
from Users.models import Users
from django.db.models import Count
from django.db.models.functions import ExtractMonth
from AppGluco.models import GlucoResultTable, VideoTable


def delete_user_request_data(values):
    return DeleteAccountRequests.objects.create(
        **values
    )

def total_no_of_users_data():
    return Users.objects.filter(
        role = 'User'
    ).count()

def total_no_of_blocked_users_data():
    return Users.objects.filter(
        role = 'User',
        status = 'B'
    ).count()

def total_no_of_gluco_test_data():
    return GlucoResultTable.objects.all().count()

def total_no_of_videos_data():
    return VideoTable.objects.all().count()

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