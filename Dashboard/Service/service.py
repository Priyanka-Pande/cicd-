from Dashboard.Data.data import *
from django.contrib.auth import authenticate, login
from django.contrib import messages
from django.template.response import TemplateResponse
from django.shortcuts import redirect
import datetime
from Dashboard.Constants.constants import MONLTY_VALUES, MONTH, MONLTY_TEST_VALUES


def request_to_delete_account(values):
    result = delete_user_request_data(values)
    return result


def authenticate_admin_user(request,username,password):
    user = authenticate(username=username,password=password)
    
    if user is not None and user.role == 'Admin':
        login(request, user)
        messages.success(request, 'Your request has been sent successfully.')
        return redirect('/admin/dashboard/')
    else:
        messages.error(request, 'Invalid Credentials')
        return TemplateResponse(request, 'login.html')


def create_dict_for_monthly_data(month_dict,total_data):
    for item in total_data:
        month_dict[MONTH[int(item['month'])]] = item['total_users']
    return month_dict


def total_number_of_users(request):
    no_of_users = total_no_of_users_data()
    no_of_user_blocked = total_no_of_blocked_users_data()
    no_of_test_condected = total_no_of_gluco_test_data()
    no_of_videos = total_no_of_videos_data()
    no_of_pictures = no_of_videos * 60 
    current_year = datetime.datetime.now().year
    total_users = users_count_by_month(current_year)
    total_tests = test_count_by_month(current_year)
    monthly_test = create_dict_for_monthly_data(MONLTY_TEST_VALUES,total_tests)
    monthly_users = create_dict_for_monthly_data(MONLTY_VALUES,total_users)

    return  {
        "total_users" : no_of_users,
        "total_blocked_users" : no_of_user_blocked,
        "total_tests" : no_of_test_condected,
        "total_videos" : no_of_videos,
        "total_pictures" : no_of_pictures,
        "labels" : list(monthly_users.keys()),
        "users_data" : list(monthly_users.values()),
        "test_data" : list(monthly_test.values())
    }