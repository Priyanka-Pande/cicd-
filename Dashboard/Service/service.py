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
        messages.success(request, 'Logged in Successfully')
        return redirect('/admin/dashboard/')
    else:
        messages.error(request, 'Invalid Credentials')
        return TemplateResponse(request, 'login.html')


def create_dict_for_monthly_data(month_dict,total_data):
    for item in total_data:
        month_dict[MONTH[int(item['month'])]] = item['total_users']
    return month_dict

def daily_report():
    date = datetime.datetime.now().date()
    no_of_users_today = total_no_of_daily_users_data(date)
    no_of_users_blocked_today = total_no_of_blocked_daily_users_data(date)
    no_of_test_condected_today = total_no_of_gluco_test_today_data(date)
    no_of_videos_today = total_no_of_videos_today_data(date)
    no_of_test_failed_today = no_of_videos_today - no_of_test_condected_today

    return {"today_user_count" : no_of_users_today, "blocked_user_today":no_of_users_blocked_today,
        "today_test_condected" : no_of_test_condected_today, "today_video_count":no_of_videos_today,
        "today_failed_test":no_of_test_failed_today}

def dashboard_total_report():
    no_of_users = total_no_of_users_data()
    no_of_user_blocked = total_no_of_blocked_users_data()
    no_of_test_condected = total_no_of_gluco_test_data()
    no_of_videos = total_no_of_videos_data()
    total_failed_tests = no_of_videos - no_of_test_condected

    return {"total_users_count":no_of_users, "total_blocked_user" :no_of_user_blocked, 
        "total_test_condected" :no_of_test_condected, "total_video_count" :no_of_videos,
        "total_failed_test" : total_failed_tests}

def montly_graphs(current_year):
    total_users = users_count_by_month(current_year)
    total_tests = test_count_by_month(current_year)
    monthly_test = create_dict_for_monthly_data(MONLTY_TEST_VALUES,total_tests)
    monthly_users = create_dict_for_monthly_data(MONLTY_VALUES,total_users)

    return {
        "labels" : list(monthly_users.keys()),
        "test_data" : list(monthly_test.values()),
        "users_data" : list(monthly_users.values()),
    }

def total_number_of_users(request):
    current_year = datetime.datetime.now().year
    total_report_data = dashboard_total_report()
    daily_report_data = daily_report()
    mothly_graph_data = montly_graphs(current_year)

    return  {
        "total_report_data" : total_report_data,
        "daily_report_data" : daily_report_data,
        "mothly_graph_data" : mothly_graph_data,
    }


def users_list_table(request):
    users_by_own = []
    personal_users = get_all_personal_users_data()
    professional_users = get_all_professional_users_data()
    users_by_own.extend(personal_users)
    users_by_own.extend(professional_users)
    personal_users_by_professional = get_all_personal_users_by_professional_data()
    
    return {
        "personal_users" : users_by_own,
        "user_by_professional" : personal_users_by_professional
    }


def block_user_by_admin(request,user_id):
    user = get_user_for_action(user_id)
    if user.status == 'A':
        user.status = 'B'
        user.updated_on = datetime.datetime.now()
        messages.error(request, 'User Blocked successfully.')
    else:
        user.status = 'A'
        user.updated_on = datetime.datetime.now()
        messages.success(request, 'User UnBlocked successfully.')
    user.save()
    return redirect('/admin/allUsers/')


def gluco_result_table(request):
    result = get_gluco_result_value_data()
    for item in result:
        item['video'] = f'https://storage.googleapis.com/glucoqr-p1-preprod/{item["video"]}'
    return {
        'gluco_result_data' : result
    }

def download_gluco_video(video_id):
    # result = get_gluco_video_data(video_id)
    # response = f'https://storage.googleapis.com/glucoqr-p1-preprod/{result.video_file}'
    # response = HttpResponse(file_content, content_type='application/octet-stream')
    # response['Content-Disposition'] = f'attachment; filename="{file_name}"'
    return 