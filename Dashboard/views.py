import logging
from django.contrib import messages
from Dashboard.Service.service import *
from django.contrib.auth import logout
from rest_framework.views import APIView
from django.contrib.auth.decorators import login_required
from django.template.response import TemplateResponse

# Create your views here.
logging.basicConfig()
logger = logging.getLogger(__name__)


class DeleteAccountRequest(APIView):

    def get(self,request):
        # Render the initial HTML page with an empty form
        return TemplateResponse(request, 'deleteAccount.html')

    def post(self, request):
        data = request.data
        patient_name = data.get('patient_name')
        phone_number = data.get('phone_number')
        message = data.get('message')
        logger.info('request for phone_number: %s', phone_number)
        if not phone_number:
            logger.critical("video_file  is required")
            messages.error(request, 'Phone number not found.')
            return TemplateResponse(request, 'deleteAccount.html')
        if not message:
            logger.critical("message is required")
            messages.error(request, 'Message not found.')
            return TemplateResponse(request, 'deleteAccount.html')
        try:
            values = {"patient_name":patient_name,"phone_number":phone_number,"message":message}
            response = request_to_delete_account(values)
            messages.success(request, 'Your request has been sent successfully.')
            logger.info('Response sent for phone_number: %s', phone_number)
            return TemplateResponse(request, 'deleteAccount.html')
        except Exception as e:
            logger.info('Response failed for phone_number: %s and Error is: %s',
                        phone_number, e)
            messages.error(request, 'Your Request is failed. Please Try again')
            return TemplateResponse(request, 'deleteAccount.html')


class AdminLoginPage(APIView):

    def get(self,request):
        # Render the initial HTML page with an empty form
        return TemplateResponse(request, 'login.html')

    def post(self,request):
        data = request.data
        username = data.get('username')
        password = data.get('password')
        logger.info('request for username: %s', username)
        if not username:
            logger.critical("user_name  is required")
            messages.error(request, 'User Name not found.')
            return TemplateResponse(request, 'login.html')
        if not password:
            logger.critical("password  is required")
            messages.error(request, 'Password not found.')
            return TemplateResponse(request, 'login.html')
        try:
            response =  authenticate_admin_user(request,username,password)
            logger.info('Response sent for username: %s', username)
            return response
        except Exception as e:
            logger.info('Response failed for username: %s and Error is: %s',
                        username, e)
            messages.error(request, 'Something went wrong. Please Try again')
            return TemplateResponse(request, 'login.html')



@login_required(login_url='/admin/login/')
def dashboard_home(request):
    user = request.user
    logger.info('request for user_id: %s', user)
    try:
        response = total_number_of_users(request)
        logger.info('Response sent for user_id: %s', user)
        return TemplateResponse(request, './Dashboard/home.html',response)
    except Exception as e:
        logger.info('Response failed for user_id: %s and Error is: %s',
                    user, e)
        
        # Render the initial HTML page with an empty form
        # return TemplateResponse(request, './Dashboard/home.html',response)


@login_required(login_url='/admin/login/')
def admin_logout(request):
    logout(request)
    return redirect('/admin/login/')
