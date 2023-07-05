import logging
import phonenumbers
from rest_framework import status
from rest_framework.response import Response
from rest_framework.views import APIView
from rest_framework.permissions import IsAuthenticated
from rest_framework_simplejwt.views import TokenViewBase
from Users.serializers import CustomTokenRefreshSerializer
from Users.Service.service import *
from Users.Data.data import is_profile_exsits_data

# Create your views here.

logging.basicConfig()
logger = logging.getLogger(__name__)


class CustomTokenRefreshView(TokenViewBase):
    """
        Renew tokens (access and refresh) with new expire time based on specific user's access token with other account
        info.
    """
    serializer_class = CustomTokenRefreshSerializer


class LoginAPI(APIView):

    def post(self, request):
        data = request.data
        phone_number = data.get('phone_number')
        otp = data.get('otp')
        logger.info('request for phone_number: %s', phone_number)
        if not phone_number:
            logger.critical("phone_number  is required")
            return Response("Phone Number is required", status=status.HTTP_400_BAD_REQUEST)
        if not otp :
            logger.critical("OTP is required")
            return Response("OTP is required", status=status.HTTP_400_BAD_REQUEST)
        try:
            parsed_number = phonenumbers.parse(phone_number, None)
        except phonenumbers.NumberParseException as e:
            logger.critical("Country code is missing")
            return Response("Country code required", status=status.HTTP_400_BAD_REQUEST)
        if not phonenumbers.is_valid_number(parsed_number):
            logger.critical("phone_number is not valid")
            return Response("Phone Number is Valid", status=status.HTTP_400_BAD_REQUEST)
        try:
            response = authorise_and_generate_token(phone_number,otp)
            logger.info('Response sent for phone_number: %s', phone_number)
            return Response(response, status=status.HTTP_200_OK)
        except Exception as e:
            logger.info('Response failed for phone_number: %s and Error is: %s',
                        phone_number, e)
            return Response("Something Went Wrong", status=status.HTTP_500_INTERNAL_SERVER_ERROR)


class ProfessionalProfile(APIView):
    permission_classes = [IsAuthenticated]

    def post(self,request):
        data = request.data
        user_id = request.user
        full_name = data.get('full_name')
        organization_name = data.get('organization_name')
        profile_type = data.get('profile_type')
        profile_pic = data.get('profile_pic')
        age = data.get('age')
        gender = data.get('gender')
        state = data.get('state')
        logger.info('request for action profile for user_id: %s', user_id)
        if not full_name:
            logger.critical("full_name  is required")
            return Response("Full Name is required", status=status.HTTP_400_BAD_REQUEST)
        if not organization_name:
            logger.critical("organization_name  is required")
            return Response("Organization Name is required", status=status.HTTP_400_BAD_REQUEST)
        if not profile_type:
            logger.critical("profile_type  is required")
            return Response("Profile Type is required", status=status.HTTP_400_BAD_REQUEST)
        if not state:
            logger.critical("state is required")
            return Response("State is required", status=status.HTTP_400_BAD_REQUEST)
        profile_data = {"user_id":user_id,"full_name":full_name,"organization_name":organization_name,
            "profile_type":profile_type,"profile_pic":profile_pic,
            "age":age,"gender":gender,"state":state}
        try:
            user_type = 'MR'
            profile = is_profile_exsits_data(user_id,user_type)
            if profile:
                response = update_personal_user_profile(profile,user_type,profile_data)
                logger.critical("professional profile update")
                return Response(response, status=status.HTTP_200_OK)
            response = create_user_profile(user_type,profile_data)
            logger.info('Response sent to create profile for user_id: %s', user_id)
            return Response(response, status=status.HTTP_200_OK)
        except Exception as e:
            logger.info('Response failed to create profile for user_id: %s and Error is: %s',
                        user_id, e)
            return Response("Something Went Wrong", status=status.HTTP_500_INTERNAL_SERVER_ERROR)


class PersonalProfile(APIView):
    permission_classes = [IsAuthenticated]

    def post(self,request):
        data = request.data
        user_id = request.user
        full_name = data.get('full_name')
        profile_pic = data.get('profile_pic')
        age = data.get('age')
        gender = data.get('gender')
        state = data.get('state')
        contact_number = data.get('contact_number')
        logger.info('request for action profile for user_id: %s', user_id)
        if not full_name:
            logger.critical("full_name  is required")
            return Response("Full Name is required", status=status.HTTP_400_BAD_REQUEST)
        if not contact_number:
            logger.critical("contact_number  is required")
            return Response("Contact Number is required", status=status.HTTP_400_BAD_REQUEST)
        if not age:
            logger.critical("age  is required")
            return Response("Age Type is required", status=status.HTTP_400_BAD_REQUEST)
        if not state:
            logger.critical("state is required")
            return Response("State is required", status=status.HTTP_400_BAD_REQUEST)
        if not gender:
            logger.critical("gender is required")
            return Response("Gender is required", status=status.HTTP_400_BAD_REQUEST)
        profile_data = {"user_id":user_id,"full_name":full_name,"profile_pic":profile_pic,
            "age":age,"gender":gender,"state":state,"contact_number":contact_number,"tester_type":2}
        try:
            user_type = 'P'
            profile = is_profile_exsits_data(user_id,user_type)
            if profile:
                response = update_personal_user_profile(profile,user_type,profile_data)
                logger.critical("personal profile update")
                return Response(response, status=status.HTTP_200_OK)
            response = create_user_profile(user_type,profile_data)
            logger.info('Response sent to create profile for user_id: %s', user_id)
            return Response(response, status=status.HTTP_200_OK)
        except Exception as e:
            logger.info('Response failed to create profile for user_id: %s and Error is: %s',
                        user_id, e)
            return Response("Something Went Wrong", status=status.HTTP_500_INTERNAL_SERVER_ERROR)


class AccountInfo(APIView):
    permission_classes = [IsAuthenticated]

    def get(self, request):
        logger.info('GET: Accounts info for user: %s', request.user.id)
        try:
            user = request.user
            profile_data = get_user_profile_data(user.id,user.type)
            response = {'accountsInfo' : profile_data}
            logger.info('Accounts info sent for user: %s', request.user.id)
            return Response(response, status=status.HTTP_200_OK)
        except Exception as e:
            logger.error("User: Error in Accounts Info for user: %s is: %s", request.user.id, e)
            return Response({"error": "Something went wrong"}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


class CreatePatient(APIView):
    permission_classes = [IsAuthenticated]

    def post(self, request):
        data = request.data
        user_id = request.user
        full_name = data.get('full_name')
        age = data.get('age')
        gender = data.get('gender')
        state = data.get('state')
        contact_number = data.get('contact_number')
        logger.info('request for action profile for user_id: %s', user_id)
        if not full_name:
            logger.critical("full_name  is required")
            return Response("Full Name is required", status=status.HTTP_400_BAD_REQUEST)
        if not age:
            logger.critical("age  is required")
            return Response("Age Type is required", status=status.HTTP_400_BAD_REQUEST)
        if not state:
            logger.critical("state is required")
            return Response("State is required", status=status.HTTP_400_BAD_REQUEST)
        if not gender:
            logger.critical("gender is required")
            return Response("Gender is required", status=status.HTTP_400_BAD_REQUEST)
        if contact_number:
            try:
                parsed_number = phonenumbers.parse(contact_number, None)
            except phonenumbers.NumberParseException as e:
                logger.critical("Country code is missing")
                return Response("Country code required", status=status.HTTP_400_BAD_REQUEST)
            if not phonenumbers.is_valid_number(parsed_number):
                logger.critical("contact_number is not valid")
                return Response("Phone Number is Valid", status=status.HTTP_400_BAD_REQUEST)
        profile_data = {"full_name":full_name,"age":age,"gender":gender,"state":state,
            "contact_number":contact_number}
        try:
            user_type = 'P'
            response = create_patient(user_id,user_type,profile_data)
            logger.info('Response sent to create profile for user_id: %s', user_id)
            return Response(response, status=status.HTTP_200_OK)
        except Exception as e:
            logger.info('Response failed to create profile for user_id: %s and Error is: %s',
                        user_id, e)
            return Response("Something Went Wrong", status=status.HTTP_500_INTERNAL_SERVER_ERROR)