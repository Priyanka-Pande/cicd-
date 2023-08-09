import logging
from rest_framework import status
from AppGluco.Service.service import *
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.permissions import IsAuthenticated
from AppGluco.validator import validate_file_extension
# Create your views here.
logging.basicConfig()
logger = logging.getLogger(__name__)


class GetAllPatientsProfessional(APIView):
    permission_classes = [IsAuthenticated]

    def post(self, request):
        user = request.user
        logger.info('request for user_id: %s', user)
        try:
            response = get_professional_all_patients(user)
            logger.info('Response sent for user_id: %s', user)
            return Response(response, status=status.HTTP_200_OK)
        except Exception as e:
            logger.info('Response failed for user_id: %s and Error is: %s',
                        user, e)
            return Response({"message":"Something Went Wrong"}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


class GetResultPersonal(APIView):
    permission_classes = [IsAuthenticated]

    def post(self, request):
        user = request.user
        logger.info('request for user_id: %s', user)
        try:
            response = get_report_for_personal(user)
            logger.info('Response sent for user_id: %s', user)
            return Response(response, status=status.HTTP_200_OK)
        except Exception as e:
            logger.info('Response failed for user_id: %s and Error is: %s',
                        user, e)
            return Response({"message":"Something Went Wrong"}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


class UploadVideo(APIView):
    permission_classes = [IsAuthenticated]

    def post(self, request):
        data = request.data
        patient_id = data.get('patient_id')
        video_file = request.FILES.get('video_file')
        user = request.user
        logger.info('request for patient_id: %s', patient_id)
        if not patient_id:
            logger.critical("patient_id  is required")
            return Response({"message":"Patient Id is required"}, status=status.HTTP_400_BAD_REQUEST)
        if not video_file:
            logger.critical("video_file  is required")
            return Response({"message":"Video File is required"}, status=status.HTTP_400_BAD_REQUEST)
        try:
            validate_file_extension(video_file)
        except Exception as e:
            logger.critical("Video Format is not valid , Exception %s",e)
            return Response({"message":"Video format is not valid"}, status=status.HTTP_400_BAD_REQUEST)
        try:
            response = upload_video_for_patient(patient_id,video_file,user)
            return response
        except Exception as e:
            logger.info('Response failed for patient_id: %s and Error is: %s',
                        patient_id, e)
            return Response({"message":"Something Went Wrong"}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


class FAQViews(APIView):
    permission_classes = [IsAuthenticated]

    def get(self, request):
        user = request.user
        logger.info('request for user_id: %s', user)
        try:
            response =  get_frequently_asked_questions()
            logger.info('Response sent for user_id: %s', user)
            return Response(response, status=status.HTTP_200_OK)
        except Exception as e:
            logger.info('Response failed for user_id: %s and Error is: %s',
                        user, e)
            return Response({"message":"Something Went Wrong"}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)