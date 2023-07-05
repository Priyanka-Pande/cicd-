import uuid
from django.db import models
from functools import partial
from django.conf import settings
from django.contrib.auth.models import AbstractUser
from phonenumber_field.modelfields import PhoneNumberField
from Users.Constants.userConstants import *

# Create your models here.

def get_unique_profile_pic_name(instance, filename):
    extension = filename.split('.')[-1]
    unique_filename = f'images/profile_pic/{uuid.uuid4()}.{extension}'
    return unique_filename

class Users(AbstractUser):
    id = models.AutoField(primary_key=True)
    phone_number = PhoneNumberField(null=False, blank=False, unique=True)
    role = models.CharField(max_length=10, default='User', choices=USER_ROLES, null=True)
    type = models.CharField(max_length=2,choices=USER_TYPE_CHOICES, null=True,blank=True)
    status = models.CharField(max_length=1, default='A', choices=USER_STATUS, null=True)
    created_on = models.DateTimeField(auto_now_add=True,null=True)
    updated_on = models.DateTimeField(auto_now_add=True,null=True)

    def __str__(self):
        return "{}".format(self.id)

    class Meta:
        db_table = 'users'


class ProfessionalProfile(models.Model):
    id = models.AutoField(primary_key=True)
    user_id = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE, db_column='user')
    full_name = models.CharField(max_length=100,null=True, blank=True)
    organization_name = models.CharField(max_length=100,null=True, blank=True)
    profile_type = models.CharField(max_length=1, choices=PROFESSIONAL_USER_TYPE_CHOICES, null=True,blank=True)
    profile_pic = models.ImageField(upload_to=partial(get_unique_profile_pic_name), blank=True, null=True)
    age = models.IntegerField(null=True, blank=True)
    gender = models.CharField(max_length=1, choices=USER_GENDER_CHOICES, null=True,blank=True)
    state = models.CharField(max_length=50,null=True, blank=True)
    created_on = models.DateTimeField(auto_now_add=True,null=True)
    updated_on = models.DateTimeField(auto_now_add=True,null=True)

    def __str__(self):
        return  "{}".format(self.id)

    class Meta:
        db_table = 'professional_profile_ids'


class PersonalProfile(models.Model):
    id = models.AutoField(primary_key=True)
    user_id = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE, db_column='user',null=True,blank=True)
    profile_id = models.ForeignKey(ProfessionalProfile, on_delete=models.CASCADE, db_column='profile',null=True,blank=True)
    full_name = models.CharField(max_length=100,null=True, blank=True)
    profile_pic = models.ImageField(upload_to=partial(get_unique_profile_pic_name), blank=True, null=True)
    age = models.IntegerField(null=True, blank=True)
    gender = models.CharField(max_length=1, choices=USER_GENDER_CHOICES, null=True,blank=True)
    state = models.CharField(max_length=50,null=True, blank=True)
    tester_type = models.CharField(max_length=1,null=True,blank=True)
    contact_number = PhoneNumberField(null=True, blank=True)
    created_on = models.DateTimeField(auto_now_add=True,null=True)
    updated_on = models.DateTimeField(auto_now_add=True,null=True)

    def __str__(self):
        return  "{}".format(self.id)

    class Meta:
        db_table = 'patient_personal_profile_ids'

