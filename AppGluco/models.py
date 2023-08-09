import uuid
from django.db import models
from functools import partial
from Users.models import PersonalProfile, ProfessionalProfile
from AppGluco.validator import validate_file_extension

# Create your models here.

def get_unique_profile_video(instance, filename):
    extension = filename.split('.')[-1]
    unique_filename = f'videos/{uuid.uuid4()}.{extension}'
    return unique_filename

class VideoTable(models.Model):
    id = models.AutoField(primary_key=True)
    profile_id = models.ForeignKey(PersonalProfile, on_delete=models.CASCADE, db_column='profile')
    video_file = models.FileField(upload_to=partial(get_unique_profile_video),validators=[validate_file_extension])
    created_on = models.DateTimeField(auto_now_add=True,null=True)
    updated_on = models.DateTimeField(auto_now_add=True,null=True)

    def __str__(self):
        return  "{}".format(self.id)

    class Meta:
        db_table = 'video_table'


class GlucoResultTable(models.Model):
    id = models.AutoField(primary_key=True)
    profile_id = models.ForeignKey(ProfessionalProfile, on_delete=models.CASCADE, db_column='profile',null=True,blank=True)
    patient_id = models.ForeignKey(PersonalProfile, on_delete=models.CASCADE, db_column='patient')
    gluco_value = models.DecimalField(max_digits=5, decimal_places=2)
    video_location = models.ForeignKey(VideoTable, on_delete=models.CASCADE, db_column='video',)
    reported_date = models.DateTimeField(auto_now_add=True,null=True)

    def __str__(self):
        return  "{}".format(self.id)

    class Meta:
        db_table = 'gluco_result_table'


class FAQTable(models.Model):
    id = models.AutoField(primary_key=True)
    question = models.TextField()
    answer = models.TextField()
    created_on = models.DateTimeField(auto_now_add=True,null=True)
    updated_on = models.DateTimeField(auto_now_add=True,null=True)


    def __str__(self):
        return  "{}".format(self.id)

    class Meta:
        db_table = 'FAQ_table'