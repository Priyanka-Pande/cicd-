# Generated by Django 4.2.2 on 2023-07-06 06:36

import Users.models
from django.db import migrations, models
import functools


class Migration(migrations.Migration):

    dependencies = [
        ('Users', '0003_users_type_alter_users_status'),
    ]

    operations = [
        migrations.AlterField(
            model_name='personalprofile',
            name='profile_pic',
            field=models.ImageField(blank=True, null=True, upload_to=functools.partial(Users.models.get_unique_profile_pic_name, *(), **{}), validators=[Users.models.validate_file_extension]),
        ),
        migrations.AlterField(
            model_name='professionalprofile',
            name='profile_pic',
            field=models.ImageField(blank=True, null=True, upload_to=functools.partial(Users.models.get_unique_profile_pic_name, *(), **{}), validators=[Users.models.validate_file_extension]),
        ),
    ]
