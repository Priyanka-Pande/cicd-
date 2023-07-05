from django.contrib import admin
from Users.models import *

# Register your models here.
class UserAdmin(admin.ModelAdmin):
    list_display = ('id','phone_number', 'role', 'status')
class PersonalProfileAdmin(admin.ModelAdmin):
    list_display = ('id','user_id','profile_id')
class ProfessionalProfileAdmin(admin.ModelAdmin):
    list_display = ('id','user_id','full_name','organization_name','profile_type')
admin.site.register(Users, UserAdmin)
admin.site.register(PersonalProfile,PersonalProfileAdmin)
admin.site.register(ProfessionalProfile,ProfessionalProfileAdmin)