from django.contrib import admin
from .models import *

# Register your models here.
class DeleteAccountRequestsAdmin(admin.ModelAdmin):
    list_display = ('id','patient_name', 'phone_number')
admin.site.register(DeleteAccountRequests,DeleteAccountRequestsAdmin)
