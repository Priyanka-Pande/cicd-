from django.contrib import admin
from AppGluco.models import *
# Register your models here.
class VideoTableAdmin(admin.ModelAdmin):
    list_display = ('id','profile_id')
class GlucoResultTableAdmin(admin.ModelAdmin):
    list_display = ('id','profile_id','patient_id','gluco_value','video_location')
class FAQTableAdmin(admin.ModelAdmin):
    list_display = ('id','created_on','updated_on')
admin.site.register(VideoTable,VideoTableAdmin)
admin.site.register(GlucoResultTable,GlucoResultTableAdmin)
admin.site.register(FAQTable,FAQTableAdmin)