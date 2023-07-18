from django.db import models


# Create your models here.

class DeleteAccountRequests(models.Model):
    id = models.AutoField(primary_key=True)
    patient_name = models.CharField(max_length=100,null=True, blank=True)
    phone_number = models.CharField(max_length=15,null=True, blank=True)
    message = models.TextField(null=True,blank=True)
    reported_at = models.DateTimeField(auto_now_add=True,null=True)

    def __str__(self):
        return  "{}".format(self.id)

    class Meta:
        db_table = 'delete_account_request'