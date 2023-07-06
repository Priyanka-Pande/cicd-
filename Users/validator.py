import os
from django.core.exceptions import ValidationError

def validate_file_extension(value):
    ext = os.path.splitext(value.name)[1] 
    valid_extensions = ['.jpeg', '.jpg', '.png', '.gif', '.svg', '.tiff', '.tif', '.bmp', '.webp', '.heic', '.heif']
    if not ext.lower() in valid_extensions:
        raise ValidationError('Unsupported file extension.')