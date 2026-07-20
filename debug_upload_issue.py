import os
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'fraud.settings')
import django
django.setup()
from django.conf import settings
settings.ALLOWED_HOSTS = ['testserver']

from django.test import Client
from django.contrib.auth import get_user_model
from django.core.files.uploadedfile import SimpleUploadedFile
from django.urls import reverse
from core.models import Upload
import traceback

user = get_user_model().objects.create_user(username='debugu7', email='debug7@example.com', password='x')
client = Client()
client.force_login(user)
csv_content = b'amt,gender,city_pop,is_fraud,trans_hour,trans_day_of_week,trans_month,age,distance_km,category_food_dining,category_gas_transport,category_grocery_net,category_grocery_pos,category_health_fitness,category_home,category_kids_pets,category_misc_net,category_misc_pos,category_personal_care,category_shopping_net,category_shopping_pos,category_travel,merchant_freq\n10.0,0,1000,0,12,2,6,25,1.5,0,0,0,0,0,0,0,0,0,0,0,0,0,0.1\n20.0,1,2000,1,18,5,8,35,2.0,1,0,0,0,0,0,0,0,0,0,0,0,0,0.2\n'
uploaded_file = SimpleUploadedFile('transactions.csv', csv_content, content_type='text/csv')
response = client.post(reverse('upload_csv'), {'csvfile': uploaded_file}, follow=True)
print('status', response.status_code)
print('redirect chain', response.redirect_chain)
print('upload count', Upload.objects.filter(user=user).count())
for upload in Upload.objects.filter(user=user):
    print(upload.original_filename, upload.status, upload.error_message)
