from io import BytesIO

from django.contrib.auth import get_user_model
from django.core.files.uploadedfile import SimpleUploadedFile
from django.test import TestCase
from django.urls import reverse

from core.models import Transaction, Upload


class AuthViewsTests(TestCase):
    def test_register_creates_user_and_logs_in(self):
        response = self.client.post(
            reverse("register"),
            {
                "full_name": "Alex Morgan",
                "email": "alex@example.com",
                "username": "alex",
                "password": "StrongPass123!",
                "password_confirm": "StrongPass123!",
                "terms": "on",
            },
            follow=True,
        )

        self.assertRedirects(response, reverse("dashboard"))
        user = get_user_model().objects.get(username="alex")
        self.assertTrue(user.check_password("StrongPass123!"))
        self.assertTrue(response.wsgi_request.user.is_authenticated)

    def test_login_authenticates_existing_user_by_email(self):
        get_user_model().objects.create_user(
            username="alex",
            email="alex@example.com",
            password="StrongPass123!",
        )

        response = self.client.post(
            reverse("login"),
            {"email": "alex@example.com", "password": "StrongPass123!"},
            follow=True,
        )

        self.assertRedirects(response, reverse("dashboard"))
        self.assertTrue(response.wsgi_request.user.is_authenticated)

    def test_dashboard_requires_login(self):
        response = self.client.get(reverse("dashboard"))

        self.assertRedirects(response, f"{reverse('login')}?next={reverse('dashboard')}")

    def test_upload_csv_creates_upload_and_transactions(self):
        user = get_user_model().objects.create_user(username="uploader", email="upload@example.com", password="StrongPass123!")
        self.client.force_login(user)

        csv_content = b"amt,gender,city_pop,is_fraud,trans_hour,trans_day_of_week,trans_month,age,distance_km,category_food_dining,category_gas_transport,category_grocery_net,category_grocery_pos,category_health_fitness,category_home,category_kids_pets,category_misc_net,category_misc_pos,category_personal_care,category_shopping_net,category_shopping_pos,category_travel,merchant_freq\n10.0,0,1000,0,12,2,6,25,1.5,0,0,0,0,0,0,0,0,0,0,0,0,0,0.1\n20.0,1,2000,1,18,5,8,35,2.0,1,0,0,0,0,0,0,0,0,0,0,0,0,0.2\n"
        uploaded_file = SimpleUploadedFile("transactions.csv", csv_content, content_type="text/csv")

        response = self.client.post(reverse("upload_csv"), {"csvfile": uploaded_file}, follow=True)

        self.assertRedirects(response, reverse("upload_history"))
        upload = Upload.objects.get(original_filename="transactions.csv", user=user)
        self.assertEqual(upload.status, Upload.Status.COMPLETED)
        self.assertEqual(upload.total_rows, 2)
        self.assertEqual(Transaction.objects.filter(upload=upload).count(), 2)
        self.assertTrue(Transaction.objects.filter(upload=upload).exists())

    def test_upload_csv_without_is_fraud_column_still_works(self):
        user = get_user_model().objects.create_user(username="predictor", email="predictor@example.com", password="StrongPass123!")
        self.client.force_login(user)

        csv_content = b"amt,gender,city_pop,trans_hour,trans_day_of_week,trans_month,age,distance_km,category_food_dining,category_gas_transport,category_grocery_net,category_grocery_pos,category_health_fitness,category_home,category_kids_pets,category_misc_net,category_misc_pos,category_personal_care,category_shopping_net,category_shopping_pos,category_travel,merchant_freq\n10.0,0,1000,12,2,6,25,1.5,0,0,0,0,0,0,0,0,0,0,0,0,0,0.1\n"
        uploaded_file = SimpleUploadedFile("predict_only.csv", csv_content, content_type="text/csv")

        response = self.client.post(reverse("upload_csv"), {"csvfile": uploaded_file}, follow=True)

        self.assertRedirects(response, reverse("upload_history"))
        upload = Upload.objects.get(original_filename="predict_only.csv", user=user)
        self.assertEqual(upload.status, Upload.Status.COMPLETED)
        self.assertEqual(Transaction.objects.filter(upload=upload).count(), 1)

    def test_transactions_page_shows_real_user_transactions(self):
        user = get_user_model().objects.create_user(username="reader", email="reader@example.com", password="StrongPass123!")
        self.client.force_login(user)

        upload = Upload.objects.create(user=user, original_filename="sample.csv", status=Upload.Status.COMPLETED, total_rows=1)
        transaction = Transaction.objects.create(
            upload=upload,
            amt=125.0,
            gender=1,
            city_pop=5000,
            trans_hour=14,
            trans_day_of_week=3,
            trans_month=7,
            age=32,
            distance_km=3.2,
            merchant_freq=2.5,
            category_food_dining=1,
            fraud_probability=0.91,
            is_fraud_pred=True,
        )

        response = self.client.get(reverse("transactions"))

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, str(transaction.pk))
        self.assertContains(response, upload.original_filename)
        self.assertContains(response, "91.00%")

    def test_transactions_page_paginates_results(self):
        user = get_user_model().objects.create_user(username="pager", email="pager@example.com", password="StrongPass123!")
        self.client.force_login(user)

        upload = Upload.objects.create(user=user, original_filename="pager.csv", status=Upload.Status.COMPLETED, total_rows=12)
        for index in range(12):
            Transaction.objects.create(
                upload=upload,
                amt=float(index + 1),
                gender=0,
                city_pop=1000 + index,
                trans_hour=12,
                trans_day_of_week=1,
                trans_month=1,
                age=30,
                distance_km=1.0,
                merchant_freq=1.0,
                fraud_probability=0.1 + index * 0.01,
                is_fraud_pred=False,
            )

        response = self.client.get(reverse("transactions"), {"page": 2})

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.context["page_obj"].number, 2)
        self.assertTrue(response.context["page_obj"].has_previous)

    def test_transaction_details_page_uses_real_transaction_data(self):
        user = get_user_model().objects.create_user(username="details", email="details@example.com", password="StrongPass123!")
        self.client.force_login(user)

        upload = Upload.objects.create(user=user, original_filename="details.csv", status=Upload.Status.COMPLETED, total_rows=1)
        transaction = Transaction.objects.create(
            upload=upload,
            amt=89.0,
            gender=0,
            city_pop=1200,
            trans_hour=9,
            trans_day_of_week=2,
            trans_month=6,
            age=28,
            distance_km=1.1,
            merchant_freq=3.2,
            category_grocery_pos=1,
            fraud_probability=0.87,
            is_fraud_pred=True,
        )

        response = self.client.get(reverse("transaction_details", args=[transaction.pk]))

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, upload.original_filename)
        self.assertContains(response, "Fraud")
        self.assertContains(response, "87.00%")

    def test_transaction_details_download_returns_report(self):
        user = get_user_model().objects.create_user(username="download", email="download@example.com", password="StrongPass123!")
        self.client.force_login(user)

        upload = Upload.objects.create(user=user, original_filename="download.csv", status=Upload.Status.COMPLETED, total_rows=1)
        transaction = Transaction.objects.create(
            upload=upload,
            amt=89.0,
            gender=0,
            city_pop=1200,
            trans_hour=9,
            trans_day_of_week=2,
            trans_month=6,
            age=28,
            distance_km=1.1,
            merchant_freq=3.2,
            category_grocery_pos=1,
            fraud_probability=0.87,
            is_fraud_pred=True,
        )

        response = self.client.get(reverse("transaction_details", args=[transaction.pk]), {"download": "1"})

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response["Content-Type"], "application/pdf")
        self.assertIn("attachment", response["Content-Disposition"])
        self.assertIn("transaction_", response["Content-Disposition"])
        self.assertIn(b"%PDF", response.content)
