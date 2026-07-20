import json
import traceback
from collections import Counter

import pandas as pd
from io import BytesIO

from django.contrib import messages
from django.contrib.auth import authenticate, get_user_model, login, logout
from django.contrib.auth.decorators import login_required
from django.db import models
from django.core.paginator import Paginator
from django.http import HttpResponse
from django.shortcuts import redirect, render
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

from ai_engine.predict import score_transactions
from core.models import Transaction, Upload


def landing(request):
    if request.user.is_authenticated:
        return redirect('dashboard')
    return render(request, 'landing.html')


def login_view(request):
    if request.user.is_authenticated:
        return redirect('dashboard')

    if request.method == 'POST':
        email = request.POST.get('email', '').strip()
        password = request.POST.get('password', '')
        remember = request.POST.get('remember')

        user = None
        if email:
            user_model = get_user_model()
            user_obj = user_model.objects.filter(email__iexact=email).first()
            if user_obj is not None:
                user = authenticate(request, username=user_obj.username, password=password)
            else:
                user = authenticate(request, username=email, password=password)

        if user is not None:
            login(request, user)
            if remember:
                request.session.set_expiry(60 * 60 * 24 * 30)
            else:
                request.session.set_expiry(0)
            next_url = request.POST.get('next') or request.GET.get('next') or 'dashboard'
            return redirect(next_url)

        messages.error(request, 'Invalid email or password.')

    return render(request, 'auth/login.html', {'next': request.GET.get('next', '')})


def register_view(request):
    if request.user.is_authenticated:
        return redirect('dashboard')

    if request.method == 'POST':
        full_name = request.POST.get('full_name', '').strip()
        email = request.POST.get('email', '').strip()
        username = request.POST.get('username', '').strip()
        password = request.POST.get('password', '')
        password_confirm = request.POST.get('password_confirm', '')
        terms = request.POST.get('terms')

        if not full_name:
            messages.error(request, 'Please enter your full name.')
        elif not email:
            messages.error(request, 'Please enter your email address.')
        elif not username:
            messages.error(request, 'Please enter a username.')
        elif not password:
            messages.error(request, 'Please enter a password.')
        elif password != password_confirm:
            messages.error(request, 'Passwords do not match.')
        elif not terms:
            messages.error(request, 'You must accept the terms and conditions.')
        else:
            user_model = get_user_model()
            if user_model.objects.filter(username__iexact=username).exists():
                messages.error(request, 'That username is already taken.')
            elif user_model.objects.filter(email__iexact=email).exists():
                messages.error(request, 'That email is already registered.')
            else:
                name_parts = full_name.split()
                first_name = name_parts[0] if name_parts else ''
                last_name = ' '.join(name_parts[1:]) if len(name_parts) > 1 else ''

                user = user_model.objects.create_user(
                    username=username,
                    email=email,
                    password=password,
                    first_name=first_name,
                    last_name=last_name,
                )
                login(request, user)
                messages.success(request, 'Account created successfully.')
                return redirect('dashboard')

    return render(request, 'auth/register.html')


@login_required
def dashboard(request):
    uploads = Upload.objects.filter(user=request.user).order_by('-uploaded_at')
    transactions = Transaction.objects.filter(upload__user=request.user)
    total_transactions = transactions.count()
    fraudulent_transactions = transactions.filter(is_fraud_pred=1).count()
    safe_transactions = total_transactions - fraudulent_transactions
    fraud_rate = round((fraudulent_transactions / total_transactions * 100), 2) if total_transactions else 0

    accuracy = None
    try:
        with open('ai_engine/model_artifacts/fraud_rf_metadata.json', encoding='utf-8') as handle:
            metadata = json.load(handle)
        report = metadata.get('classification_report', {})
        accuracy = report.get('accuracy')
    except Exception:
        accuracy = None

    if accuracy is not None:
        accuracy = round(accuracy * 100, 2)

    context = {
        'active_page': 'dashboard',
        'uploads': uploads[:4],
        'total_transactions': total_transactions,
        'fraudulent_transactions': fraudulent_transactions,
        'safe_transactions': safe_transactions,
        'fraud_rate': fraud_rate,
        'model_accuracy': accuracy,
    }
    return render(request, 'dashboard/dashboard.html', context)


@login_required
def upload_csv(request):
    if request.method == 'POST':
        csv_file = request.FILES.get('csvfile')
        if not csv_file:
            messages.error(request, 'Please select a CSV file to upload.')
            return redirect('upload_csv')

        if not csv_file.name.lower().endswith('.csv'):
            messages.error(request, 'Only CSV files are supported.')
            return redirect('upload_csv')

        try:
            df = pd.read_csv(csv_file)
        except Exception as exc:
            messages.error(request, f'Unable to read the CSV file: {exc}')
            return redirect('upload_csv')

        required_columns = Transaction.feature_field_names()
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            messages.error(request, f'CSV is missing required columns: {", ".join(missing_columns)}')
            return redirect('upload_csv')

        upload = Upload.objects.create(
            user=request.user,
            original_filename=csv_file.name,
            status=Upload.Status.PROCESSING,
        )

        try:
            scored_df = score_transactions(df)
            transaction_rows = []
            for index, row in scored_df.iterrows():
                def scalar(value):
                    if pd.isna(value):
                        return None
                    if hasattr(value, 'item'):
                        try:
                            return value.item()
                        except ValueError:
                            return value
                    return value

                transaction_rows.append(
                    Transaction(
                        upload=upload,
                        row_index=index,
                        amt=scalar(row.get('amt')),
                        gender=scalar(row.get('gender')),
                        city_pop=scalar(row.get('city_pop')),
                        trans_hour=scalar(row.get('trans_hour')),
                        trans_day_of_week=scalar(row.get('trans_day_of_week')),
                        trans_month=scalar(row.get('trans_month')),
                        age=scalar(row.get('age')),
                        distance_km=scalar(row.get('distance_km')),
                        merchant_freq=scalar(row.get('merchant_freq')),
                        category_food_dining=scalar(row.get('category_food_dining', 0)),
                        category_gas_transport=scalar(row.get('category_gas_transport', 0)),
                        category_grocery_net=scalar(row.get('category_grocery_net', 0)),
                        category_grocery_pos=scalar(row.get('category_grocery_pos', 0)),
                        category_health_fitness=scalar(row.get('category_health_fitness', 0)),
                        category_home=scalar(row.get('category_home', 0)),
                        category_kids_pets=scalar(row.get('category_kids_pets', 0)),
                        category_misc_net=scalar(row.get('category_misc_net', 0)),
                        category_misc_pos=scalar(row.get('category_misc_pos', 0)),
                        category_personal_care=scalar(row.get('category_personal_care', 0)),
                        category_shopping_net=scalar(row.get('category_shopping_net', 0)),
                        category_shopping_pos=scalar(row.get('category_shopping_pos', 0)),
                        category_travel=scalar(row.get('category_travel', 0)),
                        is_fraud=scalar(row.get('is_fraud')),
                        fraud_probability=scalar(row.get('fraud_probability')),
                        is_fraud_pred=scalar(row.get('is_fraud_pred')),
                    )
                )

            Transaction.objects.bulk_create(transaction_rows)
            upload.total_rows = len(transaction_rows)
            upload.fraud_detected_count = int((scored_df['is_fraud_pred'] == 1).sum())
            upload.status = Upload.Status.COMPLETED
            upload.save(update_fields=['total_rows', 'fraud_detected_count', 'status'])
            messages.success(request, f'Upload completed successfully. {upload.total_rows} transactions were saved.')
        except Exception as exc:
            traceback.print_exc()
            upload.status = Upload.Status.FAILED
            upload.error_message = str(exc)
            upload.save(update_fields=['status', 'error_message'])
            messages.error(request, f'Upload failed: {exc}')

        return redirect('upload_history')

    recent_uploads = Upload.objects.filter(user=request.user).order_by('-uploaded_at')[:10]
    return render(request, 'dashboard/upload_csv.html', {'active_page': 'upload', 'recent_uploads': recent_uploads})


@login_required
def transactions(request):
    transactions = Transaction.objects.filter(upload__user=request.user).select_related('upload')

    query = request.GET.get('q', '').strip()
    category = request.GET.get('category', '').strip()
    prediction = request.GET.get('prediction', '').strip()

    if query:
        transactions = transactions.filter(
            models.Q(upload__original_filename__icontains=query) |
            models.Q(pk__icontains=query)
        )

    if category:
        if category == 'groceries':
            transactions = transactions.filter(models.Q(category_grocery_net=1) | models.Q(category_grocery_pos=1))
        elif category == 'shopping':
            transactions = transactions.filter(models.Q(category_shopping_net=1) | models.Q(category_shopping_pos=1))
        elif category == 'food':
            transactions = transactions.filter(category_food_dining=1)
        elif category == 'travel':
            transactions = transactions.filter(category_travel=1)
        elif category == 'health':
            transactions = transactions.filter(category_health_fitness=1)
        elif category == 'home':
            transactions = transactions.filter(category_home=1)
        elif category == 'misc':
            transactions = transactions.filter(models.Q(category_misc_net=1) | models.Q(category_misc_pos=1))
        elif category == 'personal':
            transactions = transactions.filter(category_personal_care=1)

    if prediction == 'fraud':
        transactions = transactions.filter(is_fraud_pred=True)
    elif prediction == 'safe':
        transactions = transactions.filter(is_fraud_pred=False)

    transactions = transactions.order_by('-pk')
    paginator = Paginator(transactions, 10)
    page_number = request.GET.get('page', 1)
    page_obj = paginator.get_page(page_number)

    for transaction in page_obj.object_list:
        probability_value = transaction.fraud_probability or 0
        transaction.probability_display = f'{probability_value * 100:.2f}%'
        transaction.risk_width = min(max(probability_value * 100, 0), 100)

    return render(request, 'dashboard/transactions.html', {
        'active_page': 'transactions',
        'transactions': page_obj.object_list,
        'page_obj': page_obj,
        'query': query,
        'category': category,
        'prediction': prediction,
    })


@login_required
def transaction_details(request, transaction_id=None):
    transaction = None
    if transaction_id is not None:
        transaction = Transaction.objects.filter(upload__user=request.user, pk=transaction_id).select_related('upload').first()

    if request.GET.get('download') == '1' and transaction is not None:
        probability_value = transaction.fraud_probability or 0
        risk_width = min(max(probability_value * 100, 0), 100)
        buffer = BytesIO()
        pdf = canvas.Canvas(buffer, pagesize=letter)
        pdf.setTitle('FraudShield AI Transaction Report')
        pdf.setFont('Helvetica-Bold', 16)
        pdf.drawString(50, 760, 'FraudShield AI Transaction Report')
        pdf.setFont('Helvetica', 11)
        pdf.drawString(50, 730, f'Transaction ID: TXN-{transaction.pk}')
        pdf.drawString(50, 710, f'Upload File: {transaction.upload.original_filename}')
        pdf.drawString(50, 690, f'Amount: ${transaction.amt:.2f}')
        pdf.drawString(50, 670, f'Prediction: {"Fraud" if transaction.is_fraud_pred else "Safe"}')
        pdf.drawString(50, 650, f'Fraud Probability: {probability_value * 100:.2f}%')
        pdf.drawString(50, 630, f'Risk Width: {risk_width:.0f}%')
        pdf.drawString(50, 610, f'Upload Date: {transaction.upload.uploaded_at.strftime("%Y-%m-%d %H:%M")}')
        pdf.save()
        buffer.seek(0)

        response = HttpResponse(buffer.getvalue(), content_type='application/pdf')
        response['Content-Disposition'] = f'attachment; filename="transaction_{transaction.pk}_report.pdf"'
        return response

    if transaction is not None:
        probability_value = transaction.fraud_probability or 0
        transaction.risk_width = min(max(probability_value * 100, 0), 100)
        transaction.probability_display = f'{probability_value * 100:.2f}%'

    return render(request, 'dashboard/transaction_details.html', {'active_page': 'transactions', 'transaction': transaction})


@login_required
def upload_history(request):
    uploads = Upload.objects.filter(user=request.user).order_by('-uploaded_at')
    return render(request, 'dashboard/upload_history.html', {'active_page': 'history', 'uploads': uploads})


@login_required
def analytics(request):
    uploads = Upload.objects.filter(user=request.user).order_by('-uploaded_at')
    transactions = Transaction.objects.filter(upload__user=request.user)
    total_transactions = transactions.count()
    fraudulent_transactions = transactions.filter(is_fraud_pred=1).count()
    safe_transactions = total_transactions - fraudulent_transactions
    fraud_rate = round((fraudulent_transactions / total_transactions * 100), 2) if total_transactions else 0

    monthly_counts = Counter(
        transaction.upload.uploaded_at.strftime('%b') for transaction in transactions
    )
    months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    monthly_series = [monthly_counts.get(month, 0) for month in months]

    category_counts = Counter()
    for transaction in transactions:
        if transaction.category_grocery_net or transaction.category_grocery_pos:
            category_counts['Groceries'] += 1
        elif transaction.category_shopping_net or transaction.category_shopping_pos:
            category_counts['Shopping'] += 1
        elif transaction.category_food_dining:
            category_counts['Food & Dining'] += 1
        elif transaction.category_travel:
            category_counts['Travel'] += 1
        elif transaction.category_health_fitness:
            category_counts['Health & Fitness'] += 1
        elif transaction.category_home:
            category_counts['Home'] += 1
        elif transaction.category_personal_care:
            category_counts['Personal Care'] += 1
        else:
            category_counts['Misc'] += 1

    fraud_labels = ['Safe', 'Fraud']
    fraud_series = [safe_transactions, fraudulent_transactions]

    recent_uploads = uploads[:5]
    context = {
        'active_page': 'analytics',
        'uploads': recent_uploads,
        'total_uploads': uploads.count(),
        'total_transactions': total_transactions,
        'fraud_rate': fraud_rate,
        'fraudulent_transactions': fraudulent_transactions,
        'safe_transactions': safe_transactions,
        'monthly_labels': months,
        'monthly_series': monthly_series,
        'fraud_labels': fraud_labels,
        'fraud_series': fraud_series,
        'category_labels': list(category_counts.keys()),
        'category_series': list(category_counts.values()),
    }
    return render(request, 'dashboard/analytics.html', context)


@login_required
def profile(request):
    return render(request, 'dashboard/profile.html', {'active_page': 'profile'})


def logout_view(request):
    logout(request)
    messages.success(request, 'You have been logged out.')
    return redirect('landing')
