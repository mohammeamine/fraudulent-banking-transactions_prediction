from django.contrib import admin

from core.models import Transaction, Upload


@admin.register(Upload)
class UploadAdmin(admin.ModelAdmin):
    list_display = ("id", "original_filename", "user", "status", "total_rows", "uploaded_at")
    list_filter = ("status",)
    search_fields = ("original_filename", "user__username")


@admin.register(Transaction)
class TransactionAdmin(admin.ModelAdmin):
    list_display = ("id", "upload", "row_index", "amt", "is_fraud_pred", "fraud_probability")
    list_filter = ("is_fraud_pred",)
    search_fields = ("upload__original_filename",)
