from django.conf import settings
from django.db import models


class Upload(models.Model):
    """One record per CSV file uploaded by a user."""

    class Status(models.TextChoices):
        PENDING = "pending", "Pending"
        PROCESSING = "processing", "Processing"
        COMPLETED = "completed", "Completed"
        FAILED = "failed", "Failed"

    user = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name="uploads",
    )
    original_filename = models.CharField(max_length=255)
    status = models.CharField(
        max_length=20,
        choices=Status.choices,
        default=Status.PENDING,
    )
    total_rows = models.PositiveIntegerField(default=0)
    fraud_detected_count = models.PositiveIntegerField(default=0)
    error_message = models.TextField(blank=True)
    uploaded_at = models.DateTimeField(auto_now_add=True)
    processed_at = models.DateTimeField(null=True, blank=True)

    class Meta:
        ordering = ["-uploaded_at"]

    def __str__(self) -> str:
        return f"{self.original_filename} ({self.user.username})"


class Transaction(models.Model):
    """
    One record per CSV row — mirrors the clean dataset schema used for training.

    Feature columns must match ai_engine/model_artifacts/fraud_rf_metadata.json.
    """

    upload = models.ForeignKey(
        Upload,
        on_delete=models.CASCADE,
        related_name="transactions",
    )
    row_index = models.PositiveIntegerField(default=0)

    # --- Features (same as fraudTrain_clean.csv) ---
    amt = models.FloatField()
    gender = models.SmallIntegerField()
    city_pop = models.IntegerField()
    trans_hour = models.SmallIntegerField()
    trans_day_of_week = models.SmallIntegerField()
    trans_month = models.SmallIntegerField()
    age = models.SmallIntegerField()
    distance_km = models.FloatField()
    merchant_freq = models.FloatField()

    category_food_dining = models.SmallIntegerField(default=0)
    category_gas_transport = models.SmallIntegerField(default=0)
    category_grocery_net = models.SmallIntegerField(default=0)
    category_grocery_pos = models.SmallIntegerField(default=0)
    category_health_fitness = models.SmallIntegerField(default=0)
    category_home = models.SmallIntegerField(default=0)
    category_kids_pets = models.SmallIntegerField(default=0)
    category_misc_net = models.SmallIntegerField(default=0)
    category_misc_pos = models.SmallIntegerField(default=0)
    category_personal_care = models.SmallIntegerField(default=0)
    category_shopping_net = models.SmallIntegerField(default=0)
    category_shopping_pos = models.SmallIntegerField(default=0)
    category_travel = models.SmallIntegerField(default=0)

    # Ground truth label — only present if the uploaded CSV includes it (e.g. test data).
    is_fraud = models.BooleanField(null=True, blank=True)

    # --- Model predictions (filled after ai_engine/predict.py runs) ---
    fraud_probability = models.FloatField(null=True, blank=True)
    is_fraud_pred = models.BooleanField(null=True, blank=True)

    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["row_index"]
        indexes = [
            models.Index(fields=["upload", "is_fraud_pred"]),
        ]

    def __str__(self) -> str:
        return f"Transaction #{self.pk} (upload {self.upload_id}, row {self.row_index})"

    @classmethod
    def feature_field_names(cls) -> list[str]:
        """Column names expected by the ML model (excluding target and predictions)."""
        return [
            "amt",
            "gender",
            "city_pop",
            "trans_hour",
            "trans_day_of_week",
            "trans_month",
            "age",
            "distance_km",
            "category_food_dining",
            "category_gas_transport",
            "category_grocery_net",
            "category_grocery_pos",
            "category_health_fitness",
            "category_home",
            "category_kids_pets",
            "category_misc_net",
            "category_misc_pos",
            "category_personal_care",
            "category_shopping_net",
            "category_shopping_pos",
            "category_travel",
            "merchant_freq",
        ]
