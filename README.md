# FraudShield

## Clone the repository

The trained fraud detection model is stored with Git LFS. Install Git LFS before
cloning so the model is downloaded and available to the app:

```shell
git lfs install
git clone https://github.com/mohammeamine/fraudulent-banking-transactions_prediction.git
```

If you cloned before installing Git LFS, run `git lfs pull` from the repository
directory. The model file is `ai_engine/model_artifacts/fraud_rf_model.joblib`.

Training datasets are not included in the repository. The checked-in trained
model remains available for running predictions; a local training dataset is
needed only to retrain it.
