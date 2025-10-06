import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from imblearn.over_sampling import SMOTE
from sklearn.metrics import classification_report
import lightgbm as lgb
import joblib
import os
import numpy as np

path = "customer_booking.csv"
try:
    df = pd.read_csv(path, encoding='latin-1')
except UnicodeDecodeError:
    df = pd.read_csv(path, encoding='cp1252')

# 2. Separate features (X) and target (y)
X = df.drop('booking_complete', axis=1)
y = df['booking_complete']

# 3. Identify features
categorical_features = X.select_dtypes(include=['object']).columns
numerical_features = X.select_dtypes(include=['int64', 'float64']).columns

# 4. Handle high-cardinality categorical features
def handle_high_cardinality(df, categorical_cols, threshold=10):
    for col in categorical_cols:
        if df[col].nunique() > threshold:
            top_categories = df[col].value_counts().nlargest(threshold).index.tolist()
            df[col] = df[col].apply(lambda x: x if x in top_categories else 'Other')
    return df

X = handle_high_cardinality(X, categorical_features)

# 5. Split data before sampling
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 6. Create a preprocessing pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features),
        ('num', 'passthrough', numerical_features)
    ]
)

# 7. Apply SMOTE to training data
print("Applying SMOTE to balance the training data...")
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(preprocessor.fit_transform(X_train), y_train)

# 8. Create and train a LightGBM classifier with scale_pos_weight
print("Training LightGBM model...")
neg_count = sum(y_train_resampled == 0)
pos_count = sum(y_train_resampled == 1)
scale_pos_weight = neg_count / pos_count
lgb_classifier = lgb.LGBMClassifier(random_state=42, scale_pos_weight=scale_pos_weight)
lgb_classifier.fit(X_train_resampled, y_train_resampled)

# 9. Evaluate the model with different thresholds
print("\n" + "="*50)
print("Evaluating Model with Different Thresholds")
print("="*50)

X_test_transformed = preprocessor.transform(X_test)
y_pred_proba = lgb_classifier.predict_proba(X_test_transformed)[:, 1]

# Test a range of thresholds
thresholds = np.arange(0.1, 0.6, 0.1)

for threshold in thresholds:
    y_test_pred = (y_pred_proba >= threshold).astype(int)
    print(f"\n--- Classification Report for Threshold: {threshold:.1f} ---")
    print(classification_report(y_test, y_test_pred))

# 10. Save the final model and preprocessor
models_dir = 'models'
if not os.path.exists(models_dir):
    os.makedirs(models_dir)

joblib.dump(preprocessor, os.path.join(models_dir, 'preprocessor.joblib'))
joblib.dump(lgb_classifier, os.path.join(models_dir, 'classifier.joblib'))
print("\nModel and preprocessor saved in the 'models' directory.")