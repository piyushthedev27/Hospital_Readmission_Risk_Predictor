import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import classification_report, accuracy_score, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# --- 1. Load Data ---
# It's good practice to handle potential data type issues upon loading
# For this example, we'll assume dtypes are correct in the CSV.
try:
    data = pd.read_csv("your_data.csv")
except FileNotFoundError:
    print("Error: 'your_data.csv' not found. Please replace with your actual data file.")
    exit()

# --- 2. Define Features and Target ---
# Identify which features are numeric and which are categorical
target = 'readmitted'
numeric_features = ['age', 'length_of_stay', 'num_lab_procedures', 'num_medications']
categorical_features = ['diag_1', 'admission_type']

features = numeric_features + categorical_features
X = data[features]
y = data[target]

# --- 3. Split Data into Training and Testing Sets ---
# We use 80% for training and 20% for testing.
# `stratify=y` is important for imbalanced datasets to keep the class proportion same in train/test splits.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# --- 4. Create a Preprocessing and Modeling Pipeline ---
# Create a transformer for numeric features (scaling)
numeric_transformer = StandardScaler()

# Create a transformer for categorical features (one-hot encoding)
# `handle_unknown='ignore'` prevents errors if a category in the test set wasn't in the training set
categorical_transformer = OneHotEncoder(handle_unknown='ignore')

# Create a preprocessor object using ColumnTransformer
# This applies the correct transformation to each column type
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ],
    remainder='passthrough' # Keep other columns if any (none in this case)
)

# Create the full pipeline by combining the preprocessor and the model
model_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression(random_state=42, max_iter=1000)) # Increased max_iter for convergence
])

# --- 5. Train the Pipeline ---
print("Training the model...")
model_pipeline.fit(X_train, y_train)
print("Training complete.")

# --- 6. Evaluate the Model ---
print("\n--- Model Evaluation ---")
y_pred = model_pipeline.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy on Test Set: {accuracy:.2f}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Display Confusion Matrix
print("\nConfusion Matrix:")
ConfusionMatrixDisplay.from_estimator(model_pipeline, X_test, y_test, cmap='Blues')
plt.show()

# --- 7. Save the Final Pipeline ---
# This single file now contains all preprocessing steps and the trained model
pipeline_filename = "readmission_pipeline.pkl"
joblib.dump(model_pipeline, pipeline_filename)
print(f"\n✅ Model pipeline saved successfully as '{pipeline_filename}'")