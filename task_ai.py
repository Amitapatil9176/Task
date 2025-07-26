
# AIML RECRUITMENT TEST SOLUTIONS - By Amita Patil

# Scenario 1: Data Validation
def validate_data(data):
    invalid_entries = []
    for item in data:
        if not isinstance(item.get("age"), int):
            invalid_entries.append(item)
    return invalid_entries

# Example usage
data1 = [{"name": "Alice", "age": 30}, {"name": "Bob", "age": "25"}]
print("Scenario 1 Output:", validate_data(data1))


# Scenario 2: Logging Decorator
import time

def log_execution_time(func):
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        print(f"Execution time: {end - start:.4f} seconds")
        return result
    return wrapper

@log_execution_time
def calculate_sum(n):
    return sum(range(1, n+1))

print("Scenario 2 Output:", calculate_sum(1000000))


# Scenario 3: Missing Value Handling
import pandas as pd

df3 = pd.DataFrame({
    "income": [50000, 60000, None, 55000, None, 52000]
})

if df3["income"].skew(skipna=True) < 0.5:
    df3["income"].fillna(df3["income"].median(), inplace=True)
else:
    df3["income"].fillna(df3["income"].mode()[0], inplace=True)

print("Scenario 3 Output:\n", df3)


# Scenario 4: Text Pre-processing
import re

df4 = pd.DataFrame({
    "text": ["Hello World!", "Python@123", "AI/ML is fun!!!"]
})

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', '', text)
    tokens = text.split()
    return tokens

df4["cleaned"] = df4["text"].apply(clean_text)
print("Scenario 4 Output:\n", df4)


# Scenario 5: Hyperparameter Tuning
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.datasets import load_iris

X, y = load_iris(return_X_y=True)
params = {
    'max_depth': [3, 5, 7],
    'n_estimators': [50, 100]
}

model = RandomForestClassifier()
grid = GridSearchCV(model, params, cv=3)
grid.fit(X, y)
print("Scenario 5 Output: Best Parameters:", grid.best_params_)


# Scenario 6: Custom Evaluation Metric
from sklearn.metrics import make_scorer

def weighted_accuracy(y_true, y_pred):
    correct = 0
    for true, pred in zip(y_true, y_pred):
        if true == pred:
            correct += 2 if true == 1 else 1
    total = sum(2 if t == 1 else 1 for t in y_true)
    return correct / total

# Can be used with: scorer = make_scorer(weighted_accuracy)


# Scenario 7: Image Augmentation
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

datagen = ImageDataGenerator(
    rotation_range=20,
    zoom_range=0.2,
    horizontal_flip=True
)
# To use: datagen.flow(image_array, batch_size=1)
print("Scenario 7 Output: Image augmentation pipeline created.")


# Scenario 8: Model Callbacks
from tensorflow.keras.callbacks import EarlyStopping

early_stop = EarlyStopping(
    monitor='val_loss',
    patience=3,
    restore_best_weights=True
)
print("Scenario 8 Output: EarlyStopping callback ready.")


# Scenario 9: Structured Response Generation
import json

response = '''{
    "benefits": [
        "Easy to learn and use",
        "Has powerful libraries for data science",
        "Large community support"
    ]
}'''

try:
    data9 = json.loads(response)
    print("Scenario 9 Output:", data9)
except json.JSONDecodeError:
    print("Scenario 9 Output: Invalid JSON response")


# Scenario 10: Summarization Prompt
prompt = '''
Summarize the following news article into 2 sentences.
If the summary exceeds 50 words, truncate it to the nearest complete sentence.
'''
print("Scenario 10 Output:", prompt.strip())
