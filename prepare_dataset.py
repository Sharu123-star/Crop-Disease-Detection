import os
import pandas as pd
from sklearn.model_selection import train_test_split

DATASET_DIR = "dataset"
OUTPUT_DIR = "splits"

os.makedirs(OUTPUT_DIR, exist_ok=True)

classes = [
    "Tomato___Bacterial_spot",
    "Tomato___Early_blight",
    "Tomato___healthy",
    "Tomato___Late_blight"
]

data = []

for class_name in classes:
    class_path = os.path.join(DATASET_DIR, class_name)

    for filename in os.listdir(class_path):
        file_path = os.path.join(class_path, filename)

        if os.path.isfile(file_path):
            data.append({
                "filepath": file_path,
                "label": class_name
            })

df = pd.DataFrame(data)

print("Total images:", len(df))
print("\nImages per class:")
print(df["label"].value_counts())

# 80% training, 20% temporary
train_df, temp_df = train_test_split(
    df,
    test_size=0.20,
    stratify=df["label"],
    random_state=42
)

# Split remaining 20% into 10% validation and 10% test
val_df, test_df = train_test_split(
    temp_df,
    test_size=0.50,
    stratify=temp_df["label"],
    random_state=42
)

train_df.to_csv(os.path.join(OUTPUT_DIR, "train.csv"), index=False)
val_df.to_csv(os.path.join(OUTPUT_DIR, "validation.csv"), index=False)
test_df.to_csv(os.path.join(OUTPUT_DIR, "test.csv"), index=False)

print("\nSplit completed!")
print("Training images:", len(train_df))
print("Validation images:", len(val_df))
print("Test images:", len(test_df))