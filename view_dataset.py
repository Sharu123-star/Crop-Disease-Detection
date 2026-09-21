import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

# Load training image information
df = pd.read_csv("splits/train.csv")

# Take 8 random images
samples = df.sample(8, random_state=42)

plt.figure(figsize=(12, 8))

for i, (_, row) in enumerate(samples.iterrows()):
    image = Image.open(row["filepath"])

    plt.subplot(2, 4, i + 1)
    plt.imshow(image)
    plt.title(row["label"].replace("Tomato___", ""))
    plt.axis("off")

plt.tight_layout()
plt.show()