import os
import shutil
import torch
import torchvision.transforms as transforms
from torchvision.models import resnet18
from PIL import Image
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import sys

# Config
megadepth_data_dir = "data/megadepth"
set_id = sys.argv[1]
INPUT_DIR = f"{megadepth_data_dir}/{set_id}"
PAIR_DIR = f"data/megadepth_pairs/{set_id}"
SIMILARITY_THRESHOLD = 0.87  # Minimum cosine similarity

os.makedirs(PAIR_DIR, exist_ok=True)

# Load feature extractor
model = resnet18(pretrained=True)
model.eval()
model = torch.nn.Sequential(*(list(model.children())[:-1]))

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std =[0.229, 0.224, 0.225])
])

# Extract features
filenames = [f for f in os.listdir(INPUT_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
features = []

for fname in filenames:
    img_path = os.path.join(INPUT_DIR, fname)
    img = Image.open(img_path).convert('RGB')
    tensor = transform(img).unsqueeze(0)
    with torch.no_grad():
        embedding = model(tensor).squeeze().numpy()
    features.append(embedding)

features = np.vstack(features)
similarity_matrix = cosine_similarity(features)
np.fill_diagonal(similarity_matrix, -np.inf)

used = set()
pair_index = 0

for i, row in enumerate(similarity_matrix):
    if i in used:
        continue
    j = np.argmax(row)
    if j in used or similarity_matrix[i][j] < SIMILARITY_THRESHOLD:
        continue

    used.update([i, j])

    pair_folder = os.path.join(PAIR_DIR, f"pair_{pair_index}")
    os.makedirs(pair_folder, exist_ok=True)

    shutil.copy(os.path.join(INPUT_DIR, filenames[i]), os.path.join(pair_folder, "image1.jpg"))
    shutil.copy(os.path.join(INPUT_DIR, filenames[j]), os.path.join(pair_folder, "image2.jpg"))
    pair_index += 1
