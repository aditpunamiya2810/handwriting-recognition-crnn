import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
import numpy as np
import os
import pandas as pd
import random
from PIL import Image
import sys
import io

# --- 1. CONFIGURATION ---
class Config:
    IMAGE_DATA_DIR = "handwriting_dataset" 
    PROCESSED_DATA_DIR = "data"
    LABELS_FILE = os.path.join(PROCESSED_DATA_DIR, "labels.csv")
    CHARACTERS_FILE = os.path.join(PROCESSED_DATA_DIR, "characters.txt")
    IMG_HEIGHT = 64
    BATCH_SIZE = 32
    # Increased epochs because augmentation requires more training
    EPOCHS = 200 
    LEARNING_RATE = 0.001
    CHARACTERS = None
    VALIDATION_SPLIT = 0.1

# --- 2. PYTORCH DATASET AND LOADER ---
class HandwritingDataset(Dataset):
    def __init__(self, image_paths, labels, char_to_num, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.char_to_num = char_to_num
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        label = str(self.labels[idx])
        
        try:
            image = Image.open(image_path).convert("L")
            if self.transform:
                image = self.transform(image)
        except FileNotFoundError:
            return None, None
            
        try:
            label_encoded = torch.tensor([self.char_to_num[char] for char in label], dtype=torch.long)
        except KeyError as e:
            print(f"Error encoding label '{label}'. Character {e} not in vocab.")
            return None, None
            
        return image, label_encoded

def collate_fn(batch):
    batch = [b for b in batch if b[0] is not None]
    if not batch:
        return torch.tensor([]), torch.tensor([]), torch.tensor([]), torch.tensor([])

    images, labels = zip(*batch)
    image_widths = [img.shape[2] for img in images]
    max_width = max(image_widths)
    padded_images = torch.zeros(len(images), 1, Config.IMG_HEIGHT, max_width)
    for i, img in enumerate(images):
        padded_images[i, :, :, :img.shape[2]] = img
    label_lengths = torch.tensor([len(label) for label in labels], dtype=torch.long)
    padded_labels = nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=0)
    downsample_factor = 4
    input_lengths = torch.tensor([w // downsample_factor for w in image_widths], dtype=torch.long)
    return padded_images, padded_labels, input_lengths, label_lengths

# --- 3. THE CRNN MODEL (PYTORCH) ---
class CRNN(nn.Module):
    def __init__(self, num_chars):
        super(CRNN, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1), nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.bridge = nn.Linear(64 * (Config.IMG_HEIGHT // 4), 64)
        self.rnn = nn.LSTM(64, 128, num_layers=2, bidirectional=True, dropout=0.25, batch_first=True)
        self.output = nn.Linear(256, num_chars)

    def forward(self, x):
        x = self.cnn(x)
        x = x.permute(0, 3, 1, 2)
        b, w, c, h = x.size()
        x = x.reshape(b, w, c * h)
        x = self.bridge(x)
        x, _ = self.rnn(x)
        x = self.output(x)
        return nn.functional.log_softmax(x, dim=2)

# --- 4. DECODING ---
def decode_predictions(preds, num_to_char):
    decoded_texts = []
    pred_indices = torch.argmax(preds, dim=2)
    for indices in pred_indices:
        last_char_idx = -1
        text = ''
        for idx in indices:
            if idx.item() != 0 and idx.item() != last_char_idx:
                if idx.item() in num_to_char:
                    text += num_to_char[idx.item()]
            last_char_idx = idx.item()
        decoded_texts.append(text)
    return decoded_texts

# --- 5. MAIN EXECUTION ---
if __name__ == "__main__":
    # --- FIX for Windows Terminal Unicode Display ---
    if sys.stdout.encoding != 'utf-8':
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    # --- END FIX ---

    if not os.path.exists(Config.CHARACTERS_FILE) or not os.path.exists(Config.LABELS_FILE):
         print(f"Error: Could not find '{Config.CHARACTERS_FILE}' or '{Config.LABELS_FILE}'.")
         print("Please run 'prepare_dataset.py' to generate these files.")
         exit()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    with open(Config.CHARACTERS_FILE, "r", encoding="utf-8") as f:
        Config.CHARACTERS = [line.rstrip('\n') for line in f.readlines()]

    char_to_num = {char: i + 1 for i, char in enumerate(Config.CHARACTERS)}
    num_to_char = {i: char for char, i in char_to_num.items()}
    num_to_char[0] = '' # Blank token
    vocab_size = len(Config.CHARACTERS) + 1

    df = pd.read_csv(Config.LABELS_FILE)
    
    # --- DATA AUGMENTATION ---
    # Transform for training data (with augmentation)
    train_transform = transforms.Compose([
        transforms.Resize((Config.IMG_HEIGHT, 400)), # Resize to a larger canvas for augmentation
        transforms.RandomAffine(degrees=5, translate=(0.05, 0.05), scale=(0.9, 1.1), shear=5),
        transforms.ColorJitter(brightness=0.4, contrast=0.4),
        transforms.ToTensor()
    ])
    
    # Transform for validation data (no augmentation)
    val_transform = transforms.Compose([
        transforms.Resize((Config.IMG_HEIGHT, 400)), # Must match training size
        transforms.ToTensor()
    ])

    # Manual split of data before creating datasets
    data = list(zip(df["image_path"].values, df["label"].values))
    random.shuffle(data)
    split_idx = int(len(data) * (1 - Config.VALIDATION_SPLIT))
    train_data = data[:split_idx]
    val_data = data[split_idx:]

    train_paths, train_labels = zip(*train_data)
    val_paths, val_labels = zip(*val_data)

    # Create two separate Dataset objects
    train_dataset = HandwritingDataset(train_paths, train_labels, char_to_num, transform=train_transform)
    val_dataset = HandwritingDataset(val_paths, val_labels, char_to_num, transform=val_transform)

    # DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=Config.BATCH_SIZE, shuffle=True, collate_fn=collate_fn, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=Config.BATCH_SIZE, shuffle=False, collate_fn=collate_fn, num_workers=4, pin_memory=True)

    model = CRNN(num_chars=vocab_size).to(device)
    criterion = nn.CTCLoss(blank=0, reduction='mean', zero_infinity=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=Config.LEARNING_RATE)

    print(model)
    print("\nStarting training with data augmentation...\n")

    for epoch in range(Config.EPOCHS):
        model.train()
        total_loss = 0
        for images, labels, input_lengths, label_lengths in train_loader:
            if images.nelement() == 0: continue
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            preds = model(images).permute(1, 0, 2)
            loss = criterion(preds, labels, input_lengths, label_lengths)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        avg_train_loss = total_loss / len(train_loader) if len(train_loader) > 0 else 0
        print(f"Epoch {epoch + 1}/{Config.EPOCHS} - Training Loss: {avg_train_loss:.4f}")

        if epoch == 0 or (epoch + 1) % 20 == 0:
            model.eval()
            with torch.no_grad():
                try:
                    val_images, val_labels, _, _ = next(iter(val_loader))
                    if val_images.nelement() > 0:
                        preds = model(val_images.to(device))
                        pred_texts = decode_predictions(preds.cpu(), num_to_char)
                        true_texts = ["".join([num_to_char[i.item()] for i in lab if i.item() in num_to_char]) for lab in val_labels]
                        
                        print("-" * 100)
                        print(f"Sample predictions for epoch {epoch + 1}")
                        for i in range(min(4, len(pred_texts))):
                            print(f"True: {true_texts[i]:<25} | Pred: {pred_texts[i]}")
                        print("-" * 100)
                except StopIteration:
                    pass

    print("\nTraining complete.\n")
    torch.save(model.state_dict(), "handwriting_model_pytorch_v2.pth")
    print("Model saved to 'handwriting_model_pytorch_v2.pth'")

