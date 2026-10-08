import torch
from torchvision import transforms
import random
import os
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt

# Import the model class from your training script
from train import CRNN, Config

def test_model():
    """
    Loads the trained model and tests it on a few random images
    from the dataset.
    """
    print("--- Starting Model Test ---")

    # --- 1. Load Model and Vocabulary ---
    if not os.path.exists("handwriting_model_pytorch.pth"):
        print("Error: 'handwriting_model_pytorch.pth' not found. Please train the model first.")
        return
        
    if not os.path.exists(Config.CHARACTERS_FILE):
        print(f"Error: '{Config.CHARACTERS_FILE}' not found. Please run 'prepare_dataset.py'.")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load the character set
    with open(Config.CHARACTERS_FILE, "r", encoding="utf-8") as f:
        characters = [line.rstrip('\n') for line in f.readlines()]
    
    num_to_char = {i + 1: char for i, char in enumerate(characters)}
    num_to_char[0] = '' # Blank token
    vocab_size = len(characters) + 1

    # Initialize and load the model
    model = CRNN(num_chars=vocab_size).to(device)
    model.load_state_dict(torch.load("handwriting_model_pytorch.pth", map_location=device))
    model.eval()
    print("Model loaded successfully.")

    # --- 2. Load Image Paths ---
    df = pd.read_csv(Config.LABELS_FILE)
    image_paths = df["image_path"].values
    labels = df["label"].values
    
    # --- 3. Test on Random Images ---
    for _ in range(10): # Test on 10 random images
        # Select a random image
        idx = random.randint(0, len(image_paths) - 1)
        image_path = image_paths[idx]
        true_label = str(labels[idx])

        # Preprocess the image
        image = Image.open(image_path).convert("L")
        transform = transforms.Compose([
            transforms.Resize((Config.IMG_HEIGHT, 256)), # Resize to a consistent width for testing
            transforms.ToTensor()
        ])
        image_tensor = transform(image).unsqueeze(0).to(device) # Add batch dimension

        # Get prediction
        with torch.no_grad():
            preds = model(image_tensor)
        
        # Decode prediction
        pred_indices = torch.argmax(preds, dim=2)
        last_char_idx = -1
        pred_text = ''
        for index in pred_indices[0]: # We only have one item in the batch
            if index.item() != 0 and index.item() != last_char_idx:
                if index.item() in num_to_char:
                    pred_text += num_to_char[index.item()]
            last_char_idx = index.item()

        # Display result
        plt.figure(figsize=(10, 2))
        plt.imshow(image, cmap='gray')
        plt.title(f'True: {true_label}\nPred: {pred_text}')
        plt.axis('off')
        plt.show()

if __name__ == "__main__":
    test_model()

