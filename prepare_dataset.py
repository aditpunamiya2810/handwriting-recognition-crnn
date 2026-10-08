import os
import csv

# --- CONFIGURATION ---
ROOT_DATA_DIR = "handwriting_dataset"
OUTPUT_DIR = "data"
OUTPUT_CSV = os.path.join(OUTPUT_DIR, "labels.csv")
OUTPUT_CHARS = os.path.join(OUTPUT_DIR, "characters.txt")

# --- Updated Word Lists based on your images ---

EN_WORDS = [
    # Image 1
    "Water", "Food", "Book", "Pen", "Child", "Man", "Woman", "House", "School", "Car",
    "Door", "Chair", "Table", "Umbrella", "Mobile", "Computer", "Picture", "Flower", "Tree", "Stone",
    # Image 2
    "To eat", "To drink", "To walk", "To run", "To speak", "To read", "To write", "To think", "To see", "To listen",
    "To get up", "To sit", "To open", "To close", "To buy", "To sell", "To work", "To play", "To cook", "To wear",
    # Image 3
    "Love", "Hate", "Happiness", "Sadness", "Fear", "Anger", "Peace", "Worry", "Hope", "Tiredness",
    "Red", "Blue", "Green", "Yellow", "Black", "White", "Brown", "Pink", "Orange", "Purple",
    # Image 4
    "Mother", "Father", "Brother", "Sister", "Grandmother", "Grandfather", "Uncle", "Aunt", "Son", "Daughter",
    "Name", "Question", "Answer", "Truth", "Lie", "Road", "Market", "Language", "Music", "World",
    # Image 5
    "Miracle", "Freedom", "Reflection", "Museum", "Invention", "Imagination", "Inspiration", "Presence", "Prediction", "Sensitive",
    "Sun", "Moon", "Star", "Sky", "Cloud", "Rain", "Wind", "Time", "Day", "Night"
]

HI_WORDS = [
    # Image 1
    "पानी", "खाना", "किताब", "कलम", "बच्चा", "आदमी", "औरत", "घर", "स्कूल", "गाड़ी",
    "दरवाजा", "कुर्सी", "मेज", "छाता", "मोबाइल", "कंप्यूटर", "तस्वीर", "फूल", "पेड़", "पत्थर",
    # Image 2
    "खाना", "पीना", "चलना", "दौड़ना", "बोलना", "पढ़ना", "लिखना", "सोचना", "देखना", "सुनना",
    "उठना", "बैठना", "खोलना", "बंद करना", "खरीदना", "बेचना", "काम करना", "खेलना", "पकाना", "पहनना",
    # Image 3
    "प्यार", "नफरत", "खुशी", "दुःख", "डर", "गुस्सा", "शांति", "चिंता", "आशा", "थकावट",
    "लाल", "नीला", "हरा", "पीला", "काला", "सफेद", "भूरा", "गुलाबी", "नारंगी", "बैंगनी",
    # Image 4
    "माँ", "पिता", "भाई", "बहन", "दादी", "दादा", "चाचा", "चाची", "बेटा", "बेटी",
    "नाम", "सवाल", "जवाब", "सच्चाई", "झूठ", "रास्ता", "बाजार", "भाषा", "संगीत", "दुनिया",
    # Image 5
    "चमत्कार", "स्वतंत्रता", "प्रतिबिंब", "संग्रहालय", "आविष्कार", "कल्पना", "प्रेरणा", "उपस्थिति", "भविष्यवाणी", "संवेदनशील",
    "सूरज", "चाँद", "तारा", "आकाश", "बादल", "बारिश", "हवा", "समय", "दिन", "रात"
]

def create_dataset_files():
    print("--- Starting Dataset Preparation ---")
    
    if not os.path.isdir(ROOT_DATA_DIR):
        print(f"\n[ERROR] The directory '{ROOT_DATA_DIR}' was not found.")
        return

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    image_label_pairs = []
    all_chars = set()

    all_words = EN_WORDS + HI_WORDS
    for word in all_words:
        all_chars.update(list(word))

    style_folders = sorted([d for d in os.listdir(ROOT_DATA_DIR) if os.path.isdir(os.path.join(ROOT_DATA_DIR, d))])
    print(f"Found {len(style_folders)} style folder(s): {style_folders}")

    for style_folder in style_folders:
        style_path = os.path.join(ROOT_DATA_DIR, style_folder)
        for lang_folder in ["english", "hindi"]:
            lang_path = os.path.join(style_path, lang_folder)
            if not os.path.isdir(lang_path):
                continue
            
            word_list = EN_WORDS if lang_folder == "english" else HI_WORDS
            
            try:
                # Expects image names like '1.png', '2.png', etc.
                image_files = sorted(os.listdir(lang_path), key=lambda f: int(os.path.splitext(f)[0]))
            except ValueError:
                print(f"    - [ERROR] Could not sort files numerically in '{lang_path}'. Skipping.")
                continue

            for i, image_file in enumerate(image_files):
                if i < len(word_list):
                    image_path = os.path.join(lang_path, image_file)
                    label = word_list[i]
                    image_label_pairs.append([image_path, label])

    # --- Create labels.csv ---
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["image_path", "label"])
        writer.writerows(image_label_pairs)
    print(f"Successfully created '{OUTPUT_CSV}' with {len(image_label_pairs)} entries.")

    # --- Create characters.txt ---
    sorted_chars = sorted(list(all_chars))
    with open(OUTPUT_CHARS, "w", encoding="utf-8") as f:
        for char in sorted_chars:
            f.write(char + "\n")
    print(f"Successfully created '{OUTPUT_CHARS}' with {len(sorted_chars)} unique characters.")
    print("--- Preparation complete! ---")

if __name__ == "__main__":
    create_dataset_files()

