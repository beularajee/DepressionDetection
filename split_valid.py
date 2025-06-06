import os
import shutil
import pandas as pd

# Define paths
source_folder = r'C:\Users\hp\Desktop\project\EEG_DataSet_P_N.v2i.multiclass\valid'  # Path to your images folder
csv_file = r'C:\Users\hp\Desktop\project\EEG_DataSet_P_N.v2i.multiclass\valid\_classes.csv'  # Path to the CSV file
positive_folder = os.path.join(source_folder, 'positive')
negative_folder = os.path.join(source_folder, 'negative')

# Create subfolders if they don't exist
os.makedirs(positive_folder, exist_ok=True)
os.makedirs(negative_folder, exist_ok=True)

# Read the CSV file
df = pd.read_csv(csv_file)

# Loop through each row in the CSV
for index, row in df.iterrows():
    image_name = row['filename']
    pos_label = row['Positive']
    neg_label = row['Negative']

    # Get the full path of the image
    image_path = os.path.join(source_folder, image_name)

    # Check if the image exists
    if os.path.exists(image_path):
        # Move the image to the corresponding folder
        if pos_label == 1:
            shutil.move(image_path, os.path.join(positive_folder, image_name))
        elif neg_label == 1:
            shutil.move(image_path, os.path.join(negative_folder, image_name))
    else:
        print(f"Image {image_name} not found in the source folder.")

print("Images have been successfully moved to the respective folders.")
