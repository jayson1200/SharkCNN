import os
import shutil
from tqdm import tqdm

datasets_path = "/home/meribejayson/Desktop/SharkCNN/datasets-reduced/"
dataset_name = "megaset"

train_path = f"{dataset_name}/train/"
train_images_path = f"{dataset_name}/train/images/"
train_labels_path = f"{dataset_name}/train/labels/"

test_path = f"{dataset_name}/test/"
test_images_path = f"{dataset_name}/test/images/"
test_labels_path = f"{dataset_name}/test/labels/"

os.mkdir(dataset_name)
os.mkdir(train_path)
os.mkdir(train_images_path)
os.mkdir(train_labels_path)

os.mkdir(test_path)
os.mkdir(test_images_path)
os.mkdir(test_labels_path)

yaml_file_content = """path: /home/meribejayson/Desktop/SharkCNN/datasets-reduced/megaset

train: train/images
val: train/images
test: test/images

names:
  0: shark
"""

yaml_file = open(f"{dataset_name}/data.yaml", 'w')
yaml_file.write(yaml_file_content)

for folder_name in tqdm(os.listdir(datasets_path)):
    src_images_folder_path = datasets_path + folder_name + "/"
    curr_images_path = "train/images/" if os.path.isdir(src_images_folder_path + "train/images/") else "test/images/"
    src_images_folder_path += curr_images_path

    # Copying images
    for curr_img_name in os.listdir(src_images_folder_path):
        image_name = f"{folder_name}-{curr_img_name}"

        dest_path = shutil.copy(src_images_folder_path + curr_img_name, f"{dataset_name}/{curr_images_path}")

        os.rename(dest_path, f"{dataset_name}/{curr_images_path}/" + image_name)

    # Copying text files
    src_labels_folder_path = datasets_path + folder_name + "/"
    curr_labels_path = "train/labels/" if os.path.isdir(src_labels_folder_path + "train/labels/") else "test/labels/"
    src_labels_folder_path += curr_labels_path

    # Copying images
    for curr_labels_name in os.listdir(src_labels_folder_path):
        labels_name = f"{folder_name}-{curr_labels_name}"

        dest_path = shutil.copy(src_labels_folder_path + curr_labels_name, f"{dataset_name}/{curr_labels_path}")

        os.rename(dest_path, f"{dataset_name}/{curr_labels_path}/" + labels_name)