import os

def rename_files_in_directory(directory_path, file_extension):
    for filename in os.listdir(directory_path):
        # Check if the file has the desired extension to avoid processing any other files.
        if filename.endswith(file_extension):
            # Extract the frame number using split method.
            frame_number = int(filename.split('_')[1])
            new_filename = f"{frame_number}.{file_extension}"
            old_file_path = os.path.join(directory_path, filename)
            new_file_path = os.path.join(directory_path, new_filename)
            os.rename(old_file_path, new_file_path)
            print(f"Renamed '{filename}' to '{new_filename}'")

if(os.path.exists(os.getcwd() + "/train/images")):
    images_directory = os.getcwd() + "/train/images"
    labels_directory = os.getcwd() + "/train/labels"
else:
    images_directory = os.getcwd() + "/test/images"
    labels_directory = os.getcwd() + "/test/labels"

rename_files_in_directory(images_directory, 'jpg')
rename_files_in_directory(labels_directory, 'txt')