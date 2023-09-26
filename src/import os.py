import os

# Create the 'src' folder and subfolders
def create_folders():
    root_folder = 'src'
    subfolders = ['algorithms', 'api', 'audio_processing', 'database', 'utils', 'vst_interface']

    if not os.path.exists(root_folder):
        os.makedirs(root_folder)

    for subfolder in subfolders:
        path = os.path.join(root_folder, subfolder)
        if not os.path.exists(path):
            os.makedirs(path)

# Run the function to create folders
create_folders()

