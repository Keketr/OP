import os

# Define the directory path where the folders will be created
path = os.getcwd()

# Iterate over numbers from 1 to 12
for i in range(1, 13):
    # Create the folder name with the current number
    folder_name = f"INTG8_traces_{i}"

    # Check if the folder already exists
    if not os.path.exists(folder_name):
        # Create the folder if it doesn't exist
        os.makedirs(folder_name)
        print(f"Folder created: {folder_name}")
    else:
        print(f"The folder '{folder_name}' already exists.")
