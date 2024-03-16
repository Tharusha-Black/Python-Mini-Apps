import os

def delete_small_files(input_dir, min_size_kb=155):
    # Iterate over each file in the directory
    for root, dirs, files in os.walk(input_dir):
        for filename in files:
            if filename.endswith(".wav"):
                file_path = os.path.join(root, filename)
                # Get the size of the file in KB
                file_size_kb = os.path.getsize(file_path) / 1024
                # Delete the file if its size is less than the specified minimum size
                if file_size_kb < min_size_kb:
                    os.remove(file_path)

# Input directory containing the .wav files
input_dir = "output"

# Call the function to delete small files
delete_small_files(input_dir)
