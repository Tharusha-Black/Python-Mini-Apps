import os

def move_files_to_parent_directory(input_dir):
    # Iterate over each subdirectory
    for root, dirs, files in os.walk(input_dir):
        for filename in files:
            if filename.endswith(".wav"):
                src_file = os.path.join(root, filename)
                # Generate new filename with "audio" prefix and an incrementing number
                new_filename = f"audio{move_files_to_parent_directory.counter}.wav"
                move_files_to_parent_directory.counter += 1
                dest_file = os.path.join(input_dir, new_filename)
                os.rename(src_file, dest_file)

# Initialize the counter attribute
move_files_to_parent_directory.counter = 1

# Input directory containing subdirectories with files
input_dir = "output"

# Call the function to move files
move_files_to_parent_directory(input_dir)
