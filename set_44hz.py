from pydub import AudioSegment
import os

def convert_sample_rate(input_dir, output_dir, target_sample_rate=44100):
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Iterate over all files in the input directory
    for filename in os.listdir(input_dir):
        if filename.endswith(".wav"):
            input_file = os.path.join(input_dir, filename)
            output_file = os.path.join(output_dir, filename)
            # Load the audio file
            sound = AudioSegment.from_wav(input_file)
            # Set the sample rate
            sound = sound.set_frame_rate(target_sample_rate)
            # Export the modified audio to WAV format
            sound.export(output_file, format="wav")

# Input directory containing the .wav files
input_dir = "breed2"
# Output directory for the converted files
output_dir = "output"
# Target sample rate in Hz
target_sample_rate = 44100

# Convert the sample rate of all .wav files in the input directory
convert_sample_rate(input_dir, output_dir, target_sample_rate)
