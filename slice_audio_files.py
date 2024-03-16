import os
import subprocess
from pydub import AudioSegment

def slice_audio_files(input_dir, output_dir, slice_duration=10, ffmpeg_path=None, ffprobe_path=None):
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Iterate over all files in the input directory
    for filename in os.listdir(input_dir):
        if filename.endswith(".wav"):
            input_file = os.path.join(input_dir, filename)
            output_subdir = os.path.join(output_dir, os.path.splitext(filename)[0])
            os.makedirs(output_subdir, exist_ok=True)
            slice_audio(input_file, output_subdir, slice_duration, ffmpeg_path, ffprobe_path)

def slice_audio(input_file, output_dir, slice_duration=10, ffmpeg_path=None, ffprobe_path=None):
    # Use FFmpeg to slice the audio file
    cmd = [
        ffmpeg_path or "ffmpeg",
        "-i", input_file,
        "-f", "segment",
        "-segment_time", str(slice_duration),
        "-c", "copy",
        os.path.join(output_dir, "audio_%03d.wav")
    ]
    subprocess.run(cmd)

# Example usage
input_dir = "input"
output_dir = "output"
slice_duration = 10  # Duration of each slice in seconds
ffmpeg_path = "ffmpeg/bin/ffmpeg"  # Specify the path to FFmpeg
ffprobe_path = "ffmpeg/bin/ffprobe"  # Specify the path to FFprobe

slice_audio_files(input_dir, output_dir, slice_duration, ffmpeg_path, ffprobe_path)
