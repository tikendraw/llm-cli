from pydub import AudioSegment
from pydub.silence import split_on_silence

# Load the audio file
audio = AudioSegment.from_file("./howimanage.mp3")  # Change to your file path

# Define the split time in milliseconds
split_time = 35 * 1000  # 30 seconds

# Split the audio
first_half = audio[:split_time]
second_half = audio[split_time-3000:split_time*2]

# Add silence (e.g., 3 seconds)
silence = AudioSegment.silent(duration=15000)  # silence

# Concatenate the parts
final_audio = first_half + silence + second_half

# Export the final audio
final_audio.export("output_audio.mp3", format="mp3")
