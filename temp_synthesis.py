import sys
sys.path.append('C:/Users/New User/Downloads/cultural stuff/microtonal-carnatic')
from synthesis.synthesize_audio import synthesize_swara_sequence

swara_list = ['S', 'R1', 'G3', 'M1', 'P', 'D1', 'N3', 'S']
output_path = 'C:/Users/New User/Downloads/cultural stuff/microtonal-carnatic/data/audio/generated_audio.wav'

synthesize_swara_sequence(swara_list, output_path)
