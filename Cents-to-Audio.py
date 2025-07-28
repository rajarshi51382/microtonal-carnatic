import asyncio
import numpy as np
import pygame
import platform
from scipy.fft import fft
from scipy.io import wavfile
from math import log2
import js
import io
import base64

FPS = 60
SAMPLE_RATE = 44100
DURATION = 0.5  # Duration per note in seconds
BASE_FREQ = 440.0  # A4 note in Hz

def cents_to_freq(cents, base_freq=BASE_FREQ):
    """Convert cents to frequency."""
    return base_freq * (2 ** (cents / 1200.0))

def generate_sine_wave(freq, duration=DURATION, sample_rate=SAMPLE_RATE):
    """Generate a sine wave for a given frequency."""
    t = np.linspace(0, duration, int(sample_rate * duration), False)
    wave = 0.5 * np.sin(2 * np.pi * freq * t)
    # Convert to 16-bit stereo format for pygame
    wave = (wave * 32767).astype(np.int16)
    wave = np.column_stack((wave, wave))  # Stereo
    return wave

def freq_to_cents(freq, base_freq=BASE_FREQ):
    """Convert frequency to cents."""
    if freq <= 0:
        return 0
    return 1200 * log2(freq / base_freq)

def analyze_audio(audio_data, sample_rate=SAMPLE_RATE):
    """Analyze audio to extract dominant frequency."""
    mono = audio_data[:, 0]
    N = len(mono)
    yf = fft(mono)
    xf = np.fft.fftfreq(N, 1 / sample_rate)
    idx = np.argmax(np.abs(yf[:N//2]))
    freq = abs(xf[idx])
    return freq

def setup():
    """Initialize pygame and mixer."""
    pygame.init()
    pygame.mixer.init(frequency=SAMPLE_RATE, size=-16, channels=2)

def generate_audio_from_cents(cents_sequence):
    """Generate audio from a sequence of cents."""
    audio_segments = []
    for cents in cents_sequence:
        freq = cents_to_freq(cents)
        wave = generate_sine_wave(freq)
        audio_segments.append(wave)
    return np.concatenate(audio_segments, axis=0)

def play_audio(audio_data):
    """Play audio data using pygame."""
    sound = pygame.sndarray.make_sound(audio_data)
    sound.play()
    pygame.time.wait(int(DURATION * len(audio_data) // SAMPLE_RATE * 1000))

def cents_from_audio(audio_data):
    """Extract cents sequence from audio data."""
    samples_per_segment = int(SAMPLE_RATE * DURATION)
    num_segments = len(audio_data) // samples_per_segment
    cents_sequence = []
    for i in range(num_segments):
        start = i * samples_per_segment
        end = start + samples_per_segment
        segment = audio_data[start:end]
        freq = analyze_audio(segment)
        cents = freq_to_cents(freq)
        cents_sequence.append(round(cents, 2))
    return cents_sequence

def save_audio_to_wav(audio_data, filename="output.wav"):
    """Save audio data to a WAV file or enable download in Pyodide."""
    if platform.system() == "Emscripten":
        # In Pyodide, create a downloadable WAV file
        buffer = io.BytesIO()
        wavfile.write(buffer, SAMPLE_RATE, audio_data)
        wav_data = buffer.getvalue()
        # Convert to base64 for JavaScript
        b64_data = base64.b64encode(wav_data).decode('utf-8')
        # JavaScript to trigger download
        js_code = f"""
        var blob = new Blob([Uint8Array.from(atob('{b64_data}'), c => c.charCodeAt(0))], {{type: 'audio/wav'}});
        var url = URL.createObjectURL(blob);
        var a = document.createElement('a');
        a.href = url;
        a.download = '{filename}';
        a.click();
        URL.revokeObjectURL(url);
        """
        js.eval(js_code)
    else:
        # In local Python, save to disk
        wavfile.write(filename, SAMPLE_RATE, audio_data)
        print(f"Audio saved to {filename}")

async def main():
    """Main function to demonstrate cents to audio and back."""
    setup()
    input_cents = [1193, 1010, 784, 700, 498, 294, 100, 0, 1200, 996, 792, 702, 498, 294, 90]
    print("Input cents:", input_cents)

    # Convert cents to audio
    audio_data = generate_audio_from_cents(input_cents)
    print("Generated audio data shape:", audio_data.shape)

    # Play the audio
    play_audio(audio_data)

    # Save the audio to a WAV file
    save_audio_to_wav(audio_data, "cents_output_3.wav")

    # Convert audio back to cents
    output_cents = cents_from_audio(audio_data)
    print("Recovered cents:", output_cents)

# Handle execution in different environments
if platform.system() == "Emscripten":
    asyncio.ensure_future(main())
else:
    try:
        loop = asyncio.get_running_loop()
        asyncio.ensure_future(main())
    except RuntimeError:
        if __name__ == "__main__":
            asyncio.run(main())
