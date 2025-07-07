import pyaudio
import wave
import threading
import queue
import time
import numpy as np
from faster_whisper import WhisperModel

class HebrewSpeechToText:
    def __init__(self):
        # Audio settings
        self.CHUNK = 1024
        self.FORMAT = pyaudio.paInt16
        self.CHANNELS = 1
        self.RATE = 16000
        self.RECORD_SECONDS = 3  # Process audio in 3-second chunks
        
        # Initialize audio
        self.audio = pyaudio.PyAudio()
        self.audio_queue = queue.Queue()
        self.is_recording = False
        
        # Initialize Whisper model
        print("Loading Whisper model...")
        # Use base model first, you can change to "ivrit-ai/whisper-v2-d3-e3" for Hebrew-specific
        self.model = WhisperModel("base", device="cpu", compute_type="int8")
        print("Model loaded successfully!")
        
    def audio_callback(self, in_data, frame_count, time_info, status):
        """Callback function for audio stream"""
        self.audio_queue.put(in_data)
        return (in_data, pyaudio.paContinue)
    
    def start_recording(self):
        """Start recording audio"""
        self.is_recording = True
        
        # Open audio stream
        self.stream = self.audio.open(
            format=self.FORMAT,
            channels=self.CHANNELS,
            rate=self.RATE,
            input=True,
            frames_per_buffer=self.CHUNK,
            stream_callback=self.audio_callback
        )
        
        print("🎤 Recording started... Speak in Hebrew!")
        self.stream.start_stream()
        
        # Start processing thread
        self.processing_thread = threading.Thread(target=self.process_audio)
        self.processing_thread.daemon = True
        self.processing_thread.start()
    
    def process_audio(self):
        """Process audio chunks and transcribe"""
        frames = []
        frame_count = 0
        target_frames = int(self.RATE / self.CHUNK * self.RECORD_SECONDS)
        last_transcription = ""
        while self.is_recording:
            try:
                # Get audio data from queue
                data = self.audio_queue.get(timeout=1)
                frames.append(data)
                frame_count += 1
                # Process when we have enough frames
                if frame_count >= target_frames:
                    # Convert to numpy array
                    audio_data = b''.join(frames)
                    audio_np = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
                    # Transcribe with Whisper
                    try:
                        segments, info = self.model.transcribe(
                            audio_np, 
                            language="he",  # Hebrew language code
                            beam_size=5,
                            best_of=5,
                            temperature=0.0,
                            vad_filter=True  # Enable VAD to reduce false positives
                        )
                        # Print transcription
                        transcription = ""
                        for segment in segments:
                            transcription += segment.text
                        # Only print if not empty and not repeated
                        if transcription.strip() and transcription != last_transcription:
                            # Reverse for RTL display in terminal
                            rtl_transcription = transcription[::-1]
                            print(f"🗣️  Hebrew: {rtl_transcription}")
                            last_transcription = transcription
                    except Exception as e:
                        print(f"❌ Transcription error: {e}")
                    # Reset for next chunk
                    frames = []
                    frame_count = 0
            except queue.Empty:
                continue
            except Exception as e:
                print(f"❌ Processing error: {e}")
    
    def stop_recording(self):
        """Stop recording and clean up"""
        print("🛑 Stopping recording...")
        self.is_recording = False
        
        if hasattr(self, 'stream'):
            self.stream.stop_stream()
            self.stream.close()
        
        self.audio.terminate()
        print("✅ Recording stopped")
    
    def run(self):
        """Main run method"""
        try:
            self.start_recording()
            
            print("\n" + "="*50)
            print("🎯 Hebrew Speech-to-Text Active")
            print("🗣️  Speak in Hebrew and see the transcription")
            print("⌨️  Press Ctrl+C to stop")
            print("="*50 + "\n")
            
            # Keep running until interrupted
            while True:
                time.sleep(1)
                
        except KeyboardInterrupt:
            print("\n\n🔄 Shutting down...")
            self.stop_recording()
        except Exception as e:
            print(f"❌ Error: {e}")
            self.stop_recording()

def main():
    """Main function"""
    print("🚀 Starting Hebrew Speech-to-Text...")
    
    # Check if microphone is available
    try:
        audio_test = pyaudio.PyAudio()
        if audio_test.get_device_count() == 0:
            print("❌ No audio devices found!")
            return
        audio_test.terminate()
    except Exception as e:
        print(f"❌ Audio system error: {e}")
        return
    
    # Start the speech-to-text system
    stt = HebrewSpeechToText()
    stt.run()

if __name__ == "__main__":
    main()


    