import pyaudio
import wave


class AudioManager :
    def __init__(self) :
        self.audio = pyaudio.PyAudio()
        self.frames = list()
        self.record = False
        
        
    def initialize_stream(self) :
        self.frames = list()
        stream = self.audio.open(format=pyaudio.paInt16, channels=1, rate=44100, input=True, frames_per_buffer=1024)
        return stream
    
    
    def stop_stream(self, stream) :
        stream.stop_stream()
        stream.close()
        self.audio.terminate()
        
    
    def record_audio(self) :
        stream = self.initialize_stream()
        
        self.record = True
        
        self.start_recording(stream)
        
        self.save_audio(self.frames)
        
        self.stop_stream(stream)

    
    def start_recording(self, stream) :
        try :
            while self.record :
                data = stream.read(1024)
                self.frames.append(data)
        
        except KeyboardInterrupt :
            pass
        
        return self.frames
    

    def save_audio(self, frames) :
        sound_file = wave.open("myrecording.wav", "wb")
        sound_file.setnchannels(1)
        sound_file.setsampwidth(self.audio.get_sample_size(pyaudio.paInt16))
        sound_file.setframerate(44100)
        sound_file.writeframes(b"".join(frames))
        sound_file.close()