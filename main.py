import sounddevice as sd
import numpy as np
import scipy.io.wavfile as wav
import keyboard
import threading
import whisper

class WhisperYourThoughts:
    def __init__(self):
        # Choose a second key of your choice, for example "alt" "space" "y"
        # Default: ctrl+"alt"
        self.second_key = "alt"

        # choose model: "tiny, base, small, medium, large"
        # But you need a corresponding VRAM on your GPU. small needs for example at least 2GB
        self.model = whisper.load_model("small")
        self.key_thread = None
        self.fs = 44100 
        self.recording = None
        self.is_recording = False
        self.rec = None 

    def on_key_press(self, event):
        if not self.is_recording:  
            print("start recording...") 
            self.is_recording = True
            self.recording = sd.InputStream(samplerate=self.fs, channels=2)
            self.recording.start()
            self.data = []
            self.rec = threading.Thread(target=self.recorder)
            self.rec.start()

    def on_key_release(self, event):
        print("finished recording")
        self.is_recording = False
        self.recording.stop()
        self.recording.close()
        data = np.concatenate(self.data, axis=0)
        # Save recording to file
        wav.write('output.wav', self.fs, data)

    def recorder(self):
        while self.is_recording:
            frame, overflowed = self.recording.read(1024)
            self.data.append(frame)
        print("start transcribing...")
        result = self.model.transcribe("output.wav")
        text =  result["text"]
        print(text[1:])# In my version, result["text"] always outputs a whitespace beforehand, like: result["text"] = " hallo world"
        keyboard.write(text[1:])
        print("ready...")

    def register_keys(self): 
        keyboard.on_press_key(self.second_key, self.on_key_press, suppress=False)#supress only works on windows.
        keyboard.on_release_key(self.second_key, self.on_key_release, suppress=False)
        keyboard.wait()

    def initHotKeyThreading(self):
        self.key_thread = threading.Thread(target=self.register_keys, daemon=True)
        self.key_thread.start()

    def putInMainLoop(self,timeout):
        if self.key_thread is not None:
            self.key_thread.join(timeout=timeout)


whisperYourThoughts = WhisperYourThoughts()
whisperYourThoughts.initHotKeyThreading()

alive = True
print('finished loading, main loop started...')
while alive:
    try:
        whisperYourThoughts.putInMainLoop(1)
    except KeyboardInterrupt:
        print("Bye")
        alive = False