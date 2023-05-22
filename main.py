import sounddevice as sd
import numpy as np
import scipy.io.wavfile as wav
import keyboard
import threading
import whisper

print("Loading...")

class WhisperYourThoughts:
    '''Sample rate of the input device.'''
    fs = 44100

    '''Number of channels of the input device.'''
    channels = 2

    '''Sound device recording instance.'''
    recording = None

    '''Index of the current recording.'''
    current = 1

    '''The thread running self.recorder().'''
    rec = None

    '''The thread running self.register_keys().'''
    key_thread = None

    '''The data recorded.'''
    data = None

    def __init__(self):
        # Choose a second key of your choice, for example "alt" "space" "y"
        # Default: ctrl+"alt"
        self.second_key = "alt"

        # choose model: "tiny, base, small, medium, large"
        # But you need a corresponding VRAM on your GPU. small needs for example at least 2GB
        self.model = whisper.load_model("large")

        input_device = sd.query_devices(kind='input')
        if input_device is None:
            print("No input devices found!")
            return
        else:
            print(f'Using audio input: {input_device["name"]}')
            self.fs = int(input_device["default_samplerate"])
            self.channels = input_device["max_input_channels"]

            self.reset_recording()

    def reset_recording(self):
        self.data = np.ndarray((0, self.channels))
        self.recording = None


    def on_key_press(self, event):
        if not self.recording:  
            print("recording", end='', flush=True) 
            self.is_recording = True

            def recorder(indata, frames, time, status):
                self.data = np.concatenate((self.data, indata), axis=0)
                print('.', end='', flush=True)

            self.recording = sd.InputStream(samplerate=self.fs, channels=self.channels, callback=recorder)
            self.recording.start()

    def on_key_release(self, event):
        self.recording.stop()
        self.recording.close()

        wav.write(f'output_{self.current}.wav', self.fs, self.data)
        self.reset_recording()
        print('done')

        self.rec = threading.Thread(
            target=self.recorder, 
            args=[self.current], 
            name=f"transcription-{self.current}"
        )
        self.rec.start()
        self.current += 1

    def recorder(self, index):
        print(f"start transcribing recording {index}...")
        try:
            result = self.model.transcribe(f"output_{index}.wav", fp16=False)
            text = result["text"][1:]
            # In my version, result["text"] always outputs a whitespace beforehand, like: result["text"] = " hallo world"
            print(f"recording {index}: {text}")
            keyboard.write(text)
        except Exception as e:
            print(f"Error while transcribing recording {index}: {e}")

    def register_keys(self): 
        print(f"Press {self.second_key} to start recording")
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