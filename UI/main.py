import tkinter
from tkinter import filedialog
import soundfile as sf
import sounddevice as sd
import librosa


class main_window:
    def on_open(self):
        path = filedialog.askopenfilename(parent=self.window,
                                          title='打开一个波形文件',
                                          filetypes=[('波形文件（.wav）', '.wav')])
        if not path:
            return False

        wav, samplerate = sf.read(path)
        self.wav = librosa.resample(wav, samplerate, 16000)
        return True

    def init_window(self):
        self.window = tkinter.Tk()
        self.window.title('语音转换')
        self.window.geometry('800x600')

    def init_menu(self):
        self.menu = tkinter.Menu(self.window)
        self.window.config(menu=self.menu)
        self.file_menu = tkinter.Menu(self.menu, tearoff=False)
        self.menu.add_cascade(label='文件', menu=self.file_menu)
        self.file_menu.add_command(label='打开', command=self.on_open)

    def init_layout(self):
        pass

    def init_instance(self):
        self.init_window()
        self.init_menu()
        self.init_layout()

    def __init__(self):
        self.init_instance()
        self.declare_variable()

    def declare_variable(self):
        self.wav = None  # 总是假定采样率为 16000 Hz

    def message_loop(self):
        return self.window.mainloop()


if __name__ == '__main__':
    main = main_window()
    main.message_loop()
