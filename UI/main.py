import tkinter as tk
import tkinter.ttk
from tkinter import filedialog
import soundfile as sf
import sounddevice as sd
import librosa
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import platform


class main_window(tk.Tk):
    def on_open(self):
        path = filedialog.askopenfilename(parent=self,
                                          title='打开一个波形文件',
                                          filetypes=[('波形文件', '.wav')])
        if not path:
            return False

        wav, samplerate = sf.read(path)
        self.wav = librosa.resample(wav, samplerate, 16000)

        plt.figure(figsize=(4.8, 3.2))
        plt.plot(np.arange(wav.shape[0]), wav)
        plt.title('origin wav')
        plt.xticks([])
        plt.savefig('origin.png', dpi=100)
        self.image1 = tk.PhotoImage(file='origin.png')  # 需要对图片保持引用
        self.canvas1.create_image(0, 0, anchor='nw', image=self.image1)
        return True

    def button1_on_click(self):
        try:
            sd.play(self.wav, 16000)
        except:
            pass

    def init_window(self):
        tk.Tk.__init__(self)
        self.tk.call('tk', 'scaling', scale / 75)
        self.title('语音转换')

    def init_menu(self):
        self.menu = tk.Menu(self)
        self.config(menu=self.menu)
        self.file_menu = tk.Menu(self.menu, tearoff=False)
        self.menu.add_cascade(label='文件', menu=self.file_menu)
        self.file_menu.add_command(label='打开', command=self.on_open)

    def init_layout(self):
        self.grid_columnconfigure(0, weight=1)
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)
        # self.grid_rowconfigure(1, weight=1)

        self.canvas1 = tk.Canvas(self, width=480, height=320)
        self.canvas1.grid(row=0, column=0)
        plt.figure(figsize=(4.8, 3.2))
        plt.title('origin wav')
        plt.xticks([])
        plt.yticks([])
        plt.savefig('origin.png', dpi=100)
        self.image1 = tk.PhotoImage(file='origin.png')  # 需要对图片保持引用
        self.canvas1.create_image(0, 0, anchor='nw', image=self.image1)

        self.frame1 = tk.Frame(self)
        self.frame1.grid(row=0, column=1, sticky=tk.W+tk.E+tk.N+tk.S)
        self.frame1.grid_columnconfigure(0, weight=1)
        self.frame1.grid_rowconfigure(0, weight=1)

        self.button1 = tk.Button(
            self.frame1, text='播放', command=self.button1_on_click)
        self.button1.grid(row=0, column=0, padx=8, ipadx=8)

        self.list1 = tk.Listbox(self)
        self.list1.grid(row=1, column=0, sticky=tk.W+tk.E,
                        padx=8, ipadx=8, pady=8, ipady=8)

        # self.canvas2 = tk.Canvas(self)
        # self.canvas2.grid(row=1, column=0, sticky=tk.W+tk.E+tk.N+tk.S)
        # self.canvas2 = tk.Canvas(self)
        # self.canvas2.grid(row=1, column=1, sticky=tk.W+tk.E+tk.N+tk.S)

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
        return self.mainloop()


if __name__ == '__main__':
    if platform.system() == 'Windows':
        import ctypes
        ctypes.windll.shcore.SetProcessDpiAwareness(1)
        scale = ctypes.windll.shcore.GetScaleFactorForDevice(0)
    else:
        scale = 100
    main = main_window()
    main.message_loop()
