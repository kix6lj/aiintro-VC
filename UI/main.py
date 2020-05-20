import tkinter as tk
import tkinter.ttk as ttk
from tkinter import filedialog
import soundfile as sf
import sounddevice as sd
import librosa
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import platform
import threading


def detail_transform(wav, style):
    return np.array(wav)


class main_window(tk.Tk):
    def button1_on_click(self):
        path = filedialog.askopenfilename(parent=self,
                                          title='打开一个波形文件',
                                          filetypes=[('波形文件', '.wav')])
        if not path:
            return False

        wav, samplerate = sf.read(path)
        self.wav = wav = librosa.resample(wav, samplerate, 16000)
        self.transformed_wav = None

        plt.figure(figsize=(4.8, 3.2))
        plt.plot(np.arange(wav.shape[0]), wav)
        plt.title('origin wav')
        plt.xticks([])
        maxabs = np.max(np.abs(wav)) * 1.2
        plt.ylim([-maxabs, maxabs])
        plt.savefig('origin.png', dpi=100)
        plt.close()
        self.image1 = tk.PhotoImage(file='origin.png')  # 需要对图片保持引用
        self.canvas1.create_image(0, 0, anchor='nw', image=self.image1)

        plt.figure(figsize=(4.8, 3.2))
        plt.title('transformed wav')
        plt.xticks([])
        plt.yticks([])
        plt.savefig('transformed.png', dpi=100)
        plt.close()
        self.image2 = tk.PhotoImage(file='transformed.png')
        self.canvas2.create_image(0, 0, anchor='nw', image=self.image2)
        self.button2_on_click()
        return True

    def button2_on_click(self):
        try:
            sd.play(np.array(self.wav), 16000)
        except:
            pass

    def button3_on_click(self):
        sel_style = self.listbox1.curselection()[0]
        if self.wav is None:
            return
        self.button3.config(state=tk.DISABLED)
        self.button4.config(state=tk.DISABLED)
        self.listbox1.config(state=tk.DISABLED)
        threading.Thread(target=self.transform, args=[sel_style]).start()

    def button4_on_click(self):
        try:
            sd.play(np.array(self.transformed_wav), 16000)
        except:
            pass

    def transform(self, style):
        self.transformed_wav = wav = detail_transform(self.wav, style)
        plt.figure(figsize=(4.8, 3.2))
        plt.plot(np.arange(wav.shape[0]), wav)
        plt.title('transformed wav')
        plt.xticks([])
        maxabs = np.max(np.abs(wav)) * 1.2
        plt.ylim([-maxabs, maxabs])
        plt.savefig('transformed.png', dpi=100)
        plt.close()
        self.image2 = tk.PhotoImage(file='transformed.png')
        self.canvas2.create_image(0, 0, anchor='nw', image=self.image2)

        self.button3.config(state=tk.NORMAL)
        self.button4.config(state=tk.NORMAL)
        self.listbox1.config(state=tk.NORMAL)
        self.button4_on_click()

    def init_window(self):
        tk.Tk.__init__(self)
        self.tk.call('tk', 'scaling', scale / 75)
        self.title('语音转换')
        self.resizable(0, 0)

    def init_layout(self):
        self.grid_columnconfigure(0, weight=0)
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=1)

        # (0, 0)
        self.canvas1 = tk.Canvas(self, width=480, height=320)
        self.canvas1.grid(row=0, column=0)
        plt.figure(figsize=(4.8, 3.2))
        plt.title('origin wav')
        plt.xticks([])
        plt.yticks([])
        plt.savefig('origin.png', dpi=100)
        plt.close()
        self.image1 = tk.PhotoImage(file='origin.png')
        self.canvas1.create_image(0, 0, anchor='nw', image=self.image1)

        # (0, 1)
        self.frame1 = tk.Frame(self)
        self.frame1.grid(row=0, column=1, sticky=tk.W+tk.E+tk.N+tk.S)
        self.frame1.grid_columnconfigure(0, weight=1)
        self.frame1.grid_rowconfigure(0, weight=1)
        self.frame1 = tk.Frame(self.frame1)
        self.frame1.grid(row=0, column=0)

        self.button1 = ttk.Button(
            self.frame1, text='打开', command=self.button1_on_click)
        self.button1.pack(pady=8)

        self.button2 = ttk.Button(
            self.frame1, text='播放', command=self.button2_on_click)
        self.button2.pack(pady=8)

        self.button3 = ttk.Button(
            self.frame1, text='转换', command=self.button3_on_click)
        self.button3.pack(pady=8)

        self.button4 = ttk.Button(
            self.frame1, text='播放结果', command=self.button4_on_click)
        self.button4.pack(pady=8)

        # (1, 0)
        self.canvas2 = tk.Canvas(self, width=480, height=320)
        self.canvas2.grid(row=1, column=0)
        plt.figure(figsize=(4.8, 3.2))
        plt.title('transformed wav')
        plt.xticks([])
        plt.yticks([])
        plt.savefig('transformed.png', dpi=100)
        plt.close()
        self.image2 = tk.PhotoImage(file='transformed.png')
        self.canvas2.create_image(0, 0, anchor='nw', image=self.image2)

        # (1, 1)
        self.frame2 = tk.Frame(self)
        self.frame2.grid(row=1, column=1, sticky=tk.W+tk.E+tk.N+tk.S)
        self.frame2.grid_columnconfigure(0, weight=1)
        self.frame2.grid_rowconfigure(0, weight=1)
        self.frame2 = tk.Frame(self.frame2)
        self.frame2.grid(row=0, column=1, padx=40, pady=8)

        self.label1 = tk.Label(self.frame2, text="转换风格")
        self.label1.pack()

        self.listbox1 = tk.Listbox(self.frame2, selectmode=tk.SINGLE)
        self.listbox1.pack()
        for item in self.styles:
            self.listbox1.insert(tk.END, item)
        self.listbox1.selection_set(0)
        self.listbox1.bind("<Double-Button-1>",
                           lambda _: main.button3_on_click())

    def init_instance(self):
        self.init_window()
        self.init_layout()

    def __init__(self):
        self.declare_variable()
        self.init_instance()

    def declare_variable(self):
        self.wav = None  # 总是假定采样率为 16000 Hz
        self.styles = [f'风格 {i}' for i in range(6)]

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
