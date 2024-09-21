from tkinter import Button, Tk, Label
from tkinter import messagebox
from PIL import Image, ImageTk
from DajiYue_2 import DajiYue


class DajiYueGUI:
    def __init__(self, master):
        self.master = master
        master.title("打击乐演奏")

        # 加载背景图片
        self.bg_image = ImageTk.PhotoImage(Image.open("编钟GUI背景.png"))
        self.bg_label = Label(master, image=self.bg_image)
        self.bg_label.place(relwidth=1, relheight=1)

        self.daji_yue = DajiYue()  # 创建DajiYue类的实例

        # 创建按钮，点击时调用self.start_play方法
        self.play_button = Button(master, text="演奏编钟", command=self.start_play,
                                  font=("宋体", 14),  # 设置字体样式
                                  width=10,  # 设置按钮宽度
                                  height=2)  # 设置按钮高度
        self.play_button.place(relx=0.5, rely=0.5, anchor='center')  # 将按钮置于窗口中央

    def start_play(self):
        try:
            self.daji_yue.start_play()
        except Exception as e:
            messagebox.showerror("错误", str(e))


if __name__ == "__main__":
    root = Tk()
    root.geometry("400x170")
    app = DajiYueGUI(root)
    root.mainloop()