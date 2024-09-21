from tkinter import Button, Tk, Canvas
from tkinter import ttk
from PIL import Image, ImageTk
from ChuizouYue_2 import ChuizouYue


def load_image(path):
    image = Image.open(path)
    # 调整图片大小以适应窗口的三分之一宽度和一半高度
    base_width = 1024 // 3
    base_height = 768 // 2
    w_percent = (base_width / float(image.size[0]))
    h_size = int((float(image.size[1]) * float(w_percent)) // 2)
    image = image.resize((base_width, base_height), Image.ANTIALIAS)
    return ImageTk.PhotoImage(image)


root = Tk()
root.title("吹奏乐演奏")
root.geometry("1024x768")  # 设置窗口大小

# 创建一个Canvas作为背景
canvas = Canvas(root, width=1024, height=768)
canvas.pack(fill="both", expand=True)

# 假设你已经有了三种乐器的图片路径
taodi_image_path = '陶笛GUI背景.png'
hulusi_image_path = '葫芦丝GUI背景.png'
xun_image_path = '埙GUI背景.png'

# 加载并调整图片大小
taodi_image = load_image(taodi_image_path)
hulusi_image = load_image(hulusi_image_path)
xun_image = load_image(xun_image_path)

# 将图片作为背景
canvas.create_image(0, 0, image=taodi_image, anchor="nw")
canvas.create_image(1024 // 3, 0, image=hulusi_image, anchor="nw")
canvas.create_image(2 * 1024 // 3, 0, image=xun_image, anchor="nw")


def start_playing(instrument_name):
    instrument = ChuizouYue(instrument_name)
    instrument.start_play()


btn_taodi = Button(root, text="演奏陶笛", command=lambda: start_playing('taodi'))
btn_taodi.place(relx=0.1, rely=0.75, anchor="center")

btn_hulusi = Button(root, text="演奏葫芦丝", command=lambda: start_playing('hulusi'))
btn_hulusi.place(relx=0.5, rely=0.75, anchor="center")

btn_xun = Button(root, text="演奏埙", command=lambda: start_playing('xun'))
btn_xun.place(relx=0.9, rely=0.75, anchor="center")

# 设置按钮样式
style = ttk.Style()
style.configure("TButton", background="#0000FF", font=("宋体", 18), padding=20)

# 运行Tkinter事件循环
root.mainloop()
