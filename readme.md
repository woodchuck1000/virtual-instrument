# 爱乐智呈--多模态虚拟乐器演奏平台
主页链接：https://www.hitaiyuezhicheng.com/

主程序Github链接：https://github.com/woodchuck1000/virtual-instrument


## 简介
爱乐智呈多模态虚拟乐器演奏平台基于手势识别，乐谱音符识别，人机交互等多种模态进行设计，目前能够演奏多种打击乐器和吹奏乐器

打击乐器：编钟

![GitHub Logo](编钟GUI背景.png)

吹奏乐器：陶笛，葫芦丝，埙

![GitHub Logo](陶笛GUI背景.png)

![GitHub Logo](葫芦丝GUI背景.png)

![GitHub Logo](埙GUI背景.png)


前端平台界面：

![GitHub Logo](代码/主页.png)


## 使用教程
- 1、 先访问主页链接，获取演奏教程、演奏视频示范以及代码下载链接


- 2、 点击Github链接下载代码ZIP，阅读readme.md文件


- 3、 下载依赖库，打开终端输入：

```
$ pip install -r requirements.txt
```

- 4、 打击乐演奏：
    
```
    - 1）准备工作：在一张A4纸上写下你所选曲目中要演奏的音符
        （低音do-si, 中音do-si, 高音do-la）
         建议用较粗的马克笔书写音符，这样能够提高识别准确率哦！

    - 2）运行打击乐GUI_2.py文件，点击演奏编钟按钮

    - 3) 摄像头打开后，将A4纸对准摄像头，等待音符识别结果，出现编钟图标代表成功

    - 4）识别成功后，用手指或者笔点击对应音符即可发声，注意你演奏的节奏哦！
```

- 5、 吹奏乐演奏：
    
```
    - 1） 运行吹奏乐GUI_2.py文件
   
    - 2） 在GUI界面点击对应按钮选择你想演奏的乐器种类

    - 3） 将摄像头对准你的双手，可以用手机等物品固定你的手型方便演奏。按照画面左下的教程，控制你手指的屈伸，然后对准麦克风吹气即可发声
          

    - 4） 每演奏正确一个音后，教程跳至下一音符，当教程消失后，你就可以创作自己的乐曲啦！
          注意查看画面上方的音量曲线和音高线来调整你的吹奏音准哦!
```

## 文件附录
- 打击乐GUI_2.py , 吹奏乐GUI_2.py : 进行演奏的主GUI程序


- DajiYue_2.py , ChuizouYue.py : 演奏程序2.0版本，为两个主GUI的主调用程序


- 打击乐.py , 吹奏乐.py : 演奏程序1.0历史版本


- CNN.py , findA4.py : 打击乐的依赖文件，分别为音符识别模块和图像识别模块


- shipu.py , result.py: 可以对导入的乐谱文件进行识谱，识别结果保存在result.txt中，用于生成吹奏乐的教程
```
  如何修改乐谱识别结果：在shipu.py的第7行
  image = cv2.imread('lzlh.png')
  路径改为你导入的xx乐谱.png
```






