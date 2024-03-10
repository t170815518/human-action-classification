import tkinter as tk
from tkinter import filedialog, simpledialog, Label
import cv2
from run_webcam import infer_video
from run_image import infer_image
from PIL import ImageTk, Image


def real_time_inference():
    video_id = simpledialog.askstring("Input", "输入摄像头ID（比如\"0\"）")
    if video_id.isdigit():
        video_id = int(video_id)  # convert to int if it's a number
    infer_video(video_id, resize_out_ratio=4.0, tf_pose_estimator=None, resize_to_default=True)


def image_inference():
    filename = filedialog.askopenfilename()  # ask the user to choose a file
    if filename:
        infer_image(filename, tf_pose_estimator=None)

def video_inference():
    filename = filedialog.askopenfilename()  # ask the user to choose a file
    if filename:
        infer_video(filename, resize_out_ratio=4.0, tf_pose_estimator=None, resize_to_default=True)

root = tk.Tk()
root.title('姿势检测系统')
# root.geometry('500x555')

# Load the icons for the buttons
image_open = Image.open("assets/camera.jpg").resize([25, 25])
real_time_icon = ImageTk.PhotoImage(image_open)
image = Image.open("assets/image.png").resize([25, 25])
image_icon = ImageTk.PhotoImage(image)
image = Image.open("assets/movie.png").resize([25, 25])
video_icon = ImageTk.PhotoImage(image)


title_label = Label(root, text='姿势检测系统')
title_label.place(relx=0.5, rely=0.1, anchor='center')

button1 = tk.Button(root, text="摄像头检测", command=real_time_inference, image=real_time_icon, compound="left")
button1.place(relx=0.5, rely=0.3, anchor='center')

button2 = tk.Button(root, text="图片检测", command=image_inference, image=image_icon, compound="left")
button2.place(relx=0.5, rely=0.5, anchor='center')

button3 = tk.Button(root, text="视频检测", command=video_inference, image=video_icon, compound="left")
button3.place(relx=0.5, rely=0.7, anchor='center')

root.mainloop()
