import tkinter as tk
from tkinter import filedialog, simpledialog
import cv2
from run_webcam import infer_video
from run_image import infer_image


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

button1 = tk.Button(root, text="Real-Time Inference", command=real_time_inference)
button1.pack(side="top", fill="both", expand=True)

button2 = tk.Button(root, text="Image Inference", command=image_inference)
button2.pack(side="top", fill="both", expand=True)

button3 = tk.Button(root, text="Video Inference", command=video_inference)
button3.pack(side="top", fill="both", expand=True)

root.mainloop()
