import tkinter as tk
from tkinter import filedialog
import cv2
from run_webcam import infer_video


def real_time_inference():
    # Implement your real-time inference here
    pass

def image_inference():
    filename = filedialog.askopenfilename()  # ask the user to choose a file
    if filename:
        img = cv2.imread(filename)
        # Implement your image inference here
        pass

def video_inference():
    filename = filedialog.askopenfilename()  # ask the user to choose a file
    if filename:
        infer_video(filename,resize_out_ratio=1.0, is_resize=False)

root = tk.Tk()

button1 = tk.Button(root, text="Real-Time Inference", command=real_time_inference)
button1.pack(side="top", fill="both", expand=True)

button2 = tk.Button(root, text="Image Inference", command=image_inference)
button2.pack(side="top", fill="both", expand=True)

button3 = tk.Button(root, text="Video Inference", command=video_inference)
button3.pack(side="top", fill="both", expand=True)

root.mainloop()
