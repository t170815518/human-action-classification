"""
对视频/摄像头每一帧进行姿势识别
"""


import argparse
import logging
import time

import cv2
import numpy as np

from tf_pose.estimator import TfPoseEstimator
from tf_pose.networks import get_graph_path, model_wh
import scripts.label_image as label_img


# 设置logger
logger = logging.getLogger('TfPoseEstimator-WebCam')
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)

fps_time = 0

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='tf-pose-estimation realtime webcam')
    parser.add_argument('--camera', type=str)   # 如果是摄像头，则输入摄像头的ID，一般是0
    parser.add_argument('--resize', type=str, default='0x0',
                        help='if provided, resize images before they are processed. default=0x0, Recommends : 432x368 or 656x368 or 1312x736 ')
    parser.add_argument('--resize-out-ratio', type=float, default=4.0,
                        help='if provided, resize heatmaps before they are post-processed. default=1.0')

    parser.add_argument('--model', type=str, default='mobilenet_thin', help='cmu / mobilenet_thin')
    parser.add_argument('--show-process', type=bool, default=False,
                        help='for debug purpose, if enabled, speed for inference is dropped.')
    args = parser.parse_args()

    logger.debug('initialization %s : %s' % (args.model, get_graph_path(args.model)))
    w, h = model_wh(args.resize)    # 按照预设缩放图片

    if w > 0 and h > 0: # 加载模型
        e = TfPoseEstimator(get_graph_path(args.model), target_size=(w, h))
    else:
        e = TfPoseEstimator(get_graph_path(args.model), target_size=(432, 368))

    logger.debug('cam read+')
    if args.camera.isdigit():
        args.camera = int(args.camera)
    cam = cv2.VideoCapture(args.camera) # 加载视频/摄像头
    ret_val, image = cam.read()
    logger.info('cam image=%dx%d' % (image.shape[1], image.shape[0]))

    # count = 0
    while True: # 开始读取后面的帧
        
        logger.debug('+image processing+')
        ret_val, image = cam.read()
        
        logger.debug('+postprocessing+')
        # 检测人类和人类骨架的关键点
        humans = e.inference(image, resize_to_default=(w > 0 and h > 0), upsample_size=args.resize_out_ratio)
        img = TfPoseEstimator.draw_humans(image, humans, imgcopy=False)
        
        logger.debug('+classification+')
        # Getting only the skeletal structure (with white background) of the actual image
        image = np.zeros(image.shape,dtype=np.uint8)
        image.fill(255) 
        image = TfPoseEstimator.draw_humans(image, humans, imgcopy=False)
        
        # Classification，对姿势进行分类
        pose_class = label_img.classify(image)
        
        logger.debug('+displaying+')
        cv2.putText(img,
                    "Current predicted pose is : %s" %(pose_class),
                    (10, 10),  cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 255, 0), 2)
        
        cv2.imshow('tf-pose-estimation result', img)
        
        fps_time = time.time()
        if cv2.waitKey(1) == 27:
            break
        logger.debug('+finished+')
        
        # For gathering training data 
        # title = 'img'+str(count)+'.jpeg'
        # path = <enter any path you want>
        # cv2.imwrite(os.path.join(path , title), image)
        # count += 1

    cv2.destroyAllWindows()

# =============================================================================
# For running the script simply run the following in the cmd prompt/terminal :
# python run_webcam.py --model=mobilenet_thin --resize=432x368 --camera=0
# =============================================================================
