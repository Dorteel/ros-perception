import rospy
import cv2
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import numpy as np


# YOLO imports
import os
import sys
from pathlib import Path

import torch
#import torch.backends.cudnn as cudnn

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from yolov5.models.common import DetectMultiBackend
from datasets import LoadRosImages
from general import (LOGGER, check_img_size, non_max_suppression, scale_coords)
from plots import Annotator, colors
from torch_utils import select_device, time_sync

class cvBridgeDemo():

    def __init__(self, topic="/locobot/camera/color/image_raw"):
        self.node_name = "cv_bridge_demo"
        rospy.init_node(self.node_name)
        rospy.on_shutdown(self.cleanup) # What we do during shutdown
        self.topic = topic
        self.bridge = CvBridge()        # Create the cv_bridge object

        self.model, self.imgsz, self.device = self.prepare_model()  # Load the model
        self.image_sub = rospy.Subscriber(self.topic, Image, self.image_callback)

        rospy.loginfo("Waiting for image topics...")

    def image_callback(self, ros_image):
        try:    # Use cv_bridge() to convert the ROS image to OpenCV format
            frame = self.bridge.imgmsg_to_cv2(ros_image, "bgr8")
        except CvBridgeError:
            print(CvBridgeError)
            self.cleanup()

        # Convert the image to a Numpy array since most cv2 functions require Numpy arrays.
        #run(source=frame)
        frame = np.array(frame, dtype=np.uint8)
        self.run(self.model, self.imgsz, frame, device=self.device)
        cv2.imshow(self.node_name, frame)   # Display the image.
        self.keystroke = cv2.waitKey(5) # Process any keyboard commands
        if 32 <= self.keystroke and self.keystroke < 128:
            cc = chr(self.keystroke).lower()
            if cc == 'q':
                rospy.signal_shutdown("User hit q key to quit.") # The user has press the q key, so exit

    def cleanup(self):
        print("Shutting down vision node.")
        cv2.destroyAllWindows()  


    #@torch.no_grad()
    def prepare_model(self,
            weights=ROOT / 'yolov5s.pt',  # model.pt path(s)
            data=ROOT / 'yolov5/data/coco128.yaml',  # dataset.yaml path
            imgsz=(480, 640),  # inference size (height, width)
            device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        ):
        # Load model
        device = select_device(device)
        model = DetectMultiBackend(weights, device=device, data=data)
        imgsz = check_img_size(imgsz, s=model.stride)  # check image size
        
        # Return the model and the image sizes
        return model, imgsz, device


    @torch.no_grad()
    def run(self,
            model,
            imgsz,
            image,
            device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu,
            conf_thres=0.25,  # confidence threshold
            iou_thres=0.45,  # NMS IOU threshold
            max_det=1000,  # maximum detections per image
            agnostic_nms=False,  # class-agnostic NMS
            classes=None,
    ):
        stride, names, pt = model.stride, model.names, model.pt
        dataset = LoadRosImages(image, img_size=imgsz, stride=stride)
        bs = 1  # batch_size

        # Run inference
        model.warmup(imgsz=(1 if pt else bs, 3, *imgsz))  # warmup
        dt, seen = [0.0, 0.0, 0.0], 0
        for im, im0s in dataset:
            t1 = time_sync()
            im = torch.from_numpy(im).to(device)
            im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
            im /= 255  # 0 - 255 to 0.0 - 1.0
            if len(im.shape) == 3:
                im = im[None]  # expand for batch dim
            t2 = time_sync()
            dt[0] += t2 - t1

            # Inference
            pred = model(im)
            t3 = time_sync()
            dt[1] += t3 - t2

            # NMS
            pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
            dt[2] += time_sync() - t3

            # Process predictions
            for _, det in enumerate(pred):  # per image
                seen += 1
                im0 = im0s.copy()
                annotator = Annotator(im0, example=str(names))
                if len(det):
                    # Rescale boxes from img_size to im0 size
                    det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()

                # Stream results
                im0 = annotator.result()


def main(args):
    #run(source=0)
    # depthTopic = "/locobot/camera/depth/image_rect_raw"
    colorTopic = "/locobot/camera/color/image_raw"       
    try:
        node = cvBridgeDemo(topic=colorTopic)
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down vision node.")
        cv2.DestroyAllWindows()

if __name__ == '__main__':
    main(sys.argv)


# class RosNode():
#     # Must have __init__(self) function for a class, similar to a C++ class constructor.
#     def __init__(self,
#         name="test_node",
#         sub_topic="listener",
#         pub_topic="talker",
#         sub_msg_type=Image,
#         pub_msg_type=Image
#         ):

#         # Initialize node
#         self.node_name = name
#         rospy.init_node(self.node_name)

#         # Node cycle rate (in Hz).
#         self.loop_rate = rospy.Rate(100)

#         # Publishers
#         self.publish_topic = pub_topic
#         self.msg_to_publish = 0
#         self.pub = rospy.Publisher(self.publish_topic, pub_msg_type, queue_size=100)

#         # Subscribers
#         self.subscribe_topic = sub_topic
#         rospy.Subscriber(self.subscribe_topic, sub_msg_type, self.callback)
#         self.msg = None



#     def callback(self, msg):
#         self.msg = msg
#         rospy.loginfo("Received: {}".format(self.msg.data))


#     def start(self):
#         while not rospy.is_shutdown():
#             # Publish our custom message.
#             if self.msg:
#                 #print(self.msg)
#                 self.msg_to_publish = self.msg
#                 self.pub.publish(self.msg_to_publish)
#             # rospy.loginfo("\tPublished: {}".format(self.msg_to_publish))
#             # Sleep for a while before publishing new messages. Division is so rate != period.
#             # if self.loop_rate:
#             #     rospy.sleep(0.001)
#             # else:
#             #     rospy.sleep(1.0)
#             self.loop_rate.sleep()