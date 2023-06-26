# Copyright 2023 Intel Corporation. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from sensor_msgs.msg import Image as msg_Image
from sensor_msgs.msg import CameraInfo
from sensor_msgs.msg import Imu as msg_Imu
from cv_bridge import CvBridge, CvBridgeError
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import pyrealsense2 as rs2
if (not hasattr(rs2, 'intrinsics')):
    import pyrealsense2.pyrealsense2 as rs2
import cv2 

from detectron2.config import get_cfg
import torch
import torchvision.transforms as transforms

from PPO.agent_nav_crowd import Agent 
from PPO.buffers_nav_crowd import Buffer 
from PIL import Image 
from construct_map import Map 

from detectron_modules import DetectronModule
from compute_pose import calcOrientation, calcPosition

INPUT_SIZE = 150
INPUT_CHANNELS = 3
ACTION_LIST = ['w', 's', 'a', 'd']
ACTION_NAME = ['move forward', 'move backward', 'turn left', 'turn right']
ACTION_SIZE = len(ACTION_LIST)
device = 'cuda:0'
WIDTH = 848
HEIGHT = 480

class Pose():
    def __init__(self):
        self.pos = np.zeros((3,))
        self.orien = np.eye(3)
        self.euler = np.zeros((3,))

class ImageListener(Node):
    def __init__(self, color_image_topic, depth_image_topic, depth_info_topic, imu_topic):
        node_name = os.path.basename(sys.argv[0]).split('.')[0]
        super().__init__(node_name)

        self.publisher_ = self.create_publisher(String, 'topic', 10)
        self.bridge = CvBridge()

        self.sub = self.create_subscription(msg_Image, color_image_topic, self.imageColorCallback, 10)
        self.sub2 = self.create_subscription(msg_Image, depth_image_topic, self.imageDepthCallback, 10)
        self.sub_info = self.create_subscription(CameraInfo, depth_info_topic, self.imageDepthInfoCallback, 10)
        confidence_topic = depth_image_topic.replace('depth', 'confidence')
        self.sub_conf = self.create_subscription(msg_Image, confidence_topic, self.confidenceCallback, 1)
        self.sub_imu = self.create_subscription(msg_Imu, imu_topic, self.imuCallback, 10)
        self.intrinsics = None
        self.pix = None
        self.pix_grade = None

        cfg = get_cfg()
        cfg.merge_from_file("models/mask_rcnn_R_50_FPN_3x.yaml")
        cfg.merge_from_list(['MODEL.WEIGHTS', 'models/model_final_f10217.pkl'])
        cfg.MODEL.RETINANET.SCORE_THRESH_TEST = 0.5
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
        cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = 0.5
        cfg.device = "cuda:0"
        cfg.freeze()
        self.det_module = DetectronModule(cfg)

        
        self.pose = Pose() # To do: add the pose subscription

        self.goal = [] # To do: add the goal subscription

        obs_dim = (INPUT_CHANNELS, INPUT_SIZE, INPUT_SIZE)
        map_size = INPUT_SIZE
        lr = 3e-4
        num_epoch_step = 300
        num_env = 1
        self.agent = Agent(obs_dim, map_size, ACTION_SIZE, lr, device, is_train=False)
        
        buffer = Buffer(obs_dim, map_size, ACTION_SIZE, num_epoch_step, num_env, device=device)

        self.transform = transforms.Compose([transforms.Resize((INPUT_SIZE, INPUT_SIZE)), transforms.ToTensor(),
                                        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
        self.transform_map = transforms.Compose([transforms.Resize((INPUT_SIZE, INPUT_SIZE)), transforms.ToTensor()])
    
        state_dicts = torch.load('models/ppo_783.pt')
        self.agent.load_param(state_dicts)

        self.map_module = Map(WIDTH, HEIGHT)
        self.map_module.reset_map()
        self.map_module.reset_human_map()

        self.frames = []

        self.depth = np.zeros((HEIGHT, WIDTH))

        self.first_flag = True

    def imuCallback(self, data):
        frame_id = data.header.frame_id
        if self.first_flag:
            self.prev_time = data.header.stamp
            self.first_flag = False 
            self.gravity = np.array([data.linear_acceleration.x, data.linear_acceleration.y, data.linear_acceleration.z])
            self.vel = 0
        
        deltaT = (data.header.stamp.sec - self.prev_time.sec) + (data.header.stamp.nanosec - self.prev_time.nanosec) * 1e-9
        self.prev_time = data.header.stamp

        self.pose = calcOrientation(self.pose, data.angular_velocity, deltaT)
        self.pose, self.vel = calcPosition(self.pose, data.linear_acceleration, deltaT, self.vel, self.gravity)

        print(self.pose.euler, self.pose.pos, self.gravity)
               

    def imageColorCallback(self, data):
        try:
            cv_image = cv2.resize(self.bridge.imgmsg_to_cv2(data, data.encoding), dsize=(WIDTH, HEIGHT))
            #cv_image = self.bridge.imgmsg_to_cv2(data, data.encoding)

          
            predictions, vis_output = self.det_module.run_on_image(image=cv_image)

            

            self.frames.append(cv_image)

            #vis_output.save('output/detectron2/detect_instances_%05d.jpg' % (len(self.frames)))

            self.depth = self.depth.astype(np.float32) / 65536.

            idx = torch.where(predictions["instances"].pred_classes == 0)
            if not idx:
                return
            mask = torch.sum(predictions["instances"].pred_masks[idx].float(), dim=0).cpu().data.numpy()
            depth_human1 = self.depth * mask  # To do: check the time stamp
            mask1 = depth_human1 == 0
            depth_human1[mask1] = np.max(np.max(self.depth))


            pos_ori = [self.pose.pos[0], self.pose.pos[1], self.pose.euler[1]]
            agent_view_cropped, map_gt, agent_view_explored, explored_gt, depth, pc1 = \
                self.map_module.update_map(cv_image, self.depth, vis_output, pos_ori, flag=1)
            _, hum_map_gt, _, hum_explored_gt, hum_depth, pc2 = \
                self.map_module.update_map(cv_image, depth_human1, vis_output, pos_ori, flag=0)
            #print(np.max(np.max(depth)), np.min(np.min(depth)))

            
            plt.cla()
            plt.subplot(1, 2, 1)
            #plt.imshow(pc1)
            plt.imshow(np.array(depth).astype('uint8'))
            #plt.imshow(vis_output.get_image().astype('uint8'))
            plt.subplot(1, 2, 2)
            plt.imshow(vis_output.get_image().astype('uint8'))
            plt.pause(0.1)
            
            
            # plt.pause(0.1)            

            #del vis_output, predictions, mask

            cv_image = self.transform(Image.fromarray(cv_image).unsqueeze(0).to(device))

            action = self.agent(cv_image, self.transform_map(Image.fromarray(map_gt)).to(device).unsqueeze(0),
                                self.transform_map(Image.fromarray(hum_map_gt)).to(device).unsqueeze(0), torch.FloatTensor(self.goal).to(device))


            msg = String()
            msg.data = "Hello World! my action is: %d" % action
            self.publisher_.publish(msg)

            line = "hello, i publish the action as %d" % action 
            line += ' %d %d '%(cv_image.shape[0], cv_image.shape[1]) +  ' %s '% cv_image.dtype + '  *' + '\r'
            sys.stdout.write(line)
            sys.stdout.flush()

        except CvBridgeError as e:
            print(e)
            return
        except ValueError as e:
            return

    def imageDepthCallback(self, data):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(data, data.encoding)
            # pick one pixel among all the pixels with the closest range:
            
            self.depth = cv_image
            #print('depth: ', self.depth.dtype)

            # plt.cla()
            # plt.subplot(1, 2, 2)
            # plt.imshow(cv_image)
            # plt.pause(0.1)
            indices = np.array(np.where(cv_image == cv_image[cv_image > 0].min()))[:,0]
            pix = (indices[1], indices[0])
            self.pix = pix
            line = '\rDepth at pixel(%3d, %3d): %7.1f(mm).' % (pix[0], pix[1], cv_image[pix[1], pix[0]])

            if self.intrinsics:
                depth = cv_image[pix[1], pix[0]]
                result = rs2.rs2_deproject_pixel_to_point(self.intrinsics, [pix[0], pix[1]], depth)
                line += '  Coordinate: %8.2f %8.2f %8.2f.' % (result[0], result[1], result[2])
            if (not self.pix_grade is None):
                line += ' Grade: %2d' % self.pix_grade

        except CvBridgeError as e:
            print(e)
            return
        except ValueError as e:
            return

    def confidenceCallback(self, data):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(data, data.encoding)
            grades = np.bitwise_and(cv_image >> 4, 0x0f)
            if (self.pix):
                self.pix_grade = grades[self.pix[1], self.pix[0]]
        except CvBridgeError as e:
            print(e)
            return



    def imageDepthInfoCallback(self, cameraInfo):
        try:
            if self.intrinsics:
                return
            self.intrinsics = rs2.intrinsics()
            self.intrinsics.width = cameraInfo.width
            self.intrinsics.height = cameraInfo.height
            self.intrinsics.ppx = cameraInfo.k[2]
            self.intrinsics.ppy = cameraInfo.k[5]
            self.intrinsics.fx = cameraInfo.k[0]
            self.intrinsics.fy = cameraInfo.k[4]
            if cameraInfo.distortion_model == 'plumb_bob':
                self.intrinsics.model = rs2.distortion.brown_conrady
            elif cameraInfo.distortion_model == 'equidistant':
                self.intrinsics.model = rs2.distortion.kannala_brandt4
            self.intrinsics.coeffs = [i for i in cameraInfo.d]
        except CvBridgeError as e:
            print(e)
            return

def main():
    color_image_topic = '/camera/color/image_raw'
    depth_image_topic = '/camera/depth/image_rect_raw'
    depth_info_topic = '/camera/depth/camera_info'
    imu_topic = '/camera/imu'

    print ()
    print ('show_center_depth.py')
    print ('--------------------')
    print ('App to demontrate the usage of the /camera/depth topics.')
    print ()
    print ('Application subscribes to %s and %s topics.' % (depth_image_topic, depth_info_topic))
    print ('Application then calculates and print the range to the closest object.')
    print ('If intrinsics data is available, it also prints the 3D location of the object')
    print ('If a confedence map is also available in the topic %s, it also prints the confidence grade.' % depth_image_topic.replace('depth', 'confidence'))
    print ()
    
    listener = ImageListener(color_image_topic, depth_image_topic, depth_info_topic, imu_topic)
    rclpy.spin(listener)
    listener.destroy_node()
    rclpy.shutdown()    

if __name__ == '__main__':
    rclpy.init(args=sys.argv)
    main()
