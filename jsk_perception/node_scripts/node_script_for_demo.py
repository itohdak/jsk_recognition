#!/usr/bin/env python
# -*- coding:utf-8 -*-

# python -m demo --img_path data/random.jpg --json_path data/random_keypoints.json
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
from absl import flags
import numpy as np

import skimage.io as io
import tensorflow as tf
import cv2

# dependings
from src.util import renderer as vis_util
from src.util import image as img_util
from src.util import openpose as op_util
import src.config
from src.RunModel import RunModel

import cv_bridge
import rospy
import message_filters
from jsk_topic_tools import ConnectionBasedTransport
from jsk_recognition_msgs.msg import PeoplePoseArray
from sensor_msgs.msg import Image

flags.DEFINE_string('img_path', 'data/im1963.jpg', 'Image to run')
flags.DEFINE_string(
    'json_path', None,
    'If specified, uses the openpose output to crop the image.')

class PeopleMeshDetector(ConnectionBasedTransport):
    def __init__(self):
        super(self.__class__, self).__init__()

        self.sess = tf.Session()
        self.model = RunModel(config, sess=self.sess)

        self.with_pose = rospy.get_param('~with_pose', False)
        self.publisher = self.advertise("~output", Image, queue_size=1)

    def subscribe(self):
        queue_size = rospy.get_param('~queue_size', 10)
        sub_img = message_filters.Subscriber(
            '~input', Image, queue_size=1, buff_size=2**24)
        self.subs = [sub_img]

        if self.with_pose:
            sub_pose = message_filters.Subscriber(
                '~input_pose', PeoplePoseArray, queue_size=1, buff_size=2**24)
            self.subs.append(sub_pose)

        if rospy.get_param('~approximate_sync', False):
            slop = rospy.get_param('~slop', 0.1)
            sync = message_filters.ApproximateTimeSynchronizer(
                fs=self.subs, queue_size=queue_size, slop=slop)
        else:
            sync = message_filters.TimeSynchronizer(
                fs=self.subs, queue_size=queue_size)
        if self.with_pose:
            sync.registerCallback(self._cb_with_pose)
        else:
            sync.registerCallback(self._cb)

    def unsubscribe(self):
        for sub in self.subs:
            sub.unregister()

    def _cb(self, img_msg):
        br = cv_bridge.CvBridge()
        img = br.imgmsg_to_cv2(img_msg, desired_encoding='bgr8')
        json_path = None
        input_img, proc_param, img = self._preprocess_image(
            None, img, json_path, None)
        input_img = np.expand_dims(input_img, 0)
        joints, verts, cams, joints3d, theta = self.model.predict(
            input_img, get_theta=True)

        ret_img = self._visualize(img, proc_param, joints[0], verts[0], cams[0])

        pub_img = br.cv2_to_imgmsg(ret_img, encoding='8UC4')
        pub_img.header = img_msg.header

        self.publisher.publish(pub_img)

    def _cb_with_pose(self, img_msg, pose):
        rospy.loginfo("cb_witrh_pose")
        br = cv_bridge.CvBridge()
        img = br.imgmsg_to_cv2(img_msg, desired_encoding='bgr8')
        json_path = None
        input_img, proc_param, img = self._preprocess_image(
            None, img, json_path, pose)
        input_img = np.expand_dims(input_img, 0)
        joints, verts, cams, joints3d, theta = self.model.predict(
            input_img, get_theta=True)
        rospy.loginfo("joints3d = {}".format(1000 * joints3d))

        # print(len(joints))
        # print(joints)
        # print(joints3d)
        ret_img = self._visualize(img, proc_param, joints[0], verts[0], cams[0])

        pub_img = br.cv2_to_imgmsg(ret_img, encoding='8UC4')
        pub_img.header = img_msg.header

        self.publisher.publish(pub_img)

    def _visualize(self, img, proc_param, joints, verts, cam):
        """
        Renders the result in original image coordinate frame.
        """
        cam_for_render, vert_shifted, joints_orig = vis_util.get_original(
            proc_param, verts, cam, joints, img_size=img.shape[:2])

        # Render results
        skel_img = vis_util.draw_skeleton(img, joints_orig)
        rend_img_overlay = renderer(
            vert_shifted, cam=cam_for_render, img=img, do_alpha=True)
        rend_img = renderer(
            vert_shifted, cam=cam_for_render, img_size=img.shape[:2])
        rend_img_vp1 = renderer.rotated(
            vert_shifted, 60, cam=cam_for_render, img_size=img.shape[:2])
        rend_img_vp2 = renderer.rotated(
            vert_shifted, -60, cam=cam_for_render, img_size=img.shape[:2])
        return rend_img_overlay

        import matplotlib.pyplot as plt
        # plt.ion()
        plt.figure(1)
        plt.clf()
        plt.subplot(231)
        plt.imshow(img)
        plt.title('input')
        plt.axis('off')
        plt.subplot(232)
        plt.imshow(skel_img)
        plt.title('joint projection')
        plt.axis('off')
        plt.subplot(233)
        plt.imshow(rend_img_overlay)
        plt.title('3D Mesh overlay')
        plt.axis('off')
        plt.subplot(234)
        plt.imshow(rend_img)
        plt.title('3D mesh')
        plt.axis('off')
        plt.subplot(235)
        plt.imshow(rend_img_vp1)
        plt.title('diff vp')
        plt.axis('off')
        plt.subplot(236)
        plt.imshow(rend_img_vp2)
        plt.title('diff vp')
        plt.axis('off')
        plt.draw()
        plt.show()
        # import ipdb
        # ipdb.set_trace()


    def _preprocess_image(self, img_path=None, img=None, json_path=None, with_pose=None):
        if img_path:
            img = io.imread(img_path)

        if with_pose:
            scale, center = self._get_bbox(with_pose)
            if scale is False:
                scale = 1.
                center = np.round(np.array(img.shape[:2]) / 2).astype(int)
                # image center in (x,y)
                center = center[::-1]
        elif json_path is None:
            scale = 1.
            center = np.round(np.array(img.shape[:2]) / 2).astype(int)
            # image center in (x,y)
            center = center[::-1]
        else:
            scale, center = op_util.get_bbox(json_path)

        crop, proc_param = img_util.scale_and_crop(img, scale, center,
                                                   config.img_size)

        # Normalize image to [-1, 1]
        crop = 2 * ((crop / 255.) - 0.5)

        return crop, proc_param, img


    def _get_bbox(self, people_pose, vis_thr=0.2):
        kps = []
        for pose in people_pose.poses:
            kp = []
            for position, score in zip(pose.poses, pose.scores):
                kp.append([position.position.x, position.position.y, score])
            kps.append(np.array(kp).reshape(-1, 3))

        scores = [np.mean(kp[kp[:, 2] > vis_thr, 2]) for kp in kps]
        if len(scores) == 0:
            return False, False
        kp = kps[np.argmax(scores)]
        vis = kp[:, 2] > vis_thr
        vis_kp = kp[vis, :2]
        if len(vis_kp) == 0:
            return False, False
        min_pt = np.min(vis_kp, axis=0)
        max_pt = np.max(vis_kp, axis=0)
        person_height = np.linalg.norm(max_pt - min_pt)
        if person_height == 0:
            print('bad!')
            import ipdb
            ipdb.set_trace()
        center = (min_pt + max_pt) / 2.
        scale = 150. / person_height

        return scale, center

if __name__ == '__main__':
    config = flags.FLAGS
    config(sys.argv)
    # Using pre-trained model, change this to use your own.
    config.load_path = src.config.PRETRAINED_MODEL

    config.batch_size = 1

    renderer = vis_util.SMPLRenderer(face_path=config.smpl_face_path)

    rospy.init_node('people_mesh_detector')
    PeopleMeshDetector()
    rospy.spin()
