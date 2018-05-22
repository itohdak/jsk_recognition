#!/usr/bin/env python
# -*- coding:utf-8 -*-

import cv2
import numpy as np

import cv_bridge
import message_filters
import rospy
from jsk_topic_tools import ConnectionBasedTransport
from jsk_recognition_msgs.msg import PeoplePoseArray
from sensor_msgs.msg import Image
from std_msgs.msg import String


class PeopleMaskPublisher(ConnectionBasedTransport):

    limb_names = ["Nose",
                  "Neck",
                  "RShoulder",
                  "RElbow",
                  "RWrist",
                  "LShoulder",
                  "LElbow",
                  "LWrist",
                  "RHip",
                  "RKnee",
                  "RAnkle",
                  "LHip",
                  "LKnee",
                  "LAnkle",
                  "REye",
                  "LEye",
                  "REar",
                  "LEar",
                  'RHand',
                  'LHand']

    def __init__(self):
        super(self.__class__, self).__init__()
        self.person_indices = rospy.get_param('~person_indices', -1)
        self.limb_part = rospy.get_param('~limb_part', 'all')
        if self.limb_part == 'all':
            self.limb_part = self.limb_names
        self.arms_score_threshold = rospy.get_param(
            '~arms_score_threshold', 0.25)
        self.hand_ratio = rospy.get_param('~hand_ratio', 0.33)
        self.hand_width_ratio = rospy.get_param('~hand_width_ratio', 0.8)

        self.pub = self.advertise('~output', Image, queue_size=1)
        self.debug_pub = self.advertise('~debug/output', Image, queue_size=1)

        self.pub_direction = self.advertise('~output/direction', String, queue_size=1)
        self.prev_vector = None
        self.threshold_direction = rospy.get_param('~threshold_direction', 0.3)
        self.inertia_rate = 0.2

    def subscribe(self):
        queue_size = rospy.get_param('~queue_size', 10)
        sub_img = message_filters.Subscriber(
            '~input', Image, queue_size=1, buff_size=2**24)
        sub_pose = message_filters.Subscriber(
            '~input/pose', PeoplePoseArray, queue_size=1, buff_size=2**24)
        self.subs = [sub_img, sub_pose]
        if rospy.get_param('~approximate_sync', False):
            slop = rospy.get_param('~slop', 0.1)
            sync = message_filters.ApproximateTimeSynchronizer(
                fs=self.subs, queue_size=queue_size, slop=slop)
        else:
            sync = message_filters.TimeSynchronizer(
                fs=self.subs, queue_size=queue_size)
        sync.registerCallback(self._cb)

    def unsubscribe(self):
        for sub in self.subs:
            sub.unregister()

    def _cb(self, img_msg, people_pose_array_msg):
        br = cv_bridge.CvBridge()
        img = br.imgmsg_to_cv2(img_msg, desired_encoding='bgr8')

        if self.person_indices != -1:
            indices = np.array(self.person_indices)
            indices = indices[indices < len(people_pose_array_msg.poses)]
            people_pose = np.array(people_pose_array_msg.poses)[indices]
        else:
            people_pose = people_pose_array_msg.poses

        mask_img = np.zeros((img.shape[0], img.shape[1]), dtype=np.bool)
        arm = [limb_prefix for limb_prefix in ['R', 'L']
               if limb_prefix + 'Hand' in self.limb_part]
        # nose = ['Nose' in self.limb_part]
        # body = ['Body' in self.limb_part]
        nose = ('Nose' == self.limb_part)
        body = ('Body' == self.limb_part)
        if arm:
            arm_mask_img, debug_img = self._create_hand_mask(people_pose, img, arm)
            mask_img = np.bitwise_or(mask_img, arm_mask_img)
            direction = self.predict_direction(people_pose, img)
        if nose:
            nose_mask_img, debug_img = self._create_nose_mask(people_pose, img)
            mask_img = np.bitwise_or(mask_img, nose_mask_img)
            direction = self.predict_direction(people_pose, img)
        if body:
            body_mask_img, debug_img = self._create_body_mask(people_pose, img)
            mask_img = np.bitwise_or(mask_img, body_mask_img)
            direction = self.predict_direction(people_pose, img)

        mask_msg = br.cv2_to_imgmsg(np.uint8(mask_img * 255.0), encoding='mono8')
        mask_msg.header = img_msg.header

        debug_msg = br.cv2_to_imgmsg(debug_img, encoding='bgr8')
        debug_msg.header = img_msg.header

        self.pub.publish(mask_msg)
        self.debug_pub.publish(debug_msg)
        if self.limb_part == "Nose" or self.limb_part == "RHand":
            self.pub_direction.publish(direction)

    def _create_hand_mask(self, people_pose, img, arm=['R', 'L']):
        rectangle_thickness = 5
        rectangle_colors = [(0, 255, 0), (0, 0, 255)]

        mask_img = np.zeros((img.shape[0], img.shape[1]), dtype=np.bool)
        for person_pose in people_pose:
            for limb_prefix, color in zip(arm, rectangle_colors):
                try:
                    shoulder_index = person_pose.limb_names.index(
                        limb_prefix + 'Shoulder')
                    elbow_index = person_pose.limb_names.index(
                        limb_prefix + 'Elbow')
                    wrist_index = person_pose.limb_names.index(
                        limb_prefix + 'Wrist')
                except ValueError:
                    continue

                if not np.all(np.array(person_pose.scores)[[elbow_index, shoulder_index, wrist_index]] > self.arms_score_threshold):
                    continue

                shoulder = person_pose.poses[shoulder_index]
                elbow = person_pose.poses[elbow_index]
                wrist = person_pose.poses[wrist_index]

                x = wrist.position.x + self.hand_ratio * \
                    (wrist.position.x - elbow.position.x)
                y = wrist.position.y + self.hand_ratio * \
                    (wrist.position.y - elbow.position.y)

                wrist_to_elbow_length = ((wrist.position.x - elbow.position.x)
                                         ** 2 + (wrist.position.y - elbow.position.y) ** 2) ** 0.5
                elbow_to_shoulder_length = ((shoulder.position.x - elbow.position.x) ** 2 + (
                    shoulder.position.y - elbow.position.y) ** 2) ** 0.5
                width = self.hand_width_ratio * \
                    max(wrist_to_elbow_length, 0.9 * elbow_to_shoulder_length)
                height = width

                x -= width / 2.0
                y -= height / 2.0

                x = int(x)
                y = int(y)
                width = int(width)
                height = int(height)

                mask_img[y:y + height, x:x + width] = True
                cv2.rectangle(img, (x, y), (x + width, y + height),
                              color, rectangle_thickness)
        return mask_img, img

    def _create_nose_mask(self, people_pose, img):
        rectangle_thickness = 5
        color = (0, 255, 0)

        mask_img = np.zeros((img.shape[0], img.shape[1]), dtype=np.bool)
        for person_pose in people_pose:
            try:
                nose_index = person_pose.limb_names.index('Nose')
            except ValueError:
                continue
            nose = person_pose.poses[nose_index]
            x = int(nose.position.x)
            y = int(nose.position.y)
            margin = 15
            mask_img[y-margin : y+margin, x-margin : x+margin] = True
            cv2.rectangle(img, (x-margin, y-margin), (x+margin, y+margin),
                          color, rectangle_thickness)
        return mask_img, img

    def _create_body_mask(self, people_pose, img):
        rectangle_thickness = 5
        color = (0, 255, 0)

        mask_img = np.zeros((img.shape[0], img.shape[1]), dtype=np.bool)
        for person_pose in people_pose:
            x_max = y_max = 0
            x_min = y_min = 10000
            for limb in person_pose.poses:
                x = limb.position.x
                y = limb.position.y
                if x_max < x:
                    x_max = x
                if x_min > x:
                    x_min = x
                if y_max < y:
                    y_max = y
                if y_min > y:
                    y_min = y
            # try:
            #     rshoulder_index = person_pose.limb_names.index('RShoulder')
            #     lshoulder_index = person_pose.limb_names.index('LShoulder')
            #     rhip_index = person_pose.limb_names.index('RHip')
            #     lhip_index = person_pose.limb_names.index('LHip')
            # except ValueError:
            #     continue
            # rshoulder = person_pose.poses[rshoulder_index]
            # lshoulder = person_pose.poses[lshoulder_index]
            # rhip = person_pose.poses[rhip_index]
            # lhip = person_pose.poses[lhip_index]
            # x_min = int(min(rshoulder.position.x, rhip.position.x))
            # x_max = int(max(lshoulder.position.x, lhip.position.x))
            # y_min = int(min(lshoulder.position.y, rshoulder.position.y))
            # y_max = int(max(lhip.position.y, rhip.position.y))
            x_max = int(round(x_max))
            x_min = int(round(x_min))
            y_max = int(round(y_max))
            y_min = int(round(y_min))
            x_mid = int((x_max + x_min) / 2.)
            y_mid = int((y_max + y_min) / 2.)
            gain = 1.2
            x_margin = int(abs(x_max - x_mid) * gain)
            y_margin = int(abs(y_max - y_mid) * gain)
            mask_img[y_mid-y_margin : y_mid+y_margin, x_mid-x_margin : x_mid+x_margin] = True
            cv2.rectangle(img, (x_mid-x_margin, y_mid-y_margin), (x_mid+x_margin, y_mid+y_margin),
                          color, rectangle_thickness)
        return mask_img, img

    def predict_direction(self, people_pose, img):
        for person_pose in people_pose:
            try:
                nose_index = person_pose.limb_names.index('Nose')
                neck_index = person_pose.limb_names.index('Neck')
            except ValueError:
                continue
            nose = person_pose.poses[nose_index]
            neck = person_pose.poses[neck_index]
            nose_pos = np.array([nose.position.x, nose.position.y])
            neck_pos = np.array([neck.position.x, neck.position.y])
            vector_neck_to_nose = nose_pos - neck_pos
            if vector_neck_to_nose[0] == 0 and vector_neck_to_nose[1] == 0:
                return ""
            normalize_vector = vector_neck_to_nose / np.sqrt(vector_neck_to_nose[0] ** 2 + vector_neck_to_nose[1] ** 2)
            if self.prev_vector is None:
                normalize_vector_with_inertia = np.copy(normalize_vector)
            else:
                normalize_vector_with_inertia = normalize_vector * (1 - self.inertia_rate) + self.prev_vector * self.inertia_rate
            self.prev_vector = normalize_vector
            if normalize_vector_with_inertia[0] > self.threshold_direction:
                return "right"
            elif normalize_vector_with_inertia[0] < - self.threshold_direction:
                return "left"
            else:
                return "forward"


if __name__ == '__main__':
    rospy.init_node('people_mask_publisher')
    PeopleMaskPublisher()
    rospy.spin()
