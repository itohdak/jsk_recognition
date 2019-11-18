#!/usr/bin/env python
# -*- coding:utf-8 -*-

import cv2
import numpy as np

import cv_bridge
import message_filters
import rospy
from jsk_topic_tools import ConnectionBasedTransport
from jsk_recognition_msgs.msg import RectArray
from sensor_msgs.msg import Image


class SSDMaskPublisher(ConnectionBasedTransport):

    def __init__(self):
        super(self.__class__, self).__init__()

        self.pub = self.advertise('~output', Image, queue_size=1)
        self.debug_pub = self.advertise('~debug/output', Image, queue_size=1)

    def subscribe(self):
        queue_size = rospy.get_param('~queue_size', 10)
        sub_img = message_filters.Subscriber(
            '~input', Image, queue_size=1, buff_size=2**24)
        sub_rects = message_filters.Subscriber(
            '~input/rect', RectArray, queue_size=1, buff_size=2**24)
        self.subs = [sub_img, sub_rects]
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

    def _cb(self, img_msg, rect_array_msg):
        br = cv_bridge.CvBridge()
        img = br.imgmsg_to_cv2(img_msg, desired_encoding='bgr8')

        rects = rect_array_msg.rects

        mask_img = np.zeros((img.shape[0], img.shape[1]), dtype=np.bool)
        ssd_mask_img, debug_img = self._create_ssd_mask(rects, img)
        mask_img = np.bitwise_or(mask_img, ssd_mask_img)

        mask_msg = br.cv2_to_imgmsg(np.uint8(mask_img * 255.0), encoding='mono8')
        mask_msg.header = img_msg.header

        debug_msg = br.cv2_to_imgmsg(debug_img, encoding='bgr8')
        debug_msg.header = img_msg.header

        self.pub.publish(mask_msg)
        self.debug_pub.publish(debug_msg)

    def _create_ssd_mask(self, rects, img):
        rectangle_thickness = 5
        rectangle_color = (0, 255, 0)

        mask_img = np.zeros((img.shape[0], img.shape[1]), dtype=np.bool)
        for rect in rects:
            x = rect.x
            y = rect.y
            width = rect.width
            height = rect.height

            mask_img[y:y + height, x:x + width] = True
            cv2.rectangle(img, (x, y), (x + width, y + height),
                      rectangle_color, rectangle_thickness)
        return mask_img, img


if __name__ == '__main__':
    rospy.init_node('ssd_mask_publisher')
    SSDMaskPublisher()
    rospy.spin()
