#!/usr/bin/env python

import cv2
from cv_bridge import CvBridge
import numpy as np
import rospy

from jsk_recognition_msgs.msg import Spectrum
from sensor_msgs.msg import Image

import bayesian_changepoint_detection.online_changepoint_detection as oncd
from functools import partial
import matplotlib.pyplot as plt


# This node publish spectrogram (sensor_msgs/Image)
# from spectrum (jsk_recognition_msgs/Spectrum)

class SpectrumToSpectrogram(object):

    def __init__(self):
        super(SpectrumToSpectrogram, self).__init__()
        # Set spectrogram shape
        self.image_height = rospy.get_param('~image_height', 300)
        self.image_width = rospy.get_param('~image_width', 300)
        # Get spectrum length
        spectrum_msg = rospy.wait_for_message('~spectrum', Spectrum)
        spectrum_len = len(spectrum_msg.amplitude)
        # Buffer for spectrum topic
        self.spectrogram = np.zeros((0, spectrum_len), dtype=np.float32)
        self.spectrum_stamp = []
        # Period[s] to store audio data to create one spectrogram topic
        self.spectrogram_period = rospy.get_param('~spectrogram_period', 5)
        # ROS subscriber and publisher
        rospy.Subscriber(
            '~spectrum', Spectrum, self.audio_cb)
        self.pub_spectrogram = rospy.Publisher(
            '~spectrogram', Image, queue_size=1)
        rospy.Timer(
            rospy.Duration(
                float(self.spectrogram_period) / self.image_width),
            self.timer_cb)
        rospy.Timer(
            rospy.Duration(1),
            self.detect_changepoint
        )
        self.bridge = CvBridge()

        self.init_plot()

    def audio_cb(self, msg):
        # Add spectrum msg to buffer
        spectrum = np.array(msg.amplitude, dtype=np.float32)
        self.spectrogram = np.concatenate(
            [self.spectrogram, spectrum[None]])
        self.spectrum_stamp.append(msg.header.stamp)

    def init_plot(self):
        # Set matplotlib config
        self.fig = plt.figure(figsize=(8, 5))
        self.fig.suptitle('changepoint probability', size=12)
        self.fig.subplots_adjust(left=0.1, right=0.95, top=0.90, bottom=0.1,
                                 wspace=0.2, hspace=0.6)
        self.ax = self.fig.add_subplot(1, 1, 1)
        self.ax.grid(True)
        self.ax.set_xlabel('count', fontsize=12)
        self.ax.set_ylabel('score', fontsize=12)
        self.line, = self.ax.plot([0, 0], label='changepoint probability')

    def detect_changepoint(self, timer):
        print(self.spectrogram.transpose(1, 0)[::-1, :].shape)
        R, maxes = oncd.online_changepoint_detection(
            self.spectrogram.transpose(1, 0)[::-1, :][0, :],
            partial(oncd.constant_hazard, 250),
            oncd.StudentT(0.1, .01, 1, 0)
        )
        Nw=10;
        # Plot spectrum
        self.line.set_data(np.arange(len(R[Nw,Nw:-1])), R[Nw,Nw:-1])
        self.ax.set_xlim((0, len(R[Nw,Nw:-1])))
        self.ax.set_ylim((0.0, 1.0))
        self.ax.legend(loc='upper right')

    def timer_cb(self, timer):
        if self.spectrogram.shape[0] == 0:
            return
        # Extract spectrogram of last (self.spectrogram_period) seconds
        time_now = rospy.Time.now()
        for i, stamp in enumerate(self.spectrum_stamp):
            if (time_now - stamp).to_sec() <= self.spectrogram_period:
                self.spectrum_stamp = self.spectrum_stamp[i:]
                self.spectrogram = self.spectrogram[i:]
                break
        # Reshape spectrogram
        spectrogram = self.spectrogram.transpose(1, 0)[::-1, :]
        spectrogram = cv2.resize(
            spectrogram, (self.image_width, self.image_height))
        # Publish spectrogram
        spectrogram_msg = self.bridge.cv2_to_imgmsg(spectrogram, '32FC1')
        spectrogram_msg.header.stamp = rospy.Time.now()
        self.pub_spectrogram.publish(spectrogram_msg)


if __name__ == '__main__':
    rospy.init_node('audio_to_spectrogram')
    SpectrumToSpectrogram()
    # rospy.spin()
    while not rospy.is_shutdown():
        plt.pause(.1)  # real-time plotting

