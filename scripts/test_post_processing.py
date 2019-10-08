#!/usr/bin/env python

from __future__ import print_function

import imageio
import matplotlib.pyplot as plt
import numpy as np

synthetic_depth_img = imageio.imread('0_synthetic_depth.png')
depth_img = imageio.imread('0_depth.png')

DEPTH_SCALING_FACTOR = 10000.0
UNIT_SCALING = 1.0

synthetic_depth_img = synthetic_depth_img.astype(np.float64) / DEPTH_SCALING_FACTOR / UNIT_SCALING
depth_img = depth_img.astype(np.float64) / DEPTH_SCALING_FACTOR / UNIT_SCALING

depth_img[depth_img == 0.0] = float('nan')

# depth_img has nans, synthetic_depth_img has 0.0 for invalid depths
diffs = depth_img[synthetic_depth_img != 0.0] - synthetic_depth_img[synthetic_depth_img != 0.0]
diffs = diffs[np.logical_not(np.isnan(diffs))]

DISTANCE_THRESHOLD = 0.30  # max depth distance, TODO: make param
diffs = diffs[np.fabs(diffs) < DISTANCE_THRESHOLD]

total_pixels = len(synthetic_depth_img[synthetic_depth_img != 0.0])
valid_pixels = len(diffs)
valid_ratio = float(valid_pixels) / total_pixels
diff = diffs.mean()

print("valid_ratio: {} ({}/{})".format(valid_ratio, valid_pixels, total_pixels))
print("diff:        {}".format(diff))

MIN_VALID_RATIO = 0.0  # TODO: adjust, make param
if valid_ratio < MIN_VALID_RATIO:
    print("INVALID RATIO")

histogram, bin_edges = np.histogram(diffs, bins='auto')
diff_from_histogram = (bin_edges[histogram.argmax()] + bin_edges[histogram.argmax() + 1]) / 2.0
print("middle:      {}".format(diff_from_histogram))

_ = plt.hist(diffs, bins='auto')  # arguments are passed to np.histogram
plt.axvline(diff, color='k', linestyle='dashed', linewidth=1)
plt.axvline(diff_from_histogram, color='red', linestyle='dashed', linewidth=1)
plt.title("Histogram of diffs")
plt.show()
