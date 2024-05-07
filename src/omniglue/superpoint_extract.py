# Copyright 2024 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Wrapper for performing SuperPoint inference."""

import math
from typing import Optional, Tuple

import cv2
import numpy as np
from omniglue import utils
import tensorflow.compat.v1 as tf1


class SuperPointExtract:
  """Class to initialize SuperPoint model and extract features from an image.

  To stay consistent with SuperPoint training and eval configurations, resize
  images to (320x240) or (640x480).

  Attributes
    model_path: string, filepath to saved SuperPoint TF1 model weights.
  """

  def __init__(self, model_path: str):
    self.model_path = model_path
    self._graph = tf1.Graph()
    self._sess = tf1.Session(graph=self._graph)
    tf1.saved_model.loader.load(
        self._sess, [tf1.saved_model.tag_constants.SERVING], model_path
    )

  def __call__(
      self,
      image,
      segmentation_mask=None,
      num_features=1024,
      pad_random_features=False,
  ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    return self.compute(
        image,
        segmentation_mask=segmentation_mask,
        num_features=num_features,
        pad_random_features=pad_random_features,
    )

  def compute(
      self,
      image: np.ndarray,
      segmentation_mask: Optional[np.ndarray] = None,
      num_features: int = 1024,
      pad_random_features: bool = False,
  ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Feeds image through SuperPoint model to extract keypoints and features.

    Args:
      image: (H, W, 3) numpy array, decoded image bytes.
      segmentation_mask: (H, W) binary numpy array or None. If not None,
        extracted keypoints are restricted to being within the mask.
      num_features: max number of features to extract (or 0 to indicate keeping
        all extracted features).
      pad_random_features: if True, adds randomly sampled keypoints to the
        output such that there are exactly 'num_features' keypoints. Descriptors
        for these sampled keypoints are taken from the network's descriptor map
        output, and scores are set to 0. No action taken if num_features = 0.

    Returns:
      keypoints: (N, 2) numpy array, coordinates of keypoints as floats.
      descriptors: (N, 256) numpy array, descriptors for keypoints as floats.
      scores: (N, 1) numpy array, confidence values for keypoints as floats.
    """

    # Resize image so both dimensions are divisible by 8.
    image, keypoint_scale_factors = self._resize_input_image(image)
    if segmentation_mask is not None:
      segmentation_mask, _ = self._resize_input_image(
          segmentation_mask, interpolation=cv2.INTER_NEAREST
      )
    assert (
        segmentation_mask is None
        or image.shape[:2] == segmentation_mask.shape[:2]
    )

    # Preprocess and feed-forward image.
    image_preprocessed = self._preprocess_image(image)
    input_image_tensor = self._graph.get_tensor_by_name('superpoint/image:0')
    output_prob_nms_tensor = self._graph.get_tensor_by_name(
        'superpoint/prob_nms:0'
    )
    output_desc_tensors = self._graph.get_tensor_by_name(
        'superpoint/descriptors:0'
    )
    out = self._sess.run(
        [output_prob_nms_tensor, output_desc_tensors],
        feed_dict={input_image_tensor: np.expand_dims(image_preprocessed, 0)},
    )

    # Format output from network.
    keypoint_map = np.squeeze(out[0])
    descriptor_map = np.squeeze(out[1])
    if segmentation_mask is not None:
      keypoint_map = np.where(segmentation_mask, keypoint_map, 0.0)
    keypoints, descriptors, scores = self._extract_superpoint_output(
        keypoint_map, descriptor_map, num_features, pad_random_features
    )

    # Rescale keypoint locations to match original input image size, and return.
    keypoints = keypoints / keypoint_scale_factors
    return (keypoints, descriptors, scores)

  def _resize_input_image(self, image, interpolation=cv2.INTER_LINEAR):
    """Resizes image such that both dimensions are divisble by 8."""

    # Calculate new image dimensions and per-dimension resizing scale factor.
    new_dim = [-1, -1]
    keypoint_scale_factors = [1.0, 1.0]
    for i in range(2):
      dim_size = image.shape[i]
      mod_eight = dim_size % 8
      if mod_eight < 4:
        # Round down to nearest multiple of 8.
        new_dim[i] = dim_size - mod_eight
      elif mod_eight >= 4:
        # Round up to nearest multiple of 8.
        new_dim[i] = dim_size + (8 - mod_eight)
      keypoint_scale_factors[i] = (new_dim[i] - 1) / (dim_size - 1)

    # Resize and return image + scale factors.
    new_dim = new_dim[::-1]  # Convert from (row, col) to (x,y).
    keypoint_scale_factors = keypoint_scale_factors[::-1]
    image = cv2.resize(image, tuple(new_dim), interpolation=interpolation)
    return image, keypoint_scale_factors

  def _preprocess_image(self, image):
    """Converts image to grayscale and normalizes values for model input."""
    image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    image = np.expand_dims(image, 2)
    image = image.astype(np.float32)
    image = image / 255.0
    return image

  def _extract_superpoint_output(
      self,
      keypoint_map,
      descriptor_map,
      keep_k_points=512,
      pad_random_features=False,
  ):
    """Converts from raw SuperPoint output (feature maps) into numpy arrays.

    If keep_k_points is 0, then keep all detected keypoints. Otherwise, sort by
    confidence and keep only the top k confidence keypoints.

    Args:
      keypoint_map: (H, W, 1) numpy array, raw output confidence values from
        SuperPoint model.
      descriptor_map: (H, W, 256) numpy array, raw output descriptors from
        SuperPoint model.
      keep_k_points: int, number of keypoints to keep (or 0 to indicate keeping
        all detected keypoints).
      pad_random_features: if True, adds randomly sampled keypoints to the
        output such that there are exactly 'num_features' keypoints. Descriptors
        for these sampled keypoints are taken from the network's descriptor map
        output, and scores are set to 0. No action taken if keep_k_points = 0.

    Returns:
      keypoints: (N, 2) numpy array, image coordinates (x, y) of keypoints as
        floats.
      descriptors: (N, 256) numpy array, descriptors for keypoints as floats.
      scores: (N, 1) numpy array, confidence values for keypoints as floats.
    """

    def _select_k_best(points, k):
      sorted_prob = points[points[:, 2].argsort(), :]
      start = min(k, points.shape[0])
      return sorted_prob[-start:, :2], sorted_prob[-start:, 2]

    keypoints = np.where(keypoint_map > 0)
    prob = keypoint_map[keypoints[0], keypoints[1]]
    keypoints = np.stack([keypoints[0], keypoints[1], prob], axis=-1)

    # Keep only top k points, or all points if keep_k_points param is 0.
    if keep_k_points == 0:
      keep_k_points = keypoints.shape[0]
    keypoints, scores = _select_k_best(keypoints, keep_k_points)

    # Optionally, pad with random features (and confidence scores of 0).
    image_shape = np.array(keypoint_map.shape[:2])
    if pad_random_features and (keep_k_points > keypoints.shape[0]):
      num_pad = keep_k_points - keypoints.shape[0]
      keypoints_pad = (image_shape - 1) * np.random.uniform(size=(num_pad, 2))
      keypoints = np.concatenate((keypoints, keypoints_pad))
      scores_pad = np.zeros((num_pad))
      scores = np.concatenate((scores, scores_pad))

    # Lookup descriptors via bilinear interpolation.
    # TODO: batch descriptor lookup with bilinear interpolation.
    keypoints[:, [0, 1]] = keypoints[:, [1, 0]]  # Swap from (row,col) to (x,y).
    descriptors = []
    for kp in keypoints:
      descriptors.append(utils.lookup_descriptor_bilinear(kp, descriptor_map))
    descriptors = np.array(descriptors)
    return keypoints, descriptors, scores
