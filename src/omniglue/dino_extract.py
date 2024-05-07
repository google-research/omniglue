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

"""Wrapper for performing DINOv2 inference."""

import cv2
import numpy as np
from third_party.dinov2 import dino
from omniglue import utils
import tensorflow as tf
import torch


class DINOExtract:
  """Class to initialize DINO model and extract features from an image."""

  def __init__(self, cpt_path: str, feature_layer: int = 1):
    self.feature_layer = feature_layer
    self.model = dino.vit_base()
    state_dict_raw = torch.load(cpt_path, map_location='cpu')

    # state_dict = {}
    # for k, v in state_dict_raw.items():
    #   state_dict[k.replace('blocks', 'blocks.0')] = v

    self.model.load_state_dict(state_dict_raw)
    self.model.eval()

    self.image_size_max = 630

    self.h_down_rate = self.model.patch_embed.patch_size[0]
    self.w_down_rate = self.model.patch_embed.patch_size[1]

  def __call__(self, image) -> np.ndarray:
    return self.forward(image)

  def forward(self, image: np.ndarray) -> np.ndarray:
    """Feeds image through DINO ViT model to extract features.

    Args:
      image: (H, W, 3) numpy array, decoded image bytes, value range [0, 255].

    Returns:
      features: (H // 14, W // 14, C) numpy array image features.
    """
    image = self._resize_input_image(image)
    image_processed = self._process_image(image)
    image_processed = image_processed.unsqueeze(0).float()
    features = self.extract_feature(image_processed)
    features = features.squeeze(0).permute(1, 2, 0).cpu().numpy()
    return features

  def _resize_input_image(
      self, image: np.ndarray, interpolation=cv2.INTER_LINEAR
  ):
    """Resizes image such that both dimensions are divisble by down_rate."""
    h_image, w_image = image.shape[:2]
    h_larger_flag = h_image > w_image
    large_side_image = max(h_image, w_image)

    # resize the image with the largest side length smaller than a threshold
    # to accelerate ViT backbone inference (which has quadratic complexity).
    if large_side_image > self.image_size_max:
      if h_larger_flag:
        h_image_target = self.image_size_max
        w_image_target = int(self.image_size_max * w_image / h_image)
      else:
        w_image_target = self.image_size_max
        h_image_target = int(self.image_size_max * h_image / w_image)
    else:
      h_image_target = h_image
      w_image_target = w_image

    h, w = (
        h_image_target // self.h_down_rate,
        w_image_target // self.w_down_rate,
    )
    h_resize, w_resize = h * self.h_down_rate, w * self.w_down_rate
    image = cv2.resize(image, (w_resize, h_resize), interpolation=interpolation)
    return image

  def _process_image(self, image: np.ndarray) -> torch.Tensor:
    """Turn image into pytorch tensor and normalize it."""
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])

    image_processed = image / 255.0
    image_processed = (image_processed - mean) / std
    image_processed = torch.from_numpy(image_processed).permute(2, 0, 1)
    return image_processed

  def extract_feature(self, image):
    """Extracts features from image.

    Args:
      image: (B, 3, H, W) torch tensor, normalized with ImageNet mean/std.

    Returns:
      features: (B, C, H//14, W//14) torch tensor image features.
    """
    b, _, h_origin, w_origin = image.shape
    out = self.model.get_intermediate_layers(image, n=self.feature_layer)[0]
    h = int(h_origin / self.h_down_rate)
    w = int(w_origin / self.w_down_rate)
    dim = out.shape[-1]
    out = out.reshape(b, h, w, dim).permute(0, 3, 1, 2).detach()
    return out


def _preprocess_shape(
    h_image, w_image, image_size_max=630, h_down_rate=14, w_down_rate=14
):
  # Flatten the tensors
  h_image = tf.squeeze(h_image)
  w_image = tf.squeeze(w_image)
  # logging.info(h_image, w_image)

  h_larger_flag = tf.greater(h_image, w_image)
  large_side_image = tf.maximum(h_image, w_image)

  # Function to calculate new dimensions when height is larger
  def resize_h_larger():
    h_image_target = image_size_max
    w_image_target = tf.cast(image_size_max * w_image / h_image, tf.int32)
    return h_image_target, w_image_target

  # Function to calculate new dimensions when width is larger or equal
  def resize_w_larger_or_equal():
    w_image_target = image_size_max
    h_image_target = tf.cast(image_size_max * h_image / w_image, tf.int32)
    return h_image_target, w_image_target

  # Function to keep original dimensions
  def keep_original():
    return h_image, w_image

  h_image_target, w_image_target = tf.cond(
      tf.greater(large_side_image, image_size_max),
      lambda: tf.cond(h_larger_flag, resize_h_larger, resize_w_larger_or_equal),
      keep_original,
  )

  # resize to be divided by patch size
  h = h_image_target // h_down_rate
  w = w_image_target // w_down_rate
  h_resize = h * h_down_rate
  w_resize = w * w_down_rate

  # Expand dimensions
  h_resize = tf.expand_dims(h_resize, 0)
  w_resize = tf.expand_dims(w_resize, 0)

  return h_resize, w_resize


def get_dino_descriptors(dino_features, keypoints, height, width, feature_dim):
  """Get DINO descriptors using Superpoint keypoints.

  Args:
    dino_features: DINO features in 1-D.
    keypoints: Superpoint keypoint locations, in format (x, y), in pixels, shape
      (N, 2).
    height: image height, type tf.Tensor.int32.
    width: image width, type tf.Tensor.int32.
    feature_dim: DINO feature channel size, type tf.Tensor.int32.

  Returns:
    Interpolated DINO descriptors.
  """
  # TODO(omniglue): fix the hard-coded DINO patch size (14).
  height_1d = tf.reshape(height, [1])
  width_1d = tf.reshape(width, [1])

  height_1d_resized, width_1d_resized = _preprocess_shape(
      height_1d, width_1d, image_size_max=630, h_down_rate=14, w_down_rate=14
  )

  height_feat = height_1d_resized // 14
  width_feat = width_1d_resized // 14
  feature_dim_1d = tf.reshape(feature_dim, [1])

  size_feature = tf.concat([height_feat, width_feat, feature_dim_1d], axis=0)
  dino_features = tf.reshape(dino_features, size_feature)

  img_size = tf.cast(tf.concat([width_1d, height_1d], axis=0), tf.float32)
  feature_size = tf.cast(
      tf.concat([width_feat, height_feat], axis=0), tf.float32
  )

  keypoints_feature = (
      keypoints
      / tf.expand_dims(img_size, axis=0)
      * tf.expand_dims(feature_size, axis=0)
  )

  dino_descriptors = []
  for kp in keypoints_feature:
    dino_descriptors.append(
        utils.lookup_descriptor_bilinear(kp.numpy(), dino_features.numpy())
    )
  dino_descriptors = tf.convert_to_tensor(
      np.array(dino_descriptors), dtype=tf.float32
  )
  return dino_descriptors
