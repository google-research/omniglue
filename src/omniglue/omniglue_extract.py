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

"""Wrapper for performing OmniGlue inference, plus (optionally) SP/DINO."""

import numpy as np
from omniglue import dino_extract
from omniglue import superpoint_extract
from omniglue import utils
import tensorflow as tf

DINO_FEATURE_DIM = 768
MATCH_THRESHOLD = 1e-3


class OmniGlue:
  # TODO(omniglue): class docstring

  def __init__(
      self,
      og_export: str,
      sp_export: str | None = None,
      dino_export: str | None = None,
  ) -> None:
    self.matcher = tf.saved_model.load(og_export)
    if sp_export is not None:
      self.sp_extract = superpoint_extract.SuperPointExtract(sp_export)
    if dino_export is not None:
      self.dino_extract = dino_extract.DINOExtract(dino_export, feature_layer=1)

  def FindMatches(self, image0: np.ndarray, image1: np.ndarray):
    """TODO(omniglue): docstring."""
    height0, width0 = image0.shape[:2]
    height1, width1 = image1.shape[:2]
    sp_features0 = self.sp_extract(image0)
    sp_features1 = self.sp_extract(image1)
    dino_features0 = self.dino_extract(image0)
    dino_features1 = self.dino_extract(image1)
    dino_descriptors0 = dino_extract.get_dino_descriptors(
        dino_features0,
        tf.convert_to_tensor(sp_features0[0], dtype=tf.float32),
        tf.convert_to_tensor(height0, dtype=tf.int32),
        tf.convert_to_tensor(width0, dtype=tf.int32),
        DINO_FEATURE_DIM,
    )
    dino_descriptors1 = dino_extract.get_dino_descriptors(
        dino_features1,
        tf.convert_to_tensor(sp_features1[0], dtype=tf.float32),
        tf.convert_to_tensor(height1, dtype=tf.int32),
        tf.convert_to_tensor(width1, dtype=tf.int32),
        DINO_FEATURE_DIM,
    )

    inputs = self._construct_inputs(
        width0,
        height0,
        width1,
        height1,
        sp_features0,
        sp_features1,
        dino_descriptors0,
        dino_descriptors1,
    )

    og_outputs = self.matcher.signatures['serving_default'](**inputs)
    soft_assignment = og_outputs['soft_assignment'][:, :-1, :-1]

    match_matrix = (
        utils.soft_assignment_to_match_matrix(soft_assignment, MATCH_THRESHOLD)
        .numpy()
        .squeeze()
    )

    # Filter out any matches with 0.0 confidence keypoints.
    match_indices = np.argwhere(match_matrix)
    keep = []
    for i in range(match_indices.shape[0]):
      match = match_indices[i, :]
      if (sp_features0[2][match[0]] > 0.0) and (
          sp_features1[2][match[1]] > 0.0
      ):
        keep.append(i)
    match_indices = match_indices[keep]

    # Format matches in terms of keypoint locations.
    match_kp0s = []
    match_kp1s = []
    match_confidences = []
    for match in match_indices:
      match_kp0s.append(sp_features0[0][match[0], :])
      match_kp1s.append(sp_features1[0][match[1], :])
      match_confidences.append(soft_assignment[0, match[0], match[1]])
    match_kp0s = np.array(match_kp0s)
    match_kp1s = np.array(match_kp1s)
    match_confidences = np.array(match_confidences)
    return match_kp0s, match_kp1s, match_confidences

  ### Private methods ###

  def _construct_inputs(
      self,
      width0,
      height0,
      width1,
      height1,
      sp_features0,
      sp_features1,
      dino_descriptors0,
      dino_descriptors1,
  ):
    inputs = {
        'keypoints0': tf.convert_to_tensor(
            np.expand_dims(sp_features0[0], axis=0),
            dtype=tf.float32,
        ),
        'keypoints1': tf.convert_to_tensor(
            np.expand_dims(sp_features1[0], axis=0), dtype=tf.float32
        ),
        'descriptors0': tf.convert_to_tensor(
            np.expand_dims(sp_features0[1], axis=0), dtype=tf.float32
        ),
        'descriptors1': tf.convert_to_tensor(
            np.expand_dims(sp_features1[1], axis=0), dtype=tf.float32
        ),
        'scores0': tf.convert_to_tensor(
            np.expand_dims(np.expand_dims(sp_features0[2], axis=0), axis=-1),
            dtype=tf.float32,
        ),
        'scores1': tf.convert_to_tensor(
            np.expand_dims(np.expand_dims(sp_features1[2], axis=0), axis=-1),
            dtype=tf.float32,
        ),
        'descriptors0_dino': tf.expand_dims(dino_descriptors0, axis=0),
        'descriptors1_dino': tf.expand_dims(dino_descriptors1, axis=0),
        'width0': tf.convert_to_tensor(
            np.expand_dims(width0, axis=0), dtype=tf.int32
        ),
        'width1': tf.convert_to_tensor(
            np.expand_dims(width1, axis=0), dtype=tf.int32
        ),
        'height0': tf.convert_to_tensor(
            np.expand_dims(height0, axis=0), dtype=tf.int32
        ),
        'height1': tf.convert_to_tensor(
            np.expand_dims(height1, axis=0), dtype=tf.int32
        ),
    }
    return inputs
