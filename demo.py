#!/usr/bin/env python3
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

"""Demo script for performing OmniGlue inference."""

import time
import matplotlib.pyplot as plt
import numpy as np
import omniglue
from omniglue import utils
from PIL import Image


def main() -> None:

  print("> Loading images...")
  image0 = np.array(Image.open("./res/navi_1.png"))
  image1 = np.array(Image.open("./res/navi_2.png"))

  print("> Loading OmniGlue (and its submodules: SuperPoint & DINOv2)...")
  start = time.time()
  og = omniglue.OmniGlue(
      og_export="./models/og_export",
      sp_export="./models/sp_v6",
      dino_export="./models/dinov2_vitb14_pretrain.pth",
  )
  print(f"> \tTook {time.time() - start} seconds.")

  print("> Finding matches...")
  start = time.time()
  match_kp0, match_kp1, match_confidences = og.FindMatches(image0, image1)
  num_inliers = match_kp0.shape[0]
  print(f"> \tFound {num_inliers} inliers.")
  print(f"> \tTook {time.time() - start} seconds.")

  print("> Visualizing matches...")
  viz = utils.visualize_matches(
      image0,
      image1,
      match_kp0,
      match_kp1,
      np.eye(num_inliers),
      show_keypoints=True,
      highlight_unmatched=True,
      title=f"{num_inliers} inliers",
      line_width=2,
  )
  plt.figure(figsize=(20, 10), dpi=100, facecolor="w", edgecolor="k")
  plt.axis("off")
  plt.imshow(viz)
  plt.imsave("./demo_output.png", viz)
  print("> \tSaved visualization to ./demo_output.png")


if __name__ == "__main__":
  main()
