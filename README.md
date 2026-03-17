
## Code and Dataset for `Convolutional Occupancy Models for Dense Packing of Complex, Novel Objects`

This repo contains the dataset and model proposed in our IROS 2023 paper:

```
"Convolutional Occupancy Models for Dense Packing of Complex, Novel Objects".
Nikhil Mishra, Pieter Abbeel, Xi Chen, and Maximilan Sieb.
In the proceedings of the IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS), 2023.
```

The dataset, COB-3D-v2, can be downloaded from our [project page](https://sites.google.com/view/fcon-packing/).

The shape-completion model we proposed, F-CON, is implemented in `fcon_model.py`.

The Jupyter notebook `example.ipynb` shows how to load the data and do inference with a pretrained F-CON model we have provided.

---

## Production Improvements for Bin-Packing and Conveyor Scenes

This fork improves F-CON for **accurate volumetric shape estimation of boxes on conveyors and in totes** — where the production requirement is correct bounding-volume prediction (no over- or under-estimation that would cause placement collisions) and physically integral, connected predicted shapes.

The original model predicts each object independently from only its local frustum RGB + mask, with no information about where the support surface is. This causes depth errors: predicted boxes float above the conveyor or collapse to thin shells. The improvements below address the root causes in a principled way:

### Implemented
| Change | Why it helps |
|---|---|
| **Background depth voxelization** (`fcon_model.py`) | Background pixels in the ROI are voxelized separately and fed as a 7-channel volume alongside the foreground. The conveyor/floor depth bin is now explicitly visible to the model, giving a hard anchor for the bottom of each box. |
| **Surface normals + amodal mask as input** (`fcon_model.py`) | Normals constrain which face is visible and what direction the object extends. Amodal mask gives the full object footprint including occluded parts. Together they resolve the depth ambiguity that was the main failure mode. |
| **Depth range randomization during training** (`fcon_model.py`) | The COB3D `far_plane` is always set to ~floor depth, so the model could implicitly read box height from the frustum range. Random per-object cushion (5–35 cm near, 0–50 cm past floor) forces the model to infer depth from background/normal evidence instead. |
| **Focal loss** (`train.py`) | ~95% of frustum voxels are empty. Standard BCE biases the model toward predicting empty, causing under-filled shapes with gaps. Focal loss (γ=2) down-weights easy empty-voxel predictions, improving occupied-voxel recall and solid interiors. |
| **Cosine LR schedule** (`train.py`) | Cosine annealing from lr_max to lr_max/100 over all epochs. |
| **Largest connected component filter** (`evaluate.py`) | Post-processing: before marching cubes, keep only the single largest connected blob in the predicted binary occupancy grid. Eliminates floating fragments with zero retraining. |

### Planned
| TODO | Description |
|---|---|
| TODO-4 | RANSAC support-plane detection → encode as 4-float FiLM conditioning in 3D-UNet. Gives the model an explicit "box must touch this plane" constraint. |
| TODO-10 | Replace per-voxel binary occupancy head with two separate surface heads (near/far). Output is always a solid connected volume by construction — correct for boxes. |
| TODO-3 | Global scene encoder: lightweight 2D CNN on full RGB+depth, ROI-aligned per object, injected into 2D-UNet. Allows depth consistency across all boxes in a scene. |


### Citations

If you use COB-3D-v2 or F-CON in your work, please cite:
 ```
 @inproceedings{
     mishra2023convolutional,
     title={Convolutional Occupancy Models for Dense Packing of Complex, Novel Objects},
     author={Nikhil Mishra and Pieter Abbeel and Xi Chen and Maximilian Sieb},
     year={2023},
     booktitle={IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)},
}
```

### License
Shield: [![CC BY-NC-SA 4.0][cc-by-nc-sa-shield]][cc-by-nc-sa]

This work, including the paper, code, weights, and dataset, is licensed under a
[Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License][cc-by-nc-sa].

[![CC BY-NC-SA 4.0][cc-by-nc-sa-image]][cc-by-nc-sa]

[cc-by-nc-sa]: http://creativecommons.org/licenses/by-nc-sa/4.0/
[cc-by-nc-sa-image]: https://licensebuttons.net/l/by-nc-sa/4.0/88x31.png
[cc-by-nc-sa-shield]: https://img.shields.io/badge/License-CC%20BY--NC--SA%204.0-lightgrey.svg
