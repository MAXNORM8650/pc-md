# Any4D: Unified Feed-Forward Metric 4D Reconstruction

**Authors:** Jay Karhade; Nikhil Keetha; Yuchen Zhang; Tanisha Gupta; Akash Sharma; Sebastian Scherer; Deva Ramanan

**arXiv:** [2512.10935](https://arxiv.org/abs/2512.10935)

## Abstract

We present Any4D, a scalable multi-view transformer
for metric-scale, dense feed-forward 4D reconstruction.
Any4D directly generates per-pixel motion and geometry
predictions for N frames, in contrast to prior work that typ-
ically focuses on either 2-view dense scene flow or sparse
3D point tracking. Moreover, unlike other recent methods
for 4D reconstruction from monocular RGB videos, Any-
4D can process additional modalities and sensors such as
RGB-D frames, IMU-based egomotion, and Radar Doppler
measurements, when available.
One of the key innova-
tions that allows for such a flexible framework is a mod-
ular representation of a 4D scene; specifically, per-view
4D predictions are encoded using a variety of egocentric
factors (depthmaps and camera intrinsics) represented in
local camera coordinates, and allocentric factors (camera
extrinsics and scene flow) represented in global world co-
ordinates. We achieve superior performance across diverse
setups - both in terms of accuracy (2 −3× lower error)
and compute efficiency (15× faster) - opening avenues for
multiple downstream applications.

## Contents

- 1 Introduction
- 2 Related Work
- 2 Existing 4D Reconstruction Models
- 3 Any4D
- 3 Doppler
  - 3.1 Architecture
- 4 Any4D
  - 3.2 Training Details
- 4 Results & Analysis
  - 21.06 MASt3R + CoTracker3
  - 14.69 VGGT + CoTracker3
  - 45.77 MapAnything + CoTracker3
  - 58.01 St4RTrack
  - 28.46 SpatialTrackerV2
  - 50.70 Any4D
  - 12.93 MASt3R + SEA-RAFT
  - 10.20 VGGT + SEA-RAFT
  - 37.63 MapAnything + SEA-RAFT
  - 13.78 St4RTrack
  - 3.37 Any4D
- 7 Table 3. Any4D shows state-of-the-art video depth estimation
  - 69.30 VDA
  - 59.10 MonST3R
  - 59.40 MegaSAM
  - 74.60 SpatialTrackerV2
  - 56.00 VGGT
  - 65.90 MapAnything
  - 65.87 Any4D
  - 67.59 Table 4.
  - 68.03 Images + Geometry
  - 68.71 Images + Doppler
  - 70.32 Images + Geometry + Doppler
  - 21.87 Egocentric Scene Flow
  - 65.13 Allocentric Scene Flow
- 5 Conclusion
- 8 Any4D: Unified Feed-Forward Metric 4D Reconstruction
- 500 MegaDepth
- 275 ScanNet++
- 295 VKITTI2
- 40 Waymo-DriveTrack
- 1500 GCD-Kubric
- 5000 CoTracker3-Kubric
- 5000 Dynamic Replica
- 500 Point Odyssey
- 159 A. Training
- 64 Number of Views at inference
  - 0.26 Scene-Flow EPE
- 2007 IEEE 11th International Conference on Computer Vi-

## Key Concepts

### 4D Reconstruction

The process of reconstructing a 4D (3D + time) world from sensor observations, which aims to capture both spatial and temporal dynamics of scenes.

*Pages: 1, 2*

### Dynamic Video Synthesis

A generative AI application that creates realistic dynamic videos using 4D reconstruction techniques, enabling the generation of interactive content.

*Pages: 1, 2*

### Predictive Control

A robotics technique that uses 4D scene reconstruction to improve decision-making and motion planning for agents navigating physical environments.

*Pages: 1, 2*

### Metric-Scale Reconstruction

A reconstruction approach that produces outputs in physical metric coordinates rather than normalized or relative scales, enabling real-world applications.

*Pages: 1, 2*

### Factored 4D Representation

A modeling approach that decomposes 4D reconstruction into per-view allocentric factors (scene flow and poses) and egocentric factors (intrinsics and depth).

*Pages: 1, 2*

### Multi-Modal Inputs

A flexible input approach that allows 4D reconstruction systems to utilize various sensor modalities beyond just images, such as depth sensors, IMUs, and RADARs.

*Pages: 1, 2*

### Efficient Inference

A computational approach that enables 4D reconstruction systems to process multiple video frames in a single feed-forward pass, significantly improving speed compared to iterative optimization methods.

*Pages: 1, 2*

### Under-constrained Problem

A fundamental challenge in 4D reconstruction where the lack of sufficient constraints requires simplifying assumptions such as rigid motion or static world assumptions.

*Pages: 1, 2*

### Lack of Large-Scale Datasets

A major limitation in 4D reconstruction research due to the scarcity of large-scale, high-quality 4D datasets, which are primarily limited to simulation-based sources.

*Pages: 1, 2*

### Fragmented Sub-tasks

The tendency in existing 4D reconstruction approaches to treat dynamic attribute prediction as independent sub-tasks rather than unified 4D world modeling, leading to inconsistent datasets and benchmarks.

*Pages: 1, 2*

### Dynamic Scene Reconstruction

The process of reconstructing scenes that contain moving objects or changes over time, as opposed to static scenes. This involves both camera pose estimation and scene geometry recovery in dynamic environments.

*Pages: 2*

### Structure-from-Motion

A classical computer vision technique for reconstructing 3D structures from 2D image sequences, typically applied to static scenes without dynamic elements.

*Pages: 2*

### Simultaneous Localization and Mapping

A SLAM approach that simultaneously estimates camera poses and builds a map of the environment, commonly used for static scene reconstruction.

*Pages: 2*

### Scene Flow

The problem of recovering the 3D motion vector field for every point on surfaces observed in a scene, representing the complete motion dynamics of a dynamic environment.

*Pages: 2*

### Optical Flow

The pattern of apparent motion of objects, surfaces, and edges in a visual scene caused by the relative motion between an observer and the scene, which is the perspective projection of scene flow onto the camera plane.

*Pages: 2*

### Feed-forward Multi-view Inference

A method that performs multi-view scene reconstruction without requiring iterative optimization, making it suitable for real-time applications by processing images independently.

*Pages: 2*

### Allocentric Coordinate Frame

A coordinate system where scene flow is represented relative to a fixed world frame rather than relative to the camera, enabling more general scene flow prediction.

*Pages: 2*

### Any4D

A feed-forward model that predicts camera poses, dense 3D motion (scene flow), and geometry (pointmaps) for dynamic scenes captured by multiple frames, providing a complete description of the scene.

*Pages: 2*

### log-space conversion

A mathematical transformation using the function flog(x) = (x/||x||) log(1 + ||x||) that converts quantities to log-space for numerical stability during training.

*Pages: 5, 6*

### pointmap loss

A loss function applied to composed geometric predictions using log-space conversion to measure differences between predicted and ground truth point maps.

*Pages: 5, 6*

### scale-invariant scene flow supervision

A technique for supervising scene flow predictions in a scale-invariant manner by calculating a static-dynamic motion mask and upweighting dynamic regions 10x more than static regions.

*Pages: 5, 6*

### static-dynamic motion mask

A binary mask calculated from ground truth scene flow that distinguishes between static and dynamic regions in a scene for more effective loss calculation.

*Pages: 5, 6*

### stop-gradient operation

A technique that prevents gradient flow through certain operations (denoted by sg) to avoid scale supervision from affecting other predicted quantities during training.

*Pages: 5, 6*

### combined loss function

The final loss function that combines multiple loss components including translation, rotation, rays, depth, scene flow, and mask losses for comprehensive training.

*Pages: 5, 6*

### Any4D

A unified feed-forward metric 4D reconstruction method that can handle dynamic scenes and object-centric reconstructions.

*Pages: 8, 9*

### Dynamic Scene Flow Domain

A domain of computer vision and graphics that deals with the reconstruction and modeling of scenes that change over time, including motion and deformation.

*Pages: 8, 9*

### BlendedMVS

A dataset used for training 4D reconstruction models, specifically designed for dynamic scene reconstruction with outdoor and object-centric scenes.

*Pages: 8, 9*

### Training Datasets

A combination of static and dynamic datasets with varying levels of supervision used for training. Includes datasets for geometric quantities and scene flow supervision.

*Pages: 9*

### Scene Flow Supervision

A type of supervision that requires diverse camera and scene motion for learning good scene flow, using datasets like Kubric, PointOdyssey, and Dynamic Replica.

*Pages: 9*

### Any4D Network

The neural network architecture being trained, initialized with the public MapAnything checkpoint, with specific components like doppler scene-flow encoder and scene-flow DPT decoder.

*Pages: 9*

### DINOv2 Image Encoder

A specific image encoder component within the network that is initialized and trained separately from the rest of the network with a different learning rate.

*Pages: 9*

### Scene-flow DPT Decoder

A decoder component for scene flow that is initialized and trained from scratch with a specific learning rate during training.

*Pages: 9*

### Multi-view Training

A training approach where multiple views are sampled from each dataset during gradient steps, with 4-view training being critical for generalizing with multi-view inference.

*Pages: 9*

### Benchmarking Setup

The evaluation methodology for different datasets including TAPVID-3D PStudio, DriveTrack, Dynamic-Replica, and LSF-Odyssey, with specific filtering and frame selection criteria.

*Pages: 9*

## Figures

- **Figure 1** (p.1): Any4D is a flexible feed-forward model capable of producing dense metric 4D reconstructions using N ...
- **Figure 2** (p.3): Any4D’s unified capabilities overcome major limitations of existing 4D reconstruction models....
- **Figure 2** (p.3): ). First,...
- **Figure 3** (p.4): Any4D predicts a factorized dense metric 4D reconstruction represented as a global metric scale, per...
- **Figure 3** (p.4): ). Conceptually, it can be...
- **Figure 4** (p.5): Any4D provides dense and precise motion estimation, where on the other hand, state-of-the-art baseli...
- **Figure 4** (p.6): Dense Scene Flow:...
- **Figure 5** (p.7): Scene motion parametrized as allocentric scene flow provides the cleanest 4D reconstructions. We fin...
- **Figure 5** (p.8): We find that directly predicting allocentric motion...

## Tables

- **Table 1** (p.7): Any4D showcases state-of-the-art sparse 3D point tracking, while providing dense motion predictions ...
- **Table 2** (p.7): Any4D achieves state-of-the-art dense scene flow estimation performance. We report end-point error (...
- **Table 3** (p.8): Any4D shows state-of-the-art video depth estimation...
- **Table 4** (p.8): Auxiliary inputs improve the 4D motion estima-...
- **Table 5** (p.8): Allocentric scene flow is the optimal output repre-...

