# Astra: General Interactive World Model with Autoregressive Denoising

**Authors:** Yixuan Zhu; Jiaqi Feng; Wenzhao Zheng; Yuan Gao; Xin Tao; Pengfei Wan; Jie Zhou; Jiwen Lu

**arXiv:** [2512.08931](https://arxiv.org/abs/2512.08931)

## Abstract

Recent advances in diffusion transformers have empowered video generation
models to generate high-quality video clips from texts or images. However, world
models with the ability to predict long-horizon futures from past observations
and actions remain underexplored, especially for general-purpose scenarios and
various forms of actions. To bridge this gap, we introduce Astra, an interactive
general world model that generates real-world futures for diverse scenarios (e.g.,
autonomous driving, robot grasping) with precise action interactions (e.g., camera
motion, robot action). We propose an autoregressive denoising architecture and
use temporal causal attention to aggregate past observations and support stream-
ing outputs. We use a noise-augmented history memory to avoid over-reliance
on past frames to balance responsiveness with temporal coherence. For precise
action control, we introduce an action-aware adapter that directly injects action
signals into the denoising process. We further develop a mixture of action ex-
perts that dynamically route heterogeneous action modalities, enhancing versatil-
ity across diverse real-world tasks such as exploration, manipulation, and camera
control. Astra achieves interactive, consistent, and general long-term video pre-
diction and supports various forms of interactions. Experiments across multiple
datasets demonstrate the improvements of Astra in fidelity, long-range prediction,
and action alignment over existing state-of-the-art world models.
†Project leader.

## Contents

- 1 Tsinghua University
- 1 INTRODUCTION
- 2 RELATED WORK
- 2 Autoregressive Denoising World Framework
- 3 PROPOSED APPROACH
- 2 Training
  - 3.1 PRELIMINARY: AUTOREGRESSIVE DENOISING MODEL
  - 3.2 INTERACTIVE WORLD MODELING VIA AUTOREGRESSIVE DENOISING
  - 3.3 HISTORY CONDITION WITH NOISY MEMORY
- 5 Expert 2
  - 3.4 MIXURE OF ACTION EXPERTS FOR DIVERSE SCENARIOS
- 4 EXPERIMENT
- 850 Sekai (Li et al., 2025a)
- 9978 Multi-Cam Video (Bai et al., 2025)
  - 0.691 MatrixGame (He et al., 2025)
  - 0.748 Yume (Mao et al., 2025)
  - 0.741 Astra (Ours)
  - 4.2 MAIN RESULTS
  - 4.3 ANALYSIS
  - 4.4 EXTENDED APPLICATIONS
  - 0.727 Astra (Ours)
- 5 CONCLUSION
- 10 REFERENCES
- 14 Table A: Quantitative action-alignment comparison.
  - 0.691 YUME (Mao et al., 2025)
  - 0.741 MatrixGame (He et al., 2025)
  - 0.748 NWM (Bar et al., 2025)
  - 0.635 Astra (ours)
  - 0.632 MatrixGame (He et al., 2025)
  - 0.653 Yume (Mao et al., 2025)
  - 0.628 Astra (Ours)

## Key Concepts

### Denoising diffusion models

A dominant paradigm in generative modeling that has become popular for its high fidelity and controllability. These models work by gradually removing noise from data during a training process.

*Pages: 2*

### Text-to-image synthesis

A task in generative modeling where text descriptions are used to generate corresponding images. This approach has been successfully implemented using diffusion models.

*Pages: 2*

### Video generation models

Models designed to generate videos from various inputs such as text or images. These models extend the capabilities of image generation techniques to the temporal domain.

*Pages: 2*

### Temporal domain

The dimension representing time in video generation, which allows models to generate sequences of frames over time rather than static images.

*Pages: 2*

### UNet architecture

A type of neural network architecture originally developed for image segmentation that has been adapted for use in diffusion models for both image and video generation.

*Pages: 2*

### Text-to-video diffusion models

Advanced diffusion models that generate high-quality, high-resolution videos from text descriptions, representing a significant advancement in video generation technology.

*Pages: 2*

### Astra framework

A video generation framework that uses an Action-Aware Flow Transformer (AFT) to inject action signals into the latent space through an ACT-Adapter, enabling interactive video generation.

*Pages: 4*

### Action-Aware Flow Transformer

A transformer-based architecture that injects action signals into the latent space to guide video generation, allowing for conditioning on action streams during inference.

*Pages: 4*

### ACT-Adapter

An adapter component that aligns action features through an encoder and adds them to each transformer block, enabling the injection of action signals into the latent space.

*Pages: 4*

### Next-chunk prediction

A training mechanism where the model learns to predict subsequent video chunks based on input flow data, enabling sequential video generation.

*Pages: 4*

### Flow matching

A training approach used in the framework where the model learns to match flow patterns between consecutive video chunks during the next-chunk prediction task.

*Pages: 4*

### Autoregressive video generation

A video generation process where video chunks are generated sequentially, with each chunk conditioned on previous chunks and action streams.

*Pages: 4*

### Latent space

The internal representation space where action signals are injected and video generation occurs, serving as the intermediate representation between input and output.

*Pages: 4*

### Action stream

Input data representing action information that conditions the video generation process, allowing for interactive video creation based on action signals.

*Pages: 4*

### Autoregressive denoising framework

A framework that combines autoregression's long-horizon modeling with diffusion's high-fidelity synthesis for video generation. It models video sequences as a product of conditional probabilities where each chunk is predicted based on previous chunks.

*Pages: 4*

### Flow matching

A training method used to train the flow model by estimating the clean direction from noisy interpolations of target chunks. It minimizes the difference between the model's prediction and the ground-truth velocity.

*Pages: 4*

### Noisy interpolation

A process where noisy versions of target chunks are created using a linear interpolation between a clean chunk and random noise, with the interpolation parameter t ∈ [0, 1].

*Pages: 4*

### Ground-truth velocity

The true velocity field that represents the clean direction in the flow matching process, which serves as the target for the model to learn and approximate.

*Pages: 4*

### Autoregressive generation

An iterative process where chunks are generated one by one, with each new chunk being predicted based on previously generated chunks. In inference, this process starts from noise and proceeds by denoising.

*Pages: 4*

### Interactive World Model

A model that extends beyond standard video prediction benchmarks to real-world applications, enabling action-driven exploration with visual fidelity and coherent dynamics.

*Pages: 8, 9*

### Action-driven Exploration

The capability of generating exploration sequences based on initial images and action sequences, maintaining visual fidelity, coherent dynamics, and accurate responsiveness to user-specified actions.

*Pages: 8, 9*

### Astra Model

A specific model implementation that demonstrates the capabilities of interactive world models, showing strong visual fidelity and accurate responsiveness to user actions.

*Pages: 8, 9*

### Ablation Studies

Experimental methodology used to assess the contribution of each component in Astra by removing specific elements and measuring performance differences.

*Pages: 8, 9*

### Visual Fidelity

The quality of generated sequences maintaining strong visual similarity to the input images and actions, ensuring realistic appearance.

*Pages: 8, 9*

### Motion Smoothness

The quality of motion transitions in generated sequences being smooth and coherent, which is evaluated in ablation studies.

*Pages: 8, 9*

### Cross Attention Adapter

A component in the model that helps in adapting cross-attention mechanisms for better performance, as shown in ablation studies.

*Pages: 8, 9*

### MoAE

A component in the model that contributes to the overall performance, as demonstrated by ablation studies showing its impact on various metrics.

*Pages: 8, 9*

### Action-alignment

A quantitative measure that evaluates how well generated camera motions align with commanded actions, using rotation and translation errors as metrics.

*Pages: 14, 15*

### Rotation Error

A metric that quantifies the discrepancy between generated camera rotation and the commanded rotation, with lower values indicating better alignment.

*Pages: 14, 15*

### Translation Error

A metric that quantifies the discrepancy between generated camera translation and the commanded translation, with lower values indicating better alignment.

*Pages: 14, 15*

### Instruction Following

A metric that evaluates how well the generated camera motions follow the given instructions, with higher values indicating better performance.

*Pages: 14, 15*

### Imaging Quality

A metric that assesses the visual quality of the generated camera motions, with higher values indicating better quality.

*Pages: 14, 15*

### Astra

A lightweight interactive world model that achieves strong action-conditioned performance with low parameter and compute overhead. It introduces only two small components: ACT-Adapter and MoAE, making it the most parameter-efficient among compared methods.

*Pages: 15, 16, 17, 18*

### Action-Free Guidance

A technique used during inference where a scale s is set for action-free guidance to control the model's behavior. In Astra, this scale is set to 3.0.

*Pages: 15, 16, 17, 18*

### MoAE

A lightweight module in Astra that consists of a linear router plus small MLP experts, with only one expert active per step. It enables multi-modal control support in Astra.

*Pages: 15, 16, 17, 18*

### ACT-Adapter

A single linear layer added after each self-attention block in Astra. It's one of the two small components that make Astra lightweight.

*Pages: 15, 16, 17, 18*

### Instruction Following

The ability of a model to generate video sequences that faithfully reflect specified action directions and types. It is measured through human evaluation in Astra.

*Pages: 15, 16, 17, 18*

### Temporal Coherence

The ability of a model to maintain consistent and logical progression of actions and events over long time horizons in generated videos.

*Pages: 15, 16, 17, 18*

### Camera Motion Estimation

The process of estimating camera poses in generated videos to measure how closely the camera motion matches ground-truth trajectories, using tools like MegaSaM.

*Pages: 15, 16, 17, 18*

### Action-Conditioned Performance

The capability of a model to generate outputs conditioned on specific actions, enabling precise interactive control as demonstrated by Astra.

*Pages: 15, 16, 17, 18*

### Long-Horizon Rollout

The ability to generate extended sequences of video frames that maintain consistency and coherence over longer time spans, achieved by Astra through noisy memory and input-packing techniques.

*Pages: 15, 16, 17, 18*

### Multi-Modal Control

Support for various input modalities including camera input, keyboard/mouse, and robot pose, providing more flexible and intuitive interaction compared to traditional methods.

*Pages: 15, 16, 17, 18*

### Human Evaluation

A method of assessing model performance through human inspection, particularly used for instruction following in Astra due to limitations of automated estimators.

*Pages: 15, 16, 17, 18*

### Parameter Efficiency

The characteristic of Astra being designed to add far fewer parameters than prior interactive world models, making it computationally efficient while maintaining strong performance.

*Pages: 15, 16, 17, 18*

## Figures

- **Figure 1** (p.1): Our Astra enables interactive and versatile world modeling across exploration, robotics,...
- **Figure 2** (p.2): This design preserves the high generative quality of diffusion models while enabling...
- **Figure 1** (p.2): , Astra achieves state-of-the-...
- **Figure 2** (p.3): Overview of the proposed Astra. Our autoregressive denoising world model generates...
- **Figure 3** (p.3): 3...
- **Figure 3** (p.4): The overall framework of Astra. The Action-Aware Flow Transformer (AFT) injects...
- **Figure 3** (p.5): , we introduce an action encoder to project actions into a feature space aligned...
- **Figure 4** (p.6): Mixture of Action Experts (MoAE). Action signals from diverse modalities are projected...
- **Figure 3** (p.6): ). This design offers two advantages....
- **Figure 4** (p.6): , each action modality—continuous camera pose acam, robot pose arob,...
- **Figure 5** (p.7): ),...
- **Figure 5** (p.8): Qualitative results on action-driven real-world exploration. Starting from a single...
- **Figure 6** (p.8): ), making it particularly suitable for...
- **Figure 6** (p.9): Qualitative comparisons on action-driven real-world exploration. Given the initial im-...
- **Figure 7** (p.9): This versatility arises from unifying temporal con-...
- **Figure 8** (p.9): , we illustrate the ability to handle multi-agent scenarios...
- **Figure 7** (p.10): Extended applications of Astra. Our framework handles diverse scenarios: (a) au-...
- **Figure 8** (p.10): Multi-agent interaction of Astra. Given a specified action sequence, Astra generates...
- **Figure 7** (p.16): , our Astra framework is designed to generalize across a wide range of in-...

## Tables

- **Table 1** (p.6): Specifically,...
- **Table 1** (p.7): Datasets used in experiments, along with their actions and sample sizes. For each dataset,...
- **Table 2** (p.7): Quantitative comparison of different models. Astra demonstrates superior visual quality...
- **Table 2** (p.7): , our method consistently out-...
- **Table 3** (p.8): , cross attn. adapter) demonstrate that ACT-Adapter achieves stronger action-...
- **Table 3** (p.8): (w/o AFG), this mechanism proves particularly...
- **Table 3** (p.8): (w/o noise), the model achieves stronger responsiveness to abrupt or unexpected actions,...
- **Table 3** (p.8): (w/o MoAE), trained only on camera-action data since it cannot process...
- **Table 3** (p.9): Ablation studies. We assess the contribution of each component in Astra, ensuring all...
- **Table 1** (p.14): , with detailed descriptions reported as follows:...

