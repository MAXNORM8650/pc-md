# Research Papers: one step generative models with teacher model

Updated: 2026-01-01 22:43
Total: 188 papers

---

## 1. Towards Training One-Step Diffusion Models Without Distillation

**Authors:** Mingtian Zhang, Wenlin Chen, Jiajun He, Zijing Ou, José Miguel Hernández-Lobato

**Year:** 2026 | **Venue:** ICLR 2026 | **Citations:** N/A | **Score:** 0.627

> Recent advances in training one-step diffusion models typically follow a two-stage pipeline: first training a teacher diffusion model and then distilling it into a one-step student model. This process often depends on both the teacher’s score function for supervision and its weights for initializing the student model. In this paper, we explore whether one-step diffusion models can be trained direc...

---

## 2. MFSR: MeanFlow Distillation for One Step Real-World Image Super Resolution

**Authors:** Ruiqing Wang, Yuanzhi Zhu, Kai Zhang, Hanshu Yan, Shilin Lu

**Year:** 2026 | **Venue:** ICLR 2026 | **Citations:** N/A | **Score:** 0.570

> Diffusion- and flow-based models have advanced real-world image super-resolution (Real-ISR), but their multi-step sampling makes inference slow and hard to deploy. One-step distillation alleviates the cost, yet often degrades restoration quality and removes the option to refine with more steps. We present Mean Flows for Super-Resolution (MFSR), a new distillation framework that produces photoreali...

---

## 3. Progressive Multistep Data-free Diffusion Distillation

**Authors:** Ngoc Duy Tran, Kien Do, Duc Thanh Nguyen, Truyen Tran

**Year:** 2026 | **Venue:** ICLR 2026 | **Citations:** N/A | **Score:** 0.550

> While one-step distillation achieves strong single-step generation, these methods are not inherently flexible for multi-step sampling. Efforts to adapt them beyond one step frequently lead to reliance on training data, poor generation quality at early intermediate steps, and significant computational demands. To overcome these limitations, we propose Progressive Multi-step Diffusion Distillation (...

---

## 4. Adversarial Score identity Distillation: Rapidly Surpassing the Teacher in One Step

**Authors:** Mingyuan Zhou, Huangjie Zheng, Yi Gu, Zhendong Wang, Hai Huang

**Year:** 2025 | **Venue:** ICLR 2025 | **Citations:** N/A | **Score:** 0.550

[PDF](https://openreview.net/pdf?id=lS2SGfWizd) | > Score identity Distillation (SiD) is a data-free method that has achieved state-of-the-art performance in image generation by leveraging only a pretrained diffusion model, without requiring any training data. However, the ultimate performance of SiD is constrained by the accuracy with which the pretrained model captures the true data scores at different stages of the diffusion process. In this pap...

---

## 5. FS-DFM: Fast and Accurate Long Text Generation with Few-Step Diffusion Language Models

**Authors:** 

**Year:** 2026 | **Venue:** ICLR 2026 | **Citations:** N/A | **Score:** 0.544

> Autoregressive language models (ARMs) deliver strong likelihoods, but are inherently serial: they generate one token per forward pass, which limits throughput and inflates latency for long sequences. Diffusion Language Models (DLMs) parallelize across positions and thus appear promising for language generation, yet standard discrete diffusion typically needs hundreds to thousands of model evaluati...

---

## 6. One Step Diffusion-based Super-Resolution with Time-Aware Distillation

**Authors:** Xiao He, Huaao Tang, Zhijun Tu, Junchao Zhang, Kun Cheng

**Year:** 2025 | **Venue:** ICLR 2025 | **Citations:** N/A | **Score:** 0.538

> Diffusion-based image super-resolution (SR) methods have shown promise in reconstructing high-resolution images with fine details from low-resolution counterparts. However, these approaches typically require tens or even hundreds of iterative samplings, resulting in significant latency. Recently, techniques have been devised to enhance the sampling efficiency of diffusion-based SR models via knowl...

---

## 7. See Further When Clear: Adaptive Generative Modeling with Curriculum Consistency Model

**Authors:** Yunpeng Liu, Boxiao Liu, Yi Zhang, Xingzhong Hou, Guanglu Song

**Year:** 2025 | **Venue:** ICLR 2025 | **Citations:** N/A | **Score:** 0.536

> Significant advances have been made in the sampling efficiency of diffusion models, driven by Consistency Distillation (CD), which trains a student model to mimic the output of a teacher model at an earlier timestep. However, we found that the learning complexity of the student model varies significantly across different timesteps, leading to suboptimal performance in consistency models.
To addres...

---

## 8. Distillation Enhanced Generative Retrieval

**Authors:** Yongqi Li, Zhen Zhang, Wenjie Wang, Liqiang Nie, Wenjie Li

**Year:** 2024 | **Venue:** ACL 2024 | **Citations:** N/A | **Score:** 0.526

[PDF](https://aclanthology.org/2024.findings-acl.662.pdf) | > Generative retrieval is a promising new paradigm in text retrieval that generates identifier strings of relevant passages as the retrieval target. This paradigm leverages powerful generative language models, distinct from traditional sparse or dense retrieval methods. In this work, we identify a viable direction to further enhance generative retrieval via distillation and propose a feasible framew...

---

## 9. You Only Need One Step: Fast Super-Resolution with Stable Diffusion via Scale Distillation

**Authors:** Mehdi Noroozi*, Isma Hadji*, Brais Martinez*, Adrian Bulat*, Georgios Tzimiropoulos*

**Year:** 2024 | **Venue:** ECCV 2024 | **Citations:** N/A | **Score:** 0.523

[PDF](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/04181.pdf) | > "In this paper, we introduce YONOS-SR, a novel stable diffusion based approach for image super-resolution that yields state-of-the-art results using only a single DDIM step. Specifically, we propose a novel scale distillation approach to train our SR model. Instead of directly training our SR model on the scale factor of interest, we start by training a teacher model on a smaller magnification sca...

---

## 10. Unleashing the Power of One-Step Diffusion based Image Super-Resolution via a Large-Scale Diffusion Discriminator

**Authors:** Jianze Li, Jiezhang Cao, Zichen Zou, Xiongfei Su, Xin Yuan

**Year:** 2025 | **Venue:** NIPS 2025 | **Citations:** N/A | **Score:** 0.514

> Diffusion models have demonstrated excellent performance for real-world image super-resolution (Real-ISR), albeit at high computational costs. Most existing methods are trying to derive one-step diffusion models from multi-step counterparts through knowledge distillation (KD) or variational score distillation (VSD). However, these methods are limited by the capabilities of the teacher model, espec...

---

## 11. Generative Diffusion Prior Distillation for Long-Context Knowledge Transfer

**Authors:** 

**Year:** 2026 | **Venue:** ICLR 2026 | **Citations:** N/A | **Score:** 0.513

> While traditional time-series classifiers assume full sequences at inference, practical constraints (latency and cost) often limit inputs to partial prefixes. The absence of class-discriminative patterns in partial data can significantly hinder a classifier’s ability to generalize. This work uses knowledge distillation (KD) to equip partial time series classifiers with the generalization ability o...

---

## 12. SWITCH: Studying with Teacher for Knowledge Distillation of Large Language Models

**Authors:** Jahyun Koo, Yerin Hwang, Yongil Kim, Taegwan Kang, Hyunkyung Bae

**Year:** 2025 | **Venue:** NAACL 2025 | **Citations:** N/A | **Score:** 0.513

[PDF](https://aclanthology.org/2025.findings-naacl.206.pdf) | > Despite the success of Large Language Models (LLMs), they still face challenges related to high inference costs and memory requirements. To address these issues, Knowledge Distillation (KD) has emerged as a popular method for model compression, with the use of student-generated outputs (SGOs) as training data being particularly notable for reducing the mismatch between training and inference. Howe...

---

## 13. Generative Pre-training for Speech with Flow Matching

**Authors:** Alexander H. Liu, Matthew Le, Apoorv Vyas, Bowen Shi, Andros Tjandra

**Year:** 2024 | **Venue:** ICLR 2024 | **Citations:** N/A | **Score:** 0.511

[PDF](https://openreview.net/pdf?id=KpoQSgxbKH) | > Generative models have gained more and more attention in recent years for their remarkable success in tasks that required estimating and sampling data distribution to generate high-fidelity synthetic data. In speech, text-to-speech synthesis and neural vocoder are good examples where generative models have shined. While generative models have been applied to different applications in speech, there...

---

## 14. Direct Preference Optimization With Unobserved Preference Heterogeneity

**Authors:** Keertana Chidambaram, Karthik Vinay Seetharaman, Vasilis Syrgkanis

**Year:** 2025 | **Venue:** ICLR 2025 | **Citations:** N/A | **Score:** 0.505

> RLHF has emerged as a pivotal step in aligning language models with human objectives and values. It typically involves learning a reward model from human preference data and then using reinforcement learning to update the generative model accordingly. Conversely, Direct Preference Optimization (DPO) directly optimizes the generative model with preference data, skipping reinforcement learning. Howe...

---

## 15. One step further with Monte-Carlo sampler to guide diffusion better

**Authors:** 

**Year:** 2026 | **Venue:** ICLR 2026 | **Citations:** N/A | **Score:** 0.505

> Stochastic differential equation (SDE)-based generative models have achieved
substantial progress in conditional generation via training-free differentiable
loss-guided approaches. However, existing methodologies utilizing posterior sam-
pling typically confront a substantial estimation error, which results in inaccurate
gradients for guidance and leading to inconsistent generation results. To mit...

---

## 16. uDistil-Whisper: Label-Free Data Filtering for Knowledge Distillation in Low-Data Regimes

**Authors:** Abdul Waheed, Karima Kadaoui, Bhiksha Raj, Muhammad Abdul-Mageed

**Year:** 2025 | **Venue:** NAACL 2025 | **Citations:** N/A | **Score:** 0.504

[PDF](https://aclanthology.org/2025.naacl-long.296.pdf) | > Recent work on distilling Whisper’s knowledge into small models using pseudo-labels shows promising performance while reducing the size by up to 50%. This results in small, efficient, and dedicated models. However, a critical step of distillation using pseudo-labels involves filtering high-quality predictions and using only those during training. This step requires ground truth labels to compare w...

---

## 17. Adversarial Diffusion Distillation

**Authors:** Axel Sauer*, Dominik Lorenz, Andreas Blattmann, Robin Rombach

**Year:** 2024 | **Venue:** ECCV 2024 | **Citations:** N/A | **Score:** 0.503

[PDF](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/11557.pdf) | > "We introduce Adversarial Diffusion Distillation (ADD), a novel training approach that efficiently samples large-scale foundational image diffusion models in just 1–4 steps while maintaining high image quality. We use score distillation to leverage large-scale off-the-shelf image diffusion models as a teacher signal in combination with an adversarial loss to ensure high image fidelity even in the ...

---

## 18. Multi-Student Diffusion Distillation for Better One-Step Generators

**Authors:** Yanke Song, Jonathan Lorraine, Weili Nie, Karsten Kreis, James Lucas

**Year:** 2025 | **Venue:** ICLR 2025 | **Citations:** N/A | **Score:** 0.497

> Diffusion models achieve high-quality sample generation at the cost of a lengthy multistep inference procedure. To overcome this, diffusion distillation techniques produce student generators capable of matching or surpassing the teacher in a single step. However, the student model’s inference speed is limited by the size of the teacher architecture, preventing real-time generation for computationa...

---

## 19. Improved Techniques for Training Consistency Models

**Authors:** Yang Song, Prafulla Dhariwal

**Year:** 2024 | **Venue:** ICLR 2024 | **Citations:** N/A | **Score:** 0.495

[PDF](https://openreview.net/pdf?id=WNzy9bRDvG) | > Consistency models are a nascent family of generative models that can sample high quality data in one step without the need for adversarial training. Current consistency models achieve optimal sample quality by distilling from pre-trained diffusion models and employing learned metrics such as LPIPS. However, distillation limits the quality of consistency models to that of the pre-trained diffusion...

---

## 20. Reducing Spatial Fitting Error in Distillation of Denoising Diffusion Models

**Authors:** Shengzhe Zhou, Zejian Li, Shengyuan Zhang, Lefan Hou, Changyuan Yang

**Year:** 2024 | **Venue:** AAAI 2024 | **Citations:** N/A | **Score:** 0.495

[PDF](https://ojs.aaai.org/index.php/AAAI/article/view/28602/29171) | > Denoising Diffusion models have exhibited remarkable capabilities in image generation. However, generating high-quality samples requires a large number of iterations. Knowledge distillation for diffusion models is an effective method to address this limitation with a shortened sampling process but causes degraded generative quality. Based on our analysis with bias-variance decomposition and experi...

---

## 21. ICLEF: In-Context Learning with Expert Feedback for Explainable Style Transfer

**Authors:** Arkadiy Saakyan, Smaranda Muresan

**Year:** 2024 | **Venue:** ACL 2024 | **Citations:** N/A | **Score:** 0.491

[PDF](https://aclanthology.org/2024.acl-long.854.pdf) | > While state-of-the-art large language models (LLMs) can excel at adapting text from one style to another, current work does not address the explainability of style transfer models. Recent work has explored generating textual explanations from larger teacher models and distilling them into smaller student models. One challenge with such approach is that LLM outputs may contain errors that require e...

---

## 22. A Critical Evaluation of AI Feedback for Aligning Large Language Models

**Authors:** Archit Sharma, Sedrick Keh, Eric Mitchell, Chelsea Finn, Kushal Arora

**Year:** 2024 | **Venue:** NIPS 2024 | **Citations:** N/A | **Score:** 0.489

[PDF](https://openreview.net/pdf?id=FZQYfmsmX9) | > Learning from AI feedback (LAIF) is a popular paradigm for improving the instruction-following abilities of powerful pre-trained language models. LAIF first performs supervised fine-tuning (SFT) using demonstrations from a teacher model and then further fine-tunes the model with reinforcement learning (RL) or direct preference optimization (DPO), using feedback from a critic model. While recent po...

---

## 23. Knowledge Distillation with Perturbed Loss: From a Vanilla Teacher to a Proxy Teacher

**Authors:** Rongzhi Zhang, Jiaming Shen, Tianqi Liu, Jialu Liu, Michael Bendersky

**Year:** 2024 | **Venue:** KDD 2024 | **Citations:** N/A | **Score:** 0.484

> ...

---

## 24. AutoDisc: Automatic Distillation Schedule for Large Language Model Compression

**Authors:** Chen Zhang, Yang Yang, Qifan Wang, Jiahao Liu, Jingang Wang

**Year:** 2023 | **Venue:** ICLR 2023 | **Citations:** N/A | **Score:** 0.482

> Driven by the teacher-student paradigm, knowledge distillation is one of the de facto ways for language model compression. Recent studies have uncovered that conventional distillation is less effective when facing a large capacity gap between the teacher and the student, and introduced teacher assistant-based distillation to bridge the gap. As a connection, the scale and the performance of the tea...

---

## 25. Unified Reinforcement and Imitation Learning for Vision-Language Models

**Authors:** Byung-Kwan Lee, Ryo Hachiuma, Yong Man Ro, Yu-Chiang Frank Wang, Yueh-Hua Wu

**Year:** 2025 | **Venue:** NIPS 2025 | **Citations:** N/A | **Score:** 0.482

> Vision-Language Models (VLMs) have achieved remarkable progress, yet their large scale often renders them impractical for resource-constrained environments. This paper introduces Unified Reinforcement and Imitation Learning (RIL), a novel and efficient training algorithm designed to create powerful, lightweight VLMs. RIL distinctively combines the strengths of reinforcement learning with adversari...

---

## 26. Let’s Fuse Step by Step: A Generative Fusion Decoding Algorithm with LLMs for Robust and Instruction-Aware ASR and OCR

**Authors:** Chan-Jan Hsu, Yi-Chang Chen, Feng-Ting Liao, Pei-Chen Ho, Yu-Hsiang Wang

**Year:** 2025 | **Venue:** ACL 2025 | **Citations:** N/A | **Score:** 0.480

[PDF](https://aclanthology.org/2025.findings-acl.1281.pdf) | > We introduce “Generative Fusion Decoding” (GFD), a novel shallow fusion framework, utilized to integrate large language models(LLMs) into cross-modal text recognition systems inlculding automatic speech recognition (ASR) and optical character recognition (OCR). We derive the formulas necessary to enable GFD to operate across mismatched token spaces of different models by calculating likelihood at ...

---

## 27. How to Train the Teacher Model for Effective Knowledge Distillation

**Authors:** Shayan Mohajer Hamidi*, Xizhen Deng, Renhao Tan, Linfeng Ye, Ahmed Hussein Salamah

**Year:** 2024 | **Venue:** ECCV 2024 | **Citations:** N/A | **Score:** 0.480

[PDF](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/12363.pdf) | > "Recently, it was shown that the role of the teacher in knowledge distillation (KD) is to provide the student with an estimate of the true Bayes conditional probability density (BCPD). Notably, the new findings propose that the student’s error rate can be upper-bounded by the mean squared error (MSE) between the teacher’s output and BCPD. Consequently, to enhance KD efficacy, the teacher should be...

---

## 28. Seeking Flat Minima with Mean Teacher on Semi- and Weakly-Supervised Domain Generalization for Object Detection

**Authors:** Ryosuke Furuta, Yoichi Sato

**Year:** 2025 | **Venue:** ICLR 2025 | **Citations:** N/A | **Score:** 0.478

> Object detectors do not work well when domains largely differ between training and testing data. To overcome this domain gap in object detection without requiring expensive annotations, we consider two problem settings: semi-supervised domain generalizable object detection (SS-DGOD) and weakly-supervised DGOD (WS-DGOD). In contrast to the conventional domain generalization for object detection tha...

---

## 29. PaGoDA: Progressive Growing of a One-Step Generator from a Low-Resolution Diffusion Teacher

**Authors:** Dongjun Kim, Chieh-Hsin Lai, Wei-Hsiang Liao, Yuhta Takida, Naoki Murata

**Year:** 2024 | **Venue:** NIPS 2024 | **Citations:** N/A | **Score:** 0.476

[PDF](https://openreview.net/pdf?id=h5zYGF68KH) | > The diffusion model performs remarkable in generating high-dimensional content but is computationally intensive, especially during training. We propose Progressive Growing of Diffusion Autoencoder (PaGoDA), a novel pipeline that reduces the training costs through three stages: training diffusion on downsampled data, distilling the pretrained diffusion, and progressive super-resolution. With the pr...

---

## 30. PFGuard: A Generative Framework with Privacy and Fairness Safeguards

**Authors:** Soyeon Kim, Yuji Roh, Geon Heo, Steven Euijong Whang

**Year:** 2025 | **Venue:** ICLR 2025 | **Citations:** N/A | **Score:** 0.476

[PDF](https://openreview.net/pdf?id=8rbkePAapb) | > Generative models must ensure both privacy and fairness for Trustworthy AI. While these goals have been pursued separately, recent studies propose to combine existing privacy and fairness techniques to achieve both goals. However, naively combining these techniques can be insufficient due to privacy-fairness conflicts, where a sample in a minority group may be represented in ways that support fair...

---

## 31. DiffusionBERT: Improving Generative Masked Language Models with Diffusion Models

**Authors:** Zhengfu He, Tianxiang Sun, Qiong Tang, Kuanning Wang, Xuanjing Huang

**Year:** 2023 | **Venue:** ACL 2023 | **Citations:** N/A | **Score:** 0.473

[PDF](https://aclanthology.org/2023.acl-long.248.pdf) | > We present DiffusionBERT, a new generative masked language model based on discrete dif- fusion models. Diffusion models and many pre- trained language models have a shared training objective, i.e., denoising, making it possible to combine the two powerful models and enjoy the best of both worlds. On the one hand, dif- fusion models offer a promising training strat- egy that helps improve the gener...

---

## 32. EM Distillation for One-step Diffusion Models

**Authors:** Sirui Xie, Zhisheng Xiao, Diederik P Kingma, Tingbo Hou, Ying Nian Wu

**Year:** 2024 | **Venue:** NIPS 2024 | **Citations:** N/A | **Score:** 0.466

[PDF](https://openreview.net/pdf?id=rafVvthuxD) | > While diffusion models  can learn complex distributions, sampling requires a computationally expensive iterative process.  Existing distillation methods enable efficient sampling, but have notable limitations, such as performance degradation with very few sampling steps, reliance on training data access, or mode-seeking optimization that may fail to capture the full distribution. We propose EM Dis...

---

## 33. Hot PATE: Private Aggregation of Distributions for Diverse Tasks

**Authors:** Edith Cohen, Xin Lyu, Jelani Nelson, Tamas Sarlos, Uri Stemmer

**Year:** 2024 | **Venue:** ICLR 2024 | **Citations:** N/A | **Score:** 0.465

> The Private Aggregation of Teacher Ensembles (PATE) framework~\cite{PapernotAEGT:ICLR2017} is a versatile approach to privacy-preserving machine learning. In PATE, teacher models are trained on distinct portions of sensitive data, and their predictions are privately aggregated to label new training examples for a student model.
 Until now, PATE has primarily been explored with classification-like ...

---

## 34. Should Under-parameterized Student Networks Copy or Average Teacher Weights?

**Authors:** Berfin Simsek, Amire Bendjeddou, Wulfram Gerstner, Johanni Brea

**Year:** 2023 | **Venue:** NIPS 2023 | **Citations:** N/A | **Score:** 0.462

[PDF](https://openreview.net/pdf?id=MG0mYskXN2) | > Any continuous function $f^*$ can be approximated arbitrarily well by a neural network with sufficiently many neurons $k$. We consider the case when $f^*$ itself is a neural network with one hidden layer and $k$ neurons. Approximating $f^*$ with a neural network with $n< k$ neurons can thus be seen as fitting an under-parameterized "student" network with $n$ neurons to a "teacher" network with $k$...

---

## 35. Stare at What You See: Masked Image Modeling Without Reconstruction

**Authors:** Hongwei Xue, Peng Gao, Hongyang Li, Yu Qiao, Hao Sun

**Year:** 2023 | **Venue:** CVPR 2023 | **Citations:** N/A | **Score:** 0.462

[PDF](https://openaccess.thecvf.com/content/CVPR2023/papers/Xue_Stare_at_What_You_See_Masked_Image_Modeling_Without_Reconstruction_CVPR_2023_paper.pdf) | > Masked Autoencoders (MAE) have been prevailing paradigms for large-scale vision representation pre-training. By reconstructing masked image patches from a small portion of visible image regions, MAE forces the model to infer semantic correlation within an image. Recently, some approaches apply semantic-rich teacher models to extract image features as the reconstruction target, leading to better pe...

---

## 36. Tackling the Generative Learning Trilemma with Denoising Diffusion GANs

**Authors:** Zhisheng Xiao, Karsten Kreis, Arash Vahdat

**Year:** 2022 | **Venue:** ICLR 2022 | **Citations:** N/A | **Score:** 0.455

[PDF](https://openreview.net/pdf?id=JprM0p-q0Co) | > A wide variety of deep generative models has been developed in the past decade. Yet, these models often struggle with simultaneously addressing three key requirements including: high sample quality, mode coverage, and fast sampling. We call the challenge imposed by these requirements the generative learning trilemma, as the existing models often trade some of them for others. Particularly, denoisi...

---

## 37. Multi-Scale Distillation from Multiple Graph Neural Networks

**Authors:** Chunhai Zhang, Jie Liu, Kai Dang, Wenzheng Zhang

**Year:** 2022 | **Venue:** AAAI 2022 | **Citations:** N/A | **Score:** 0.444

[PDF](https://cdn.aaai.org/ojs/20354/20354-13-24367-1-2-20220628.pdf) | > Knowledge Distillation (KD), which is an effective model compression and acceleration technique, has been successfully applied to graph neural networks (GNNs) recently. Existing approaches utilize a single GNN model as the teacher to distill knowledge. However, we notice that GNN models with different number of layers demonstrate different classification abilities on nodes with different degrees. ...

---

## 38. Generative Fairness Teaching

**Authors:** Rongmei Lin, Hanjun Dai, Li Xiong, Wei Wei

**Year:** 2021 | **Venue:** ICLR 2021 | **Citations:** N/A | **Score:** 0.441

> Increasing evidences has shown that data biases towards sensitive features such as gender or race are often inherited or even amplified by machine learning models. Recent advancements in fairness mitigate such biases by adjusting the predictions across sensitive groups during the training. Such a correction, however, can only take advantage of samples in a fixed dataset, which usually has limited ...

---

## 39. Misspecified Phase Retrieval with Generative Priors

**Authors:** Zhaoqiang Liu, Xinshao Wang, Jiulong Liu

**Year:** 2022 | **Venue:** NIPS 2022 | **Citations:** N/A | **Score:** 0.438

[PDF](https://openreview.net/pdf?id=--aQNMdJc9x) | > In this paper, we study phase retrieval under model misspecification and generative priors. In particular, we aim to estimate an $n$-dimensional signal $\mathbf{x}$ from $m$ i.i.d.~realizations of the single index model $y = f(\mathbf{a}^T\mathbf{x})$, where $f$ is an unknown and possibly random nonlinear link function and $\mathbf{a} \in \mathbb{R}^n$ is a standard Gaussian vector. We make the as...

---

## 40. High-dimensional Asymptotics of Feature Learning: How One Gradient Step Improves the Representation

**Authors:** Jimmy Ba, Murat A Erdogdu, Taiji Suzuki, Zhichao Wang, Denny Wu

**Year:** 2022 | **Venue:** NIPS 2022 | **Citations:** N/A | **Score:** 0.438

[PDF](https://openreview.net/pdf?id=akddwRG6EGi) | > We study the first gradient descent step on the first-layer parameters $\boldsymbol{W}$ in a two-layer neural network: $f(\boldsymbol{x}) = \frac{1}{\sqrt{N}}\boldsymbol{a}^\top\sigma(\boldsymbol{W}^\top\boldsymbol{x})$, where $\boldsymbol{W}\in\mathbb{R}^{d\times N}, \boldsymbol{a}\in\mathbb{R}^{N}$ are randomly initialized, and the training objective is the empirical MSE loss: $\frac{1}{n}\sum_{...

---

## 41. MixTeacher: Mining Promising Labels With Mixed Scale Teacher for Semi-Supervised Object Detection

**Authors:** Liang Liu, Boshen Zhang, Jiangning Zhang, Wuhao Zhang, Zhenye Gan

**Year:** 2023 | **Venue:** CVPR 2023 | **Citations:** N/A | **Score:** 0.437

[PDF](https://openaccess.thecvf.com/content/CVPR2023/papers/Liu_MixTeacher_Mining_Promising_Labels_With_Mixed_Scale_Teacher_for_Semi-Supervised_CVPR_2023_paper.pdf) | > Scale variation across object instances is one of the key challenges in object detection. Although modern detection models have achieved remarkable progress in dealing with the scale variation, it still brings trouble in the semi-supervised case. Most existing semi-supervised object detection methods rely on strict conditions to filter out high-quality pseudo labels from the network predictions. H...

---

## 42. OSOA: One-Shot Online Adaptation of Deep Generative Models for Lossless Compression

**Authors:** Chen Zhang, Shifeng Zhang, Fabio Maria Carlucci, Zhenguo Li

**Year:** 2021 | **Venue:** NIPS 2021 | **Citations:** N/A | **Score:** 0.423

[PDF](https://openreview.net/pdf?id=Me-tuhUjhKK) | > Explicit deep generative models (DGMs), e.g., VAEs and Normalizing Flows, have shown to offer an effective data modelling alternative for lossless compression. 
However, DGMs themselves normally require large storage space and thus contaminate the advantage brought by accurate data density estimation.
To eliminate the requirement of saving separate models for different target datasets, we propose ...

---

## 43. NewsBERT: Distilling Pre-trained Language Model for Intelligent News Application

**Authors:** Chuhan Wu, Fangzhao Wu, Yang Yu, Tao Qi, Yongfeng Huang

**Year:** 2021 | **Venue:** EMNLP 2021 | **Citations:** N/A | **Score:** 0.418

[PDF](https://aclanthology.org/2021.findings-emnlp.280.pdf) | > Pre-trained language models (PLMs) like BERT have made great progress in NLP. News articles usually contain rich textual information, and PLMs have the potentials to enhance news text modeling for various intelligent news applications like news recommendation and retrieval. However, most existing PLMs are in huge size with hundreds of millions of parameters. Many online news applications need to s...

---

## 44. G-PATE: Scalable Differentially Private Data Generator via Private Aggregation of Teacher Discriminators

**Authors:** Yunhui Long, Boxin Wang, Zhuolin Yang, Bhavya Kailkhura, Aston Zhang

**Year:** 2021 | **Venue:** NIPS 2021 | **Citations:** N/A | **Score:** 0.417

[PDF](https://openreview.net/pdf?id=_CmrI7UrmCl) | > Recent advances in machine learning have largely benefited from the massive accessible training data. However, large-scale data sharing has raised great privacy concerns. In this work, we propose a novel privacy-preserving data Generative model based on the PATE framework (G-PATE), aiming to train a scalable differentially private data generator that preserves high generated data utility. Our appr...

---

## 45. Near-Zero-Cost Differentially Private Deep Learning with Teacher Ensembles

**Authors:** Lichao Sun, Yingbo Zhou, Jia Li, Richard Socher, Philip S. Yu

**Year:** 2020 | **Venue:** ICLR 2020 | **Citations:** N/A | **Score:** 0.412

> Ensuring the privacy of sensitive data used to train modern machine learning models is of paramount importance in many areas of practice. One approach to study these concerns is through the lens of differential privacy. In this framework, privacy guarantees are generally obtained by perturbing models in such a way that specifics of data used to train the model are made ambiguous. A particular inst...

---

## 46. P-KDGAN: Progressive Knowledge Distillation with GANs for One-class Novelty Detection

**Authors:** Zhiwei Zhang, Shifeng Chen, Lei Sun

**Year:** 2020 | **Venue:** IJCAI 2020 | **Citations:** N/A | **Score:** 0.406

[PDF](https://www.ijcai.org/proceedings/2020/0448.pdf) | > One-class novelty detection is to identify anomalous instances that do not conform to the expected normal instances. In this paper, the Generative Adversarial Networks (GANs) based on encoder-decoder-encoder pipeline are used for detection and achieve state-of-the-art performance. However, deep neural networks are too over-parameterized to deploy on resource-limited devices. Therefore, Progressive...

---

## 47. TIDOT: A Teacher Imitation Learning Approach for Domain Adaptation with Optimal Transport

**Authors:** Tuan Nguyen, Trung Le, Nhan Dam, Quan Hung Tran, Truyen Nguyen

**Year:** 2021 | **Venue:** IJCAI 2021 | **Citations:** N/A | **Score:** 0.404

[PDF](https://www.ijcai.org/proceedings/2021/0394.pdf) | > Using the principle of imitation learning and the theory of optimal transport we propose in this paper a novel model for unsupervised domain adaptation named Teacher Imitation Domain Adaptation with Optimal Transport (TIDOT). Our model includes two cooperative agents: a teacher and a student. The former agent is trained to be an expert on labeled data in the source domain, whilst the latter one ai...

---

## 48. One Generation Knowledge Distillation by Utilizing Peer Samples

**Authors:** Xingjian Li, Haozhe An, Haoyi Xiong, Jun Huan, Dejing Dou

**Year:** 2020 | **Venue:** ICLR 2020 | **Citations:** N/A | **Score:** 0.401

> Knowledge Distillation (KD) is a widely used technique in recent deep learning research to obtain small and simple models whose performance is on a par with their large and complex counterparts. Standard Knowledge Distillation tends to be time-consuming because of the training time spent to obtain a teacher model that would then provide guidance for the student model. It might be possible to cut s...

---

## 49. EchoDistill: Bidirectional Concept Distillation for One-Step Diffusion Personalization

**Authors:** Yixiong Yang, Tao Wu, Shiqi Yang

**Year:** 2025 | **Venue:** arXiv (Cornell University) | **Citations:** N/A | **Score:** 0.394

[PDF](https://arxiv.org/pdf/2510.20512) | [DOI](https://doi.org/10.48550/arxiv.2510.20512)

> Recent advances in accelerating text-to-image (T2I) diffusion models have enabled the synthesis of high-fidelity images even in a single step. However, personalizing these models to incorporate novel concepts remains a challenge due to the limited capacity of one-step models to capture new concept distributions effectively. We propose a bidirectional concept distillation framework, EchoDistill, to...

---

## 50. Mean Flows for One-step Generative Modeling

**Authors:** Zhengyang Geng, Mingyang Deng, Xingjian Bai, J. Z. Kolter, Kaiming He

**Year:** 2025 | **Venue:** arXiv.org | **Citations:** 134 | **Score:** 0.394

[DOI](https://doi.org/10.48550/arXiv.2505.13447)

> We propose a principled and effective framework for one-step generative modeling. We introduce the notion of average velocity to characterize flow fields, in contrast to instantaneous velocity modeled by Flow Matching methods. A well-defined identity between average and instantaneous velocities is derived and used to guide neural network training. Our method, termed the MeanFlow model, is self-con...

---

## 51. Revisiting Diffusion Models: From Generative Pre-training to One-Step Generation

**Authors:** Bowen Zheng, Tianming Yang

**Year:** 2025 | **Venue:** International Conference on Machine Learning | **Citations:** 3 | **Score:** 0.385

[DOI](https://doi.org/10.48550/arXiv.2506.09376)

> Diffusion distillation is a widely used technique to reduce the sampling cost of diffusion models, yet it often requires extensive training, and the student performance tends to be degraded. Recent studies show that incorporating a GAN objective may alleviate these issues, yet the underlying mechanism remains unclear. In this work, we first identify a key limitation of distillation: mismatched ste...

---

## 52. One Step Diffusion via Shortcut Models

**Authors:** Kevin Frans, Danijar Hafner, Sergey Levine, Pieter Abbeel

**Year:** 2024 | **Venue:** International Conference on Learning Representations | **Citations:** 167 | **Score:** 0.384

[DOI](https://doi.org/10.48550/arXiv.2410.12557)

> Diffusion models and flow-matching models have enabled generating diverse and realistic images by learning to transfer noise to data. However, sampling from these models involves iterative denoising over many neural network passes, making generation slow and expensive. Previous approaches for speeding up sampling require complex training regimes, such as multiple training phases, multiple networks...

---

## 53. Di$\mathtt{[M]}$O: Distilling Masked Diffusion Models into One-step Generator

**Authors:** Yuanzhi Zhu, Xi Wang, Stéphane Lathuilière, Vicky Kalogeiton

**Year:** 2025 | **Venue:**  | **Citations:** N/A | **Score:** 0.377

> Masked Diffusion Models (MDMs) have emerged as a powerful generative modeling technique. Despite their remarkable results, they typically suffer from slow inference with several steps. In this paper, we propose Di$\mathtt{[M]}$O, a novel approach that distills masked diffusion models into a one-step generator. Di$\mathtt{[M]}$O addresses two key challenges: (1) the intractability of using intermed...

---

## 54. Phased DMD: Few-step Distribution Matching Distillation via Score Matching within Subintervals

**Authors:** Xiangyu Fan, Zesong Qiu, Zhuguanyu Wu, Fanzhou Wang, Zhiqian Lin

**Year:** 2025 | **Venue:** arXiv (Cornell University) | **Citations:** N/A | **Score:** 0.370

[PDF](https://arxiv.org/pdf/2510.27684) | [DOI](https://doi.org/10.48550/arxiv.2510.27684)

> Distribution Matching Distillation (DMD) distills score-based generative models into efficient one-step generators, without requiring a one-to-one correspondence with the sampling trajectories of their teachers. However, limited model capacity causes one-step distilled models underperform on complex generative tasks, e.g., synthesizing intricate object motions in text-to-video generation. Directly...

---

## 55. Teacher Guided Architecture Search

**Authors:** Pouya Bashivan, Mark Tensen, James J DiCarlo

**Year:** 2019 | **Venue:** ICLR 2019 | **Citations:** N/A | **Score:** 0.370

> Strong improvements in neural network performance in vision tasks have resulted from the search of alternative network architectures, and prior work has shown that this search process can be automated and guided by evaluating candidate network performance following limited training (“Performance Guided Architecture Search” or PGAS).  However, because of the large architecture search spaces and the...

---

## 56. SlimFlow: Training Smaller One-Step Diffusion Models with Rectified Flow

**Authors:** Yuanzhi Zhu, Xingchao Liu, Qiang Liu

**Year:** 2024 | **Venue:** European Conference on Computer Vision | **Citations:** 23 | **Score:** 0.368

[DOI](https://doi.org/10.48550/arXiv.2407.12718)

> Diffusion models excel in high-quality generation but suffer from slow inference due to iterative sampling. While recent methods have successfully transformed diffusion models into one-step generators, they neglect model size reduction, limiting their applicability in compute-constrained scenarios. This paper aims to develop small, efficient one-step diffusion models based on the powerful rectifie...

---

## 57. Score-of-Mixture Training: One-Step Generative Model Training Made Simple via Score Estimation of Mixture Distributions

**Authors:** Tejas Jayashankar, J. J. Ryu, Greg Wornell

**Year:** 2025 | **Venue:** International Conference on Machine Learning | **Citations:** 1 | **Score:** 0.364

> ...

---

## 58. Distilled Decoding 2: One-step Sampling of Image Auto-regressive Models with Conditional Score Distillation

**Authors:** En-hao Liu, Qian Chen, Xuefei Ning, Shengen Yan, Guohao Dai

**Year:** 2025 | **Venue:** arXiv.org | **Citations:** 1 | **Score:** 0.364

[DOI](https://doi.org/10.48550/arXiv.2510.21003)

> Image Auto-regressive (AR) models have emerged as a powerful paradigm of visual generative models. Despite their promising performance, they suffer from slow generation speed due to the large number of sampling steps required. Although Distilled Decoding 1 (DD1) was recently proposed to enable few-step sampling for image AR models, it still incurs significant performance degradation in the one-ste...

---

## 59. Denoising Score Distillation: From Noisy Diffusion Pretraining to One-Step High-Quality Generation

**Authors:** Tianyu Chen, Yasi Zhang, Zhendong Wang, Yingnian Wu, Oscar Leong

**Year:** 2025 | **Venue:** arXiv.org | **Citations:** 5 | **Score:** 0.364

[DOI](https://doi.org/10.48550/arXiv.2503.07578)

> Diffusion models have achieved remarkable success in generating high-resolution, realistic images across diverse natural distributions. However, their performance heavily relies on high-quality training data, making it challenging to learn meaningful distributions from corrupted samples. This limitation restricts their applicability in scientific domains where clean data is scarce or costly to obt...

---

## 60. pi-Flow: Policy-Based Few-Step Generation via Imitation Distillation

**Authors:** Hansheng Chen, Kai Zhang, Hao Tan, Leonidas Guibas, Gordon Wetzstein

**Year:** 2025 | **Venue:** arXiv (Cornell University) | **Citations:** N/A | **Score:** 0.363

[PDF](https://arxiv.org/pdf/2510.14974) | [DOI](https://doi.org/10.48550/arxiv.2510.14974)

> Few-step diffusion or flow-based generative models typically distill a velocity-predicting teacher into a student that predicts a shortcut towards denoised data. This format mismatch has led to complex distillation procedures that often suffer from a quality-diversity trade-off. To address this, we propose policy-based flow models ($π$-Flow). $π$-Flow modifies the output layer of a student flow mo...

---

## 61. Toward Theoretical Insights into Diffusion Trajectory Distillation via Operator Merging

**Authors:** Wei Gao, Ming Li

**Year:** 2025 | **Venue:** arXiv.org | **Citations:** 2 | **Score:** 0.362

[DOI](https://doi.org/10.48550/arXiv.2505.16024)

> Diffusion trajectory distillation methods aim to accelerate sampling in diffusion models, which produce high-quality outputs but suffer from slow sampling speeds. These methods train a student model to approximate the multi-step denoising process of a pretrained teacher model in a single step, enabling one-shot generation. However, theoretical insights into the trade-off between different distilla...

---

## 62. MeanFlowSE: One-Step Generative Speech Enhancement via MeanFlow

**Authors:** Yike Zhu, Boyi Kang, Ziqian Wang, Xingchen Li, Zihan Zhang

**Year:** 2025 | **Venue:** arXiv.org | **Citations:** 1 | **Score:** 0.362

[DOI](https://doi.org/10.48550/arXiv.2509.23299)

> Speech enhancement (SE) recovers clean speech from noisy signals and is vital for applications such as telecommunications and automatic speech recognition (ASR). While generative approaches achieve strong perceptual quality, they often rely on multi-step sampling (diffusion/flow-matching) or large language models, limiting real-time deployment. To mitigate these constraints, we present MeanFlowSE,...

---

## 63. Universal Inverse Distillation for Matching Models with Real-Data Supervision (No GANs)

**Authors:** Nikita Kornilov, David Li, Tikhon Mavrin, Aleksei Leonov, Nikita Gushchin

**Year:** 2025 | **Venue:** arXiv.org | **Citations:** N/A | **Score:** 0.360

[DOI](https://doi.org/10.48550/arXiv.2509.22459)

> While achieving exceptional generative quality, modern diffusion, flow, and other matching models suffer from slow inference, as they require many steps of iterative generation. Recent distillation methods address this by training efficient one-step generators under the guidance of a pre-trained teacher model. However, these methods are often constrained to only one specific framework, e.g., only ...

---

## 64. One-Step Diffusion Distillation through Score Implicit Matching

**Authors:** Weijian Luo, Zemin Huang, Zhengyang Geng, J. Z. Kolter, Guo-Jun Qi

**Year:** 2024 | **Venue:** Neural Information Processing Systems | **Citations:** 41 | **Score:** 0.360

[DOI](https://doi.org/10.48550/arXiv.2410.16794)

> Despite their strong performances on many generative tasks, diffusion models require a large number of sampling steps in order to generate realistic samples. This has motivated the community to develop effective methods to distill pre-trained diffusion models into more efficient models, but these methods still typically require few-step inference or perform substantially worse than the underlying ...

---

## 65. SoFlow: Solution Flow Models for One-Step Generative Modeling

**Authors:** Tianze Luo, Haotian Yuan, Zhuang Liu

**Year:** 2025 | **Venue:**  | **Citations:** N/A | **Score:** 0.358

> The multi-step denoising process in diffusion and Flow Matching models causes major efficiency issues, which motivates research on few-step generation. We present Solution Flow Models (SoFlow), a framework for one-step generation from scratch. By analyzing the relationship between the velocity function and the solution function of the velocity ordinary differential equation (ODE), we propose a Flo...

---

## 66. Consistency Trajectory Matching for One-Step Generative Super-Resolution

**Authors:** Weiyi You, Mingyang Zhang, Leheng Zhang, Xingyu Zhou, Kexuan Shi

**Year:** 2025 | **Venue:** arXiv.org | **Citations:** 1 | **Score:** 0.357

[DOI](https://doi.org/10.48550/arXiv.2503.20349)

> Current diffusion-based super-resolution (SR) approaches achieve commendable performance at the cost of high inference overhead. Therefore, distillation techniques are utilized to accelerate the multi-step teacher model into one-step student model. Nevertheless, these methods significantly raise training costs and constrain the performance of the student model by the teacher model. To overcome the...

---

## 67. MeanFlowSE: one-step generative speech enhancement via conditional mean flow

**Authors:** Duojia Li, Shenghui Lu, Hongchen Pan, Zongyi Zhan, Q. Hong

**Year:** 2025 | **Venue:** arXiv.org | **Citations:** N/A | **Score:** 0.357

[DOI](https://doi.org/10.48550/arXiv.2509.14858)

> Multistep inference is a bottleneck for real-time generative speech enhancement because flow- and diffusion-based systems learn an instantaneous velocity field and therefore rely on iterative ordinary differential equation (ODE) solvers. We introduce MeanFlowSE, a conditional generative model that learns the average velocity over finite intervals along a trajectory. Using a Jacobian-vector product...

---

## 68. Restoration Score Distillation: From Corrupted Diffusion Pretraining to One-Step High-Quality Generation

**Authors:** Yasi Zhang, Tianyu Chen, Zhendong Wang, Yingnian Wu, Mingyuan Zhou

**Year:** 2025 | **Venue:** arXiv.org | **Citations:** 2 | **Score:** 0.355

[DOI](https://doi.org/10.48550/arXiv.2505.13377)

> Learning generative models from corrupted data is a fundamental yet persistently challenging task across scientific disciplines, particularly when access to clean data is limited or expensive. Denoising Score Distillation (DSD) \cite{chen2025denoising} recently introduced a novel and surprisingly effective strategy that leverages score distillation to train high-fidelity generative models directly...

---

## 69. Score identity Distillation: Exponentially Fast Distillation of Pretrained Diffusion Models for One-Step Generation

**Authors:** Mingyuan Zhou, Huangjie Zheng, Zhendong Wang, Mingzhang Yin, Hai Huang

**Year:** 2024 | **Venue:** International Conference on Machine Learning | **Citations:** 136 | **Score:** 0.354

[DOI](https://doi.org/10.48550/arXiv.2404.04057)

> We introduce Score identity Distillation (SiD), an innovative data-free method that distills the generative capabilities of pretrained diffusion models into a single-step generator. SiD not only facilitates an exponentially fast reduction in Fr\'echet inception distance (FID) during distillation but also approaches or even exceeds the FID performance of the original teacher diffusion models. By re...

---

## 70. SwiftBrush v2: Make Your One-step Diffusion Model Better Than Its Teacher

**Authors:** T. Dao, Thuan Hoang Nguyen, Van Thanh Le, D. Vu, Khoi Nguyen

**Year:** 2024 | **Venue:** European Conference on Computer Vision | **Citations:** 33 | **Score:** 0.354

[DOI](https://doi.org/10.48550/arXiv.2408.14176)

> In this paper, we aim to enhance the performance of SwiftBrush, a prominent one-step text-to-image diffusion model, to be competitive with its multi-step Stable Diffusion counterpart. Initially, we explore the quality-diversity trade-off between SwiftBrush and SD Turbo: the former excels in image diversity, while the latter excels in image quality. This observation motivates our proposed modificat...

---

## 71. Generative Classifiers Avoid Shortcut Solutions

**Authors:** Alexander C. Li, Ananya Kumar, Deepak Pathak

**Year:** 2025 | **Venue:** arXiv | **Citations:** N/A | **Score:** 0.353

[PDF](https://arxiv.org/pdf/2512.25034v1) | > Discriminative approaches to classification often learn shortcuts that hold in-distribution but fail even under minor distribution shift. This failure mode stems from an overreliance on features that are spuriously correlated with the label. We show that generative classifiers, which use class-conditional generative models, can avoid this issue by modeling all features, both core and spurious, ins...

---

## 72. TARFVAE: Efficient One-Step Generative Time Series Forecasting via TARFLOW based VAE

**Authors:** Jiawen Wei, Lan Jiang, Pengbo Wei, Ziwen Ye, Teng Song

**Year:** 2025 | **Venue:**  | **Citations:** N/A | **Score:** 0.353

> Time series data is ubiquitous, with forecasting applications spanning from finance to healthcare. Beyond popular deterministic methods, generative models are gaining attention due to advancements in areas like image synthesis and video generation, as well as their inherent ability to provide probabilistic predictions. However, existing generative approaches mostly involve recurrent generative ope...

---

## 73. One-Step Generative Channel Estimation via Average Velocity Field

**Authors:** Zehua Jiang, Fenghao Zhu, Siming Jiang, Chongwen Huang, Zhaohui Yang

**Year:** 2025 | **Venue:**  | **Citations:** N/A | **Score:** 0.352

> Generative models have shown immense potential for wireless communication by learning complex channel data distributions. However, the iterative denoising process associated with these models imposes a significant challenge in latency-sensitive wireless communication scenarios, particularly in channel estimation. To address this challenge, we propose a novel solution for one-step generative channe...

---

## 74. Flow Straighter and Faster: Efficient One-Step Generative Modeling via MeanFlow on Rectified Trajectories

**Authors:** Xinxi Zhang, Shiwei Tan, Quang Nguyen, Quan Dao, Ligong Han

**Year:** 2025 | **Venue:**  | **Citations:** N/A | **Score:** 0.352

> Flow-based generative models have recently demonstrated strong performance, yet sampling typically relies on expensive numerical integration of ordinary differential equations (ODEs). Rectified Flow enables one-step sampling by learning nearly straight probability paths, but achieving such straightness requires multiple computationally intensive reflow iterations. MeanFlow achieves one-step genera...

---

## 75. OFTSR: One-Step Flow for Image Super-Resolution with Tunable Fidelity-Realism Trade-offs

**Authors:** Yuanzhi Zhu, Ruiqing Wang, Shilin Lu, Junnan Li, Hanshu Yan

**Year:** 2024 | **Venue:** arXiv.org | **Citations:** 14 | **Score:** 0.351

[DOI](https://doi.org/10.48550/arXiv.2412.09465)

> Recent advances in diffusion and flow-based generative models have demonstrated remarkable success in image restoration tasks, achieving superior perceptual quality compared to traditional deep learning approaches. However, these methods either require numerous sampling steps to generate high-quality images, resulting in significant computational overhead, or rely on common model distillation, whi...

---

## 76. One-step Diffusion Models with Bregman Density Ratio Matching

**Authors:** Yuanzhi Zhu, Eleftherios Tsonis, Lucas Degeorge, Vicky Kalogeiton

**Year:** 2025 | **Venue:** arXiv (Cornell University) | **Citations:** N/A | **Score:** 0.351

[PDF](https://arxiv.org/pdf/2510.16983) | [DOI](https://doi.org/10.48550/arxiv.2510.16983)

> Diffusion and flow models achieve high generative quality but remain computationally expensive due to slow multi-step sampling. Distillation methods accelerate them by training fast student generators, yet most existing objectives lack a unified theoretical foundation. In this work, we propose Di-Bregman, a compact framework that formulates diffusion distillation as Bregman divergence-based densit...

---

## 77. Score-of-Mixture Training: Training One-Step Generative Models Made Simple via Score Estimation of Mixture Distributions

**Authors:** Tejas Jayashankar, J. J. Ryu, Gregory W. Wornell

**Year:** 2025 | **Venue:** arXiv.org | **Citations:** 1 | **Score:** 0.350

[DOI](https://doi.org/10.48550/arXiv.2502.09609)

> We propose Score-of-Mixture Training (SMT), a novel framework for training one-step generative models by minimizing a class of divergences called the $\alpha$-skew Jensen--Shannon divergence. At its core, SMT estimates the score of mixture distributions between real and fake samples across multiple noise levels. Similar to consistency models, our approach supports both training from scratch (SMT) ...

---

## 78. Distillation of Discrete Diffusion by Exact Conditional Distribution Matching

**Authors:** Yansong Gao, Yu Sun

**Year:** 2025 | **Venue:** arXiv (Cornell University) | **Citations:** N/A | **Score:** 0.350

[PDF](https://arxiv.org/pdf/2512.12889) | [DOI](https://doi.org/10.48550/arxiv.2512.12889)

> Discrete diffusion models (DDMs) are a powerful class of generative models for categorical data, but they typically require many function evaluations for a single sample, making inference expensive. Existing acceleration methods either rely on approximate simulators, such as $τ$-leaping, or on distillation schemes that train new student models and auxiliary networks with proxy objectives. We propo...

---

## 79. You Only Sample Once: Taming One-Step Text-To-Image Synthesis by Self-Cooperative Diffusion GANs

**Authors:** Yihong Luo, Xiaolong Chen, Tianyang Hu, Jing Tang

**Year:** 2024 | **Venue:** International Conference on Learning Representations | **Citations:** 17 | **Score:** 0.348

[DOI](https://doi.org/10.48550/arXiv.2403.12931)

> Recently, some works have tried to combine diffusion and Generative Adversarial Networks (GANs) to alleviate the computational cost of the iterative denoising inference in Diffusion Models (DMs). However, existing works in this line suffer from either training instability and mode collapse or subpar one-step generation learning efficiency. To address these issues, we introduce YOSO, a novel genera...

---

## 80. One-Step Effective Diffusion Network for Real-World Image Super-Resolution

**Authors:** Rongyuan Wu, Lingchen Sun, Zhiyuan Ma, Lei Zhang

**Year:** 2024 | **Venue:** Neural Information Processing Systems | **Citations:** 151 | **Score:** 0.347

[DOI](https://doi.org/10.48550/arXiv.2406.08177)

> The pre-trained text-to-image diffusion models have been increasingly employed to tackle the real-world image super-resolution (Real-ISR) problem due to their powerful generative image priors. Most of the existing methods start from random noise to reconstruct the high-quality (HQ) image under the guidance of the given low-quality (LQ) image. While promising results have been achieved, such Real-I...

---

## 81. Universal Audio Generation

**Authors:** Antoine Laurent, Sameer Khurana, Anthony Larcher, Dominik Klement, Mickaël Rouvier

**Year:** 2026 | **Venue:** HAL (Le Centre pour la Communication Scientifique Directe) | **Citations:** N/A | **Score:** 0.346

[PDF](https://hal.science/hal-05110014v1/document) | > This report describe the research done during the third ESPERANTO/JSALT workshop from the 10th June 2024 to the 2nd of August 2024....

---

## 82. MeanFlow-TSE: One-Step Generative Target Speaker Extraction with Mean Flow

**Authors:** Riki Shimizu, Xilin Jiang, N. Mesgarani

**Year:** 2025 | **Venue:**  | **Citations:** N/A | **Score:** 0.345

> Target speaker extraction (TSE) aims to isolate a desired speaker's voice from a multi-speaker mixture using auxiliary information such as a reference utterance. Although recent advances in diffusion and flow-matching models have improved TSE performance, these methods typically require multi-step sampling, which limits their practicality in low-latency settings. In this work, we propose MeanFlow-...

---

## 83. Accelerating Diffusion Models with One-to-Many Knowledge Distillation

**Authors:** Linfeng Zhang, Kaisheng Ma

**Year:** 2024 | **Venue:** arXiv.org | **Citations:** 7 | **Score:** 0.345

[DOI](https://doi.org/10.48550/arXiv.2410.04191)

> Significant advancements in image generation have been made with diffusion models. Nevertheless, when contrasted with previous generative models, diffusion models face substantial computational overhead, leading to failure in real-time generation. Recent approaches have aimed to accelerate diffusion models by reducing the number of sampling steps through improved sampling techniques or step distil...

---

## 84. David and Goliath: Small One-step Model Beats Large Diffusion with Score Post-training

**Authors:** Weijian Luo, Colin Zhang, Debing Zhang, Zhengyang Geng

**Year:** 2024 | **Venue:** International Conference on Machine Learning | **Citations:** 4 | **Score:** 0.344

> We propose Diff-Instruct* (DI*), a data-efficient post-training approach for one-step text-to-image generative models to improve its human preferences without requiring image data. Our method frames alignment as online reinforcement learning from human feedback (RLHF), which optimizes the one-step model to maximize human reward functions while being regularized to be kept close to a reference diff...

---

## 85. Distill, Forget, Repeat: A Framework for Continual Unlearning in Text-to-Image Diffusion Models

**Authors:** Naoki Murata, Yuhta Takida, Konda Reddy Mopuri

**Year:** 2025 | **Venue:** arXiv (Cornell University) | **Citations:** N/A | **Score:** 0.343

[PDF](https://arxiv.org/pdf/2512.02657) | [DOI](https://doi.org/10.48550/arxiv.2512.02657)

> The recent rapid growth of visual generative models trained on vast web-scale datasets has created significant tension with data privacy regulations and copyright laws, such as GDPR's ``Right to be Forgotten.'' This necessitates machine unlearning (MU) to remove specific concepts without the prohibitive cost of retraining. However, existing MU techniques are fundamentally ill-equipped for real-wor...

---

## 86. Diffusion Models Are Innate One-Step Generators

**Authors:** Bowen Zheng, Tianming Yang

**Year:** 2024 | **Venue:** arXiv.org | **Citations:** 15 | **Score:** 0.342

[DOI](https://doi.org/10.48550/arXiv.2405.20750)

> Diffusion Models (DMs) have achieved great success in image generation and other fields. By fine sampling through the trajectory defined by the SDE/ODE solver based on a well-trained score model, DMs can generate remarkable high-quality results. However, this precise sampling often requires multiple steps and is computationally demanding. To address this problem, instance-based distillation method...

---

## 87. Modular MeanFlow: Towards Stable and Scalable One-Step Generative Modeling

**Authors:** Haochen You, Baojing Liu, Hongyang He

**Year:** 2025 | **Venue:** arXiv.org | **Citations:** 3 | **Score:** 0.342

[DOI](https://doi.org/10.48550/arXiv.2508.17426)

> One-step generative modeling seeks to generate high-quality data samples in a single function evaluation, significantly improving efficiency over traditional diffusion or flow-based models. In this work, we introduce Modular MeanFlow (MMF), a flexible and theoretically grounded approach for learning time-averaged velocity fields. Our method derives a family of loss functions based on a differentia...

---

## 88. Generative Adversarial Networks as Variational Training of Energy Based Models

**Authors:** Shuangfei Zhai, Yu Cheng, Rogerio Feris, Zhongfei Zhang

**Year:** 2017 | **Venue:** ICLR 2017 | **Citations:** N/A | **Score:** 0.341

> In this paper, we study deep generative models for effective unsupervised learning. We propose VGAN, which works by minimizing a variational lower bound of the negative log likelihood (NLL) of an energy based model (EBM), where the model density $p(\mathbf{x})$ is approximated by a variational distribution $q(\mathbf{x})$ that is easy to sample from. The training of VGAN takes a two step procedure...

---

## 89. Research on the influencing factors of generative artificial intelligence usage intent in post-secondary education: an empirical analysis based on the AIDUA extended model

**Authors:** Xue Bai, Lin F. Yang

**Year:** 2025 | **Venue:** Frontiers in Psychology | **Citations:** 1 | **Score:** 0.341

[PDF](https://www.frontiersin.org/journals/psychology/articles/10.3389/fpsyg.2025.1644209/pdf) | [DOI](https://doi.org/10.3389/fpsyg.2025.1644209)

> Objective Generative Artificial Intelligence (AIGC) presents a profound dialectic in higher education: its transformative potential is challenged by deep-seated psychological and ethical barriers. Traditional adoption models fail to capture this complexity. To bridge this gap, this study develops and tests an integrated cognitive-behavioral framework. We posit that AIGC acceptance is a three-stage...

---

## 90. Scaling Open-Ended Reasoning to Predict the Future

**Authors:** Nikhil Chandak, Shashwat Goel, Ameya Prabhu, Moritz Hardt, Jonas Geiping

**Year:** 2025 | **Venue:** arXiv | **Citations:** N/A | **Score:** 0.341

[PDF](https://arxiv.org/pdf/2512.25070v1) | > High-stakes decision making involves reasoning under uncertainty about the future. In this work, we train language models to make predictions on open-ended forecasting questions. To scale up training data, we synthesize novel forecasting questions from global events reported in daily news, using a fully automated, careful curation recipe. We train the Qwen3 thinking models on our dataset, OpenFore...

---

## 91. From Diffusion to One-Step Generation: A Comparative Study of Flow-Based Models with Application to Image Inpainting

**Authors:** Umang Agarwal, Rudraksh Sangore, Sumit Laddha

**Year:** 2025 | **Venue:**  | **Citations:** N/A | **Score:** 0.339

> We present a comprehensive comparative study of three generative modeling paradigms: Denoising Diffusion Probabilistic Models (DDPM), Conditional Flow Matching (CFM), and MeanFlow. While DDPM and CFM require iterative sampling, MeanFlow enables direct one-step generation by modeling the average velocity over time intervals. We implement all three methods using a unified TinyUNet architecture (<1.5...

---

## 92. Many Minds from One Model: Bayesian Transformers for Population Intelligence

**Authors:** Diji Yang, Yi Zhang

**Year:** 2025 | **Venue:** arXiv | **Citations:** N/A | **Score:** 0.337

[PDF](https://arxiv.org/pdf/2512.25063v1) | > Despite their scale and success, modern transformers are almost universally trained as single-minded systems: optimization produces one deterministic set of parameters, representing a single functional hypothesis about the data. Motivated by the idea that intelligence emerge from many minds, we propose Population Bayesian Transformers (B-Trans), which transform a standard Large Language Model into...

---

## 93. Large Neutrino-Dark Matter Interactions: From Effective Field Theory to Ultraviolet Completions

**Authors:** K. S. Babu, P. S. Bhupal Dev, Anil Thapa

**Year:** 2025 | **Venue:** arXiv | **Citations:** N/A | **Score:** 0.337

[PDF](https://arxiv.org/pdf/2512.25035v1) | > We develop a general effective field theory (EFT) framework for neutrino-dark matter (DM) interactions, and apply it to systematically find all possible gauge-invariant ultraviolet (UV) completions at a given EFT operator dimension. Our goal here is to find simple UV-complete models that can realize potentially large neutrino-DM interactions, while being consistent with all existing theoretical an...

---

## 94. MeanFlow Transformers with Representation Autoencoders

**Authors:** Zheyuan Hu, Yuki Mitsufuji

**Year:** 2025 | **Venue:** arXiv (Cornell University) | **Citations:** N/A | **Score:** 0.337

[PDF](https://arxiv.org/pdf/2511.13019) | [DOI](https://doi.org/10.48550/arxiv.2511.13019)

> MeanFlow (MF) is a diffusion-motivated generative model that enables efficient few-step generation by learning long jumps directly from noise to data. In practice, it is often used as a latent MF by leveraging the pre-trained Stable Diffusion variational autoencoder (SD-VAE) for high-dimensional data modeling. However, MF training remains computationally demanding and is often unstable. During inf...

---

## 95. Imagine Flash: Accelerating Emu Diffusion Models with Backward Distillation

**Authors:** Jonas Kohler, Albert Pumarola, Edgar Schönfeld, Artsiom Sanakoyeu, Roshan Sumbaly

**Year:** 2024 | **Venue:** arXiv.org | **Citations:** 34 | **Score:** 0.337

[DOI](https://doi.org/10.48550/arXiv.2405.05224)

> Diffusion models are a powerful generative framework, but come with expensive inference. Existing acceleration methods often compromise image quality or fail under complex conditioning when operating in an extremely low-step regime. In this work, we propose a novel distillation framework tailored to enable high-fidelity, diverse sample generation using just one to three steps. Our approach compris...

---

## 96. Values of the T-shaped Leader: Applying the 4M Framework to Address SoTL Grand Challenges and Foster Sustainable Development Goals in Higher Education

**Authors:** Earle Abrahamson, Lisa J. Hatfield, Corinne A. Green, Nina Namaste, Mayi Arcellana‐Panlilio

**Year:** 2025 | **Venue:** Journal of University Teaching and Learning Practice | **Citations:** N/A | **Score:** 0.334

[PDF](https://open-publishing.org/journals/index.php/jutlp/article/download/1235/1091) | [DOI](https://doi.org/10.53761/0z732113)

> This paper follows the conceptual T-shaped model, introduced by (the Authors), serving as a conduit for the dissemination of empirical data derived from engaging discussions with international Scholarship of Teaching and Learning (SoTL) scholars. The authors not only present the model, but also offer auto-ethnographic reflections, unravelling the key values embedded within it. These values, namely...

---

## 97. Feeling Blue: Constructing a Robust SALT3 UV Template and Constraining its Redshift Dependency

**Authors:** Qinan Wang, David O. Jones, Justin D. R. Pierel, Matthew R. Siebert, W. D'Arcy Kenworthy

**Year:** 2025 | **Venue:** arXiv | **Citations:** N/A | **Score:** 0.333

[PDF](https://arxiv.org/pdf/2512.25064v1) | > Upcoming cosmological surveys will obtain numerous rest-frame ultraviolet (UV) observations of Type Ia supernovae (SNe Ia), yet there is concern about how standardizable SNe Ia are in the UV. In this work, we train a robust optical--UV SED model for SNe Ia (SALT3-UV) with the open-source model-training software $\texttt{SALTshaker}$. We incorporate a spectroscopic UV data sample from HST, includin...

---

## 98. Fluid dynamics as intersection problem

**Authors:** Nikita Nekrasov, Paul Wiegmann

**Year:** 2025 | **Venue:** arXiv | **Citations:** N/A | **Score:** 0.332

[PDF](https://arxiv.org/pdf/2512.25053v1) | > We formulate the covariant hydrodynamics equations describing the fluid dynamics as the problem of intersection theory on the infinite dimensional symplectic manifold associated with spacetime. This point of view separates the structures related to the equation of state, the geometry of spacetime, and structures related to the (differential) topology of spacetime. We point out a five-dimensional o...

---

## 99. Designing Sustainable Digital Platforms for Ageing Societies: A User-Centred Multi-Level Theoretical Framework

**Authors:** Lü Pan, Xin Hu

**Year:** 2025 | **Venue:** Sustainability | **Citations:** N/A | **Score:** 0.332

[PDF](https://www.mdpi.com/2071-1050/17/18/8305/pdf?version=1758025435) | [DOI](https://doi.org/10.3390/su17188305)

> With the intensification of population ageing and the increasingly diverse service needs of older adults, existing digital elderly care platforms generally exhibit fragmentation in functional integration, understanding of needs, and service coordination, making it difficult to effectively respond to the complex challenges faced by urban ageing populations. To fill this gap, this study starts from ...

---

## 100. Edit3r: Instant 3D Scene Editing from Sparse Unposed Images

**Authors:** Jiageng Liu, Weijie Lyu, Xueting Li, Yejie Guo, Ming-Hsuan Yang

**Year:** 2025 | **Venue:** arXiv | **Citations:** N/A | **Score:** 0.332

[PDF](https://arxiv.org/pdf/2512.25071v1) | > We present Edit3r, a feed-forward framework that reconstructs and edits 3D scenes in a single pass from unposed, view-inconsistent, instruction-edited images. Unlike prior methods requiring per-scene optimization, Edit3r directly predicts instruction-aligned 3D edits, enabling fast and photorealistic rendering without optimization or pose estimation. A key challenge in training such a model lies i...

---

## 101. Sequential Bayesian parameter-state estimation in dynamical systems with noisy and incomplete observations via a variational framework

**Authors:** Liliang Wang, Alex Gorodetsky

**Year:** 2025 | **Venue:** arXiv | **Citations:** N/A | **Score:** 0.332

[PDF](https://arxiv.org/pdf/2512.25056v1) | > Online joint estimation of unknown parameters and states in a dynamical system with uncertainty quantification is crucial in many applications. For example, digital twins dynamically update their knowledge of model parameters and states to support prediction and decision-making. Reliability and computational speed are vital for DTs. Online parameter-state estimation ensures computational efficienc...

---

## 102. Automated Summarization of Software Documents: An LLM-based Multi-Agent Approach

**Authors:** Dan Phuoc Nguyen, Minh T. Nguyen, Phuong T. Nguyen, Juri Di Rocco, Davide Di Ruscio

**Year:** 2025 | **Venue:**  | **Citations:** N/A | **Score:** 0.332

[PDF](https://www.researchsquare.com/article/rs-7539426/latest.pdf) | [DOI](https://doi.org/10.21203/rs.3.rs-7539426/v1)

> <title>Abstract</title> Large Language Models (LLMs) and LLM-based Multi-Agent Sys-tems (MAS) are revolutionizing software engineering (SE) by advancing automation, decision-making, and knowledge processing. Their recent application to SE tasks has already shown promising results. In this paper, we focus on summarization as a key application area. We present Metagente, an LLM-based MAS designed to...

---

## 103. SpaceTimePilot: Generative Rendering of Dynamic Scenes Across Space and Time

**Authors:** Zhening Huang, Hyeonho Jeong, Xuelin Chen, Yulia Gryaditskaya, Tuanfeng Y. Wang

**Year:** 2025 | **Venue:** arXiv | **Citations:** N/A | **Score:** 0.332

[PDF](https://arxiv.org/pdf/2512.25075v1) | > We present SpaceTimePilot, a video diffusion model that disentangles space and time for controllable generative rendering. Given a monocular video, SpaceTimePilot can independently alter the camera viewpoint and the motion sequence within the generative process, re-rendering the scene for continuous and arbitrary exploration across space and time. To achieve this, we introduce an effective animati...

---

## 104. Computational Analysis of Disease Progression in Pediatric Pulmonary Arterial Hypertension

**Authors:** Omar Said, Christopher Tossas-Betancourt, Mary K. Olive, Jimmy C. Lu, Adam Dorfman

**Year:** 2025 | **Venue:** arXiv | **Citations:** N/A | **Score:** 0.332

[PDF](https://arxiv.org/pdf/2512.25027v1) | > Pulmonary arterial hypertension (PAH) is a progressive cardiopulmonary disease that leads to increased pulmonary pressures, vascular remodeling, and eventual right ventricular (RV) failure. Pediatric PAH remains understudied due to limited data and the lack of targeted diagnostic and therapeutic strategies. In this study, we developed and calibrated multi-scale, patient-specific cardiovascular mod...

---

## 105. Anticipatory Semantics with Bidirectional Guidance for Image Captioning

**Authors:** Nathalie Laurent, Elodie Fairchild, Arthur Delvaux

**Year:** 2025 | **Venue:** Preprints.org | **Citations:** N/A | **Score:** 0.332

[PDF](https://www.preprints.org/frontend/manuscript/3a90020e8e0ebfd0b1934a961c7a89ba/download_pub) | [DOI](https://doi.org/10.20944/preprints202509.1497.v1)

> Producing captions that are not only grammatically fluent but also semantically faithful to visual content has long stood as a central problem at the junction of computer vision and natural language processing. Conventional encoder-decoder frameworks with attention modules, although powerful, typically confine the decoding process to a retrospective scope: every prediction is conditioned solely on...

---

## 106. SwiftBrush: One-Step Text-to-Image Diffusion Model with Variational Score Distillation

**Authors:** Thuan Hoang Nguyen, Anh Tran

**Year:** 2023 | **Venue:** Computer Vision and Pattern Recognition | **Citations:** 89 | **Score:** 0.332

[PDF](http://arxiv.org/pdf/2312.05239) | [DOI](https://doi.org/10.1109/CVPR52733.2024.00746)

> Despite their ability to generate high-resolution and diverse images from text prompts, text-to-image diffusion models often suffer from slow iterative sampling processes. Model distillation is one of the most effective directions to accelerate these models. However, previous distillation methods fail to retain the generation quality while requiring a significant amount of images for training, eit...

---

## 107. Differentiated Assessment Strategies: Best Practices in a Multi-Level Learning Manitoban Classroom

**Authors:** Eric. K. Appiah-Odame

**Year:** 2025 | **Venue:** European Journal of Mathematics and Science Education | **Citations:** N/A | **Score:** 0.332

[PDF](https://www.ejmse.com/articles/EJMSE_6_3_179.pdf) | [DOI](https://doi.org/10.12973/ejmse.6.3.179)

> Introduction: This study explores the effectiveness of differentiated assessment as a strategy to support diverse learners in multi-level K–12 classrooms in Manitoba, Canada. Literature Review: Articles published from 2005 onward were sourced from ProQuest, ERIC, Google Scholar, ResearchGate, and Taylor &amp; Francis databases. Methodology: A qualitative document review was employed by analyzing p...

---

## 108. Generative AI-Supported Project-Based Learning in EFL: Impacts on Student Engagement and Learner Agency

**Authors:** Safaa M. Abdelhalim, Maram Almaneea

**Year:** 2025 | **Venue:** Forum for Linguistic Studies | **Citations:** N/A | **Score:** 0.331

[PDF](https://journals.bilpubgroup.com/index.php/fls/article/download/10855/7006) | [DOI](https://doi.org/10.30564/fls.v7i9.10855)

> The growing integration of Generative AI (GenAI) tools in language education presents new opportunities for enhancing learner engagement. However, empirical evidence on their effectiveness remains limited, particularly in project-based and collaborative contexts. This study examined the impact of integrating GenAI tools into collaborative Project-Based Language Learning (PBLL) on EFL undergraduate...

---

## 109. Universal Seesaw Pati-Salam Model with P for Strong CP

**Authors:** K. S. Babu, Sumit Biswas

**Year:** 2025 | **Venue:** arXiv | **Citations:** N/A | **Score:** 0.331

[PDF](https://arxiv.org/pdf/2512.25028v1) | > We develop a universal seesaw version of the Pati-Salam model wherein quarks and leptons of each family are unified into common multiplets transforming as $ψ_L(2,1,4))+ ψ_R((1,2,4)$ under the $SU(2)_L \times SU(2)_R \times SU(4)_c$ gauge symmetry. Parity symmetry is spontaneously broken in the model, which helps in solving the strong CP problem without the axion. The Higgs sector of the model is v...

---

## 110. Evaluating Time-series Augmentation Techniques for Deep Learning–Based Solar Flare Prediction

**Authors:** Peiyu Li, Omar Bahri, Soukaïna Filali Boubrahimi, Shah Muhammad Hamdi

**Year:** 2025 | **Venue:** The Astrophysical Journal Supplement Series | **Citations:** N/A | **Score:** 0.331

[PDF](https://iopscience.iop.org/article/10.3847/1538-4365/adfa2a/pdf) | [DOI](https://doi.org/10.3847/1538-4365/adfa2a)

> Abstract Accurate forecasting of solar flares is crucial for mitigating their severe impacts on space-based and communication systems. Deep learning models have shown promise in predicting flare activity using magnetic field measurements of solar active regions. However, the scarcity and imbalance of flare occurrence data—particularly for major flares—pose significant challenges to model robustnes...

---

## 111. Randomization Times under Quantum Chaotic Hamiltonian Evolution

**Authors:** Souradeep Ghosh, Nicholas Hunter-Jones, Joaquin F. Rodriguez-Nieva

**Year:** 2025 | **Venue:** arXiv | **Citations:** N/A | **Score:** 0.331

[PDF](https://arxiv.org/pdf/2512.25074v1) | > Randomness generation through quantum-chaotic evolution underpins foundational questions in statistical mechanics and applications across quantum information science, including benchmarking, tomography, metrology, and demonstrations of quantum computational advantage. While statistical mechanics successfully captures the temporal averages of local observables, understanding randomness at the level...

---

## 112. Vulcan: Instance-Optimal Systems Heuristics Through LLM-Driven Search

**Authors:** Rohit Dwivedula, Divyanshu Saxena, Sujay Yadalam, Daehyeok Kim, Aditya Akella

**Year:** 2025 | **Venue:** arXiv | **Citations:** N/A | **Score:** 0.331

[PDF](https://arxiv.org/pdf/2512.25065v1) | > Resource-management tasks in modern operating and distributed systems continue to rely primarily on hand-designed heuristics for tasks such as scheduling, caching, or active queue management. Designing performant heuristics is an expensive, time-consuming process that we are forced to continuously go through due to the constant flux of hardware, workloads and environments.
  We propose a new alter...

---

## 113. All optical Lithography for Spatiotemporal Patterning of Azopolymer Microreliefs

**Authors:** I Komang Januariyasa, Francesco Reda, Nikolai Liubimtsev, Marina Saphiannikova, Fabio Borbone

**Year:** 2025 | **Venue:** arXiv | **Citations:** N/A | **Score:** 0.330

[PDF](https://arxiv.org/pdf/2512.25048v1) | > Microstructured surfaces are central to photonics, biointerfaces, and functional coatings, yet they are typically fabricated through multi-step lithographic workflows requiring masks or molds and post-processing. Azopolymers provide an alternative route by converting structured optical fields into surface reliefs via light-induced mass migration, but existing approaches have been limited to smooth...

---

## 114. Modewise Additive Factor Model for Matrix Time Series

**Authors:** Elynn Chen, Yuefeng Han, Jiayu Li, Ke Xu

**Year:** 2025 | **Venue:** arXiv | **Citations:** N/A | **Score:** 0.330

[PDF](https://arxiv.org/pdf/2512.25025v1) | > We introduce a Modewise Additive Factor Model (MAFM) for matrix-valued time series that captures row-specific and column-specific latent effects through an additive structure, offering greater flexibility than multiplicative frameworks such as Tucker and CP factor models. In MAFM, each observation decomposes into a row-factor component, a column-factor component, and noise, allowing distinct sourc...

---

## 115. School Leadership Approach to Teacher Collaboration: A Qualitative Investigation in the Secondary School Context of Bangladesh

**Authors:** Beauty Debnath

**Year:** 2025 | **Venue:** European Journal of Educational Management | **Citations:** N/A | **Score:** 0.330

[PDF](https://www.eujem.com/articles/EUJEM_8_3_185.pdf) | [DOI](https://doi.org/10.12973/eujem.8.3.185)

> Teacher collaboration appears essential for creating a dynamic and effective educational environment that supports teachers' professional learning and growth. In various research studies, supportive school leadership has been identified as a key condition for collaborative teacher learning, which enhances teachers’ engagement and professional learning by fostering a climate of trust. Bangladesh is...

---

## 116. Machine Learning-Based Classification of Student Adaptability in Online Learning with Feature Engineering

**Authors:** Yasin Efendi

**Year:** 2025 | **Venue:** TIERS Information Technology Journal | **Citations:** N/A | **Score:** 0.329

[PDF](https://journal.undiknas.ac.id/index.php/tiers/article/download/6806/1898) | [DOI](https://doi.org/10.38043/tiers.v6i1.6806)

> Student adaptability in online learning environments has become increasingly important in contemporary education. This study introduces a feature engineering approach guided by SHAP (SHapley Additive exPlanations) to enhance the classification of student adaptability levels. Unlike prior studies that primarily utilize exploratory analysis or statistical importance scores, this method leverages SHA...

---

## 117. Bayesian Elastic Net Regression with Structured Prior Dependence

**Authors:** Christopher M. Hans, Ningyi Liu

**Year:** 2025 | **Venue:** arXiv | **Citations:** N/A | **Score:** 0.329

[PDF](https://arxiv.org/pdf/2512.25045v1) | > Many regularization priors for Bayesian regression assume the regression coefficients are a priori independent. In particular this is the case for standard Bayesian treatments of the lasso and the elastic net. While independence may be reasonable in some data-analytic settings, incorporating dependence in these prior distributions provides greater modeling flexibility. This paper introduces the or...

---

## 118. Multivariate Generalized Counting Process via Gamma Subordination

**Authors:** Manisha Dhillon, Kuldeep Kumar Kataria, Shyan Ghosh

**Year:** 2025 | **Venue:** arXiv | **Citations:** N/A | **Score:** 0.329

[PDF](https://arxiv.org/pdf/2512.25030v1) | > In this paper, we study a multivariate gamma subordinator whose components are independent gamma processes subject to a random time governed by an independent negative binomial process. We derive the explicit expressions for its joint Laplace-Stieltjes transform, its probability density function and the associated governing differential equations. Also, we study a time-changed variant of the multi...

---

## 119. Degradation-Guided One-Step Image Super-Resolution with Diffusion Priors

**Authors:** Aiping Zhang, Zongsheng Yue, Renjing Pei, Wenqi Ren, Xiaochun Cao

**Year:** 2024 | **Venue:** arXiv.org | **Citations:** 27 | **Score:** 0.329

[DOI](https://doi.org/10.48550/arXiv.2409.17058)

> Diffusion-based image super-resolution (SR) methods have achieved remarkable success by leveraging large pre-trained text-to-image diffusion models as priors. However, these methods still face two challenges: the requirement for dozens of sampling steps to achieve satisfactory results, which limits efficiency in real scenarios, and the neglect of degradation models, which are critical auxiliary in...

---

## 120. ResponseRank: Data-Efficient Reward Modeling through Preference Strength Learning

**Authors:** Timo Kaufmann, Yannick Metz, Daniel Keim, Eyke Hüllermeier

**Year:** 2025 | **Venue:** arXiv | **Citations:** N/A | **Score:** 0.329

[PDF](https://arxiv.org/pdf/2512.25023v1) | > Binary choices, as often used for reinforcement learning from human feedback (RLHF), convey only the direction of a preference. A person may choose apples over oranges and bananas over grapes, but which preference is stronger? Strength is crucial for decision-making under uncertainty and generalization of preference models, but hard to measure reliably. Metadata such as response times and inter-an...

---

## 121. A Qualitative Approach to EFL Postgraduates’ GenAI-Assisted Research Writing Within Social Sciences

**Authors:** Alejandro Blas Curado Fuentes

**Year:** 2025 | **Venue:** Preprints.org | **Citations:** N/A | **Score:** 0.329

[PDF](https://www.preprints.org/frontend/manuscript/aaf3bab4376598387d8c7f223129592b/download_pub) | [DOI](https://doi.org/10.20944/preprints202509.1514.v1)

> In academic L2 English / EFL (English as a Foreign Language) writing, GenAI (Generative Artificial Intelligence) and other digital tools are being extensively explored. However, this AI exploration for academic / research writing has been addressed less at postgraduate levels, and even less so, according to different scientific fields. This study examines this topic within Social Sciences at Unive...

---

## 122. ImaginateAR: AI-Assisted In-Situ Authoring in Augmented Reality

**Authors:** Jaewook Lee, Filippo Aleotti, Diego Mazala, Guillermo Garcia-Hernando, Sara Vicente

**Year:** 2025 | **Venue:**  | **Citations:** N/A | **Score:** 0.328

[PDF](https://doi.org/10.1145/3746059.3747635) | [DOI](https://doi.org/10.1145/3746059.3747635)

> While augmented reality (AR) enables new ways to play, tell stories, and explore ideas rooted in the physical world, authoring personalized AR content remains difficult for non-experts, often requiring professional tools and time. Prior systems have explored AI-driven XR design but typically rely on manually defined VR environments and fixed asset libraries, limiting creative flexibility and real-...

---

## 123. Levelling Up Writing: Investigating Duolingo’s Gamification Effect on EFL Students’ Writing Skills

**Authors:** Annisa Hafizah Gamal, Afrianto Daud, Eliwarti Eliwarti

**Year:** 2025 | **Venue:** European Journal of English Language Studies | **Citations:** N/A | **Score:** 0.328

[PDF](https://www.ejels.com/articles/EJELS_5_3_191.pdf) | [DOI](https://doi.org/10.12973/ejels.5.3.191)

> Duolingo has become one of the most widely used gamification apps for learning English, mostly for vocabulary and grammar. However, there is limited research on its effectiveness in enhancing writing skills. This study aimed to investigate the effect of Duolingo on junior high school students' learning English as a Foreign Language (EFL) writing skills, focusing on the overall score and five compo...

---

## 124. Anomalous (3+1)d Fermionic Topological Quantum Field Theories via Symmetry Extension

**Authors:** Zheyan Wan, Juven Wang

**Year:** 2025 | **Venue:** arXiv | **Citations:** N/A | **Score:** 0.328

[PDF](https://arxiv.org/pdf/2512.25038v1) | > Discrete finite-group global symmetries may suffer from nonperturbative 't-Hooft anomalies. Such global anomalies can be canceled by anomalous symmetry-preserving topological quantum field theories (TQFTs), which contain no local point operators but only extended excitations such as line and surface operators. In this work, we study mixed gauge-gravitational nonperturbative global anomalies of Wey...

---

## 125. Modeling Language as a Sequence of Thoughts

**Authors:** Nasim Borazjanizadeh, James McClelland

**Year:** 2025 | **Venue:** arXiv | **Citations:** N/A | **Score:** 0.327

[PDF](https://arxiv.org/pdf/2512.25026v1) | > Transformer language models can generate strikingly natural text by modeling language as a sequence of tokens. Yet, by relying primarily on surface-level co-occurrence statistics, they fail to form globally consistent latent representations of entities and events, lack of which contributes to brittleness in relational direction (e.g., reversal curse), contextualization errors, and data inefficienc...

---

## 126. EF(X) Orientations: A Parameterized Complexity Perspective

**Authors:** Sotiris Kanellopoulos, Edouard Nemery, Christos Pergaminelis, Minas Marios Sotiriou, Manolis Vasilakis

**Year:** 2025 | **Venue:** arXiv | **Citations:** N/A | **Score:** 0.327

[PDF](https://arxiv.org/pdf/2512.25033v1) | > The concept of fair orientations in graphs was introduced by Christodoulou, Fiat, Koutsoupias, and Sgouritsa in 2023, naturally modeling fair division scenarios in which resources are only contested by neighbors. In this model, vertices represent agents and undirected edges represent goods; edges have to be oriented towards one of their endpoints, i.e., allocated to one of their adjacent agents. A...

---

## 127. Loop-Level Lepton Flavor Violation and Diphoton Signals in the Minimal Left-Right Symmetric Model

**Authors:** Shufang Qiang, Peiwen Wu, Yongchao Zhang

**Year:** 2025 | **Venue:** arXiv | **Citations:** N/A | **Score:** 0.327

[PDF](https://arxiv.org/pdf/2512.25019v1) | > The left-right symmetric model (LRSM) could not only restore parity of the weak interaction, but also provide natural explanations of the tiny active neutrino masses via the seesaw mechanisms. The $SU(2)_R$-breaking scalar $H_3$ can induce lepton flavor violating (LFV) effects in the minimal version of LRSM at the 1-loop order, originating from the mixing of heavy right-handed neutrinos. If $H_3$ ...

---

## 128. On Nonlinear Inertial Transformations

**Authors:** Nicholas Agia

**Year:** 2025 | **Venue:** arXiv | **Citations:** N/A | **Score:** 0.327

[PDF](https://arxiv.org/pdf/2512.25024v1) | > It is often assumed that the most general transformation between two inertial reference frames is affine linear in their Cartesian coordinates, an assumption which is however not true. We provide a complete derivation of the most general inertial frame transformation, which is indeed nonlinear; along the way, we shall find that the conditions of preserving the Law of Inertia take the form of Schwa...

---

## 129. Crying in the algorithm: modeling academic stress via multilayer topic construction and ERA effect

**Authors:** Liwei Ding, Hongfeng Zhang, Jinqiao Zhou

**Year:** 2025 | **Venue:** Frontiers in Psychology | **Citations:** N/A | **Score:** 0.326

[PDF](https://www.frontiersin.org/journals/psychology/articles/10.3389/fpsyg.2025.1673559/pdf) | [DOI](https://doi.org/10.3389/fpsyg.2025.1673559)

> Amid intensifying educational competition and societal expectations, academic stress has emerged as a multidimensional force influencing student mental health. While prior research has explored individual and institutional factors, limited attention has been paid to how learners semantically construct and express academic stress in digital environments. Addressing this gap, this study introduces a...

---

## 130. Ethics, Privacy, and Transparency in AI‐Assisted Teaching: Evaluating Notegrade.ai Against Global Standards

**Authors:** Akinwumi Fakokunde

**Year:** 2025 | **Venue:** Preprints.org | **Citations:** N/A | **Score:** 0.326

[PDF](https://www.preprints.org/frontend/manuscript/d7cdbe30a70999e65be97d02adfff621/download_pub) | [DOI](https://doi.org/10.20944/preprints202509.1148.v1)

> This article provides a deep, standards-based analysis of Notegrade.ai –an AI teaching tool that provides lesson plan generation, rubric based grading, plagiarism check, and student assessment tools- within several of the world’s most pertinent legal and ethical frameworks in education including the GDPR, the EU AI Act, the UNESCO Recommendation on the Ethics of AI, COPPA/FERPA recommendations and...

---

## 131. Fractal conduction pathways governing ionic transport in a glass

**Authors:** J. L. Iguain, F. O. Sanchez-Varreti, M. A. Frechero

**Year:** 2025 | **Venue:** arXiv | **Citations:** N/A | **Score:** 0.326

[PDF](https://arxiv.org/pdf/2512.25031v1) | > We present a systematic characterization of the fractal conduction pathways governing ionic transport in a non-crystalline solid below the glass-transition temperature. Using classical molecular dynamics simulations of lithium metasilicate, we combine mobility-resolved dynamical analysis with a real-space description of the regions explored by lithium ions. Ensemble-averaged velocity autocorrelati...

---

## 132. From perception to policy: course coordinators’ views on generative artificial intelligence in higher education

**Authors:** Nick den Hollander, Annemie Struyf, Despoina Georgiou, Jacqueline Wong

**Year:** 2025 | **Venue:** International Journal of Educational Management | **Citations:** N/A | **Score:** 0.326

[PDF](https://www.emerald.com/ijem/article-pdf/doi/10.1108/IJEM-12-2024-0842/10298203/ijem-12-2024-0842en.pdf) | [DOI](https://doi.org/10.1108/ijem-12-2024-0842)

> Purpose As key figures in curriculum design and implementation, course coordinators operate at the intersection of institutional goals, the needs of students, institutional policy and classroom practice, balancing educators alike (Stark and Lattuca, 1997). Their perspective is especially valuable in understanding the broader implications of GenAI integration, as they are often responsible for shap...

---

## 133. One-Step Diffusion Distillation via Deep Equilibrium Models

**Authors:** Zhengyang Geng, Ashwini Pokle, J. Z. Kolter, Paul Micaelli, Arash Vahdat

**Year:** 2023 | **Venue:** Neural Information Processing Systems | **Citations:** 47 | **Score:** 0.326

[DOI](https://doi.org/10.48550/arXiv.2401.08639)

> Diffusion models excel at producing high-quality samples but naively require hundreds of iterations, prompting multiple attempts to distill the generation process into a faster network. However, many existing approaches suffer from a variety of challenges: the process for distillation training can be complex, often requiring multiple training stages, and the resulting models perform poorly when ut...

---

## 134. Real Riemann Surfaces: Smooth and Discrete

**Authors:** Johanna Düntsch, Felix Günther

**Year:** 2025 | **Venue:** arXiv | **Citations:** N/A | **Score:** 0.326

[PDF](https://arxiv.org/pdf/2512.25022v1) | > This paper develops a discrete theory of real Riemann surfaces based on quadrilateral cellular decompositions (quad-graphs) and a linear discretization of the Cauchy-Riemann equations. We construct a discrete analogue of an antiholomorphic involution and classify the topological types of discrete real Riemann surfaces, recovering the classical results on the number of real ovals and the separation...

---

## 135. Primordial black hole dark matter from ultra-slow-roll inflation in Horndeski gravity

**Authors:** Despina Totolou, Theodoros Papanikolaou, Emmanuel N. Saridakis

**Year:** 2025 | **Venue:** arXiv | **Citations:** N/A | **Score:** 0.326

[PDF](https://arxiv.org/pdf/2512.25044v1) | > Primordial black holes (PBHs) provide a well-motivated non-particle candidate for dark matter, requiring an enhancement of curvature perturbations on small inflationary scales consistent with observational constraints. In this work we study PBH production within Horndeski gravity, accounting for compatibility with the GW170817 constraint on the gravitational-wave speed and imposing a constant coup...

---

## 136. Transitions in development – an interview with Mijo Šimunović

**Authors:** 

**Year:** 2025 | **Venue:** Development | **Citations:** N/A | **Score:** 0.325

[PDF](https://journals.biologists.com/dev/article-pdf/152/18/dev205185/3665207/dev205185.pdf) | [DOI](https://doi.org/10.1242/dev.205185)

> Mijo Šimunović is an Assistant Professor of Chemical Engineering at Columbia University, USA. With two PhDs – one in Chemistry and the other in Physics – and postdoctoral experience in Stem Cell and Developmental Biology, Mijo's research group now generates quantitative models of embryogenesis using human pluripotent stem cells to study the molecular mechanisms and the biomechanics of embryo impla...

---

## 137. Context-aware LLM-based AI Agents for Human-centered Energy Management Systems in Smart Buildings

**Authors:** Tianzhi He, Farrokh Jazizadeh

**Year:** 2025 | **Venue:** arXiv | **Citations:** N/A | **Score:** 0.325

[PDF](https://arxiv.org/pdf/2512.25055v1) | > This study presents a conceptual framework and a prototype assessment for Large Language Model (LLM)-based Building Energy Management System (BEMS) AI agents to facilitate context-aware energy management in smart buildings through natural language interaction. The proposed framework comprises three modules: perception (sensing), central control (brain), and action (actuation and user interaction),...

---

## 138. Melting curve of correlated iron at Earth's core conditions from machine-learned DFT+DMFT

**Authors:** Rishi Rao, Li Zhu

**Year:** 2025 | **Venue:** arXiv | **Citations:** N/A | **Score:** 0.325

[PDF](https://arxiv.org/pdf/2512.25061v1) | > Reliable constraints on iron's melting curve at Earth's inner-core boundary require accurate finite-temperature electronic correlations, yet DFT+DMFT calculations remain too costly for large-scale thermodynamic sampling. Here, we develop a machine-learning accelerator for charge self-consistent DFT+DMFT by training E(3)-equivariant graph neural networks to predict the local self-energy and Fermi l...

---

## 139. Source-Free Domain Adaptation for Medical Image Segmentation via Mutual Information Maximization and Prediction Bank

**Authors:** Hongzhen Wu, Yue Zhou, Xiaoqiang Li

**Year:** 2025 | **Venue:** Electronics | **Citations:** N/A | **Score:** 0.325

[PDF](https://www.mdpi.com/2079-9292/14/18/3656/pdf?version=1758004270) | [DOI](https://doi.org/10.3390/electronics14183656)

> Medical image segmentation faces significant challenges due to domain shift between different clinical centers and data privacy restrictions. Current source-free domain adaptation methods for medical images suffer from three critical limitations, including unstable training caused by noisy pseudo-labels and poor handling of foreground-background imbalance where critical structures like optic cup o...

---

## 140. Enhancing Monkeypox Diagnosis with Transformers: Bridging Explainability and Performance with Quantitative Validation

**Authors:** Delal Şeker, Abdulnasır Yildiz

**Year:** 2025 | **Venue:** Diagnostics | **Citations:** N/A | **Score:** 0.325

[PDF](https://www.mdpi.com/2075-4418/15/18/2354/pdf?version=1758032312) | [DOI](https://doi.org/10.3390/diagnostics15182354)

> Background/Objectives: Monkeypox is a zoonotic virus that presents with smallpox-like symptoms, making visual diagnosis challenging due to overlap with other dermatological conditions. Existing AI-based studies on monkeypox classification have largely relied on Convolutional Neural Networks (CNNs), with limited exploration of Transformer architectures or robust interpretability frameworks. Moreove...

---

## 141. Integrating problem-based learning and computational thinking: cultivating creative thinking in primary education

**Authors:** Ji Wang, Gary K. W. Wong

**Year:** 2025 | **Venue:** Frontiers in Education | **Citations:** 1 | **Score:** 0.325

[PDF](https://www.frontiersin.org/journals/education/articles/10.3389/feduc.2025.1625105/pdf) | [DOI](https://doi.org/10.3389/feduc.2025.1625105)

> This study examines how integrating problem-based learning (PBL) with computational thinking (CT) contributes to cultivating creative thinking in senior primary school students (grades 5–6). Creativity is a critical skill for addressing complex, real-world problems, yet its development in education remains challenging. A four-week “Unmanned Supermarket” project was designed, incorporating CT skill...

---

## 142. Integrating Cross-Modal Semantic Learning with Generative Models for Gesture Recognition

**Authors:** Shuangjiao Zhai, Zixin Dai, Zanxia Jin, Pinle Qin, Jianchao Zeng

**Year:** 2025 | **Venue:** Sensors | **Citations:** N/A | **Score:** 0.325

[PDF](https://www.mdpi.com/1424-8220/25/18/5783/pdf?version=1758100174) | [DOI](https://doi.org/10.3390/s25185783)

> Radio frequency (RF)-based human activity sensing is an essential component of ubiquitous computing, with WiFi sensing providing a practical and low-cost solution for gesture and activity recognition. However, challenges such as manual data collection, multipath interference, and poor cross-domain generalization hinder real-world deployment. Existing data augmentation approaches often neglect the ...

---

## 143. FineTec: Fine-Grained Action Recognition Under Temporal Corruption via Skeleton Decomposition and Sequence Completion

**Authors:** Dian Shao, Mingfei Shi, Like Liu

**Year:** 2025 | **Venue:** arXiv | **Citations:** N/A | **Score:** 0.325

[PDF](https://arxiv.org/pdf/2512.25067v1) | > Recognizing fine-grained actions from temporally corrupted skeleton sequences remains a significant challenge, particularly in real-world scenarios where online pose estimation often yields substantial missing data. Existing methods often struggle to accurately recover temporal dynamics and fine-grained spatial structures, resulting in the loss of subtle motion cues crucial for distinguishing simi...

---

## 144. FlashEval: Towards Fast and Accurate Evaluation of Text-to-Image Diffusion Generative Models

**Authors:** Lin Zhao, Tianchen Zhao, Zinan Lin, Xuefei Ning, Guohao Dai

**Year:** 2024 | **Venue:** Computer Vision and Pattern Recognition | **Citations:** 13 | **Score:** 0.325

[PDF](http://arxiv.org/pdf/2403.16379) | [DOI](https://doi.org/10.1109/CVPR52733.2024.01526)

> In recent years, there has been significant progress in the development of text-to-image generative models. Evaluating the quality of the generative models is one essential step in the development process. Unfortunately, the evaluation process could consume a significant amount of computational resources, making the required periodic evaluation of model performance (e.g., monitoring training progr...

---

## 145. Mod $p$ Poincaré duality for $p$-adic period domains

**Authors:** Guillaume Pignon-Ywanne

**Year:** 2025 | **Venue:** arXiv | **Citations:** N/A | **Score:** 0.325

[PDF](https://arxiv.org/pdf/2512.25029v1) | > In this article, we introduce a new class of smooth partially proper rigid analytic varieties over a $p$-adic field that satisfy Poincaré duality for étale cohomology with mod $p$-coefficients : the varieties satisfying "primitive comparison with compact support". We show that almost proper varieties, as well as p-adic (weakly admissible) period domains in the sense of Rappoport-Zink belong to thi...

---

## 146. Extreme nonlinear optics in optical fibers

**Authors:** Mario Ferraro, Bertrand Kibler, Pierre Béjot, Frédéric Gérome, Benoit Debord

**Year:** 2025 | **Venue:** arXiv | **Citations:** N/A | **Score:** 0.324

[PDF](https://arxiv.org/pdf/2512.25046v1) | > This paper reviews the field of extreme nonlinear optics in optical fibers, highlighting key phenomena and advancements. It discusses multiple ionization effects caused by femtosecond laser pulses that generate plasma and induce permanent material modifications, as well as plasma luminescence and its dependence on material imperfections. The formation and dynamics of plasma filaments, including he...

---

## 147. CoMoSpeech: One-Step Speech and Singing Voice Synthesis via Consistency Model

**Authors:** Zhe Ye, Wei Xue, Xuejiao Tan, Jie Chen, Qi-fei Liu

**Year:** 2023 | **Venue:** ACM Multimedia | **Citations:** 54 | **Score:** 0.324

[PDF](https://arxiv.org/pdf/2305.06908) | [DOI](https://doi.org/10.1145/3581783.3612061)

> Denoising diffusion probabilistic models (DDPMs) have shown promising performance for speech synthesis. However, a large number of iterative steps are required to achieve high sample quality, which restricts the inference speed. Maintaining sample quality while increasing sampling speed has become a challenging task. In this paper, we propose a Consistency Model-based Speech synthesis method, CoMo...

---

## 148. Towards precision cosmology with Voids x CMB correlations (I): Roman-Agora mock catalogs and pipeline validation

**Authors:** Mar Pérez Sar, Carlos Hernández Monteagudo, András Kovács, Alice Pisani

**Year:** 2025 | **Venue:** arXiv | **Citations:** N/A | **Score:** 0.324

[PDF](https://arxiv.org/pdf/2512.25040v1) | > We construct and validate a set of multi-purpose mock galaxy catalogs designed to capture, to different degrees of accuracy, the main characteristics of the Nancy Grace Roman Space Telescope survey. These catalogs provide a foundation for void statistics and various CMB cross-correlation analyses. Our approach differs from traditional halo occupation or abundance matching methods by directly trans...

---

## 149. Expectancy-value theories applied in Korean physical activity contexts: a meta-analysis

**Authors:** Jihyun Song, Wonseok Choi

**Year:** 2025 | **Venue:** Frontiers in Psychology | **Citations:** N/A | **Score:** 0.324

[PDF](https://www.frontiersin.org/journals/psychology/articles/10.3389/fpsyg.2025.1678503/pdf) | [DOI](https://doi.org/10.3389/fpsyg.2025.1678503)

> Introduction Recognizing expectancy-value theory as an influential framework for explaining the mechanism of motivation, relevant research has increased in physical activity contexts. This theory assumes that individuals’ motivation is situated and context-specific, shaped by the culture in which they live. Guided by expectancy-value theory, the purpose of this study was to synthesize the determin...

---

## 150. On exact Observability for Compactly perturbed infinite dimension system

**Authors:** Nisrine Charaf, Faouzi Triki

**Year:** 2025 | **Venue:** arXiv | **Citations:** N/A | **Score:** 0.324

[PDF](https://arxiv.org/pdf/2512.25041v1) | > In this paper, we study the observability of compactly perturbed infinite dimensional systems. Assuming that a given infinite-dimensional system with self-adjoint generator is exactly observable we derive sufficient conditions on a compact self adjoint perturbation to guarantee that the perturbed system stays exactly observable. The analysis is based on a careful asymptotic estimation of the spect...

---

## 151. Evaluating accuracy and bias of different comparative judgment equating methods against traditional statistical equating

**Authors:** Milja Curcin, Melissa Zhang Lee

**Year:** 2025 | **Venue:** Frontiers in Education | **Citations:** N/A | **Score:** 0.324

[PDF](https://www.frontiersin.org/journals/education/articles/10.3389/feduc.2025.1538486/pdf) | [DOI](https://doi.org/10.3389/feduc.2025.1538486)

> Traditional common-item or common-person statistical equating cannot always be used for standard maintaining or linking between test forms. In some contexts, comparative judgment (CJ) methods which capture expert judgment of quality of student work on different test forms have been trialed for this purpose. While plausibility, reliability and replicability of CJ outcomes has been shown to be high,...

---

## 152. A solvable generative model with a linear, one-step denoiser

**Authors:** Indranil Halder

**Year:** 2024 | **Venue:**  | **Citations:** N/A | **Score:** 0.324

> We develop an analytically tractable single-step diffusion model based on a linear denoiser and present an explicit formula for the Kullback-Leibler divergence between the generated and sampling distribution, taken to be isotropic Gaussian, showing the effect of finite diffusion time and noise scale. Our study further reveals that the monotonic fall phase of Kullback-Leibler divergence begins when...

---

## 153. AdaGReS:Adaptive Greedy Context Selection via Redundancy-Aware Scoring for Token-Budgeted RAG

**Authors:** Chao Peng, Bin Wang, Zhilei Long, Jinfang Sheng

**Year:** 2025 | **Venue:** arXiv | **Citations:** N/A | **Score:** 0.324

[PDF](https://arxiv.org/pdf/2512.25052v1) | > Retrieval-augmented generation (RAG) is highly sensitive to the quality of selected context, yet standard top-k retrieval often returns redundant or near-duplicate chunks that waste token budget and degrade downstream generation. We present AdaGReS, a redundancy-aware context selection framework for token-budgeted RAG that optimizes a set-level objective combining query-chunk relevance and intra-s...

---

## 154. Coordinated Humanoid Manipulation with Choice Policies

**Authors:** Haozhi Qi, Yen-Jen Wang, Toru Lin, Brent Yi, Yi Ma

**Year:** 2025 | **Venue:** arXiv | **Citations:** N/A | **Score:** 0.324

[PDF](https://arxiv.org/pdf/2512.25072v1) | > Humanoid robots hold great promise for operating in human-centric environments, yet achieving robust whole-body coordination across the head, hands, and legs remains a major challenge. We present a system that combines a modular teleoperation interface with a scalable learning framework to address this problem. Our teleoperation design decomposes humanoid control into intuitive submodules, which i...

---

## 155. The process of student engagement in Health-Promoting Schools: a co-design approach with Youth Engagement Coordinators in Nova Scotia, Canada

**Authors:** Julia Kontak, Rebecca Feicht, Stephanie Heath, Şebnem Özbek, Camille L. Hancock Friesen

**Year:** 2025 | **Venue:** Health Education | **Citations:** N/A | **Score:** 0.323

[PDF](https://www.emerald.com/he/article-pdf/doi/10.1108/HE-02-2025-0028/10287034/he-02-2025-0028en.pdf) | [DOI](https://doi.org/10.1108/he-02-2025-0028)

> Purpose Evidence supports meaningful student engagement in Health-Promoting Schools (HPS) approaches as an essential condition for enhancing student well-being, yet little is known about this process. This study sought to interpret the process of student engagement in HPS through the perspectives of Youth Engagement Coordinators (YECs). Design/methodology/approach This qualitative exploratory stud...

---

## 156. No-cost Bell Nonlocality Certification from Quantum Tomography and Its Applications in Quantum Magic Witnessing

**Authors:** Pawel Cieslinski, Lukas Knips, Harald Weinfurter, Wieslaw Laskowski

**Year:** 2025 | **Venue:** arXiv | **Citations:** N/A | **Score:** 0.323

[PDF](https://arxiv.org/pdf/2512.25068v1) | > Tomographic measurements are the standard tool for characterizing quantum states, yet they are usually regarded only as means for state reconstruction or fidelity measurement. Here, we show that the same Pauli-basis measurements (X, Y, Z) can be directly employed for the certification of nonlocality at no additional experimental cost. Our framework allows any tomographic data - including archival ...

---

## 157. Optimizing GPT-Based Distractor Generation for the Korean CSAT English Exam

**Authors:** Corinna Jung, Sanghoun Song

**Year:** 2025 | **Venue:**  | **Citations:** N/A | **Score:** 0.323

[PDF](https://www.researchsquare.com/article/rs-6680435/latest.pdf) | [DOI](https://doi.org/10.21203/rs.3.rs-6680435/v1)

> <title>Abstract</title> High-quality distractors are essential in multiple-choice questions to assess student understanding and diagnose misconceptions; however, constructing these distractors manually is labor-intensive. This study presents the first large-scale investigation of automated distractor generation (ADG) for the English section of Korea’s College Scholastic Ability Test (CSAT), a high...

---

## 158. Mitigating AI Bias in School Psychology: Toward Equitable and Ethical Implementation

**Authors:** Adam Lockwood, Jeffrey Brown

**Year:** 2025 | **Venue:**  | **Citations:** N/A | **Score:** 0.323

[PDF](https://osf.io/mh4rj_v2/download) | [DOI](https://doi.org/10.31234/osf.io/mh4rj_v2)

> The integration of Artificial Intelligence (AI) into school psychology is evolving rapidly, presenting both opportunities and challenges. AI has the potential to enhance educational and mental health services by facilitating data-driven decision-making, streamlining administrative tasks, and offering personalized interventions for students. However, biases inherent in AI systems—reflecting the pre...

---

## 159. OSDFace: One-Step Diffusion Model for Face Restoration

**Authors:** Jingkai Wang, Jue Gong, Lin Zhang, Zheng Chen, Xingang Liu

**Year:** 2024 | **Venue:** Computer Vision and Pattern Recognition | **Citations:** 16 | **Score:** 0.323

[DOI](https://doi.org/10.1109/CVPR52734.2025.01178)

> Diffusion models have demonstrated impressive performance in face restoration. Yet, their multi-step inference process remains computationally intensive, limiting their applicability in real-world scenarios. Moreover, existing methods often struggle to generate face images that are harmonious, realistic, and consistent with the subject’s identity. In this work, we propose OSDFace, a novel one-step...

---

## 160. Adaptive Probabilistic Inference of Human Intentions in Smart Manufacturing via Discrete Active Inference

**Authors:** Diluna Adeesha Warnakulasuriya, Juha Plosila, Mohammad-Hashem Haghbayan

**Year:** 2025 | **Venue:**  | **Citations:** N/A | **Score:** 0.323

[PDF](https://www.techrxiv.org/doi/pdf/10.36227/techrxiv.175825615.50962299/v1) | [DOI](https://doi.org/10.36227/techrxiv.175825615.50962299/v1)

> ...

---

## 161. From Inpainting to Editing: A Self-Bootstrapping Framework for Context-Rich Visual Dubbing

**Authors:** Xu He, Haoxian Zhang, Hejia Chen, Changyuan Zheng, Liyang Chen

**Year:** 2025 | **Venue:** arXiv | **Citations:** N/A | **Score:** 0.323

[PDF](https://arxiv.org/pdf/2512.25066v1) | > Audio-driven visual dubbing aims to synchronize a video's lip movements with new speech, but is fundamentally challenged by the lack of ideal training data: paired videos where only a subject's lip movements differ while all other visual conditions are identical. Existing methods circumvent this with a mask-based inpainting paradigm, where an incomplete visual conditioning forces models to simulta...

---

## 162. Intergenerational transmission of mental health problems : observational and quasi-experimental studies

**Authors:** Mengping Zhou

**Year:** 2025 | **Venue:**  | **Citations:** N/A | **Score:** 0.323

[PDF](https://openarchive.ki.se/articles/thesis/Intergenerational_transmission_of_mental_health_problems_observational_and_quasi-experimental_studies/29064848/2/files/57970879.pdf) | [DOI](https://doi.org/10.69622/29064848.v2)

> &lt;p dir="ltr"&gt;Mental health problems run in families, likely due to complex transmission mechanisms. Using nationwide Swedish registers following millions of parents and their children over decades, we investigated how and why mental health problems are transmitted across generations and whether modifiable factors can mitigate these intergenerational risks.&lt;/p&gt;&lt;p dir="ltr"&gt;Study I...

---

## 163. Chatbots, Bible Apps and Theological Bias

**Authors:** Jonas Kurlberg, Zoltán Schwáb, Daniel Washbrook, Ximian Xu

**Year:** 2025 | **Venue:** Cursor_ Zeitschrift für explorative Theologie | **Citations:** N/A | **Score:** 0.323

[PDF](https://cursor.pubpub.org/pub/azvojqqj/download/pdf) | [DOI](https://doi.org/10.21428/fb61f6aa.4ca2c070)

> ...

---

## 164. Grade Level and Gender Perspectives: Middle Grade Mathematics Affect and Identity Stabilization

**Authors:** Ruby L. Lynch-Arroyo, Scott A. Chamberlin, William Medina‐Jerez

**Year:** 2025 | **Venue:** European Journal of Mathematics and Science Education | **Citations:** N/A | **Score:** 0.323

[PDF](https://www.ejmse.com/articles/EJMSE_6_3_193.pdf) | [DOI](https://doi.org/10.12973/ejmse.6.3.191)

> Data from over 1,500 middle-grade mathematics students were used to investigate their mathematical affect and identity. Early secondary students were asked if they considered themselves mathematicians and a prompt was employed to substantiate their mathematical identity. Separating by gender and grade affiliation (6, 7, and 8), Chi-square and Z-score analyses were used to compare subgroups. Data s...

---

## 165. Vision transformers in precision agriculture: A comprehensive survey

**Authors:** Saber Mehdipour, Seyed Abolghasem Mirroshandel, Seyed Amirhossein Tabatabaei

**Year:** 2025 | **Venue:** Intelligent Systems with Applications | **Citations:** N/A | **Score:** 0.323

[PDF](https://doi.org/10.1016/j.iswa.2025.200617) | [DOI](https://doi.org/10.1016/j.iswa.2025.200617)

> ...

---

## 166. Perturbative Kondo destruction and global phase diagram of heavy fermion metals

**Authors:** Yiming Wang, Shouvik Sur, Chia-Chuan Liu, Qimiao Si

**Year:** 2025 | **Venue:** arXiv | **Citations:** N/A | **Score:** 0.323

[PDF](https://arxiv.org/pdf/2512.25036v1) | > Strange metals represent a foundational problem in quantum condensed matter physics, and heavy fermion systems provide a canonical setting to advance a general understanding. The concept of a Kondo destruction quantum critical point is widely invoked to describe the competition of the Kondo effect and the local-moment magnetism. Here, we develop a unified field-theoretic approach, analyzing this c...

---

## 167. HexaGen3D: StableDiffusion is just one step away from Fast and Diverse Text-to-3D Generation

**Authors:** Antoine Mercier, Ramin Nakhli, Mahesh Reddy, R. Yasarla, Hong Cai

**Year:** 2024 | **Venue:** arXiv.org | **Citations:** 23 | **Score:** 0.323

[DOI](https://doi.org/10.48550/arXiv.2401.07727)

> Despite the latest remarkable advances in generative modeling, efficient generation of high-quality 3D assets from textual prompts remains a difficult task. A key challenge lies in data scarcity: the most extensive 3D datasets encompass merely millions of assets, while their 2D counterparts contain billions of text-image pairs. To address this, we propose a novel approach which harnesses the power...

---

## 168. Tackling reflexivity in human-centred design research to benefit user needs and sensitivities

**Authors:** Antonia Clasina Södergren

**Year:** 2025 | **Venue:** Base Diseño e Innovación | **Citations:** N/A | **Score:** 0.323

[PDF](https://revistas.udd.cl/index.php/BDI/article/download/1121/969) | [DOI](https://doi.org/10.52611/bdi.num11.2025.1121)

> Longevity research highlights complexities of assessing the elderly’s future needs and impact of design solutions. Especially, the potential for (self-) transformation within human relations or interactions through design work (e.g., through technology) highlights the importance of design (researchers) being more aware of their design decisions and research impact and finding ways to discuss their...

---

## 169. Bilinear tau forms of quantum Painlevé equations and $\mathbb{C}^2/\mathbb{Z}_2$ blowup relations in SUSY gauge theories

**Authors:** Giulio Bonelli, Anton Shchechkin, Alessandro Tanzini

**Year:** 2025 | **Venue:** arXiv | **Citations:** N/A | **Score:** 0.322

[PDF](https://arxiv.org/pdf/2512.25051v1) | > We derive bilinear tau forms of the canonically quantized Painlevé equations, thereby relating them to those previously obtained from the $\mathbb{C}^2/\mathbb{Z}_2$ blowup relations for the $\mathcal{N}=2$ supersymmetric gauge theory partition functions on a general $Ω$-background. We fully fix the refined Painlevé/gauge theory dictionary by formulating the proper equations for the quantum nonaut...

---

## 170. Motivating change-oriented behavior through coaching leadership: the role of psychological entitlement and knowledge management

**Authors:** Jing Hu, Myeong-Cheol Choi, Hann Earl Kim

**Year:** 2025 | **Venue:** Frontiers in Psychology | **Citations:** N/A | **Score:** 0.322

[PDF](https://www.frontiersin.org/journals/psychology/articles/10.3389/fpsyg.2025.1626507/pdf) | [DOI](https://doi.org/10.3389/fpsyg.2025.1626507)

> Introduction In today’s fast-changing organizational environment, leadership styles such as coaching leadership are attracting attention for their potential to inspire innovative and proactive behaviors among employees. Coaching leadership focuses on employee development, motivation, and support, which makes it an ideal leadership style to address organizational challenges and drive change. This s...

---

## 171. The PDE-ODI principle and cylindrical mean curvature flows

**Authors:** Richard H. Bamler, Yi Lai

**Year:** 2025 | **Venue:** arXiv | **Citations:** N/A | **Score:** 0.322

[PDF](https://arxiv.org/pdf/2512.25050v1) | > We introduce a new approach for analyzing ancient solutions and singularities of mean curvature flow that are locally modeled on a cylinder. Its key ingredient is a general mechanism, called the \emph{PDE--ODI principle}, which converts a broad class of parabolic differential equations into systems of ordinary differential inequalities. This principle bypasses many delicate analytic estimates used...

---

## 172. Compound Estimation for Binomials

**Authors:** Yan Chen, Lihua Lei

**Year:** 2025 | **Venue:** arXiv | **Citations:** N/A | **Score:** 0.322

[PDF](https://arxiv.org/pdf/2512.25042v1) | > Many applications involve estimating the mean of multiple binomial outcomes as a common problem -- assessing intergenerational mobility of census tracts, estimating prevalence of infectious diseases across countries, and measuring click-through rates for different demographic groups. The most standard approach is to report the plain average of each outcome. Despite simplicity, the estimates are no...

---

## 173. From Religious Representation to Conceptual Truth: The Role of Religion in Hegel’s Philosophical System

**Authors:** Guanyu Guo

**Year:** 2025 | **Venue:** Religions | **Citations:** N/A | **Score:** 0.322

[PDF](https://www.mdpi.com/2077-1444/16/9/1187/pdf?version=1757917520) | [DOI](https://doi.org/10.3390/rel16091187)

> The present study interprets the indispensable mediating role of religion within Hegel’s monistic system. This study undertakes a systematic investigation of the development of Hegel’s religious thought in different periods, his logical reconstruction of multiple religions, and the positioning of religion within his system. The central argument posits that religions, particularly Christianity, ser...

---

## 174. The Logical Structure of Physical Laws: A Fixed Point Reconstruction

**Authors:** Eren Volkan Küçük

**Year:** 2025 | **Venue:** arXiv | **Citations:** N/A | **Score:** 0.322

[PDF](https://arxiv.org/pdf/2512.25057v1) | > We formalise the self referential definition of physical laws using monotone operators on a lattice of theories, resolving the pathologies of naive set theoretic formulations. By invoking Tarski fixed point theorem, we identify physical theories as least fixed points of admissibility constraints derived from Galois connections. We demonstrate that QED and General Relativity can be represented in s...

---

## 175. GaMO: Geometry-aware Multi-view Diffusion Outpainting for Sparse-View 3D Reconstruction

**Authors:** Yi-Chuan Huang, Hao-Jen Chien, Chin-Yang Lin, Ying-Huan Chen, Yu-Lun Liu

**Year:** 2025 | **Venue:** arXiv | **Citations:** N/A | **Score:** 0.321

[PDF](https://arxiv.org/pdf/2512.25073v1) | > Recent advances in 3D reconstruction have achieved remarkable progress in high-quality scene capture from dense multi-view imagery, yet struggle when input views are limited. Various approaches, including regularization techniques, semantic priors, and geometric constraints, have been implemented to address this challenge. Latest diffusion-based methods have demonstrated substantial improvements b...

---

## 176. Universal polar dual pairs of spherical codes found in $E_8$ and $Λ_{24}$

**Authors:** S. V. Borodachov, P. G. Boyvalenkov, P. D. Dragnev, D. P. Hardin, E. B. Saff

**Year:** 2025 | **Venue:** arXiv | **Citations:** N/A | **Score:** 0.321

[PDF](https://arxiv.org/pdf/2512.25037v1) | > We identify universal polar dual pairs of spherical codes $C$ and $D$ such that for a large class of potential functions $h$ the minima of the discrete $h$-potential of $C$ on the sphere occur at the points of $D$ and vice versa. Moreover, the minimal values of their normalized potentials are equal. These codes arise from the known sharp codes embedded in the even unimodular extremal lattices $E_8...

---

## 177. The Hochschild homology of a noncommutative symmetric quotient stack

**Authors:** Rina Anno, Vladimir Baranovsky, Timothy Logvinenko

**Year:** 2025 | **Venue:** arXiv | **Citations:** N/A | **Score:** 0.321

[PDF](https://arxiv.org/pdf/2512.25039v1) | > We prove an orbifold type decomposition theorem for the Hochschild homology of the symmetric powers of a small DG category $\mathcal{A}$. In noncommutative geometry, these can be viewed as the noncommutative symmetric quotient stacks of $\mathcal{A}$. We use this decomposition to show that the total Hochschild homology of the symmetric powers of $\mathcal{A}$ is isomorphic to the symmetric algebra...

---

## 178. One-step data-driven generative model via Schr\"odinger Bridge

**Authors:** Hanwen Huang

**Year:** 2024 | **Venue:**  | **Citations:** 4 | **Score:** 0.321

> Generating samples from a probability distribution is a fundamental task in machine learning and statistics. This article proposes a novel scheme for sampling from a distribution for which the probability density $\mu({\bf x})$ for ${\bf x}\in{\mathbb{R}}^d$ is unknown, but finite independent samples are given. We focus on constructing a Schr\"odinger Bridge (SB) diffusion process on finite horizo...

---

## 179. Testing Monotonicity in a Finite Population

**Authors:** Jiafeng Chen, Jonathan Roth, Jann Spiess

**Year:** 2025 | **Venue:** arXiv | **Citations:** N/A | **Score:** 0.321

[PDF](https://arxiv.org/pdf/2512.25032v1) | > We consider the extent to which we can learn from a completely randomized experiment whether everyone has treatment effects that are weakly of the same sign, a condition we call monotonicity. From a classical sampling perspective, it is well-known that monotonicity is untestable. By contrast, we show from the design-based perspective -- in which the units in the population are fixed and only treat...

---

## 180. Classification of Interacting Topological Crystalline Superconductors in Three Dimensions and Beyond

**Authors:** Shang-Qiang Ning, Xing-Yu Ren, Qing-Rui Wang, Yang Qi, Zheng-Cheng Gu

**Year:** 2025 | **Venue:** arXiv | **Citations:** N/A | **Score:** 0.320

[PDF](https://arxiv.org/pdf/2512.25069v1) | > Although classification for free-fermion topological superconductors (TSC) is established, systematically understanding the classification of 3D interacting TSCs remains difficult, especially those protected by crystalline symmetries like the 230 space groups. We build up a general framework for systematically classifying 3D interacting TSCs protected by crystalline symmetries together with discre...

---

## 181. Design and Preliminary Application of a CV-Based Multimodal Teaching Support System for Higher Vocational Education

**Authors:** Xiaoxue Yang

**Year:** 2025 | **Venue:** International Journal of Educational Review | **Citations:** N/A | **Score:** 0.320

[PDF](https://janfs-journals.org/index.php/ijer/article/download/35/16) | [DOI](https://doi.org/10.64583/b0j1xn67)

> In recent years, the steady progress of artificial intelligence has brought computer vision (CV) into the spotlight of educational research. Compared with traditional approaches that mainly depend on text or speech, CV is capable of processing multiple information streams—such as images, recognized text, and structural features—and reorganizing them into meaningful teaching resources. This ability...

---

## 182. Exploring the impact of role-playing exercises on cognitive and emotional processes: a social- and educational psychological perspective

**Authors:** Pär Löfstrand, Ingrid Zakrisson

**Year:** 2025 | **Venue:** Frontiers in Psychology | **Citations:** N/A | **Score:** 0.320

[PDF](https://www.frontiersin.org/journals/psychology/articles/10.3389/fpsyg.2025.1645213/pdf) | [DOI](https://doi.org/10.3389/fpsyg.2025.1645213)

> This study investigates the educational potential of role-playing exercises in addressing complex social phenomena such as attitude change, stereotyping, conformity, and racism. Building on prior research demonstrating the efficacy of role-play in reducing prejudice, this study aims to explore how such activities influence both cognitive and emotional engagement with these issues. The methodology ...

---

## 183. Impact of Artificial Intelligence on Education and Research: Pedagogy, Learning Analytics, and Academic Transformation

**Authors:** Sasmita Padhy

**Year:** 2025 | **Venue:**  | **Citations:** N/A | **Score:** 0.320

[PDF](https://www.deepscienceresearch.com/dsr/catalog/download/304/1459/2845) | [DOI](https://doi.org/10.70593/978-93-7185-525-9)

> This book discusses the impact of artificial intelligence on academic practice and research. This book demonstrates how AI and its applications in teaching, learning, and discovery impact opportunities for educational and scientific innovation. The description raises the good, and the bad, moral considerations, academic honesty obligations that border on cheating and human judgment colliding with ...

---

## 184. Learning to Attune and Revise: A Response to Silvis, Clarke‐Midura, Lee, and Shumway

**Authors:** Shirin Vossoughi

**Year:** 2025 | **Venue:** Science Education | **Citations:** 1 | **Score:** 0.319

[PDF](https://onlinelibrary.wiley.com/doi/pdfdirect/10.1002/sce.70017) | [DOI](https://doi.org/10.1002/sce.70017)

> ABSTRACT In this response to Silvis, Clarke‐Midura, Lee and Shumway as part of the special issue Centering Affect and Emotion Toward Justice and Dignity in Science Education, I discuss some of the key openings and complexities involved in interpreting the affective, powered, embodied, historical, and cultural layers present within the learning processes of young children in a computer science sett...

---

## 185. The variety of orthogonal frames

**Authors:** Laura Casabella, Alessio Sammartano

**Year:** 2025 | **Venue:** arXiv | **Citations:** N/A | **Score:** 0.319

[PDF](https://arxiv.org/pdf/2512.25058v1) | > An orthogonal n-frame is an ordered set of n pairwise orthogonal vectors. The set of all orthogonal n-frames in a d-dimensional quadratic vector space is an algebraic variety V(d,n). In this paper, we investigate the variety V(d,n) as well as the quadratic ideal I(d,n) generated by the orthogonality relations, which cuts out V(d,n). We classify the irreducible components of V(d,n), give criteria f...

---

## 186. Reliable and Resilient Collective Communication Library for LLM Training and Serving

**Authors:** Wei Wang, Nengneng Yu, Sixian Xiong, Zaoxing Liu

**Year:** 2025 | **Venue:** arXiv | **Citations:** N/A | **Score:** 0.319

[PDF](https://arxiv.org/pdf/2512.25059v1) | > Modern ML training and inference now span tens to tens of thousands of GPUs, where network faults can waste 10--15\% of GPU hours due to slow recovery. Common network errors and link fluctuations trigger timeouts that often terminate entire jobs, forcing expensive checkpoint rollback during training and request reprocessing during inference. We present R$^2$CCL, a fault-tolerant communication libr...

---

## 187. On the geometry and topology of representations: the manifolds of modular addition

**Authors:** Gabriela Moisescu-Pareja, Gavin McCracken, Harley Wiltzer, Vincent Létourneau, Colin Daniels

**Year:** 2025 | **Venue:** arXiv | **Citations:** N/A | **Score:** 0.317

[PDF](https://arxiv.org/pdf/2512.25060v1) | > The Clock and Pizza interpretations, associated with architectures differing in either uniform or learnable attention, were introduced to argue that different architectural designs can yield distinct circuits for modular addition. In this work, we show that this is not the case, and that both uniform attention and trainable attention architectures implement the same algorithm via topologically and...

---

## 188. Teachers Do More Than Teach: Compressing Image-to-Image Models

**Authors:** Qing Jin, Jian Ren, Oliver J. Woodford, Jiazhuo Wang, Geng Yuan

**Year:** 2021 | **Venue:** Computer Vision and Pattern Recognition | **Citations:** 63 | **Score:** 0.298

[PDF](http://arxiv.org/pdf/2103.03467) | [DOI](https://doi.org/10.1109/CVPR46437.2021.01339)

> Generative Adversarial Networks (GANs) have achieved huge success in generating high-fidelity images, however, they suffer from low efficiency due to tremendous computational cost and bulky memory usage. Recent efforts on compression GANs show noticeable progress in obtaining smaller generators by sacrificing image quality or involving a time-consuming searching process. In this work, we aim to ad...

---

