---
permalink: /projects/
title: "Projects"
excerpt: "Projects"
author_profile: true
redirect_from: 
  - /work/
---

**Using Interleaved Ensemble Unlearning to Keep Backdoors at Bay** \
Many backdoor defences exist for image classification tasks in litertature but prior work have not focused on developing backdoor defenses tailored to Vision Transformers (ViT) and existing defenses result in worse performance on ViTs compared to Convolutional Neural Networks (CNNs). To fill this gap, I developed a novel backdoor defence in this independent project that uses a trigger-only mini-ViT to capture easily learned samples and unlearn them during fine-tuning, preventing the injection of backdoor triggers when fine-tuning the ViT. My defence is not limited to defending ViTs and also serves as an alternative unlearning-based defence for other model architectures. I sincerely thank Prof. Tingjun Chen for providing most of this project's computing resources. Under review. \
Link to the project's [paper](https://arxiv.org/abs/2410.01128) (code is coming soon...)

**High-Fidelity RF Signal Mapping Using Geographic Databases** \
I worked with Professor Tingjun Chen to create an open-source tool using Sionna, Blender, and OpenStreetMap for generating signal-strength maps for mobile communication systems using real-world geographic data. The simulated signal strength maps are used to train a cascaded U-Net for predicting signal strength maps. Existing literature used proprietary software to generate signal-strength maps, so this open-source pipeline reduces the barrier for researchers in this area. I am immensely grateful to have worked with [Yiming Li](https://www.linkedin.com/in/yiming-li-824839231), who led this project. Published in IEEE DySPAN 2024. \
Link to the project's [paper](https://ieeexplore.ieee.org/document/10632773) and [code](https://github.com/functions-lab/geo2sigmap)

**Side-Channel Attacks on Mixture of Experts (MoE) models** \
This project investigates side-channel attacks on MoE layers that are present within some vision models. The attack's goal is to extract information about the inputs' class. \
Link to the project's [code](https://github.com/lizeyu090312/cs585_project_vmoe)