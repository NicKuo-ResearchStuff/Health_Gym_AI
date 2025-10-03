# Health Gym Implementation Series

<img src="Supporting_Images/ZFig016_ImplementationSeries.png" alt="Health + Data Illustration" width="300"/>

Hey, hello, and Kia Ora!

Welcome to the Implementation Series of the Health Gym.  
This is where we roll up our sleeves and walk through step-by-step worked examples, showing exactly how to move from raw tables to model-ready data and embeddings, and how feature schemas and data loaders intertwine in practice.

If you’ve seen the Health Gym overview, you’ll know the big picture.  
Here, we zoom in to the nuts and bolts of how synthetic ART for HIV datasets are prepared and used.

---

## What is this series?

This series is created to help you understand the mechanics behind our worked examples.
* Each post is a Colab-friendly walkthrough with runnable code; and
* each step builds on the last, so you can follow sequentially or jump in where relevant.

---

## Posts in the Series

### [Implementation 01: Pre-processing the ART for HIV Dataset](https://github.com/NicKuo-ResearchStuff/Health_Gym_AI/tree/main/Blogs/Blogs_Z_Implementation/Implementation01)

A step-by-step walkthrough of the Health Gym ART for HIV dataset pre-processing pipeline, covering categorical mapping, Box–Cox normalisation, sanity checks, and preparing model-ready data.

---

### [Implementation 02: From Table to DataLoader](https://github.com/NicKuo-ResearchStuff/Health_Gym_AI/tree/main/Blogs/Blogs_Z_Implementation/Implementation02)

A step-by-step guide showing how to reshape the ART for HIV dataset into patient–timestep sequences, of shape `((-1, Cur_Len, Feats_Len))`.

---

### [Implementation 03: Embedding Features for ART for HIV](https://github.com/NicKuo-ResearchStuff/Health_Gym_AI/tree/main/Blogs/Blogs_Z_Implementation/Implementation03)

A walkthrough on `ARTFeatureEmbedding` to embed mixed clinical features of the ART for HIV dataset into dense vectors using PyTorch, preparing them for sequence models.

---

### [Implementation 04: Rethinking Feature Schema and Data Loading](https://github.com/NicKuo-ResearchStuff/Health_Gym_AI/tree/main/Blogs/Blogs_Z_Implementation/Implementation04)

A reflective blog post uncovering the "white lies" we told about reshaping and feature schemas, showing how data loading and embedding are in fact inseparably linked.

---

### [Implementation 05: Shuffling Feature Schema + DataLoader](https://github.com/NicKuo-ResearchStuff/Health_Gym_AI/tree/main/Blogs/Blogs_Z_Implementation/Implementation05)

This post shows how feature schema and DataLoader come together, with one-hot expansion handled cleanly inside a unified pipeline.

---

### [Implementation 06: Curriculum Learning with Nested DataLoaders](https://github.com/NicKuo-ResearchStuff/Health_Gym_AI/tree/main/Blogs/Blogs_Z_Implementation/Implementation06)

We build multiple DataLoaders of increasing sequence lengths and use them in curriculum style, so the WGAN trains from short to long horizons within each epoch.

---

### [Implementation 07: Health Gym v1: An LSTM-based WGAN](https://github.com/NicKuo-ResearchStuff/Health_Gym_AI/tree/main/Blogs/Blogs_Z_Implementation/Implementation07)

This post discusses how LSTM is embedded a GAN architecture to power the generation of realistic sequences of synthetic EHR time series.

---

### [Implementation 08: Health Gym v1: The Eye of the Critic, Part 1](https://github.com/NicKuo-ResearchStuff/Health_Gym_AI/tree/main/Blogs/Blogs_Z_Implementation/Implementation08)

This post dives into how the Health Gym v1 critic "rates realism" by scoring real data, fake data, and generator outputs during WGAN-GP training.

---

### [Implementation 09: Health Gym v1: The Eye of the Critic, Part 2](https://github.com/NicKuo-ResearchStuff/Health_Gym_AI/tree/main/Blogs/Blogs_Z_Implementation/Implementation09)

This post unpacks the critic’s roles in -- scoring interpolated real–fake sequences and applying the gradient penalty via input-space derivatives -- to enforce 1-Lipschitzness and stabilise training.

(Last Edit: 2025-10-03)

