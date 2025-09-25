# ART for HIV using the Health Gym v2.5 Model (WGAN-GP + 3EOT + VAE + Buffer)

<img src="Supporting_Images/ZFig029_HandsOnV2_5.png" width="600"/>

Hey, hello, and Kia Ora!

In this walkthrough, we’ll explore the Health Gym v2.5 model.
Where v1 relied on a plain GAN, and v2 introduced a meta-learning autoencoder with latent buffers, v2.5 takes an extra step to swap the recurrent generator for a Transformer encoder.

This change means the generator can capture long-range dependencies across patient trajectories more efficiently, while still benefiting from the contextualised latent prior introduced in v2.

---

## About this Example

As before, this notebook is a worked sample (see the Colab script in this folder) designed to illustrate the model’s workflow.

A few notes up front:

* v1 (Scientific Data 2022): GAN only, sampling from random noise, see [here](https://github.com/NicKuo-ResearchStuff/Health_Gym_AI/tree/main/Blogs/Blogs003_HandsOn(HealthGymV1)).
* v2 (JBI 2023): GAN + VAE + latent buffer, adding contextualised noise, see [here](https://github.com/NicKuo-ResearchStuff/Health_Gym_AI/tree/main/Blogs/Blogs004_HandsOn(HealthGymV2)).
* v2.5 (JBI 2023 as well): keeps the meta-learning prior of v2 but replaces the generator with an encoder-only Transformer.

At a high level, this change allows the mode to directly attend across the entire timeline of a patient.

---

## What This Workflow Does

At a glance, the main steps are:

1. Load the ART for HIV cohort.
2. Standardise categorical and numeric variables (as before).
3. Meta-learning step: the autoencoder builds latent prototypes and scales.
4. Train the generative model: now with a Transformer encoder backbone (3EOT) in the generator.
5. Generate synthetic data: richer, more stable trajectories with better global coherence.
6. Evaluate realism: distributions, correlations, and subgroup coverage.

---

## Results

After 100 epochs

<img src="Supporting_Images/ZFig030_HealthGymV2_5Results_Epoch100.png" width="900"/>  

Already at 100 epochs, we see the benefit of the Transformer backbone: the categorical distributions track the real cohort more closely (*e.g.,* ethnicity).

After 200 epochs

<img src="Supporting_Images/ZFig031_HealthGymV2_5Results_Epoch200.png" width="900"/>  

By 200 epochs, v2.5 delivers stable overlaps across lab values and demographic variables. Importantly, minority regimen categories are preserved with less distortion.
Personally I am not too worried about the weird double-spikes in CD4 counts, that is more of a feature relating to the use of a memory buffer, which is out of the scope of this blog.


### Takeaway

Health Gym v2.5 shows how Transformer-based generators can enhance synthetic health data.
By combining meta-learning from v2 with the sequence-modelling power of attention, we achieve:

* More realistic lab value distributions.
* Better representation of minority categories.

---

## What’s Next

This walkthrough is still high-level. In the upcoming Implementation Series, we’ll unpack details of the 3EOT Transformer design.

Cheers,</br>
\- Nic

(Last Edit: 2025-09-26)
