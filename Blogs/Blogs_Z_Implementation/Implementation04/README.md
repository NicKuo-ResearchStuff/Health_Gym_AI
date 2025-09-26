# Rethinking Feature Schema and Data Loading

<img src="Supporting_Images/ZFig032_LiarLiarPantsOnFire.png" width="600"/>

Hey, hello, and Kia Ora!

If you’ve been following along, you’ve seen how in [Implementation 02](https://github.com/NicKuo-ResearchStuff/Health_Gym_AI/tree/main/Blogs/Blogs_Z_Implementation/Implementation02) we reshaped the ART for HIV dataset into `(patients, timesteps, features)`, and in [Implementation 03](https://github.com/NicKuo-ResearchStuff/Health_Gym_AI/tree/main/Blogs/Blogs_Z_Implementation/Implementation03) we built a feature schema to embed those features into dense vectors.

In both posts, I told you a couple of convenient lies. These simplifications were intentional -- they let us focus on one idea at a time. But it’s worth pausing here to reflect: in reality, data loading and feature embedding are deeply entwined. You can’t really think about one without the other.

Let’s unpack what I glossed over.

---

## The (WHITE) Lies We Told

### In Implementation 02

I said:

> "`reshape((-1, Cur_Len, Feats_Len))` reorganises row-based data into `(patients, timesteps, features)`."

That’s true only at a superficial level.
In reality:

* What the model actually consumes is `(patients, timesteps, embedded_dim)`.
* That embedded dimension only appears after we apply the feature schema and embedding layers.

Here’s the simplified flow I previously showed you:

```
Figshare CSV  ──▶  DataFrame
     │
     └─▶ drop helper cols + map categoricals + scale reals
     │
     └─▶ reshape((-1, Cur_Len, Feats_Len))
           = (patients, timesteps, features)
     │
     └─▶ TensorDataset + DataLoader
```

---

### In Implementation 03

I said:

> “Each feature occupies a single slot, so `index_start` and `index_end` always differ by 1.”

That’s also a mental shortcut.
In reality:

* Many categorical features expand into multiple one-hot columns.
* To stabilise the schema, we often pad blocks with "phantom" zero-columns so every category group always has the same width.
* As a result, `index_start` and `index_end` don’t line up as neatly as 0,1,2,3 ...
* Those ranges later become essential when we compute correlations in the embedded space.

Here’s the simplified flow I previously showed you:

```
Reshaped data (patients, timesteps, Feats_Len)
     │
     └─▶ define dtype schema
           • each feature maps to one slot
           • index_start / index_end differ by 1
     │
     └─▶ embedding module
           • nn.Linear for real
           • nn.Embedding for categorical/bin
     │
     └─▶ forward pass → (B, T, embedded_dim)
```

---

## Wrapping Up

In a future blog, we’ll bring these strands together:

* How the DataLoader isn’t just about `(patients, timesteps, features)` but about structuring inputs for embedding.
* How the Feature Schema isn’t just about embeddings, but also about how we organise and validate the data we load.
* And how correlation analysis in embedded space forces us to treat loading and embedding as one joint problem.

So if Blog 02 was about "reshaping" and Blog 03 was about "embedding", the next step is to rethink them as a single pipeline.

Cheers,</br>
\- Nic

(Last Edit: 2025-09-26)
