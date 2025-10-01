# Health Gym v1: An LSTM-based WGAN

<img src="Supporting_Images/ZFig036_WganV1Logo.png" width="600"/>  

Hey, hello, and Kia Ora!

Over the next few blogs, we will break down the core components of the Health Gym WGAN-GP model.

---

## Step 1: LSTM From Scratch

Long Short-Term Memory (LSTM) remains one of the most widely used tools for sequence modelling. Its defining feature is the use of gates to regulate information flow across time.

An LSTM maintains both a hidden state (h_t) and a cell state (c_t). At each time step (t), the equations are:

$f_t = \sigma(W_f x_t + U_f h_{t-1}),$ \
$i_t = \sigma(W_i x_t + U_i h_{t-1}),$ \
$A_t = \tanh(W_a x_t + U_a h_{t-1}),$ \
$o_t = \sigma(W_o x_t + U_o h_{t-1}),$ \
$c_t = f_t \odot c_{t-1} + i_t \odot A_t,$ \
$h_t = o_t \odot \tanh(c_t).$

or

```python
class MyLSTM(nn.Module):
    def __init__(self, ID, HD):
        super().__init__()
        self.i2h = nn.Linear(ID, HD * 4)
        self.h2h = nn.Linear(HD, HD * 4)
        self.HD = HD

    def forward(self, x0):
        h_t = torch.zeros(x0.shape[0], self.HD, device=x0.device)
        c_t = torch.zeros(x0.shape[0], self.HD, device=x0.device)
        h_seq = []

        for t in range(x0.shape[1]):
            x_t = x0[:, t, :]
            i2h_out = self.i2h(x_t).chunk(4, dim=1)
            h2h_out = self.h2h(h_t).chunk(4, dim=1)

            F_i, I_i, A_i, O_i = i2h_out
            F_h, I_h, A_h, O_h = h2h_out

            f_t = torch.sigmoid(F_i + F_h)
            i_t = torch.sigmoid(I_i + I_h)
            a_t = torch.tanh(A_i + A_h)
            o_t = torch.sigmoid(O_i + O_h)

            c_t = f_t * c_t + i_t * a_t
            h_t = o_t * torch.tanh(c_t)

            h_seq.append(h_t.unsqueeze(1))

        h_seq = torch.cat(h_seq, dim=1)
        return h_seq, (h_t, c_t)
```

---

## Step 2: The GAN Architecture

In original the Health Gym project (Kuo et al., *Scientific Data* 2022), LSTMs are embedded inside a Wasserstein GAN with Gradient Penalty (WGAN-GP) to generate realistic clinical time series. Here’s how it works.

---

### (a) Overview

The adversarial loop:

```
Real clinical dataset ──▶ Critic ──▶ Critic Loss
                      │
Latent Gaussian noise ──▶ Generator ──▶ Synthetic data ──▶ Critic ──▶ Generator Loss
```

* Critic loss: Wasserstein objective + gradient penalty (GP)
* Generator loss: fool the critic + correlation alignment
* Updates alternate between critic and generator.

---

### (b) Generator Pipeline

```
Latent Gaussian inputs
   │
   └─▶ biLSTM
         │
         └─▶ Fully Connected
               │
               └─▶ Fully Connected
                     │
                     └─▶ Fully Connected
                           │
                           └─▶ Synthetic clinical sequence
```

Role: interpolate temporal patterns from latent space → map to realistic patient trajectories.
Design: one biLSTM + three dense layers.

---

### (c) Critic Pipeline

```
Real or synthetic sequences
   │
   ├─▶ Numeric vars ──▶ Fully Connected ──▶┐
   │                                       ├─▶ Fully Connected
   └─▶ Categorical/Binary vars ──▶ Soft Embedding ──▶┘
                                               │
                                               └─▶ biLSTM
                                                     │
                                                     └─▶ Fully Connected
                                                           │
                                                           └─▶ Realism score
```

Design: embeddings for discrete variables, dense mixing, a biLSTM for temporal relations, final FC for realism score.

---

## Step 3: Wrapping Up

In this post, we explored how LSTMs can be embedded within GANs, showing how low-level sequence modules underpin high-level generative tasks such as synthetic EHR creation.

In the next post, we’ll take a closer look at the discriminator (the critic in WGAN), breaking down how it learns to rank the realism of generated sequences.

Cheers,</br>
\- Nic

(Last Edit: 2025-10-01)
