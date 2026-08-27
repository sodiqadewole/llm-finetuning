# KL Divergence, Explained Simply

KL divergence measures how different one probability distribution is from another.

It is used throughout machine learning to compare predictions, compress knowledge from one model into another, regularize learned distributions, and limit how far a trained policy moves from a reference policy.

## 1. A Simple Example

Suppose two models predict tomorrow's weather:

| Weather | Distribution $P$ | Distribution $Q$ |
|---|---:|---:|
| Sunny | 70% | 50% |
| Cloudy | 20% | 30% |
| Rainy | 10% | 20% |

The distributions are not identical. KL divergence summarizes their difference in one number.

- If $P$ and $Q$ assign nearly identical probabilities, the KL divergence is close to zero.
- If they assign substantially different probabilities, the KL divergence is larger.

## 2. The Discrete Definition

For discrete outcomes, the KL divergence from $Q$ to $P$ is

```math
D_{\mathrm{KL}}(P\|Q)
=
\sum_{x\in\mathcal{X}}
P(x)\log\frac{P(x)}{Q(x)}.
```

The terms mean:

| Term | Meaning |
|---|---|
| $\mathcal{X}$ | Set of all possible outcomes. |
| $x$ | One possible outcome. |
| $P(x)$ | Probability of $x$ under the distribution being measured. |
| $Q(x)$ | Probability of $x$ under the comparison distribution. |
| $\log(P(x)/Q(x))$ | Log probability ratio for outcome $x$. |
| $D_{\mathrm{KL}}(P\|Q)$ | Expected log ratio when outcomes come from $P$. |

An equivalent expectation form is

```math
D_{\mathrm{KL}}(P\|Q)
=
\mathbb{E}_{x\sim P}
\left[
\log P(x)-\log Q(x)
\right].
```

This says: sample outcomes according to $P$, then measure how much more or less probable each outcome is under $P$ than under $Q$.

## 3. The Continuous Definition

For continuous probability densities $p(x)$ and $q(x)$, the sum becomes an integral:

```math
D_{\mathrm{KL}}(P\|Q)
=
\int
p(x)
\log\frac{p(x)}{q(x)}
\,dx.
```

The interpretation remains the same: it is the expected log-density ratio when samples come from $P$.

## 4. What the Value Means

KL divergence is always non-negative:

```math
D_{\mathrm{KL}}(P\|Q)\geq 0.
```

It equals zero exactly when the distributions agree almost everywhere:

```math
D_{\mathrm{KL}}(P\|Q)=0
\quad\Longleftrightarrow\quad
P=Q.
```

In practical terms:

- $0$: the distributions are identical.
- Small positive value: they are similar.
- Large value: they disagree substantially on important outcomes.

The numerical magnitude depends on the problem and logarithm base. Natural logarithms produce values in **nats**; base-2 logarithms produce values in **bits**.

## 5. Why KL Is Not a Distance

KL divergence is not symmetric:

```math
D_{\mathrm{KL}}(P\|Q)
\neq
D_{\mathrm{KL}}(Q\|P)
```

in general.

It also does not satisfy the triangle inequality. It is therefore called a divergence rather than a mathematical distance.

Direction matters because the first distribution determines which outcomes receive weight:

```math
D_{\mathrm{KL}}(P\|Q)
=
\sum_x
\underbrace{P(x)}_{\text{weight}}
\log\frac{P(x)}{Q(x)}.
```

## 6. Forward and Reverse KL

The names depend on context, but a common convention is:

```math
\text{Forward KL:}
qquad
D_{\mathrm{KL}}(P_{\mathrm{target}}\|Q_\theta),
```

```math
\text{Reverse KL:}
qquad
D_{\mathrm{KL}}(Q_\theta\|P_{\mathrm{target}}).
```

### Forward KL

Forward KL averages according to the target distribution. It heavily penalizes the learned model when it assigns very little probability to outcomes that the target considers likely. This often encourages broad **mode covering**.

### Reverse KL

Reverse KL averages according to the learned distribution. It strongly discourages the learned model from placing probability where the target probability is low. This can produce **mode-seeking** behavior.

Neither direction is universally better. The correct direction depends on what is sampled, what is fixed, and which errors matter in the application.

## 7. Why Zero Probabilities Matter

If an outcome can occur under $P$ but is impossible under $Q$, then

```math
P(x)>0
\quad\text{and}\quad
Q(x)=0
```

causes

```math
D_{\mathrm{KL}}(P\|Q)=\infty.
```

The comparison distribution must assign nonzero probability everywhere that the measured distribution can produce samples.

Neural-network softmax outputs are theoretically positive for every class, although finite-precision arithmetic and explicit masking still require care.

## 8. Relation to Entropy and Cross-Entropy

Entropy measures the uncertainty inside $P$:

```math
H(P)
=
-\sum_x P(x)\log P(x).
```

Cross-entropy measures how well $Q$ represents outcomes from $P$:

```math
H(P,Q)
=
-\sum_x P(x)\log Q(x).
```

They are related by

```math
D_{\mathrm{KL}}(P\|Q)
=
H(P,Q)-H(P).
```

When $P$ is fixed, $H(P)$ is constant. Therefore, minimizing cross-entropy with respect to $Q$ also minimizes

```math
D_{\mathrm{KL}}(P\|Q).
```

This is why ordinary classification training with cross-entropy can be understood as fitting the predicted distribution to the data distribution.

## 9. A Numerical Calculation

Consider two binary distributions:

```math
P=(0.8,0.2),
\qquad
Q=(0.6,0.4).
```

Using natural logarithms,

```math
D_{\mathrm{KL}}(P\|Q)
=
0.8\log\frac{0.8}{0.6}
+
0.2\log\frac{0.2}{0.4}.
```

Numerically,

```math
D_{\mathrm{KL}}(P\|Q)
\approx
0.8(0.2877)+0.2(-0.6931)
\approx
0.0915\ \text{nats}.
```

An individual term may be negative, but the complete sum cannot be negative.

Reversing the distributions gives a different result:

```math
D_{\mathrm{KL}}(Q\|P)
=
0.6\log\frac{0.6}{0.8}
+
0.4\log\frac{0.4}{0.2}
\approx
0.1046\ \text{nats}.
```

## 10. Estimating KL with Samples

For large spaces, summing over every possible outcome may be impractical. The expectation form supports Monte Carlo estimation.

Draw samples

```math
x_1,\ldots,x_N\sim P.
```

Then estimate

```math
D_{\mathrm{KL}}(P\|Q)
\approx
\frac{1}{N}
\sum_{i=1}^{N}
\left[
\log P(x_i)-\log Q(x_i)
\right].
```

This estimator may be noisy and can even be negative for a finite batch, although the exact KL divergence is non-negative.

## 11. KL Divergence in Language Models

A language model produces logits for every possible next token. For a hidden state $\mathbf{h}_t$, the language-model head computes

```math
\mathbf{z}_t
=
W_{\mathrm{LM}}\mathbf{h}_t.
```

Softmax converts these logits into a next-token distribution:

```math
\pi_\theta(v\mid x_{\leq t})
=
\frac{\exp(z_{t,v})}
{\sum_{u\in\mathcal{V}}\exp(z_{t,u})}.
```

Two language models can be compared at that position with

```math
D_{\mathrm{KL}}
\left(
\pi_\theta(\cdot\mid x_{\leq t})
\|\pi_{\mathrm{ref}}(\cdot\mid x_{\leq t})
\right)
=
\sum_{v\in\mathcal{V}}
\pi_\theta(v\mid x_{\leq t})
\log
\frac{\pi_\theta(v\mid x_{\leq t})}
{\pi_{\mathrm{ref}}(v\mid x_{\leq t})}.
```

For an autoregressive completion $y=(y_1,\ldots,y_T)$,

```math
\log\pi_\theta(y\mid x)
=
\sum_{t=1}^{T}
\log\pi_\theta(y_t\mid x,y_{<t}).
```

The sampled sequence log-ratio is

```math
s_\theta(x,y)
=
\log\pi_\theta(y\mid x)
-
\log\pi_{\mathrm{ref}}(y\mid x).
```

This value measures the change for one completion. It is not itself the full KL divergence. Taking its expectation over completions from the policy produces sequence-level KL:

```math
D_{\mathrm{KL}}
\left(
\pi_\theta(\cdot\mid x)
\|\pi_{\mathrm{ref}}(\cdot\mid x)
\right)
=
\mathbb{E}_{y\sim\pi_\theta(\cdot\mid x)}
\left[
s_\theta(x,y)
\right].
```

## 12. Common Machine-Learning Applications

### Classification

Cross-entropy training minimizes forward KL from the target-label distribution to the model prediction:

```math
\min_\theta
D_{\mathrm{KL}}
\left(
P_{\mathrm{data}}\|P_\theta
\right).
```

### Knowledge distillation

A student model learns to match a teacher model's softened output distribution:

```math
\mathcal{L}_{\mathrm{KD}}
=
T^2
D_{\mathrm{KL}}
\left(
P_{\mathrm{teacher}}^{(T)}
\|P_{\mathrm{student}}^{(T)}
\right),
```

where $T$ is the softmax temperature.

### Variational autoencoders

A VAE encourages its learned latent distribution to remain close to a prior:

```math
\mathcal{L}_{\mathrm{VAE}}
=
\mathcal{L}_{\mathrm{reconstruction}}
+
D_{\mathrm{KL}}
\left(
q_\phi(z\mid x)\|p(z)
\right).
```

### Policy optimization and RLHF

Policy-training methods often penalize movement away from a reference model:

```math
\mathcal{L}
=
\mathcal{L}_{\mathrm{policy}}
+
\beta
D_{\mathrm{KL}}
\left(
\pi_\theta\|\pi_{\mathrm{ref}}
\right).
```

The coefficient $\beta$ controls the tradeoff between maximizing reward and preserving the reference model's behavior.

### GRPO

GRPO commonly uses a sampled per-token KL estimator to keep the policy near a frozen reference model. Conceptually, the regularizer remains

```math
\beta
D_{\mathrm{KL}}
\left(
\pi_\theta\|\pi_{\mathrm{ref}}
\right).
```

### KTO

KTO uses estimated KL as a reference point rather than only as an additive penalty. A labeled completion is evaluated relative to the model's typical drift:

```math
z_0
=
\mathbb{E}_{x}
\left[
D_{\mathrm{KL}}
\left(
\pi_\theta(\cdot\mid x)
\|\pi_{\mathrm{ref}}(\cdot\mid x)
\right)
\right].
```

Desirable completions are pushed above this reference point, while undesirable completions are pushed below it.

## 13. Practical Cautions

1. **Always state the direction.** $D_{\mathrm{KL}}(P\|Q)$ and $D_{\mathrm{KL}}(Q\|P)$ optimize different behavior.
2. **Do not call one log-ratio a KL divergence.** KL is an expectation of log-ratios.
3. **Mask invalid positions.** For language models, prompt and padding tokens may need exclusion depending on the objective.
4. **Expect noisy estimates.** Sample-based KL estimates vary across batches.
5. **Check the reduction.** Per-token means, per-sequence means, and batch sums have different scales.
6. **Track the logarithm base.** Natural logs give nats; base-2 logs give bits.
7. **Avoid zero support.** If $Q(x)=0$ where $P(x)>0$, forward KL is infinite.

## 14. Summary

KL divergence answers:

> When outcomes are generated according to $P$, how different or surprising would they be under $Q$?

Its central equation is

```math
D_{\mathrm{KL}}(P\|Q)
=
\mathbb{E}_{x\sim P}
\left[
\log P(x)-\log Q(x)
\right].
```

The essential facts are:

- KL divergence compares complete probability distributions.
- It is non-negative but not symmetric.
- Its direction changes what kinds of errors are emphasized.
- Cross-entropy minimization is closely related to KL minimization.
- Exact KL can be calculated by summation or integration; large problems often use sample estimates.
- In language-model alignment, KL helps improve desired behavior without allowing the trained policy to drift arbitrarily far from a reference model.