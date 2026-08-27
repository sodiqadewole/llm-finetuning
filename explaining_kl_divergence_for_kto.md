# KL Divergence in KTO, Explained Simply

KL divergence measures **how much one probability distribution differs from another**.

In KTO, it helps compare:

- the trainable Qwen model, $\pi_\theta$,
- the original frozen Qwen model, $\pi_{\mathrm{ref}}$.

The reference model acts as an anchor. KTO teaches the trainable model to prefer desirable responses and avoid undesirable ones without changing its overall behavior unnecessarily.

## 1. A Simple Probability Example

Suppose the two models predict the next word after:

> The capital of France is ...

| Next token | Reference Qwen | Trainable Qwen |
|---|---:|---:|
| `Paris` | 70% | 50% |
| `London` | 20% | 30% |
| `Berlin` | 10% | 20% |

The models no longer make identical predictions. KL divergence summarizes the size of this difference in a single number.

If both models assign nearly the same probabilities, KL divergence is close to zero. If their probabilities are very different, KL divergence is larger.

## 2. The KL Divergence Equation

For one next-token prediction, the forward KL divergence is

```math
D_{\mathrm{KL}}
\left(
\pi_\theta\|\pi_{\mathrm{ref}}
\right)
=
\sum_{v\in\mathcal{V}}
\pi_\theta(v)
\log
\frac{\pi_\theta(v)}
{\pi_{\mathrm{ref}}(v)}.
```

The terms mean:

| Term | Meaning |
|---|---|
| $\mathcal{V}$ | Qwen's vocabulary containing every possible next token. |
| $v$ | One possible next token. |
| $\pi_\theta(v)$ | Probability assigned to token $v$ by the trainable Qwen model. |
| $\pi_{\mathrm{ref}}(v)$ | Probability assigned to the same token by the frozen reference model. |
| $\log(\pi_\theta(v)/\pi_{\mathrm{ref}}(v))$ | How much the probability of that token changed. |
| $D_{\mathrm{KL}}$ | Weighted total of the changes across the vocabulary. |

Each token's change is weighted by the probability that the trainable model assigns to it. Changes involving tokens the trainable model considers likely therefore matter more.

## 3. How to Interpret Its Value

KL divergence has these properties:

```math
D_{\mathrm{KL}}
\left(
\pi_\theta\|\pi_{\mathrm{ref}}
\right)
\geq 0.
```

- A value of $0$ means the two distributions are identical.
- A small positive value means the trainable model remains close to the reference model.
- A large value means the trainable model has changed its predictions substantially.

KL divergence is not an ordinary distance because it is not symmetric:

```math
D_{\mathrm{KL}}(P\|Q)
\neq
D_{\mathrm{KL}}(Q\|P)
```

in general. Asking "how surprising is $Q$ when samples come from $P$?" is different from asking the reverse question.

## 4. Where Qwen's Probabilities Come From

Qwen is a causal Transformer language model. For a prompt $x$ and previous completion tokens $y_{<t}$, its decoder produces a hidden state:

```math
\mathbf{h}_{t-1}^{(L)}.
```

The language-model head turns that hidden state into one logit per vocabulary token:

```math
\mathbf{z}_t
=
W_{\mathrm{LM}}\mathbf{h}_{t-1}^{(L)}.
```

Softmax converts the logits into next-token probabilities:

```math
\pi_\theta(v\mid x,y_{<t})
=
\frac{\exp(z_{t,v})}
{\sum_{u\in\mathcal{V}}\exp(z_{t,u})}.
```

The reference Qwen model performs the same calculation with frozen parameters:

```math
\pi_{\mathrm{ref}}(v\mid x,y_{<t}).
```

KL divergence compares these two distributions. KTO does not require a special KL head or classifier; the ordinary Qwen vocabulary logits provide everything needed.

## 5. From Token Probabilities to a Completion Probability

A completion $y$ contains tokens

```math
y=(y_1,y_2,\ldots,y_T).
```

Because Qwen generates text one token at a time, the probability of the complete response is

```math
\pi_\theta(y\mid x)
=
\prod_{t=1}^{T}
\pi_\theta(y_t\mid x,y_{<t}).
```

Implementations use log probabilities because sums are more numerically stable than products:

```math
\log\pi_\theta(y\mid x)
=
\sum_{t=1}^{T}
\log\pi_\theta(y_t\mid x,y_{<t}).
```

Only completion tokens are included. Prompt tokens and padding positions are masked out.

The reference model computes

```math
\log\pi_{\mathrm{ref}}(y\mid x)
=
\sum_{t=1}^{T}
\log\pi_{\mathrm{ref}}(y_t\mid x,y_{<t}).
```

## 6. KTO's Sequence Log-Ratio

For each labeled completion, KTO calculates

```math
s_\theta(x,y)
=
\log\pi_\theta(y\mid x)
-
\log\pi_{\mathrm{ref}}(y\mid x).
```

This can also be written as

```math
s_\theta(x,y)
=
\log
\frac{\pi_\theta(y\mid x)}
{\pi_{\mathrm{ref}}(y\mid x)}.
```

Its interpretation is straightforward:

- $s_\theta(x,y)>0$: the trainable model makes this completion more likely than the reference model does.
- $s_\theta(x,y)=0$: both models assign the same completion probability.
- $s_\theta(x,y)<0$: the trainable model makes this completion less likely.

This single-completion log-ratio is related to KL divergence, but it is **not itself the full KL divergence**. KL divergence is the expected log-ratio over responses sampled from the trainable policy:

```math
D_{\mathrm{KL}}
\left(
\pi_\theta(\cdot\mid x)
\|\pi_{\mathrm{ref}}(\cdot\mid x)
\right)
=
\mathbb{E}_{y'\sim\pi_\theta(\cdot\mid x)}
\left[
s_\theta(x,y')
\right].
```

A useful distinction is:

- $s_\theta(x,y)$ measures the change for **one particular completion**.
- KL divergence measures the **average change across possible completions** generated by the policy.

## 7. The KL Reference Point in KTO

KTO uses the model's average divergence as a reference point:

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

In simple terms, $z_0$ represents the trainable model's **typical amount of movement away from the reference model**.

KTO does not ask only whether a completion became more or less likely. It asks whether the completion moved in the desired direction relative to this typical amount of model drift.

For a desirable completion, KTO uses

```math
s_\theta(x,y)-z_0.
```

For an undesirable completion, it reverses the comparison:

```math
z_0-s_\theta(x,y).
```

Therefore:

- desirable completion: push its score above the KL reference point;
- undesirable completion: push its score below the KL reference point.

## 8. How the Reference Point Enters the KTO Loss

For a desirable example, KTO defines

```math
v_D(x,y)
=
\sigma
\left(
\beta[s_\theta(x,y)-z_0]
\right),
```

and minimizes

```math
\mathcal{L}_D
=
\lambda_D[1-v_D(x,y)].
```

For an undesirable example, it defines

```math
v_U(x,y)
=
\sigma
\left(
\beta[z_0-s_\theta(x,y)]
\right),
```

and minimizes

```math
\mathcal{L}_U
=
\lambda_U[1-v_U(x,y)].
```

Here:

- $\sigma$ is the sigmoid function;
- $\beta$ controls sensitivity to deviation from the reference model;
- $\lambda_D$ weights desirable examples;
- $\lambda_U$ weights undesirable examples.

The resulting behavior is

```text
Desirable completion   -> make it more likely than the reference
Undesirable completion -> make it less likely than the reference
KL reference point     -> account for the policy's typical overall drift
```

## 9. Why KTO Needs the KL Reference Point

Imagine that the trainable model has shifted all completion probabilities somewhat during training. A raw log-ratio of zero is no longer the most informative dividing line.

The KL reference point gives KTO a moving baseline:

- A desirable response should improve by more than ordinary background drift.
- An undesirable response should fall below that baseline.
- The policy remains anchored to the original Qwen model.
- Training is less likely to gain reward by changing unrelated language behavior.

A useful analogy is comparing a student's score with the class average rather than only checking whether the score is positive. The KL reference point represents the normal amount of change, and KTO evaluates each completion relative to it.

## 10. How TRL Estimates the KL Term

Calculating the exact expectation over every possible completion is impossible because the space of text sequences is enormous.

TRL estimates the reference point from a batch. It rotates completions to construct mismatched pairs: a prompt from one example is paired with a completion from another example. Let $y_i'$ denote the rotated completion for prompt $x_i$.

TRL approximately computes

```math
\widehat z_0
=
\max
\left(
0,
\frac{1}{B}
\sum_{i=1}^{B}
\left[
\log\pi_\theta(y_i'\mid x_i)
-
\log\pi_{\mathrm{ref}}(y_i'\mid x_i)
\right]
\right).
```

The maximum with zero ensures that the estimated reference point is non-negative, as a true KL divergence must be.

The estimate is detached before it is used:

```math
z_0
=
\operatorname{stopgrad}(\widehat z_0).
```

Detaching means that $z_0$ is treated as a fixed baseline during that gradient update. Gradients flow through the matched completion score $s_\theta(x,y)$, not through the KL estimate.

Because TRL rotates completions within each batch, the actual per-device batch size must be greater than one. With only one example, rotation produces the original completion and does not create a useful mismatched pair. The notebook uses a batch size of two.

## 11. A Small Numerical Example

Suppose a completion receives these sequence log probabilities:

```math
\log\pi_\theta(y\mid x)=-8,
\qquad
\log\pi_{\mathrm{ref}}(y\mid x)=-10.
```

Its sequence log-ratio is

```math
s_\theta(x,y)
=-8-(-10)
=2.
```

The trainable Qwen model therefore makes the completion

```math
\exp(2)\approx7.39
```

times as likely as the reference model does.

Assume the estimated KL reference point is

```math
z_0=0.4.
```

For a desirable label, KTO compares

```math
s_\theta-z_0
=2-0.4
=1.6.
```

This is positive, meaning the desirable completion has moved above the policy's typical drift level.

For an undesirable completion with $s_\theta=-1$, KTO instead compares

```math
z_0-s_\theta
=0.4-(-1)
=1.4.
```

This is also positive because the undesirable completion has become less likely than the reference baseline.

## 12. KL in KTO Versus a Direct KL Penalty

Many reinforcement-learning objectives add KL divergence directly to the loss:

```math
\mathcal{L}
=
\mathcal{L}_{\mathrm{task}}
+
\beta D_{\mathrm{KL}}
\left(
\pi_\theta\|\pi_{\mathrm{ref}}
\right).
```

In that form, any movement away from the reference model creates an explicit penalty.

KTO uses KL differently. It places the estimated divergence **inside the sigmoid comparison as a reference point**:

```math
\sigma
\left(
\beta[s_\theta-z_0]
\right).
```

Thus, the KL term is not simply an extra punishment added after the preference loss. It helps define what counts as a meaningful gain for a desirable response or a meaningful reduction for an undesirable response.

## 13. Summary

KL divergence answers:

> How different is the trainable Qwen model from the original reference Qwen model?

KTO applies it as follows:

1. Qwen produces next-token logits.
2. Softmax converts the logits into token probabilities.
3. Token log probabilities are summed to obtain completion log probabilities.
4. KTO compares trainable and reference completion probabilities using $s_\theta(x,y)$.
5. A batch-based KL estimate supplies the reference point $z_0$.
6. Desirable completions are pushed above that reference point.
7. Undesirable completions are pushed below it.

The key distinction is

```math
\underbrace{s_\theta(x,y)}_{\text{change for one completion}}
\qquad\text{versus}\qquad
\underbrace{z_0}_{\text{average policy drift}}.
```

Together, they let KTO change Qwen in the direction indicated by binary feedback while keeping the model anchored to its original behavior.
