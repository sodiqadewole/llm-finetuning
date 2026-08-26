# Understanding the DPO Loss

The Direct Preference Optimization (DPO) loss compares how strongly a trainable model prefers a **chosen response** over a **rejected response**, relative to the same preference under a frozen reference model:

```math
\mathcal{L}_{\mathrm{DPO}}(\theta)
=
-\mathbb{E}_{(x,y^+,y^-)}
\left[
\log \sigma
\left(
\beta
\left[
\log \frac{\pi_\theta(y^+\mid x)}{\pi_{\mathrm{ref}}(y^+\mid x)}
-
\log \frac{\pi_\theta(y^-\mid x)}{\pi_{\mathrm{ref}}(y^-\mid x)}
\right]
\right)
\right]
```

## Inputs

- $x$: the prompt, such as a user message.
- $y^+$: the chosen or preferred response.
- $y^-$: the rejected response.
- $(x,y^+,y^-)$: one preference example from the dataset.
- $\mathbb{E}$: the average loss over examples in the training batch or dataset.

For example:

```text
x  = "Explain photosynthesis."
y+ = "Photosynthesis converts light energy..."
y- = "Photosynthesis is how animals digest food."
```

## From Qwen to Probabilities

Qwen2.5 is a causal Transformer language model. Given a token sequence, its final hidden states pass through a language-model output head:

```math
\mathbf{z}_t = W_{\mathrm{LM}}\mathbf{h}_t
```

where:

- $\mathbf{h}_t$ is Qwen's final hidden representation at position $t$.
- $W_{\mathrm{LM}}$ is its vocabulary output matrix.
- $\mathbf{z}_t$ contains one logit for every token in Qwen's vocabulary.

A softmax converts those logits into next-token probabilities:

```math
P_\theta(w_t\mid w_{<t})
=
\operatorname{softmax}(\mathbf{z}_{t-1})_{w_t}
```

Qwen generates a response autoregressively, so the probability of an entire response is the product of its token probabilities:

```math
\pi_\theta(y\mid x)
=
\prod_{t=1}^{T}
P_\theta(y_t\mid x,y_{<t})
```

In practice, DPO uses log probabilities, turning the product into a numerically stable sum:

```math
\log \pi_\theta(y\mid x)
=
\sum_{t=1}^{T}
\log P_\theta(y_t\mid x,y_{<t})
```

Only response tokens are normally included. Prompt and padding positions are masked out.

## Policy and Reference Models

### Trainable Policy

```math
\pi_\theta(y\mid x)
```

This is the Qwen2.5 model being fine-tuned. Its parameters $\theta$ receive gradients and change during training.

For the chosen response:

```math
\log\pi_\theta(y^+\mid x)
```

For the rejected response:

```math
\log\pi_\theta(y^-\mid x)
```

### Reference Policy

```math
\pi_{\mathrm{ref}}(y\mid x)
```

This is normally a frozen copy of the original pretrained or supervised-fine-tuned Qwen model. Its parameters are not updated.

It acts as an anchor, preventing the policy from moving too far from the behavior of the original model.

## Log-Ratio Terms

For the chosen response:

```math
r^+
=
\log\frac{\pi_\theta(y^+\mid x)}
{\pi_{\mathrm{ref}}(y^+\mid x)}
=
\log\pi_\theta(y^+\mid x)
-
\log\pi_{\mathrm{ref}}(y^+\mid x)
```

This measures how much more or less likely the trainable Qwen model makes the chosen response compared with the reference model.

Similarly:

```math
r^-
=
\log\frac{\pi_\theta(y^-\mid x)}
{\pi_{\mathrm{ref}}(y^-\mid x)}
```

The central DPO score is:

```math
\Delta = r^+ - r^-
```

Interpretation:

- $\Delta>0$: the policy has shifted toward the chosen response relative to the rejected response.
- $\Delta=0$: its relative preference has not changed from the reference.
- $\Delta<0$: it has shifted in the wrong direction.

An equivalent expanded form is:

```math
\Delta
=
\underbrace{
\left[
\log\pi_\theta(y^+\mid x)
-
\log\pi_\theta(y^-\mid x)
\right]}_{\text{policy preference}}
-
\underbrace{
\left[
\log\pi_{\mathrm{ref}}(y^+\mid x)
-
\log\pi_{\mathrm{ref}}(y^-\mid x)
\right]}_{\text{reference preference}}
```

DPO therefore trains Qwen to prefer $y^+$ over $y^-$ **more strongly than the reference model does**.

## Beta, Sigmoid, and Loss

$\beta$ controls the strength of the comparison:

```math
s = \beta\Delta
```

A larger $\beta$ makes the loss more sensitive to preference differences. It also corresponds to stronger pressure to stay near the reference policy in the reward-model interpretation of DPO.

The sigmoid converts the score into a value between zero and one:

```math
\sigma(s)=\frac{1}{1+e^{-s}}
```

It can be interpreted as the predicted probability that $y^+$ should be preferred over $y^-$. Finally,

```math
-\log\sigma(s)
```

is a binary logistic loss:

- If Qwen strongly favors $y^+$, then $s$ is positive, $\sigma(s)$ approaches $1$, and the loss approaches $0$.
- If Qwen favors $y^-$, then $s$ is negative, $\sigma(s)$ approaches $0$, and the loss becomes large.

## Training Flow

For every preference pair, DPO:

1. Tokenizes $(x,y^+)$ and $(x,y^-)$ using Qwen's tokenizer and chat template.
2. Runs both sequences through the trainable Qwen model.
3. Runs them through the frozen reference Qwen model.
4. Applies `log_softmax` to the vocabulary logits.
5. Selects the log probability assigned to each actual response token.
6. Sums those token log probabilities over each response.
7. Calculates $\Delta$, applies $\beta$, and computes $-\log\sigma(\beta\Delta)$.
8. Backpropagates only through $\pi_\theta$.

The Qwen architecture supplies token logits and probabilities, but the DPO relationship itself comes from the preference dataset and loss function. DPO does not require adding a separate reward-model head to Qwen.
