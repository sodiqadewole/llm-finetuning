# Understanding the DPO Loss

![alt text](image.png)

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
P_\theta(w_t\mid w_1,\ldots,w_{t-1})
=
\mathrm{softmax}(\mathbf{z}_{t-1})_{w_t}
```

Qwen generates a response autoregressively, so the probability of an entire response is the product of its token probabilities:

```math
\pi_\theta(y\mid x)
=
\prod_{t=1}^{T}
P_\theta(y_t\mid x,y_1,\ldots,y_{t-1})
```

In practice, DPO uses log probabilities, turning the product into a numerically stable sum:

```math
\log \pi_\theta(y\mid x)
=
\sum_{t=1}^{T}
\log P_\theta(y_t\mid x,y_1,\ldots,y_{t-1})
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

## Simple PyTorch Implementation

The following helper converts Qwen vocabulary logits into one log probability per response. `response_mask` is `1` for response tokens and `0` for prompt or padding tokens.

```python
import torch
import torch.nn.functional as F


def sequence_log_probs(logits, token_ids, response_mask):
	"""Sum log probabilities of the actual response tokens."""
	token_log_probs = F.log_softmax(logits, dim=-1)
	selected_log_probs = token_log_probs.gather(
		dim=-1, index=token_ids.unsqueeze(-1)
	).squeeze(-1)
	return (selected_log_probs * response_mask).sum(dim=-1)


def dpo_loss(
	policy_chosen_logps,
	policy_rejected_logps,
	reference_chosen_logps,
	reference_rejected_logps,
	beta=0.1,
):
	policy_preference = policy_chosen_logps - policy_rejected_logps
	reference_preference = reference_chosen_logps - reference_rejected_logps
	preference_improvement = policy_preference - reference_preference
	losses = -F.logsigmoid(beta * preference_improvement)
	return losses.mean(), preference_improvement
```

A small sequence-level example is:

```python
policy_chosen = torch.tensor([-4.0], requires_grad=True)
policy_rejected = torch.tensor([-6.0], requires_grad=True)
reference_chosen = torch.tensor([-4.5])
reference_rejected = torch.tensor([-5.5])

loss, improvement = dpo_loss(
	policy_chosen,
	policy_rejected,
	reference_chosen,
	reference_rejected,
	beta=0.1,
)

print("policy preference:", (policy_chosen - policy_rejected).item())  # 2.0
print("reference preference:", (reference_chosen - reference_rejected).item())  # 1.0
print("preference improvement:", improvement.item())  # 1.0
print("loss:", loss.item())  # approximately 0.6444

loss.backward()
print("chosen gradient:", policy_chosen.grad.item())   # negative: increase chosen log-probability
print("rejected gradient:", policy_rejected.grad.item())  # positive: decrease rejected log-probability
```

In actual training, the four sequence log probabilities are produced by `sequence_log_probs` from policy and frozen-reference Qwen forward passes. The reference tensors are computed under `torch.no_grad()`.

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
