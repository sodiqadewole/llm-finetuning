# Understanding the KTO Loss with Qwen2

Kahneman-Tversky Optimization (KTO) aligns a language model using **unpaired binary feedback**. Each training example needs only:

- a prompt $x$,
- one completion $y$,
- a label saying whether that completion is desirable or undesirable.

Unlike DPO, KTO does not require a chosen and rejected response for the same prompt. The notebook's `trl-lib/kto-mix-14k` dataset supplies independent examples of the form $(x,y,l)$, where $l$ is the binary desirability label.

## 1. Compact KTO Objective

The compact loss is

```math
\mathcal{L}_{\mathrm{KTO}}(\theta)
=
\mathbb{E}_{(x,y,l)\sim\mathcal{D}}
\left[
w(l)\left(1-v_\theta(x,y,l)\right)
\right].
```

The user's notation writes $w(y)$ because a completion's label determines its weight. Writing $w(l)$ makes explicit that the weight depends on the **desirable/undesirable label**, not directly on the text or token IDs in $y$.

The label-dependent weight is

```math
w(l)
=
\begin{cases}
\lambda_D, & l=1 \quad \text{(desirable)},\\
\lambda_U, & l=0 \quad \text{(undesirable)}.
\end{cases}
```

The value term is

```math
v_\theta(x,y,l)
=
\begin{cases}
\sigma\!\left(\beta\left[s_\theta(x,y)-z_0\right]\right),
& l=1,\\[4pt]
\sigma\!\left(\beta\left[z_0-s_\theta(x,y)\right]\right),
& l=0.
\end{cases}
```

Here,

```math
s_\theta(x,y)
=
\log\frac{\pi_\theta(y\mid x)}{\pi_{\mathrm{ref}}(y\mid x)}
=
\log\pi_\theta(y\mid x)-\log\pi_{\mathrm{ref}}(y\mid x)
```

is the sequence-level policy-to-reference log-ratio, and $z_0$ is a detached estimate of the average divergence from the reference policy.

Substituting $v_\theta$ gives the two losses used by TRL:

```math
\mathcal{L}_D(x,y)
=
\lambda_D
\left[
1-
\sigma\!\left(\beta\left[s_\theta(x,y)-z_0\right]\right)
\right]
```

for desirable examples, and

```math
\mathcal{L}_U(x,y)
=
\lambda_U
\left[
1-
\sigma\!\left(\beta\left[z_0-s_\theta(x,y)\right]\right)
\right]
```

for undesirable examples.

Because $1-\sigma(a)=\sigma(-a)$, equivalent forms are

```math
\mathcal{L}_D(x,y)
=
\lambda_D\,
\sigma\!\left(\beta\left[z_0-s_\theta(x,y)\right]\right),
```

```math
\mathcal{L}_U(x,y)
=
\lambda_U\,
\sigma\!\left(\beta\left[s_\theta(x,y)-z_0\right]\right).
```

The batch loss is the mean of these label-dependent example losses.

## 2. Definition of Every Term

| Term | Definition |
|---|---|
| $x$ | Prompt or conversation context. |
| $y$ | One completion, tokenized as $(y_1,\ldots,y_T)$. |
| $l$ | Binary feedback label: desirable ($1$) or undesirable ($0$). |
| $\mathcal{D}$ | Dataset of unpaired labeled examples $(x,y,l)$. |
| $\pi_\theta$ | Trainable Qwen policy with parameters $\theta$. |
| $\pi_{\mathrm{ref}}$ | Frozen reference Qwen policy, normally initialized from the same checkpoint. |
| $s_\theta(x,y)$ | Sequence log-ratio comparing the trainable and reference policies. |
| $z_0$ | Detached KL-based reference point representing typical policy drift. |
| $\sigma(a)$ | Sigmoid function $1/(1+e^{-a})$. |
| $v_\theta(x,y,l)$ | Label-dependent prospect value in $[0,1]$. Higher is better for the observed label. |
| $w(l)$ | Class weight: $\lambda_D$ or $\lambda_U$. |
| $\beta$ | Controls sensitivity to the policy-reference log-ratio and constrains drift. |
| $\mathbb{E}$ | Average over labeled examples in the training distribution or batch. |

The loss is bounded per example:

```math
0 < \mathcal{L}(x,y,l) < w(l).
```

This bounded sigmoid objective is one practical difference from ordinary negative log-likelihood, whose loss can grow without bound.

## 3. How Qwen Defines $\pi_\theta(y\mid x)$

Qwen2 and Qwen2.5 are decoder-only causal Transformer language models. KTO does not add a classifier or value head. It constructs the entire loss from Qwen's normal next-token logits.

### Tokenization and causal input

The chat template first converts the conversation and completion to token IDs:

```math
(x_1,\ldots,x_P,y_1,\ldots,y_T).
```

At completion position $t$, causal attention allows Qwen to condition only on the prompt and earlier completion tokens:

```math
p_\theta(y_t\mid x,y_{<t}).
```

Prompt tokens provide context, but only completion-token probabilities are accumulated into the KTO sequence score.

### Decoder hidden states

Token embeddings enter a stack of causal Transformer decoder blocks. In simplified pre-normalized form, layer $k$ computes

```math
\widetilde{\mathbf{h}}^{(k)}
=
\mathbf{h}^{(k-1)}
+
\operatorname{GQA}^{(k)}
\left(
\operatorname{RMSNorm}(\mathbf{h}^{(k-1)})
\right),
```

```math
\mathbf{h}^{(k)}
=
\widetilde{\mathbf{h}}^{(k)}
+
\operatorname{SwiGLU}^{(k)}
\left(
\operatorname{RMSNorm}(\widetilde{\mathbf{h}}^{(k)})
\right).
```

Relevant Qwen components include:

- **Causal self-attention**, which prevents access to future tokens.
- **RoPE**, which injects relative position information into attention queries and keys.
- **Grouped-query attention (GQA)**, which shares key/value heads across groups of query heads.
- **RMSNorm**, which stabilizes hidden activations.
- **SwiGLU feed-forward layers**, which provide gated nonlinear transformations.
- **Residual connections**, which carry information through the decoder stack.

### Vocabulary logits and probabilities

After the final decoder block, the language-model head maps each hidden state to vocabulary logits:

```math
\mathbf{z}_t
=
W_{\mathrm{LM}}\mathbf{h}_{t-1}^{(L)}.
```

For batch size $B$, sequence length $S$, hidden width $d$, and vocabulary size $V$:

```math
\mathbf{H}\in\mathbb{R}^{B\times S\times d},
\qquad
W_{\mathrm{LM}}\in\mathbb{R}^{V\times d},
\qquad
\mathbf{Z}\in\mathbb{R}^{B\times S\times V}.
```

Softmax converts the logits into a distribution over the next token:

```math
p_\theta(v\mid x,y_{<t})
=
\frac{\exp(z_{t,v})}
{\sum_{u=1}^{V}\exp(z_{t,u})}.
```

The log probability assigned to the actual completion token is

```math
\ell_t^\theta
=
\log p_\theta(y_t\mid x,y_{<t})
=
\operatorname{logsoftmax}(\mathbf{z}_t)_{y_t}.
```

The probability of the whole completion factorizes autoregressively:

```math
\pi_\theta(y\mid x)
=
\prod_{t=1}^{T}p_\theta(y_t\mid x,y_{<t}).
```

TRL computes its log probability as a masked sum:

```math
\log\pi_\theta(y\mid x)
=
\sum_{t=1}^{T}m_t\ell_t^\theta,
```

where $m_t=1$ for valid completion tokens and $m_t=0$ for prompt or padding positions. The reference model computes the same quantity with frozen parameters:

```math
\log\pi_{\mathrm{ref}}(y\mid x)
=
\sum_{t=1}^{T}m_t\ell_t^{\mathrm{ref}}.
```

Therefore, the KTO score is directly determined by Qwen's LM-head outputs:

```math
s_\theta(x,y)
=
\sum_{t=1}^{T}m_t
\left(
\ell_t^\theta-\ell_t^{\mathrm{ref}}
\right).
```

No reward model, critic, or binary classification head is required.

## 4. What the Sequence Log-Ratio Means

The ratio

```math
\frac{\pi_\theta(y\mid x)}{\pi_{\mathrm{ref}}(y\mid x)}
```

measures how the fine-tuned Qwen model has changed the probability of the full completion relative to the original model.

- $s_\theta(x,y)>0$: the policy makes $y$ more likely than the reference does.
- $s_\theta(x,y)=0$: both models assign the same sequence probability.
- $s_\theta(x,y)<0$: the policy makes $y$ less likely than the reference does.

For a desirable example, KTO wants $s_\theta(x,y)$ to rise above the reference point $z_0$. For an undesirable example, it wants the score to fall below $z_0$.

Because KTO sums token log probabilities, every completion token contributes:

```math
\frac{\partial s_\theta(x,y)}{\partial\theta}
=
\sum_{t=1}^{T}
\frac{\partial\log p_\theta(y_t\mid x,y_{<t})}{\partial\theta}.
```

The binary label is sequence-level feedback. It does not identify which individual token was good or bad, so credit or blame is distributed over all completion tokens.

## 5. The KL Reference Point $z_0$

KTO compares an example's implicit reward with the policy's typical divergence from the reference model. Conceptually,

```math
z_0
=
\mathbb{E}_{x}
\left[
D_{\mathrm{KL}}
\left(
\pi_\theta(\cdot\mid x)
\,\|\,
\pi_{\mathrm{ref}}(\cdot\mid x)
\right)
\right].
```

Expanding the sequence-level KL gives

```math
D_{\mathrm{KL}}
\left(
\pi_\theta\|\pi_{\mathrm{ref}}
\right)
=
\mathbb{E}_{y'\sim\pi_\theta(\cdot\mid x)}
\left[
\log\pi_\theta(y'\mid x)
-
\log\pi_{\mathrm{ref}}(y'\mid x)
\right].
```

### TRL's batch estimator

The offline dataset does not contain fresh samples from the current policy for every update. TRL approximates the expectation by creating mismatched prompt-completion pairs. Within a batch, completions are shifted so that a prompt $x_i$ is paired with another example's completion $y'_{i}$.

It then estimates

```math
\widehat z_0
=
\max\left(
0,
\frac{1}{B}\sum_{i=1}^{B}
\left[
\log\pi_\theta(y'_i\mid x_i)
-
\log\pi_{\mathrm{ref}}(y'_i\mid x_i)
\right]
\right).
```

In the loss, this estimate is detached:

```math
z_0=\operatorname{stopgrad}(\widehat z_0).
```

Thus, gradients flow through the matched example score $s_\theta(x,y)$, not through the batch baseline. TRL also clamps the estimated value to be non-negative, consistent with a KL divergence.

This estimator requires an actual training batch size greater than one. With batch size one, rotating completions produces the original matched pair rather than a useful mismatched sample. The notebook uses `per_device_train_batch_size=2`, which satisfies this local requirement.

## 6. Desirable Examples

For $l=1$,

```math
v_D(x,y)
=
\sigma\!\left(\beta[s_\theta(x,y)-z_0]\right).
```

The loss is

```math
\mathcal{L}_D
=
\lambda_D(1-v_D).
```

If the trainable Qwen policy makes a desirable completion sufficiently more likely than the reference policy, then

```math
s_\theta(x,y)-z_0 \gg 0.
```

Consequently, $v_D\rightarrow 1$ and $\mathcal{L}_D\rightarrow 0$. Minimizing the loss therefore pushes up the Qwen token log probabilities that compose the desirable completion.

The derivative with respect to the sequence score is

```math
\frac{\partial\mathcal{L}_D}{\partial s_\theta}
=
-\lambda_D\beta\,
\sigma(a_D)\left[1-\sigma(a_D)\right],
\qquad
a_D=\beta(s_\theta-z_0).
```

It is negative, so gradient descent increases $s_\theta$.

## 7. Undesirable Examples

For $l=0$,

```math
v_U(x,y)
=
\sigma\!\left(\beta[z_0-s_\theta(x,y)]\right).
```

The loss is

```math
\mathcal{L}_U
=
\lambda_U(1-v_U).
```

If the policy makes an undesirable completion less likely than the reference, then $s_\theta(x,y)$ decreases. This makes $z_0-s_\theta(x,y)$ positive, so $v_U\rightarrow1$ and the loss approaches zero.

Its derivative is

```math
\frac{\partial\mathcal{L}_U}{\partial s_\theta}
=
\lambda_U\beta\,
\sigma(a_U)\left[1-\sigma(a_U)\right],
\qquad
a_U=\beta(z_0-s_\theta).
```

It is positive, so gradient descent decreases $s_\theta$.

## 8. Why KTO Is Called Prospect-Theoretic

KTO is motivated by Kahneman and Tversky's prospect theory. Outcomes are evaluated relative to a **reference point**, rather than only by their absolute value.

In KTO:

- the implicit outcome is $s_\theta(x,y)$,
- the reference point is $z_0$,
- desirable and undesirable labels use opposite branches around that point,
- the sigmoid creates a bounded, saturating value.

This gives the piecewise value

```math
v_\theta(x,y,l)
=
\begin{cases}
\sigma\!\left(\beta[s_\theta-z_0]\right), & l=1,\\
\sigma\!\left(\beta[z_0-s_\theta]\right), & l=0.
\end{cases}
```

KTO's two branches are mathematically symmetric unless different class weights are chosen. The prospect-theory motivation mainly supplies the reference-dependent, gain-versus-loss framing and the use of separate desirable and undesirable weighting.

## 9. Roles of $\beta$, $\lambda_D$, and $\lambda_U$

### Beta

The parameter $\beta$ scales both the example log-ratio and KL reference point:

```math
\sigma\!\left(\beta[s_\theta-z_0]\right).
```

A larger $\beta$ makes the sigmoid transition sharper. It also imposes stronger effective pressure to remain near the reference model because smaller policy-reference changes are enough to saturate the objective. In the installed TRL version, the default is

```math
\beta=0.1.
```

### Class weights

The weights compensate for unequal numbers or unequal importance of desirable and undesirable examples:

```math
\lambda_D=\texttt{desirable\_weight},
\qquad
\lambda_U=\texttt{undesirable\_weight}.
```

If there are many more desirable examples, each undesirable example can be given greater weight, or vice versa. The installed defaults are

```math
\lambda_D=1,
\qquad
\lambda_U=1.
```

These weights scale gradient magnitude but do not change the target direction: desirable sequence scores move upward and undesirable scores move downward.

## 10. A Numerical Example

Assume

```math
\beta=0.1,
\qquad
z_0=0.4,
\qquad
\lambda_D=\lambda_U=1.
```

### Desirable completion

Suppose Qwen assigns

```math
\log\pi_\theta(y\mid x)=-8.0,
\qquad
\log\pi_{\mathrm{ref}}(y\mid x)=-10.0.
```

Then

```math
s_\theta(x,y)=-8-(-10)=2.
```

The desirable value and loss are

```math
v_D
=
\sigma(0.1[2-0.4])
=
\sigma(0.16)
\approx 0.540,
```

```math
\mathcal{L}_D
=1-v_D
\approx 0.460.
```

Gradient descent increases the completion's policy log probability relative to the reference.

### Undesirable completion

Suppose another example has

```math
s_\theta(x,y)=-1.0.
```

Its undesirable value and loss are

```math
v_U
=
\sigma(0.1[0.4-(-1.0)])
=
\sigma(0.14)
\approx0.535,
```

```math
\mathcal{L}_U
=1-v_U
\approx0.465.
```

Gradient descent lowers its policy-relative sequence score further.

## 11. Simple PyTorch Implementation

The function below expects one policy and reference sequence log probability per example, a Boolean desirability label, and the detached KL reference point used by KTO.

```python
import torch
import torch.nn.functional as F


def sequence_log_probs(logits, token_ids, completion_mask):
    """Convert vocabulary logits to masked completion log probabilities."""
    all_logps = F.log_softmax(logits, dim=-1)
    token_logps = all_logps.gather(
        dim=-1, index=token_ids.unsqueeze(-1)
    ).squeeze(-1)
    return (token_logps * completion_mask).sum(dim=-1)


def kto_loss(
    policy_logps,
    reference_logps,
    desirable,
    kl_reference_point,
    beta=0.1,
    desirable_weight=1.0,
    undesirable_weight=1.0,
):
    log_ratios = policy_logps - reference_logps
    kl = kl_reference_point.detach()

    desirable_losses = 1 - torch.sigmoid(beta * (log_ratios - kl))
    undesirable_losses = 1 - torch.sigmoid(beta * (kl - log_ratios))

    losses = torch.where(
        desirable,
        desirable_weight * desirable_losses,
        undesirable_weight * undesirable_losses,
    )
    return losses.mean(), log_ratios
```

A minimal batch containing one desirable and one undesirable example is:

```python
policy_logps = torch.tensor([-8.0, -11.0], requires_grad=True)
reference_logps = torch.tensor([-10.0, -10.0])
labels = torch.tensor([True, False])
kl_reference_point = torch.tensor(0.4)

loss, log_ratios = kto_loss(
    policy_logps,
    reference_logps,
    labels,
    kl_reference_point,
    beta=0.1,
)

print("log ratios:", log_ratios)  # [2.0, -1.0]
print("per-batch loss:", loss.item())  # approximately (0.460 + 0.465) / 2

loss.backward()
print("gradients:", policy_logps.grad)
# Desirable gradient is negative, so its log-probability increases.
# Undesirable gradient is positive, so its log-probability decreases.
```

TRL estimates `kl_reference_point` from mismatched prompt-completion pairs in the batch. The example supplies it directly so the label-dependent KTO calculation remains easy to see.

## 12. End-to-End Training Flow

For a batch of labeled examples, `KTOTrainer` performs the following:

1. Apply Qwen's chat template and tokenizer to each prompt and completion.
2. Concatenate prompt and completion token IDs.
3. Run the trainable Qwen model and obtain vocabulary logits.
4. Apply `log_softmax`, gather each actual completion token's log probability, mask prompt and padding positions, and sum over the completion.
5. Run the frozen reference Qwen model and calculate the same completion log probabilities.
6. Compute $s_\theta(x,y)=\log\pi_\theta(y\mid x)-\log\pi_{\mathrm{ref}}(y\mid x)$.
7. Rotate completions within the batch to make mismatched pairs and estimate $z_0$.
8. Use the desirable or undesirable sigmoid branch according to the binary label.
9. Multiply by the corresponding class weight.
10. Average the losses and backpropagate only through $\pi_\theta$.

In compact form:

```math
(x,y,l)
\longrightarrow
\text{Qwen logits}
\longrightarrow
\log\pi_\theta(y\mid x),\log\pi_{\mathrm{ref}}(y\mid x)
\longrightarrow
s_\theta,z_0
\longrightarrow
v_\theta
\longrightarrow
\mathcal{L}_{\mathrm{KTO}}
\longrightarrow
\nabla_\theta\mathcal{L}.
```

## 13. Relation to the Notebook

The notebook creates

```python
training_args = KTOConfig(
    output_dir="Qwen2-0.5B-KTO",
    num_train_epochs=1,
    per_device_train_batch_size=2,
)

trainer = KTOTrainer(
    model=model,
    args=training_args,
    processing_class=tokenizer,
    train_dataset=train_dataset,
)
```

The code maps to the equations as follows:

| Notebook component | Mathematical role |
|---|---|
| `model` | Trainable policy $\pi_\theta$. |
| Automatically created frozen copy | Reference policy $\pi_{\mathrm{ref}}$. |
| `processing_class=tokenizer` | Produces prompt and completion token IDs and masks. |
| `train_dataset` | Supplies $(x,y,l)$ examples. |
| `label=True` | Selects the desirable branch. |
| `label=False` | Selects the undesirable branch. |
| `beta` | Sets $\beta$; defaults to $0.1$. |
| `desirable_weight` | Sets $\lambda_D$; defaults to $1.0$. |
| `undesirable_weight` | Sets $\lambda_U$; defaults to $1.0$. |
| `per_device_train_batch_size=2` | Allows mismatched pairs for the batch KL estimate. |
| `trainer.train()` | Executes forward passes, KTO loss, backpropagation, and optimizer updates. |

An explicit equivalent configuration is

```python
training_args = KTOConfig(
    output_dir="Qwen2-0.5B-KTO",
    num_train_epochs=1,
    per_device_train_batch_size=2,
    loss_type="kto",
    beta=0.1,
    desirable_weight=1.0,
    undesirable_weight=1.0,
)
```

## 14. What Comes from Qwen and What Comes from KTO

| Component | Qwen architecture | KTO trainer |
|---|---:|---:|
| Token embeddings and decoder hidden states | Yes | No |
| Causal attention and position handling | Yes | No |
| Vocabulary logits | Yes | No |
| Token and sequence log probabilities | Defines them | Selects and sums them |
| Frozen reference model | Same architecture | Creates and evaluates it |
| Desirable/undesirable label | No | Reads it from the dataset |
| Sequence log-ratio $s_\theta$ | Supplies both models' log probabilities | Subtracts them |
| KL reference point $z_0$ | Supplies mismatched-pair log probabilities | Estimates and detaches it |
| Prospect value $v_\theta$ | No | Computes it |
| Class weight $w(l)$ | No | Configures it |
| Final loss and batch reduction | No | Computes it |

The essential idea is that KTO converts binary feedback into a reference-relative training signal for Qwen's ordinary next-token probabilities. A desirable label pushes the full completion above the KL reference point; an undesirable label pushes it below that point. The Qwen architecture supplies the logits, while KTO determines how those logits should change.
