# Understanding Reward-Model Loss with Qwen3

A reward model learns to assign a higher scalar score to a preferred response than to a non-preferred response.

Given a prompt $x$, a preferred response $y^+$, and a rejected response $y^-$, the standard pairwise reward-model loss is

```math
\mathcal{L}(\theta)
=
-\mathbb{E}_{(x,y^+,y^-)\sim\mathcal{D}}
\left[
\log\sigma
\left(
 r_\theta(x,y^+)-r_\theta(x,y^-)
\right)
\right].
```

The loss does not require a known numerical reward. It only requires the human or synthetic preference

```math
y^+ \succ y^-.
```

The model learns a scalar scoring function $r_\theta(x,y)$ that makes this ordering likely.

## 1. Definition of Every Term

| Term | Meaning |
|---|---|
| $x$ | Prompt or conversation context. |
| $y^+$ | Chosen or preferred response to $x$. |
| $y^-$ | Rejected or less-preferred response to $x$. |
| $(x,y^+,y^-)$ | One pairwise preference example. |
| $\mathcal{D}$ | Distribution or dataset of preference examples. |
| $r_\theta(x,y)$ | Scalar reward predicted by the trainable model for response $y$ given $x$. |
| $\theta$ | Trainable Qwen3 decoder and reward-head parameters. |
| $r_\theta(x,y^+)-r_\theta(x,y^-)$ | Predicted reward margin between the two responses. |
| $\sigma$ | Sigmoid function, which maps a real number to $(0,1)$. |
| $\log$ | Natural logarithm. |
| $\mathbb{E}$ | Average over preference pairs in the dataset or minibatch. |
| $\mathcal{L}$ | Scalar loss minimized during training. |

## 2. The Reward Margin

Define the predicted reward margin

```math
\Delta r_\theta
=
 r_\theta(x,y^+)-r_\theta(x,y^-).
```

Its sign tells us whether the reward model ranks the pair correctly:

- $\Delta r_\theta>0$: the chosen response receives the higher score.
- $\Delta r_\theta=0$: both responses receive the same score.
- $\Delta r_\theta<0$: the rejected response incorrectly receives the higher score.

Only this difference enters the basic loss. The individual reward values do not have an absolute interpretation.

For example, these score pairs produce the same margin and the same loss:

```math
(r^+,r^-)=(3,1),
\qquad
(r^+,r^-)=(102,100).
```

Both have

```math
\Delta r=2.
```

## 3. From Reward Difference to Preference Probability

The sigmoid function is

```math
\sigma(z)
=
\frac{1}{1+\exp(-z)}.
```

The reward model interprets

```math
P_\theta(y^+\succ y^-\mid x)
=
\sigma
\left(
 r_\theta(x,y^+)-r_\theta(x,y^-)
\right)
```

as the predicted probability that $y^+$ is preferred over $y^-$. This is the Bradley-Terry preference model.

The complementary probability is

```math
P_\theta(y^-\succ y^+\mid x)
=
1-
P_\theta(y^+\succ y^-\mid x).
```

Because

```math
\sigma(-z)=1-\sigma(z),
```

swapping the responses reverses the predicted preference probability.

## 4. Why the Negative Logarithm Is Used

The observed training label says that $y^+$ should win. Its likelihood is

```math
\sigma(\Delta r_\theta).
```

Maximum-likelihood training would maximize

```math
\log\sigma(\Delta r_\theta).
```

Optimizers conventionally minimize losses, so the sign is reversed:

```math
\mathcal{L}
=
-\log\sigma(\Delta r_\theta).
```

This is binary cross-entropy written in terms of the reward margin. A numerically stable implementation uses

```python
loss = -torch.nn.functional.logsigmoid(reward_chosen - reward_rejected)
```

rather than computing `log(sigmoid(...))` as two separate operations.

## 5. How the Loss Behaves

The pairwise loss can also be written as softplus:

```math
-\log\sigma(\Delta r)
=
\log(1+\exp(-\Delta r)).
```

Its behavior is:

| Reward margin $\Delta r$ | Predicted preference $\sigma(\Delta r)$ | Loss |
|---:|---:|---:|
| Large positive | Close to $1$ | Close to $0$ |
| $0$ | $0.5$ | $\log 2\approx0.693$ |
| Negative | Below $0.5$ | Greater than $0.693$ |
| Large negative | Close to $0$ | Very large |

Therefore, minimizing the loss encourages

```math
r_\theta(x,y^+)>r_\theta(x,y^-).
```

It also encourages a confident positive margin rather than merely requiring the chosen score to be infinitesimally larger.

## 6. Gradients of the Loss

For one pair, let

```math
\mathcal{L}_{\mathrm{pair}}
=
-\log\sigma(r^+-r^-).
```

Its derivatives are

```math
\frac{\partial\mathcal{L}_{\mathrm{pair}}}{\partial r^+}
=
\sigma(r^+-r^-)-1,
```

```math
\frac{\partial\mathcal{L}_{\mathrm{pair}}}{\partial r^-}
=
1-\sigma(r^+-r^-).
```

When the ranking is wrong or uncertain, gradient descent:

- increases $r^+$,
- decreases $r^-$.

When the model already predicts a large positive margin, both gradients approach zero. Easy, confidently ranked pairs therefore contribute less to later updates.

## 7. How Qwen3 Becomes a Reward Model

Qwen3 is normally a decoder-only causal language model. For reward modeling, its vocabulary-generating language-model head is replaced by a sequence-classification head with one output.

Conceptually, the architecture changes from

```text
Qwen3 decoder -> vocabulary logits for every token
```

to

```text
Qwen3 decoder -> one scalar score for the complete sequence
```

In the notebook, passing a model ID to `RewardTrainer` causes TRL to load

```python
AutoModelForSequenceClassification.from_pretrained(
    "Qwen/Qwen3-0.6B",
    num_labels=1,
)
```

The one label dimension is not a binary class probability. It is an unrestricted scalar reward.

## 8. Building the Qwen3 Input

The prompt and one response are formatted as a single token sequence. For the chosen response:

```math
s^+=(x,y^+),
```

and for the rejected response:

```math
s^-=(x,y^-).
```

After applying the tokenizer and chat template, these become token IDs:

```math
s^+=(u_1^+,u_2^+,\ldots,u_{T_+}^+),
```

```math
s^-=(u_1^-,u_2^-,\ldots,u_{T_-}^-).
```

Each sequence includes both the prompt and response because response quality depends on the prompt. The same response can deserve different rewards for different prompts.

Batches are padded to a common length. An attention mask identifies real and padding tokens:

```math
m_t
=
\begin{cases}
1, & \text{real token},\\
0, & \text{padding token}.
\end{cases}
```

## 9. Qwen3 Decoder Hidden States

Qwen3 maps each token to an embedding and processes the sequence through causal Transformer decoder layers.

In simplified pre-normalized form, decoder layer $k$ performs

```math
\widetilde{\mathbf{h}}^{(k)}
=
\mathbf{h}^{(k-1)}
+
\operatorname{Attention}^{(k)}
\left(
\operatorname{RMSNorm}(\mathbf{h}^{(k-1)})
\right),
```

```math
\mathbf{h}^{(k)}
=
\widetilde{\mathbf{h}}^{(k)}
+
\operatorname{MLP}^{(k)}
\left(
\operatorname{RMSNorm}(\widetilde{\mathbf{h}}^{(k)})
\right).
```

Important architectural components include:

- **Causal self-attention:** each position can attend only to itself and earlier positions.
- **RoPE:** rotary position embeddings encode token positions in attention.
- **Grouped-query attention:** multiple query heads can share key and value heads, reducing memory use.
- **RMSNorm:** normalizes hidden activations.
- **Gated MLP layers:** transform each position nonlinearly.
- **Residual connections:** combine information across decoder layers.

After the final layer, Qwen3 produces hidden states

```math
\mathbf{H}^{(L)}
=
(\mathbf{h}_1^{(L)},\ldots,\mathbf{h}_T^{(L)}),
```

with shape

```math
\mathbf{H}^{(L)}
\in
\mathbb{R}^{B\times T\times d},
```

where $B$ is batch size, $T$ is padded sequence length, and $d$ is Qwen3's hidden width.

## 10. Why the Last Token Represents the Sequence

Because Qwen3 uses causal attention, the hidden state at position $t$ can contain information from all positions up to $t$:

```math
\mathbf{h}_t^{(L)}
=
f_\theta(u_1,\ldots,u_t).
```

The hidden state at the final real token can therefore summarize the entire prompt-response sequence:

```math
\mathbf{h}_{T_*}^{(L)}
=
f_\theta(x,y),
```

where $T_*$ is the index of the rightmost non-padding token.

Transformers finds this index using the configured padding token. That is why `RewardTrainer` ensures that `pad_token_id` is defined.

For a sequence ending in an end-of-sequence token, the pooled state is typically the hidden state corresponding to that final token. Padding states are not selected.

## 11. The Scalar Reward Head

The sequence-classification head is a linear projection from hidden width $d$ to one scalar:

```math
r_\theta(x,y)
=
\mathbf{w}_r^\top
\mathbf{h}_{T_*}^{(L)}.
```

In the installed Transformers implementation, this head has no bias:

```math
\mathbf{w}_r\in\mathbb{R}^{d}.
```

Before pooling, applying the same head to all token states gives

```math
\mathbf{R}
=
\mathbf{H}^{(L)}\mathbf{w}_r,
\qquad
\mathbf{R}\in\mathbb{R}^{B\times T\times1}.
```

The score at the last non-padding position is selected:

```math
r_\theta(x,y)
=
R_{T_*}.
```

The head is newly initialized when a causal Qwen3 checkpoint is loaded as a sequence-classification model. During training, gradients update the reward head and, unless frozen or adapter-based training is configured, the Qwen3 decoder parameters as well.

## 12. Computing Both Rewards

The chosen and rejected sequences are passed through the same model:

```math
r^+
=
r_\theta(x,y^+),
```

```math
r^-
=
r_\theta(x,y^-).
```

The parameters are shared. There are not separate chosen and rejected networks.

`RewardTrainer` may concatenate the chosen and rejected sequences into one larger batch for efficiency:

```math
\begin{bmatrix}
s_1^+\\
\vdots\\
s_B^+\\
s_1^-\\
\vdots\\
s_B^-
\end{bmatrix}
\xrightarrow{\text{Qwen3 reward model}}
\begin{bmatrix}
r_1^+\\
\vdots\\
r_B^+\\
r_1^-\\
\vdots\\
r_B^-
\end{bmatrix}.
```

It then splits the scalar outputs into chosen and rejected rewards and computes

```math
\mathcal{L}_{\mathrm{batch}}
=
-\frac{1}{B}
\sum_{i=1}^{B}
\log\sigma(r_i^+-r_i^-).
```

## 13. A Numerical Example

Suppose Qwen3 predicts

```math
r_\theta(x,y^+)=1.2,
\qquad
r_\theta(x,y^-)=0.2.
```

The reward margin is

```math
\Delta r=1.2-0.2=1.0.
```

The predicted preference probability is

```math
P(y^+\succ y^-\mid x)
=
\sigma(1)
\approx0.731.
```

The pairwise loss is

```math
\mathcal{L}_{\mathrm{pair}}
=
-\log(0.731)
\approx0.313.
```

If the model ranks the pair incorrectly,

```math
r^+=-0.5,
\qquad
r^-=0.5,
```

then

```math
\Delta r=-1,
\qquad
\sigma(-1)\approx0.269,
```

and

```math
\mathcal{L}_{\mathrm{pair}}
=-\log(0.269)
\approx1.313.
```

The incorrect ranking receives a much larger loss.

## 14. What the Model Learns

The objective learns relative ordering:

```math
r_\theta(x,y^+)>r_\theta(x,y^-).
```

It does not directly guarantee that:

- a reward of $2$ has a universal real-world meaning,
- rewards are calibrated probabilities,
- scores are comparable across unrelated datasets,
- the model understands why one response is preferred,
- the learned ranking generalizes outside the preference-data distribution.

The sigmoid output is a pairwise preference probability. The raw reward $r_\theta(x,y)$ itself is an unbounded score, not a probability.

## 15. Reward-Shift Ambiguity

Adding the same constant to every reward does not change the loss:

```math
(r^++c)-(r^-+c)
=
r^+-r^-.
```

Therefore, the basic pairwise objective cannot determine the absolute reward offset. This is called reward-shift or translation ambiguity.

TRL optionally adds a centering regularizer:

```math
\mathcal{L}_{\mathrm{center}}
=
\alpha
\mathbb{E}
\left[
(r^++r^-)^2
\right].
```

The complete configured loss becomes

```math
\mathcal{L}
=
\mathcal{L}_{\mathrm{pair}}
+
\mathcal{L}_{\mathrm{center}}.
```

This encourages rewards to remain centered around zero. In `RewardConfig`, it is controlled by `center_rewards_coefficient`. It is not enabled unless a coefficient is provided.

## 16. Preference Margins

Some datasets say not only which response won, but also how strongly it was preferred. If a target margin $m$ is provided, TRL uses

```math
\mathcal{L}_{\mathrm{margin}}
=
-\log\sigma(r^+-r^--m).
```

A positive $m$ asks the model to separate the rewards by more than merely zero:

```math
r^+-r^->m.
```

Without a `margin` field, the notebook uses the standard equation with $m=0$.

## 17. Relationship to Binary Classification

For each pair, define a preference logit

```math
z=r^+-r^-.
```

The observed label is always $1$ because the dataset places the preferred response in the chosen position. Binary cross-entropy is

```math
\operatorname{BCE}(z,1)
=
-\log\sigma(z).
```

Thus, pairwise reward-model training is binary logistic classification over **reward differences**, not independent classification of each response as absolutely good or bad.

This distinction matters. A response may be chosen only because it is better than its paired alternative; it need not be perfect in isolation.

## 18. Relationship to Language Modeling

A causal language model predicts tokens using vocabulary logits:

```math
P_\theta(y_t\mid x,y_{<t}).
```

A reward model predicts one scalar for the complete prompt-response sequence:

```math
r_\theta(x,y)\in\mathbb{R}.
```

The underlying Qwen3 decoder is similar, but the output heads and objectives differ:

| Model | Output head | Training target |
|---|---|---|
| Causal language model | $d\rightarrow |\mathcal{V}|$ | Correct next token |
| Reward model | $d\rightarrow1$ | Correct pairwise ranking |

The reward-model loss does not use token-level next-token likelihood directly. It backpropagates the pairwise ranking signal through the scalar head into the sequence representation and decoder.

## 19. Simple PyTorch Implementation

At its core, the pairwise loss needs only the two scalar rewards:

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


def pairwise_reward_loss(chosen_rewards, rejected_rewards, margin=None):
    reward_difference = chosen_rewards - rejected_rewards
    if margin is not None:
        reward_difference = reward_difference - margin
    losses = -F.logsigmoid(reward_difference)
    return losses.mean()
```

This small module demonstrates how final Qwen hidden states become scalar rewards. In Transformers, the last non-padding position is selected after applying the score head; selecting the hidden state first is equivalent because the head is linear.

```python
class SimpleRewardHead(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.score = nn.Linear(hidden_size, 1, bias=False)

    def forward(self, hidden_states, attention_mask):
        # hidden_states: (batch, sequence_length, hidden_size)
        # attention_mask: (batch, sequence_length)
        positions = torch.arange(
            attention_mask.size(1), device=attention_mask.device
        )
        last_token_index = (positions * attention_mask).argmax(dim=1)
        batch_index = torch.arange(hidden_states.size(0), device=hidden_states.device)
        final_hidden_state = hidden_states[batch_index, last_token_index]
        return self.score(final_hidden_state).squeeze(-1)
```

A direct reward example is:

```python
chosen_rewards = torch.tensor([1.2, 0.8], requires_grad=True)
rejected_rewards = torch.tensor([0.2, 1.0], requires_grad=True)

loss = pairwise_reward_loss(chosen_rewards, rejected_rewards)

print("reward differences:", chosen_rewards - rejected_rewards)  # [1.0, -0.2]
print("preference probabilities:", torch.sigmoid(chosen_rewards - rejected_rewards))
print("loss:", loss.item())

loss.backward()
print("chosen gradients:", chosen_rewards.grad)
print("rejected gradients:", rejected_rewards.grad)
```

The first pair is correctly ranked, while the second is incorrectly ranked and contributes the larger loss. In actual training, Qwen3 produces `hidden_states`; the learned one-output head produces both reward tensors.

## 20. End-to-End Training Flow

For each preference pair, reward-model training performs these steps:

1. Read a prompt, chosen response, and rejected response from the dataset.
2. Apply the Qwen3 tokenizer and chat template to $(x,y^+)$ and $(x,y^-)$. 
3. Pad both token sequences and build attention masks.
4. Pass both sequences through the same Qwen3 decoder.
5. Select the final hidden state at each sequence's rightmost non-padding token.
6. Project each selected state through the one-output reward head.
7. Calculate the margin $r^+-r^-$. 
8. Convert the margin to a preference probability with sigmoid.
9. Calculate $-\log\sigma(r^+-r^-)$. 
10. Average over the batch and backpropagate through the reward head and trainable Qwen3 parameters.

In compact form:

```math
(x,y^+,y^-)
\longrightarrow
\text{Qwen3 hidden states}
\longrightarrow
(r^+,r^-)
\longrightarrow
\Delta r
\longrightarrow
\sigma(\Delta r)
\longrightarrow
\mathcal{L}
\longrightarrow
\nabla_\theta\mathcal{L}.
```

## 21. Mapping to the Notebook

The notebook constructs

```python
trainer = RewardTrainer(
    model="Qwen/Qwen3-0.6B",
    train_dataset=load_dataset(
        "trl-lib/ultrafeedback_binarized",
        split="train",
    ),
)
```

The code maps to the mathematics as follows:

| Notebook component | Mathematical role |
|---|---|
| `Qwen/Qwen3-0.6B` | Qwen3 decoder used inside $r_\theta(x,y)$. |
| Automatically added one-output score head | Maps the final sequence state to scalar reward $r_\theta(x,y)$. |
| `ultrafeedback_binarized` | Supplies preference examples $(x,y^+,y^-)$. |
| `chosen` sequence | Supplies $(x,y^+)$ and reward $r^+$. |
| `rejected` sequence | Supplies $(x,y^-)$ and reward $r^-$. |
| `RewardTrainer` | Tokenizes pairs, runs both sequences, and computes the pairwise loss. |
| `trainer.train()` | Runs forward passes, backpropagation, and optimizer updates. |

The installed TRL implementation computes the central loss as

```python
loss = -torch.nn.functional.logsigmoid(
    rewards_chosen - rewards_rejected
).mean()
```

which is exactly

```math
-\frac{1}{B}
\sum_{i=1}^{B}
\log\sigma
\left(
 r_\theta(x_i,y_i^+)-r_\theta(x_i,y_i^-)
\right).
```

## 22. Evaluation Metrics

A basic ranking-accuracy metric is

```math
\operatorname{accuracy}
=
\frac{1}{N}
\sum_{i=1}^{N}
\mathbf{1}[r_i^+>r_i^-].
```

TRL also reports the mean reward margin:

```math
\operatorname{mean\ margin}
=
\frac{1}{N}
\sum_{i=1}^{N}(r_i^+-r_i^-).
```

Useful evaluation should also consider held-out preference accuracy, performance across prompt categories, score distributions, ties, calibration, and robustness to response length or superficial style.

## 23. Practical Considerations

### Length bias

Because the final hidden state summarizes the entire sequence, the model may learn correlations between response length and preference. The preference dataset should contain enough counterexamples to prevent length from becoming an easy shortcut.

### Position and truncation

If a sequence exceeds the configured maximum length, preprocessing may filter or truncate it. Losing the end of a response can change the final representation and remove information needed to judge quality.

### Padding token

The model must know which token ID represents padding so that it can select the last real token rather than a padded position.

### Data quality

The reward model reproduces patterns in its preference labels. Inconsistent, biased, or style-dominated labels produce a reward function with the same weaknesses.

### Overoptimization

A policy trained against a fixed reward model can exploit accidental features that receive high reward without genuinely improving response quality. Reward-model accuracy alone does not eliminate reward hacking.

## 24. Summary

The pairwise reward-model loss is

```math
\mathcal{L}(\theta)
=
-\mathbb{E}
\left[
\log\sigma
\left(
 r_\theta(x,y^+)-r_\theta(x,y^-)
\right)
\right].
```

Its essential logic is:

```text
chosen reward > rejected reward  -> small loss
chosen reward = rejected reward  -> loss of log(2)
chosen reward < rejected reward  -> large loss
```

For Qwen3:

- the tokenizer forms complete prompt-response sequences;
- the causal decoder produces contextual hidden states;
- the final non-padding hidden state summarizes the sequence;
- a one-output linear head converts that state to $r_\theta(x,y)$;
- the loss compares two scalar rewards and trains their ordering.

The architecture supplies the reward scores. The preference dataset supplies the desired ordering. The pairwise logistic loss connects the two.
