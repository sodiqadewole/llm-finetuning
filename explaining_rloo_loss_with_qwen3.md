# Understanding RLOO Loss with Qwen3

REINFORCE Leave-One-Out (RLOO) is an online reinforcement-learning method for aligning a language model from human, learned, or verifiable feedback.

For each prompt, the policy generates several responses. Each response receives a reward, and its baseline is the average reward of the **other** responses generated for that prompt. The policy then increases the probability of responses that outperform this leave-one-out baseline and decreases the probability of responses that underperform it.

A clipped RLOO objective is

```math
\mathcal{L}_{\mathrm{RLOO}}(\theta)
=
-\frac{1}{G}
\sum_{i=1}^{G}
\min
\left(
\rho_i(\theta)\widehat A_i,
\operatorname{clip}
\left(
\rho_i(\theta),1-\varepsilon,1+\varepsilon
\right)
\widehat A_i
\right),
```

where

```math
\rho_i(\theta)
=
\frac{\pi_\theta(o_i\mid q)}
{\pi_{\theta_{\mathrm{old}}}(o_i\mid q)}.
```

This is REINFORCE with a leave-one-out baseline, plus a PPO-style clipped importance ratio that permits controlled reuse of sampled responses.

## 1. Definition of Every Term

| Term | Meaning |
|---|---|
| $q$ | Prompt or conversation context. |
| $G$ | Number of responses generated for the same prompt. |
| $o_i$ | The $i$th generated response, represented as a token sequence. |
| $\pi_\theta$ | Current trainable Qwen3 policy. |
| $\pi_{\theta_{\mathrm{old}}}$ | Frozen rollout policy that generated the responses. |
| $\pi_{\mathrm{ref}}$ | Optional frozen reference Qwen3 policy used for KL regularization. |
| $R_i$ | Feedback reward assigned to response $o_i$. |
| $\widehat A_i$ | Leave-one-out advantage for response $i$. |
| $\rho_i$ | Sequence-level importance ratio between current and rollout policies. |
| $\varepsilon$ | Clipping width around ratio $1$. |
| $\operatorname{clip}$ | Restricts a ratio to the interval $[1-\varepsilon,1+\varepsilon]$. |
| $\min$ | Chooses the more conservative unclipped or clipped objective. |
| $\theta$ | Trainable Qwen3 parameters. |
| $\mathcal{L}$ | Scalar loss minimized by the optimizer. |

The leading minus sign converts reward maximization into loss minimization.

## 2. Why It Is Called RLOO

RLOO stands for **REINFORCE Leave-One-Out**.

- **REINFORCE** refers to the score-function policy gradient.
- **Leave-one-out** refers to constructing each response's baseline from all responses in its group except itself.

The basic REINFORCE gradient is

```math
\nabla_\theta J(\theta)
=
\mathbb{E}
\left[
\widehat A
\nabla_\theta\log\pi_\theta(o\mid q)
\right].
```

A baseline reduces gradient variance without changing the expected policy gradient, provided that the baseline does not depend on the sampled action whose gradient it multiplies. Excluding response $i$ from its own baseline preserves this property.

## 3. Generating a Group of Responses

For one prompt $q$, the rollout policy samples $G$ responses:

```math
 o_1,o_2,\ldots,o_G
\sim
\pi_{\theta_{\mathrm{old}}}(\cdot\mid q).
```

For example, Qwen3 might generate four candidate answers to the same instruction. A feedback function then scores each answer:

```math
R_i=r(q,o_i).
```

The feedback can come from:

- a learned reward model trained from human preferences;
- direct human ratings, if available during data collection;
- verifiable correctness, such as exact math answers or unit tests;
- rule-based format or safety checks;
- a weighted combination of several reward functions.

RLOO training is online because the current or recent policy generates the responses used for updates.

## 4. Leave-One-Out Baseline

For response $i$, the baseline is the mean reward of the other $G-1$ responses:

```math
b_i
=
\frac{1}{G-1}
\sum_{\substack{j=1\\j\neq i}}^{G}R_j.
```

The advantage is

```math
\widehat A_i
=
R_i-b_i.
```

Equivalently,

```math
\widehat A_i
=
R_i
-
\frac{\sum_{j=1}^{G}R_j-R_i}{G-1}.
```

Interpretation:

- $\widehat A_i>0$: response $i$ scored above its siblings, so increase its probability.
- $\widehat A_i<0$: response $i$ scored below its siblings, so decrease its probability.
- $\widehat A_i=0$: response $i$ supplies no policy-gradient signal.

At least two generations are needed during training because $G-1$ appears in the denominator.

### Numerical example

Suppose four responses receive

```math
(R_1,R_2,R_3,R_4)=(1.0,0.6,0.2,0.0).
```

For the first response,

```math
b_1
=
\frac{0.6+0.2+0.0}{3}
\approx0.267,
```

so

```math
\widehat A_1
=1.0-0.267
\approx0.733.
```

For the fourth response,

```math
b_4
=
\frac{1.0+0.6+0.2}{3}
=0.6,
```

so

```math
\widehat A_4
=0.0-0.6
=-0.6.
```

The same numerical reward can produce a different advantage in another group. RLOO learns from **relative performance among responses to the same prompt**.

## 5. How Qwen3 Defines the Policy

In RLOO, Qwen3 remains a causal language model. It uses its normal vocabulary language-model head, unlike a reward model that replaces that head with one scalar output.

For a prompt and partial response, Qwen3's decoder produces a contextual hidden state:

```math
\mathbf{h}_{i,t-1}^{(L)}
=
f_\theta(q,o_{i,<t}).
```

The language-model head projects it to one logit for every vocabulary token:

```math
\mathbf{z}_{i,t}
=
W_{\mathrm{LM}}\mathbf{h}_{i,t-1}^{(L)}.
```

For batch size $B$, sequence length $T$, hidden width $d$, and vocabulary size $V$:

```math
\mathbf{H}\in\mathbb{R}^{B\times T\times d},
\qquad
W_{\mathrm{LM}}\in\mathbb{R}^{V\times d},
\qquad
\mathbf{Z}\in\mathbb{R}^{B\times T\times V}.
```

Softmax turns logits into next-token probabilities:

```math
\pi_\theta(v\mid q,o_{i,<t})
=
\frac{\exp(z_{i,t,v})}
{\sum_{u=1}^{V}\exp(z_{i,t,u})}.
```

The probability assigned to the actual generated token $o_{i,t}$ is selected from this distribution.

Qwen3 architectural components such as causal self-attention, RoPE, grouped-query attention, RMSNorm, gated MLPs, and residual connections produce the hidden states. RLOO operates on the token probabilities supplied by the ordinary Qwen3 language-model head.

## 6. Probability of a Complete Response

If response $o_i$ contains $T_i$ tokens, its sequence probability factorizes autoregressively:

```math
\pi_\theta(o_i\mid q)
=
\prod_{t=1}^{T_i}
\pi_\theta(o_{i,t}\mid q,o_{i,<t}).
```

Implementations work in log space:

```math
\log\pi_\theta(o_i\mid q)
=
\sum_{t=1}^{T_i}
\log\pi_\theta(o_{i,t}\mid q,o_{i,<t}).
```

With a completion mask $m_{i,t}$,

```math
\log\pi_\theta(o_i\mid q)
=
\sum_t
m_{i,t}
\log\pi_\theta(o_{i,t}\mid q,o_{i,<t}).
```

Prompt tokens, padding, and tokens after the first end-of-sequence token do not contribute.

## 7. Current, Rollout, and Reference Policies

Three model roles can appear in RLOO.

### Current policy

```math
\pi_\theta
```

This is the Qwen3 model being optimized. Gradients flow through its token log probabilities.

### Rollout policy

```math
\pi_{\theta_{\mathrm{old}}}
```

This policy generated the stored responses. Its sequence log probabilities are saved without gradients. It forms the denominator of the importance ratio.

### Reference policy

```math
\pi_{\mathrm{ref}}
```

This optional frozen Qwen3 model anchors the policy's language behavior through KL regularization. It is generally distinct from the rollout policy:

```math
\pi_{\theta_{\mathrm{old}}}
\neq
\pi_{\mathrm{ref}}.
```

The rollout policy tracks recent training parameters, while the reference policy normally remains fixed at the initial pretrained or supervised-fine-tuned checkpoint.

## 8. Sequence-Level Importance Ratio

RLOO compares the current and rollout probabilities of the complete response:

```math
\rho_i(\theta)
=
\frac{\pi_\theta(o_i\mid q)}
{\pi_{\theta_{\mathrm{old}}}(o_i\mid q)}.
```

Using log probabilities,

```math
\rho_i(\theta)
=
\exp
\left(
\log\pi_\theta(o_i\mid q)
-
\log\pi_{\theta_{\mathrm{old}}}(o_i\mid q)
\right).
```

Because sequence probabilities are products of token probabilities, the ratio is also a product of token-level ratios:

```math
\rho_i(\theta)
=
\prod_{t=1}^{T_i}
\frac{
\pi_\theta(o_{i,t}\mid q,o_{i,<t})
}{
\pi_{\theta_{\mathrm{old}}}(o_{i,t}\mid q,o_{i,<t})
}.
```

Interpretation:

- $\rho_i=1$: current and rollout policies assign the same probability to the response.
- $\rho_i>1$: the current policy makes the response more likely.
- $\rho_i<1$: the current policy makes the response less likely.

Sequence ratios can move far from $1$ because small token-level changes multiply across a long completion. This makes clipping especially important when generated batches are reused for multiple updates.

When the rollout and current policies are initially identical, the numerical ratio is $1$. The objective still has a gradient because the old log probability is detached:

```math
\nabla_\theta\rho_i
=
\rho_i
\nabla_\theta\log\pi_\theta(o_i\mid q).
```

## 9. The Unclipped REINFORCE Objective

Ignoring clipping, the surrogate objective is

```math
J(\theta)
=
\frac{1}{G}
\sum_{i=1}^{G}
\rho_i(\theta)\widehat A_i.
```

The minimized loss is

```math
\mathcal{L}(\theta)
=
-J(\theta).
```

For a positive advantage, minimizing the loss increases the response probability. For a negative advantage, it decreases the response probability.

If responses are used immediately and the policies match, $\rho_i=1$ numerically and the gradient reduces to the familiar REINFORCE form:

```math
\nabla_\theta J
=
\frac{1}{G}
\sum_{i=1}^{G}
\widehat A_i
\nabla_\theta\log\pi_\theta(o_i\mid q).
```

## 10. PPO-Style Clipping

The clipped ratio is

```math
\overline\rho_i
=
\operatorname{clip}
\left(
\rho_i,1-\varepsilon,1+\varepsilon
\right).
```

The objective uses

```math
\min
\left(
\rho_i\widehat A_i,
\overline\rho_i\widehat A_i
\right).
```

The `min` must be interpreted together with the sign of the advantage.

### Positive advantage

When $\widehat A_i>0$, making an already-good response much more likely should not create unlimited improvement. Ratios above $1+\varepsilon$ are capped by the conservative branch.

### Negative advantage

When $\widehat A_i<0$, making a bad response dramatically less likely should not create unlimited improvement. Ratios below $1-\varepsilon$ are capped by the conservative branch.

Clipping does not force the ratio to remain in the interval. It limits the objective's incentive to move farther in a beneficial direction once the policy update becomes too large.

With the common value

```math
\varepsilon=0.2,
```

the clipping interval is

```math
[1-\varepsilon,1+\varepsilon]=[0.8,1.2].
```

## 11. KL-Regularized Rewards

RLOO can penalize deviation from a frozen reference model before computing advantages.

For each response token, TRL uses the first-order sampled log-ratio

```math
k_{i,t}
=
\log\pi_{\theta_{\mathrm{old}}}(o_{i,t}\mid q,o_{i,<t})
-
\log\pi_{\mathrm{ref}}(o_{i,t}\mid q,o_{i,<t}).
```

Summing over valid response tokens gives

```math
K_i
=
\sum_t m_{i,t}k_{i,t}.
```

The adjusted reward is

```math
\widetilde R_i
=
R_i-\beta K_i,
```

where $\beta$ controls the strength of the reference-model constraint.

The leave-one-out advantage is then computed from the adjusted rewards:

```math
\widehat A_i
=
\widetilde R_i
-
\frac{1}{G-1}
\sum_{j\neq i}\widetilde R_j.
```

This sampled $K_i$ can be negative for an individual response even though exact KL divergence is non-negative. Its expectation under the policy corresponds to KL divergence. In RLOO, the KL estimate modifies the sequence reward; it is not added separately to the final clipped loss.

## 12. Reward and Advantage Normalization

A trainer may combine several reward functions:

```math
R_i
=
\sum_{k=1}^{K}w_k r_k(q,o_i).
```

It may also clip rewards before constructing the leave-one-out baseline. Optionally, advantages can be normalized across the generation batch:

```math
\widehat A_i^{\mathrm{norm}}
=
\frac{
\widehat A_i-\mu_A
}{
\sigma_A+\epsilon_{\mathrm{num}}
}.
```

Normalization changes gradient scale but preserves whether each advantage is positive or negative. In the installed TRL version, advantage normalization is disabled by default.

## 13. Simple PyTorch Implementation

The implementation below matches the central TRL calculation. It accepts sequence log probabilities, grouped rewards, and computes leave-one-out advantages followed by the clipped sequence-level loss.

```python
import torch


def leave_one_out_advantages(rewards):
    """Compute RLOO advantages for rewards shaped (batch, generations)."""
    num_generations = rewards.size(1)
    if num_generations < 2:
        raise ValueError("RLOO requires at least two generations per prompt")

    other_reward_mean = (
        rewards.sum(dim=1, keepdim=True) - rewards
    ) / (num_generations - 1)
    return rewards - other_reward_mean


def rloo_loss(
    current_logps,
    old_logps,
    rewards,
    epsilon=0.2,
    normalize_advantages=False,
):
    """
    Args:
        current_logps: current sequence log-probs, shape (batch, generations)
        old_logps: detached rollout sequence log-probs, same shape
        rewards: one reward per completion, same shape
    """
    advantages = leave_one_out_advantages(rewards)

    if normalize_advantages:
        advantages = (
            advantages - advantages.mean()
        ) / (advantages.std(unbiased=False) + 1e-4)

    ratios = torch.exp(current_logps - old_logps)
    clipped_ratios = ratios.clamp(1 - epsilon, 1 + epsilon)

    unclipped_objective = ratios * advantages
    clipped_objective = clipped_ratios * advantages
    per_sequence_loss = -torch.minimum(
        unclipped_objective, clipped_objective
    )

    return per_sequence_loss.mean(), advantages, ratios
```

A helper for obtaining sequence log probabilities from Qwen vocabulary logits is:

```python
import torch.nn.functional as F


def sequence_log_probs(logits, completion_ids, completion_mask):
    """Sum selected completion-token log-probabilities."""
    all_token_logps = F.log_softmax(logits, dim=-1)
    selected_logps = all_token_logps.gather(
        dim=-1,
        index=completion_ids.unsqueeze(-1),
    ).squeeze(-1)
    return (selected_logps * completion_mask).sum(dim=-1)
```

In a causal model, logits and target IDs must be shifted so each position predicts the following token. A trainer normally handles that alignment before passing completion-only logits and IDs to a helper like this one.

### Worked tensor example

```python
# One prompt with four sampled responses.
rewards = torch.tensor([[1.0, 0.6, 0.2, 0.0]])

# Log-probabilities saved when the rollout policy generated each response.
old_logps = torch.tensor([[-5.0, -5.5, -6.0, -6.5]])

# Current policy log-probabilities after a small parameter update.
current_logps = torch.tensor(
    [[-4.90, -5.45, -6.10, -6.70]],
    requires_grad=True,
)

loss, advantages, ratios = rloo_loss(
    current_logps=current_logps,
    old_logps=old_logps,
    rewards=rewards,
    epsilon=0.2,
)

print("advantages:", advantages)
# approximately [[0.7333, 0.2000, -0.3333, -0.6000]]

print("ratios:", ratios)
# approximately [[1.1052, 1.0513, 0.9048, 0.8187]]

print("loss:", loss.item())

loss.backward()
print("current log-prob gradients:", current_logps.grad)
# Positive-advantage responses get negative gradients, increasing log-probability.
# Negative-advantage responses get positive gradients, decreasing log-probability.
```

The `old_logps` tensor must be detached or computed under `torch.no_grad()`. In actual Qwen3 training, `current_logps` are obtained by running the sampled prompt-completion tokens through the current policy and summing the selected completion-token log probabilities.

## 14. Detailed Numerical Interpretation

For the first response in the example,

```math
\widehat A_1\approx0.7333.
```

Its sequence log-probability changed from $-5.0$ to $-4.9$, so

```math
\rho_1
=
\exp(-4.9-(-5.0))
=
\exp(0.1)
\approx1.1052.
```

This ratio is within $[0.8,1.2]$, so it is not clipped. Its objective contribution is

```math
\rho_1\widehat A_1
\approx
1.1052(0.7333)
\approx0.8105.
```

For the fourth response,

```math
\widehat A_4=-0.6,
\qquad
\rho_4=\exp(-0.2)\approx0.8187.
```

It also remains just inside the clipping interval. Because its advantage is negative, gradient descent pushes its sequence probability downward.

## 15. How Gradients Reach Qwen3

The loss depends on the current sequence log probability:

```math
\log\pi_\theta(o_i\mid q)
=
\sum_t
m_{i,t}
\operatorname{logsoftmax}(\mathbf z_{i,t})_{o_{i,t}}.
```

Backpropagation follows this path:

```math
\mathcal{L}_{\mathrm{RLOO}}
\longrightarrow
\log\pi_\theta(o_i\mid q)
\longrightarrow
\mathbf z_{i,t}
\longrightarrow
W_{\mathrm{LM}},\mathbf h_{i,t}^{(L)}
\longrightarrow
\theta.
```

The scalar reward and leave-one-out advantage are treated as fixed feedback during the policy update. Gradients do not pass through human ratings, a separate reward model, the old policy, or the reference policy.

## 16. End-to-End Training Flow

For each generation and optimization cycle, RLOO performs the following:

1. Apply Qwen3's chat template and tokenizer to prompt $q$.
2. Use the rollout Qwen3 policy to generate $G$ responses.
3. Save each response's rollout token log probabilities.
4. Score each response using a reward model, human-derived signal, verifier, or custom reward function.
5. Optionally subtract a reference-model KL penalty from each reward.
6. For each response, average the rewards of its $G-1$ siblings to obtain the leave-one-out baseline.
7. Subtract the baseline from the response reward to obtain $\widehat A_i$.
8. Run the sampled sequences through the current Qwen3 policy.
9. Gather and sum completion-token log probabilities into sequence log probabilities.
10. Form $\rho_i$ from current and rollout sequence probabilities.
11. Compute the clipped surrogate loss and average it across responses.
12. Backpropagate through the current Qwen3 policy and update $\theta$.

In compact form:

```math
q
\longrightarrow
\{o_i\}_{i=1}^{G}
\longrightarrow
\{R_i\}_{i=1}^{G}
\longrightarrow
\{\widehat A_i\}_{i=1}^{G}
\longrightarrow
\{\rho_i\}_{i=1}^{G}
\longrightarrow
\mathcal{L}_{\mathrm{RLOO}}
\longrightarrow
\nabla_\theta\mathcal{L}.
```

## 17. Mapping to TRL Configuration

A typical setup is

```python
from trl import RLOOConfig, RLOOTrainer

training_args = RLOOConfig(
    output_dir="Qwen3-RLOO",
    num_generations=4,
    epsilon=0.2,
    beta=0.05,
)

trainer = RLOOTrainer(
    model="Qwen/Qwen3-0.6B",
    args=training_args,
    reward_funcs=reward_function,
    train_dataset=train_dataset,
)
```

The configuration maps to the equations as follows:

| Configuration | Mathematical role |
|---|---|
| `model` | Current Qwen3 policy $\pi_\theta$ and initial rollout policy. |
| `num_generations` | Group size $G$; installed default is $2$. |
| `epsilon` | Lower and upper clipping width $\varepsilon$; default is $0.2$. |
| `epsilon_high` | Optional distinct upper clipping width. |
| `beta` | Reference KL coefficient $\beta$; installed default is $0.05$. |
| `reward_funcs` | Defines the feedback rewards $R_i$. |
| `reward_weights` | Weights multiple reward functions. |
| `normalize_advantages` | Optionally standardizes $\widehat A_i$; default is `False`. |
| `num_iterations` | Number of updates that reuse one generated batch. |
| `max_completion_length` | Maximum number of generated response tokens. |

## 18. RLOO Compared with Related Methods

### RLOO versus basic REINFORCE

Basic REINFORCE may use no baseline or a general moving baseline. RLOO uses sibling responses to the same prompt as a prompt-specific baseline.

### RLOO versus PPO

PPO commonly trains a separate value network to estimate advantages. RLOO removes the critic and uses leave-one-out rewards instead. The clipped policy ratio remains PPO-like.

### RLOO versus GRPO

Both generate multiple responses per prompt and avoid a learned value model. Their common formulations differ in two central details:

- RLOO's baseline for response $i$ excludes $R_i$.
- RLOO uses one importance ratio for the complete response, while standard GRPO commonly applies token-level ratios and reductions.

For the RLOO baseline,

```math
b_i^{\mathrm{RLOO}}
=
\frac{1}{G-1}\sum_{j\neq i}R_j.
```

For a group-mean baseline,

```math
b^{\mathrm{group}}
=
\frac{1}{G}\sum_{j=1}^{G}R_j.
```

These produce related but differently scaled advantages.

## 19. Practical Considerations

### Sequence-ratio variance

Multiplying probability ratios across many tokens can create very large or very small sequence ratios. Keep rollout data fresh, use clipping, and monitor ratio statistics.

### Number of generations

With $G=2$, each response uses the other response's reward as its baseline. Larger groups generally produce a more stable baseline but require more generation compute.

### Reward quality

RLOO optimizes whatever the reward function measures. A biased or exploitable reward model can encourage undesirable shortcuts or reward hacking.

### Completion length

Sequence log probabilities are sums over tokens. Longer responses can exhibit larger policy log-ratio changes, so length distributions and truncation behavior should be monitored.

### Stale rollout data

If many optimizer updates reuse the same generations, the current policy may drift far from the rollout policy. More ratios then hit the clipping boundary, reducing useful learning signal.

### No scalar reward head in the policy

Qwen3's RLOO policy uses the ordinary language-model head. A separate reward model may score completions, but its scalar head is not part of the policy being optimized.

## 20. Summary

The RLOO loss is

```math
\mathcal{L}_{\mathrm{RLOO}}(\theta)
=
-\frac{1}{G}
\sum_{i=1}^{G}
\min
\left(
\rho_i\widehat A_i,
\operatorname{clip}(\rho_i,1-\varepsilon,1+\varepsilon)
\widehat A_i
\right).
```

Its essential logic is:

```text
response beats its siblings    -> positive advantage -> increase probability
response trails its siblings   -> negative advantage -> decrease probability
policy changes too far         -> clipping limits the incentive
policy drifts from reference   -> optional KL term lowers the reward
```

Qwen3 supplies autoregressive token probabilities through its causal decoder and vocabulary head. The feedback system supplies scalar rewards. RLOO converts those rewards into leave-one-out advantages and applies them to complete-response probability ratios, training Qwen3 without a separate value network.
