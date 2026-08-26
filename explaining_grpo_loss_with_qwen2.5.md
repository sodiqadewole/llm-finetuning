# Understanding the GRPO Loss with Qwen2.5

Group Relative Policy Optimization (GRPO) trains a language model by generating a **group of completions for the same prompt**, scoring them, and increasing the probabilities of completions that score above the group's baseline.

GRPO resembles PPO, but it does not require a learned value model. Instead, the other completions in the group provide the baseline used to estimate each completion's advantage.

## 1. The Objective

A clear version of the token-level GRPO objective is:

```math
\mathcal{L}_{\mathrm{GRPO}}(\theta)
=
-\mathbb{E}_{q,\{o_i\}_{i=1}^{G}}
\left[
\frac{1}{G}
\sum_{i=1}^{G}
\frac{1}{|o_i|}
\sum_{t=1}^{|o_i|}
\left(
\rho_{i,t}(\theta)\,\hat A_i
-
\beta\,D_{i,t}
\right)
\right].
```

The importance ratio is

```math
\rho_{i,t}(\theta)
=
\frac{
\pi_\theta(o_{i,t}\mid q,o_{i,<t})
}{
\pi_{\theta_{\mathrm{old}}}(o_{i,t}\mid q,o_{i,<t})
}.
```

Here, the denominator is treated as a constant during backpropagation:

```math
\rho_{i,t}(\theta)
=
\exp\left(
\log\pi_\theta(o_{i,t}\mid q,o_{i,<t})
-
\operatorname{stopgrad}\left[
\log\pi_{\theta_{\mathrm{old}}}(o_{i,t}\mid q,o_{i,<t})
\right]
\right).
```

Some presentations write the denominator as `no grad` applied to the same policy. This means that its saved rollout probability is detached from the computation graph; it does **not** mean that the numerator and denominator cancel during differentiation.

### The clipped objective used in practice

Modern GRPO implementations generally use PPO-style clipping:

```math
\mathcal{L}_{\mathrm{policy}}(\theta)
=
-\mathbb{E}
\left[
\frac{1}{G}
\sum_{i=1}^{G}
\frac{1}{|o_i|}
\sum_{t=1}^{|o_i|}
\min\left(
\rho_{i,t}\hat A_i,
\operatorname{clip}(\rho_{i,t},1-\varepsilon,1+\varepsilon)\hat A_i
\right)
\right].
```

Clipping prevents one batch of sampled text from causing an excessively large policy update. If the new Qwen policy changes a token probability too far from the rollout policy, the clipped branch limits the benefit of moving farther.

## 2. Meaning of Every Symbol

| Symbol | Definition |
|---|---|
| $q$ | One input prompt, such as a math problem. |
| $G$ | Number of completions sampled for the same prompt. |
| $o_i$ | Completion $i$ in the group, represented by tokens $(o_{i,1},\ldots,o_{i,|o_i|})$. |
| $|o_i|$ | Number of valid completion tokens, normally including the first EOS token and excluding padding. |
| $o_{i,<t}$ | Completion tokens before position $t$. |
| $\pi_\theta$ | The trainable Qwen2.5 policy. Its parameters $\theta$ receive gradients. |
| $\pi_{\theta_{\mathrm{old}}}$ | The frozen rollout snapshot that generated the sampled completions. |
| $\pi_{\mathrm{ref}}$ | A frozen reference Qwen model, usually the initial pretrained or SFT model. It is distinct from the rollout policy. |
| $\rho_{i,t}$ | Importance ratio between the current and rollout policies for one sampled token. |
| $R_i$ | Reward assigned to completion $o_i$. |
| $\hat A_i$ | Group-relative advantage derived from rewards. It is usually shared by every token in completion $i$. |
| $D_{i,t}$ | Per-token estimator of divergence from the reference policy. |
| $\beta$ | Strength of the reference-policy KL penalty. |
| $\varepsilon$ | Clipping width for the importance ratio. |
| $\mathbb{E}$ | Average over prompts and sampled groups in the training batch. |

The leading minus sign converts a maximization objective into a loss minimized by PyTorch.

## 3. From Group Rewards to Advantages

For one prompt $q$, Qwen generates $G$ different completions:

```math
\{o_1,o_2,\ldots,o_G\}
\sim
\pi_{\theta_{\mathrm{old}}}(\cdot\mid q).
```

A reward function scores each complete response:

```math
R_i = r(q,o_i).
```

In the notebook, `accuracy_reward` supplies this signal by checking a generated math answer against the reference answer. With multiple reward functions, TRL can form a weighted sum:

```math
R_i=\sum_{k=1}^{K}w_k r_k(q,o_i).
```

The group mean and standard deviation are

```math
\mu_R
=
\frac{1}{G}\sum_{j=1}^{G}R_j,
\qquad
\sigma_R
=
\sqrt{
\frac{1}{G}\sum_{j=1}^{G}(R_j-\mu_R)^2
}.
```

The normalized group-relative advantage is then

```math
\hat A_i
=
\frac{R_i-\mu_R}{\sigma_R+\epsilon_{\mathrm{num}}},
```

where $\epsilon_{\mathrm{num}}$ is a small numerical constant, such as $10^{-4}$.

Interpretation:

- $\hat A_i>0$: completion $i$ scored above the group average, so GRPO increases the probabilities of its sampled tokens.
- $\hat A_i<0$: it scored below the group average, so GRPO decreases those probabilities.
- $\hat A_i=0$: it produces no policy-gradient signal.

The advantage is usually computed once per **completion** and broadcast over its tokens:

```math
\hat A_{i,t}=\hat A_i.
```

A token-level advantage $\hat A_{i,t}$ is possible if a task supplies intermediate or process rewards, but an outcome reward such as answer accuracy naturally produces one value per completion.

### Why this removes the value network

PPO commonly estimates

```math
A_i \approx R_i - V_\phi(q),
```

where $V_\phi$ is a separately trained critic. GRPO replaces that learned baseline with the empirical mean of completions for the same prompt:

```math
A_i \approx R_i-\frac{1}{G}\sum_{j=1}^{G}R_j.
```

This avoids adding a value head or a second value-model network to Qwen.

## 4. How Qwen2.5 Defines the Policy

Qwen2.5-0.5B-Instruct is a decoder-only causal Transformer. GRPO does not add a special GRPO head to it. The model's ordinary language-model head defines every probability in the loss.

For a prompt and partial completion, tokenization produces

```math
x_i=(q,o_{i,<t}).
```

If the token IDs are $(x_1,\ldots,x_s)$, Qwen maps each token to an embedding:

```math
\mathbf{h}^{(0)}_s=E[x_s]+\text{position information}.
```

Qwen applies a stack of causal Transformer decoder blocks. In simplified pre-normalized form, layer $\ell$ computes

```math
\widetilde{\mathbf{h}}^{(\ell)}
=
\mathbf{h}^{(\ell-1)}
+
\operatorname{GQA}^{(\ell)}
\left(
\operatorname{RMSNorm}(\mathbf{h}^{(\ell-1)})
\right),
```

```math
\mathbf{h}^{(\ell)}
=
\widetilde{\mathbf{h}}^{(\ell)}
+
\operatorname{SwiGLU}^{(\ell)}
\left(
\operatorname{RMSNorm}(\widetilde{\mathbf{h}}^{(\ell)})
\right).
```

The relevant architectural pieces are:

- **Causal self-attention:** position $s$ can attend only to positions at or before $s$, which enforces the conditioning $(q,o_{i,<t})$.
- **RoPE:** rotary position embeddings encode token positions inside the attention queries and keys.
- **GQA:** grouped-query attention uses more query heads than key/value heads to reduce key-value-cache memory.
- **RMSNorm:** normalizes hidden activations before attention and feed-forward sublayers.
- **SwiGLU MLP:** provides the gated nonlinear transformation in each decoder block.
- **Residual connections:** preserve and combine information across layers.

After the final block and normalization, the language-model head projects the hidden vector to one logit per vocabulary item:

```math
\mathbf{z}_{i,t}
=
W_{\mathrm{LM}}\mathbf{h}_{i,t-1}^{(L)}.
```

For batch size $B$, sequence length $S$, hidden width $d$, and vocabulary size $V$, the main shapes are

```math
\mathbf{H}\in\mathbb{R}^{B\times S\times d},
\qquad
W_{\mathrm{LM}}\in\mathbb{R}^{V\times d},
\qquad
\mathbf{Z}\in\mathbb{R}^{B\times S\times V}.
```

A softmax turns logits into the policy distribution:

```math
\pi_\theta(v\mid q,o_{i,<t})
=
\frac{\exp(z_{i,t,v})}
{\sum_{u=1}^{V}\exp(z_{i,t,u})}.
```

The loss selects the log probability assigned to the token that was actually sampled:

```math
\ell_{i,t}^{\theta}
=
\log\pi_\theta(o_{i,t}\mid q,o_{i,<t})
=
\operatorname{logsoftmax}(\mathbf{z}_{i,t})_{o_{i,t}}.
```

Thus, the importance ratio used by GRPO is computed directly from two Qwen forward-pass log probabilities:

```math
\rho_{i,t}
=
\exp\left(\ell_{i,t}^{\theta}-\ell_{i,t}^{\mathrm{old}}\right).
```

The architecture creates the logits; the GRPO trainer creates the grouped samples, rewards, advantages, ratios, clipping, masks, and KL penalty.

## 5. Policy, Rollout, and Reference Models

These three symbols have different roles.

### Current policy

```math
\pi_\theta
```

This is the Qwen model currently being optimized. Gradients flow through its hidden states, LM head, and any trainable parameters or adapters.

### Rollout policy

```math
\pi_{\theta_{\mathrm{old}}}
```

This is the policy state used to generate the stored completions. Its saved token log probabilities form the denominator of $\rho_{i,t}$. They are detached because sampled data are fixed while one or more optimizer updates are performed.

When there is only one update and generation uses the same model implementation, a trainer may use

```math
\ell_{i,t}^{\mathrm{old}}
=
\operatorname{stopgrad}(\ell_{i,t}^{\theta}).
```

The numerical ratio is then $1$ in the forward pass, but the surrogate has a nonzero gradient because only the denominator is detached:

```math
\nabla_\theta
\frac{\pi_\theta(a\mid s)}
{\operatorname{stopgrad}(\pi_\theta(a\mid s))}
=
\nabla_\theta\log\pi_\theta(a\mid s).
```

### Reference policy

```math
\pi_{\mathrm{ref}}
```

This is normally a frozen copy of Qwen before GRPO training. It appears only in the KL regularizer and prevents the optimized model from drifting too far from its initial language behavior.

Therefore,

```math
\pi_{\theta_{\mathrm{old}}}
\neq
\pi_{\mathrm{ref}}
```

in general. The rollout policy changes as training progresses; the reference policy normally stays fixed.

## 6. The KL Penalty

Conceptually, the regularizer is

```math
D_{\mathrm{KL}}
\left[
\pi_\theta(\cdot\mid q,o_{i,<t})
\,\|\,
\pi_{\mathrm{ref}}(\cdot\mid q,o_{i,<t})
\right]
=
\sum_{v=1}^{V}
\pi_\theta(v\mid q,o_{i,<t})
\log
\frac{\pi_\theta(v\mid q,o_{i,<t})}
{\pi_{\mathrm{ref}}(v\mid q,o_{i,<t})}.
```

Computing this full sum over Qwen's large vocabulary at every completion position is expensive. GRPO implementations commonly use a sampled non-negative estimator. Define

```math
\Delta_{i,t}
=
\log\pi_{\mathrm{ref}}(o_{i,t}\mid q,o_{i,<t})
-
\log\pi_\theta(o_{i,t}\mid q,o_{i,<t}).
```

Then TRL uses

```math
D_{i,t}
=
\exp(\Delta_{i,t})-\Delta_{i,t}-1.
```

This quantity is zero when the probabilities match and non-negative because $e^x\geq 1+x$. The minimized loss adds the penalty:

```math
\mathcal{L}
=
\mathcal{L}_{\mathrm{policy}}
+
\beta\,\mathbb{E}[D_{i,t}].
```

This is equivalent to writing $-\beta D_{i,t}$ inside a reward-maximization objective and then negating the whole objective.

A larger $\beta$ keeps Qwen closer to the reference model. Setting $\beta=0$ disables the term and allows implementations to avoid loading a reference model.

## 7. Masking and Length Normalization

Prompt tokens are conditioning context, not sampled actions, so they are not included in the GRPO sum. Padding and tokens after the first EOS token are also masked.

Let $m_{i,t}\in\{0,1\}$ be the completion mask. The effective completion length is

```math
T_i=\sum_t m_{i,t}.
```

The classic per-completion reduction is

```math
\mathcal{L}
=
\frac{1}{G}
\sum_{i=1}^{G}
\frac{1}{T_i}
\sum_t m_{i,t}\mathcal{L}_{i,t}.
```

This gives each completion equal weight regardless of length. Other reductions normalize over all active batch tokens or by a fixed maximum length. Those choices matter because response length can otherwise change the effective weight of an example.

## 8. A Small Numerical Example

Suppose one prompt produces four completions with rewards

```math
(R_1,R_2,R_3,R_4)=(1,1,0,0).
```

The group mean is $0.5$. Using the population standard deviation, $\sigma_R=0.5$, so the advantages are approximately

```math
(\hat A_1,\hat A_2,\hat A_3,\hat A_4)
=(1,1,-1,-1).
```

Consider one token from the first completion:

```math
\pi_{\theta_{\mathrm{old}}}(o_{1,t}\mid q,o_{1,<t})=0.20,
\qquad
\pi_\theta(o_{1,t}\mid q,o_{1,<t})=0.22.
```

Its importance ratio is

```math
\rho_{1,t}=\frac{0.22}{0.20}=1.10.
```

With $\varepsilon=0.2$, the ratio is inside $[0.8,1.2]$, so clipping does nothing. Because $\hat A_1=1$, minimizing the negative surrogate encourages Qwen to increase this token's log probability.

For a below-average completion with $\hat A_i=-1$, the direction reverses: minimizing the loss lowers the probability of its sampled tokens. The reward does not identify one individually incorrect token; it assigns credit or blame to all generated tokens through the shared sequence advantage.

## 9. End-to-End Training Flow

For each optimizer cycle, GRPO performs the following:

1. Apply Qwen's chat template and tokenizer to prompt $q$.
2. Use the rollout Qwen policy to sample $G$ completions for that prompt.
3. Save each sampled completion and its rollout token log probabilities.
4. Score complete responses with `accuracy_reward` or other reward functions.
5. Normalize rewards within each prompt group to obtain $\hat A_i$.
6. Concatenate prompt and completion IDs and run the current Qwen policy.
7. Apply `log_softmax` to Qwen's vocabulary logits and gather each sampled token's log probability.
8. Form $\rho_{i,t}$ from current and rollout log probabilities.
9. If enabled, run the frozen reference Qwen model and calculate the per-token KL estimator.
10. Apply clipping, completion masks, and the selected length normalization.
11. Backpropagate only through the current policy and update $\theta$.

In compact form:

```math
q
\longrightarrow
\{o_i\}_{i=1}^{G}
\longrightarrow
\{R_i\}_{i=1}^{G}
\longrightarrow
\{\hat A_i\}_{i=1}^{G}
\longrightarrow
\text{Qwen token log probabilities}
\longrightarrow
\mathcal{L}_{\mathrm{GRPO}}
\longrightarrow
\nabla_\theta\mathcal{L}.
```

## 10. Relation to the Notebook

The notebook constructs

```python
trainer = GRPOTrainer(
    model="Qwen/Qwen2.5-0.5B-Instruct",
    reward_funcs=accuracy_reward,
    train_dataset=dataset,
    args=args,
)
```

The arguments map to the mathematics as follows:

- `model` defines $\pi_\theta$ and, when needed, the initial reference model.
- `reward_funcs=accuracy_reward` defines $r(q,o_i)$ and therefore $R_i$.
- `train_dataset` supplies prompts $q$ and reference answers used by the reward function.
- `num_generations` in `GRPOConfig` defines $G$.
- `epsilon` defines the clipping parameter $\varepsilon$.
- `beta` defines the KL coefficient $\beta$.
- `num_iterations` determines how many updates reuse a generated batch.
- `loss_type` determines how token losses are normalized and may select a modern GRPO variant.

With the installed TRL version, important defaults are `beta=0.0` and `loss_type="dapo"`. Therefore, the notebook as written does **not** necessarily optimize the exact historical equation shown at the top: by default it omits the reference KL term and uses DAPO's global active-token normalization. To request the classic per-completion GRPO reduction with an explicit reference penalty, configure it directly, for example:

```python
args = GRPOConfig(
    num_train_epochs=1,
    per_device_train_batch_size=1,
    loss_type="grpo",
    beta=0.001,
    epsilon=0.2,
)
```

## 11. What Comes from Qwen and What Comes from GRPO

| Component | Qwen2.5 architecture | GRPO trainer |
|---|---:|---:|
| Tokenization and chat formatting | Yes | Orchestrates |
| Causal hidden states | Yes | No |
| Vocabulary logits | Yes | No |
| $\pi_\theta(o_{i,t}\mid q,o_{i,<t})$ | Yes | Extracts it |
| Sampling $G$ completions | Provides generation | Organizes groups |
| Reward $R_i$ | No | Calls reward functions |
| Advantage $\hat A_i$ | No | Computes from grouped rewards |
| Importance ratio $\rho_{i,t}$ | Supplies log probabilities | Computes ratio |
| Clipping | No | Yes |
| Reference KL | Supplies reference logits | Computes estimator and weighting |
| Final loss reduction | No | Yes |

The central idea is that GRPO does not change what a Qwen language model is. It changes the **training signal applied to Qwen's ordinary next-token probabilities**: tokens belonging to relatively successful completions become more likely, tokens belonging to relatively unsuccessful completions become less likely, and optional clipping and KL regularization constrain how far each update can move the policy.
