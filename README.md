# One Layer Deeper
An architecture-and-optimizer competition from **Core Automation × Tilde Research**.

Build the best function-composition model under a fixed persistent-state ceiling and H100 training-time budget. Participants control architecture, depth, optimizer, learning-rate schedule, and training loss. The evaluator controls data, the outer loop, and final evaluation.

> **Beta period:** July 31 through Sunday, August 2 at 10:00 PM PT.
>
> **Submission deadline:** Monday, August 31 at 10:00 PM PT. The service will not accept submissions after this time.

For competition updates, join [discord.gg/gpumode](https://discord.gg/gpumode) and follow the `#one-layer-deeper` channel.

We are grateful to [Modal](https://modal.com/) for supporting the GPU evaluation infrastructure and to [Northflank](https://northflank.com/) for supporting the competition service and leaderboard. Thank you both for helping make this research competition possible.


## Fork research log

This fork records our experiments. The upstream project still defines the competition.

Our question is simple:

> Can a small model learn `x <- x^2 mod N`, then apply that same step `T` times?

The model receives `x`, `N`, and `T`. It learns only from final answers. We cannot put a modular-arithmetic solver in the model.

We started with Easy because each model gets 60 H100 training seconds. This lets us reject weak ideas quickly. It also exposes an important trap: a model can get some answers right without learning a reusable step.

### How we judge an experiment

- Measure score, update count, time, and model size separately.
- Change one important choice at a time.
- Build experiments from typed configs so comparisons stay clean.
- Save the exact source, settings, result, workload estimate, and checkpoint.
- Test for prompt shortcuts, frozen state, wrong `T` handling, changed evaluation state, and broken gradients.
- Use internal probes only when they can change the next experiment.
- Treat score and learned behavior as separate evidence. A better score does not prove a better algorithm.

An idea moves through these checks:

1. Local behavior and gradient tests.
2. Competition validation on the exact file we plan to run.
3. Rough calculations for memory, work, and expected updates.
4. One clean H100 run.
5. State traces and controlled state changes on the saved model.
6. Transfer from fixed-modulus E1 to variable-modulus E5.
7. Medium only after the model passes the Easy behavior checks.

### What the H100 profile told us

Our first recurrent Transformer had about 202,000 model-state elements. At batch 512 and sequence length 12, one training update was roughly 7.36 GFLOP. Attention was only about 1.5% of that work.

The matched public register model took 430.5 ms per H100 step:

- Batch fetch: 280.8 ms.
- Forward pass and loss: 85.9 ms.
- Backward pass: 24.7 ms.

The optimizer was only about 3.2% of an earlier measured step. Full attention was not the bottleneck. This ruled out custom attention kernels, KDA, sliding-window attention, and sparse attention as useful next steps.

More updates also did not guarantee a better result. The 8.0% model scored exactly 8.0% after 129, 186, and 231 updates.

Our later causal-state model had 68,736 model-state elements and completed 196 updates in 60.30 seconds. Its rough throughput was 23 GFLOP/s. This small model spends much of its time launching operations instead of using the H100's arithmetic units. That is an estimate, not a full roofline profile.

### Experiment log

#### 1. Recurrent register models

We tested tied Transformer blocks, field and decimal-place features, continuous state, and hard token feedback.

| Model | E1 test | E1 OOD | E1 mean | Decision |
|---|---:|---:|---:|---|
| Continuous recurrent register | 4.67% | 3.00% | 3.83% | Stop |
| Discrete recurrent register | 2.00% | 1.00% | 1.50% | Stop |
| Matched public register model | 6.00% | 10.00% | 8.00% | Keep as reference |

The discrete model had lower validation loss but worse exact answers. Token loss and full-answer accuracy were telling us different things.

A public Square-then-Reduce model had already scored 1.67%, or 2.00% with Muon. We did not repeat that weak branch.

#### 2. Reproducing the 8.0% model

Our copy matched the public model's parameter count, logits, and loss. The audit also found a batch-wide maximum in its final readout. We made that choice explicit in the experiment config.

We then changed one choice at a time:

| Change from the 8.0% reference | E1 mean | Result |
|---|---:|---|
| Per-example final readout | 7.33% | Worse |
| Standard `0.02` initialization | 7.00% | Worse |
| Entropy loss only on valid tokens | 8.00% | No change |
| Train through every recurrent step | **8.50%** | Repeated twice |

The 8.5% model is our best E1 reference. It did not transfer:

| Dataset | Mean exact accuracy | Certified depth |
|---|---:|---:|
| E1 | 8.50% | None |
| E2 | 1.21% | None |
| E3 | 0.875% | None |
| E5 | 0.375% | None |

E1 has a fixed modulus, so shortcuts can work. E5 changes the modulus and is a better test of the learned operation.

#### 3. Why 8.5% did not prove a learned loop

On the fixed `T=8` check, the model reached 10.53% after one update and never improved. After step two, it entered an exact two-step cycle.

CKA and Procrustes showed that later states were almost identical. A model-specific Jacobian lens showed that influence from early recurrent states shrank almost to zero. These measurements supported the diagnosis, but they did not prove it.

The direct test was decisive: disabling feedback state changed E1 test accuracy, E1 OOD accuracy, and the `T=8` result by exactly zero. Removing field and decimal-place features caused a large drop.

The 8.5% model is a good one-step classifier. It does not use feedback state to repeat the calculation.

#### 4. Forcing the model to use state

We then removed the direct path from the prompt to the answer:

```text
x -> encode once -> mutable residue state y0
N --------------> fixed modulus memory m
                  scratch state z0

(y1, z1) = tied_cell(y0, z0, m)
(y2, z2) = tied_cell(y1, z1, m)
...
answer = decode(yT)
```

The answer head reads only the final residue state. The model sees `x` once. It keeps `N` because every step needs the modulus.

This design passed 167 local contract tests, standalone loading, competition validation, checksum checks, and a real CPU training step.

One clean E1 H100 run produced:

| Measurement | Result |
|---|---:|
| Updates | 196 |
| Training time | 60.30 s |
| Training loss | 2.860 -> 1.834 |
| Test exact accuracy | 3.33% |
| OOD exact accuracy | 0% |
| E1 mean | 1.67% |
| Certified depth | None |

The loss fell normally. AdamW was not the main failure.

#### 5. The state mattered, but the updates were wrong

This model really used its state. At `T=4`, freezing state after step 1 or 2 changed every prediction. At `T=2`, freezing after step 1 changed 92.1% of predictions. But later updates made answers worse:

| Evaluation | Full model | Freeze after step 1 | Freeze after step 2 |
|---|---:|---:|---:|
| E1 fixed `T=4` | 0% | 7.89% | 7.89% |
| E1 held-out `T=6` | 0% | 6.00% | 9.00% |

After training, we compared each state with the correct answer for that step. These answers were never used for training.

| State at fixed `T=4` | Accuracy for the correct state at that step |
|---|---:|
| Encoded input `x` | 60.53% |
| After update 1 | 5.26% |
| After update 2 | 2.63% |
| After update 3 | 2.63% |
| After update 4 | 0% |

The model uses its state, but the state does not hold the correct calculation. It stores changing guesses.

Scratch state also looked like a shared clock or bias. Swapping it between examples changed only 2.63% of `T=4` predictions. Removing it changed every prediction and improved the loss.

The local Jacobians showed stable or shrinking updates, not exploding values:

| Transition | Spectral radius | Median singular value |
|---|---:|---:|
| 0 -> 1 | 0.731 | 0.501 |
| 1 -> 2 | 0.718 | 0.482 |
| 2 -> 3 | 0.831 | 0.537 |
| 3 -> 4 | 0.937 | 0.651 |

The update was stable enough. It had learned the wrong function. A different optimizer would only train that wrong function differently.

### What we ruled out

- A higher E1 score does not prove learned recurrence. The 8.5% model ignored feedback state.
- Lower token loss does not guarantee more exact answers.
- More updates do not guarantee a better score.
- The attention type is not the main issue for these short inputs.
- A Jacobian lens does not read the model's thoughts. It measures local sensitivity and needs direct state tests beside it.
- This is not normal in-context learning. The prompt gives no examples from which to infer a new rule. The weights learn the rule; the state must apply it.
- A more advanced optimizer is not useful until the model learns a useful update.
- Medium stays blocked until a model works on E5 and passes the state tests.

### Papers that changed the plan

- [Improving the Neural GPU Architecture for Algorithm Learning](https://huggingface.co/papers/1702.08727): use a shared recurrent digit grid that can move information left and right.
- [Looped Transformers for Length Generalization](https://huggingface.co/papers/2409.15647): train one shared update at different depths. We do not reuse its direct prompt injection.
- [Less is More: Recursive Reasoning with Tiny Networks](https://huggingface.co/papers/2510.04871): keep the fixed problem, changing answer, and scratch memory separate.

Tabular-model ideas helped us encode `N`, `x`, `T`, and decimal place. Those features improved E1, but they did not perform the calculation.

### What we will try next

We will stop tuning the current model. Each digit updates mostly on its own, and the digits share only a pooled summary. Squaring needs digits to exchange products and carries. Modular reduction also needs comparison and subtraction across digits.

The next model will use a shared recurrent digit grid:

- One lane for each decimal place.
- Information moves left and right between nearby digits.
- Modulus digits stay available at matching positions.
- The answer comes only from the final grid.
- A small global controller is added only if local updates work but modular reduction fails.
- Start with AdamW and train through every step.

We will continue only if:

- Changing the state changes the answer.
- Later states move toward the correct intermediate answers.
- The state does not get stuck or enter a short cycle.
- E5 reaches at least 0.75% and gets at least one OOD-modulus answer right.
- The model passes these checks before we try Medium.

### Code and evidence map

- [`submissions/recurrent/submission.py`](submissions/recurrent/submission.py): experiment configs and models.
- [`research/runner.py`](research/runner.py): checks, workload estimates, H100 runs, and receipts.
- [`research/step_benchmark.py`](research/step_benchmark.py): step timing.
- [`research/interventions.py`](research/interventions.py): state tests.
- [`research/geometry.py`](research/geometry.py): per-step results, CKA, Procrustes, and cycle checks.
- [`research/jacobian.py`](research/jacobian.py): Jacobian lens and local transition measurements.
- [`autoresearch/orchestrator-260804-2322/results.tsv`](autoresearch/orchestrator-260804-2322/results.tsv): result ledger.
- [`autoresearch/orchestrator-260804-2322/handoff.json`](autoresearch/orchestrator-260804-2322/handoff.json): final verdict for the 8.5% model.

Large checkpoints, profiles, and detailed traces stay in the ignored `.artifacts/` directory. They are not part of this public fork. The source, tests, result ledger, and decisions are versioned here.

No official competition submission was created by this research run.


## Install

#### CLI only for remote GPU use

```bash
uv tool install git+https://github.com/tilde-research/one-layer-deeper.git
one-layer --help
```

[See the full CLI instructions.](#cli)


#### CLI and local and remote GPU use

```bash
git clone https://github.com/tilde-research/one-layer-deeper.git
cd one-layer-deeper
uv venv .venv
source .venv/bin/activate
uv sync
python -m unittest discover -s tests
```

[See the full CLI](#cli) and [local development](#local-development) instructions.


## Rules

1. Submit exactly one UTF-8 file named `submission.py`. It exports one `benchmark.Submission` with model and optimizer factories and an optional training loss.
2. The submission must be self-contained. It may import the public `benchmark` API and pinned evaluator dependencies, but it may not depend on repository `model` or `optim` modules, extra files, package installation, or external services.
3. Participant code defines the model, optimizer bundle, optional learning-rate scheduler, optional loss, training and evaluation batch sizes, and maximum training steps. Recurrence, adaptive computation, and depth curricula are allowed.
4. The evaluator fixes data order and owns the outer loop, model and loss invocations, backward passes, gradient clipping, optimizer cadence, seeds, deadline, final evaluation, and aggregation. A submission may declare bounded evaluator-owned forward/backward passes within one optimizer step and may dynamically request bounded reuse of the current batch. This does not otherwise restrict computation within a submitted model, loss, or optimizer: recurrent/iterative mechanisms, TRMs, and optimizer-side curvature or Hessian approximations are allowed. Participants may choose the training and evaluation batch size and a lower maximum step count; evaluator ceilings still apply.
5. The model may contain at most 500,000,000 trainable parameters. Shared state counts once; persistent buffers and frozen state still count toward the model-state ceiling.
6. No hard-coded weights. Trainable weights must use a random initialization and be updated during training. For example, `torch.load` is not allowed.
7. No hard-coded algorithm in the forward pass. Outputs must be produced by the learned model.
8. End-to-end learning only. Final logits must be produced entirely by the submitted model from its inputs and learned PyTorch state, with all input-dependent computation inside the autograd graph and an unbroken gradient path from the loss to the parameters responsible for the prediction.
9. Everything stays on the GPU. Model state and computation must remain on the GPU throughout training and evaluation; CPU offloading is not allowed.
10. Optimizer state, activations, and temporary workspace may use remaining VRAM. OOM or timeout fails the run.
11. Easy provides 60 H100 training seconds, Medium 600 seconds, and Hard 3,600 seconds. Model construction, submission import, and compilation consume the budget.
12. Token tasks may use legacy `training_loss`, which receives flattened valid logits and labels plus the model's auxiliary output, or `token_training_loss`, which receives a boundary-preserving `TokenLossBatch`. A custom loss returns one differentiable finite scalar for every evaluator-owned pass; the evaluator performs backward.
13. Each final checkpoint is evaluated once with a separate time budget equal to half its training allowance. Easy and Medium score mean exact accuracy. Hard ranks by the largest consecutively certified T on fresh prompts using modulus identities seen during training, then by the largest consecutively certified T on unseen modulus identities, then by accuracy at each profile's first uncertified rung. Both use T=1,2,4,8,16,32,64; every example in a rung must be exactly correct, and certification must form a consecutive prefix.
14. Data inspection, data augmentation, task-specific solvers, custom training loops, participant-controlled backward passes, and manifest overrides are not allowed. Participant code must not invoke derivative-engine entry points such as `Tensor.backward`, `torch.autograd.backward`, or `torch.autograd.grad`; ordinary differentiable tensor operations inside the submitted model and loss remain allowed. Model and loss code, autograd hooks, and `OptimizerBundle` callbacks must not initiate nested model or loss calls, derivative-engine entry points, optimizer or scheduler steps, or other hidden training work. The documented intermediate gradient, parameter, and optimizer-state transformation is the only exception.
15. Repeated rule-breaking will get you banned. We still encourage creativity: discussing possible loopholes on Discord or testing one in a submission won't get you banned.
16. The metric recorder for a Hard run must not be exploited. Any attempt to exploit it will result in an immediate ban.

Depth is deliberately unconstrained. Fixed stacks, tied recurrence, iterative refinement, routing, adaptive halting, memory tokens, and parameter-free work are all valid if the model-state ceiling is respected. A deeper forward completes fewer optimizer updates under the same clock.

### Submission contract

The file is limited to 256 KiB. `build_model(spec)` receives `vocab_size`, `max_seq_len`, and `maximum_model_state_elements`. It returns a `torch.nn.Module` whose `config` exposes the first two matching fields. The model accepts evaluator tensor arguments and returns `(logits, auxiliary_value)`.

The evaluator calls `model.train()` for optimization and `model.eval()` for final evaluation. If the model should behave differently during evaluation, use PyTorch's inherited `self.training` flag inside `forward` (for example, `if self.training: ... else: ...`).

`build_optimizer(model, spec)` receives the per-seed time allowance and device type. It returns an `OptimizerBundle`; its optimizer must include every trainable parameter exactly once. An optional scheduler is stepped after every update. The bundle may also declare bounded multi-pass and batch-reuse callbacks described below.

```python
from benchmark import ModelSpec, OptimizerBundle, OptimizerSpec, Submission, assert_model_state

def build_model(spec: ModelSpec):
    model = MyModel(spec)
    assert_model_state(model, spec)
    return model

def build_optimizer(model, spec: OptimizerSpec) -> OptimizerBundle:
    return OptimizerBundle(MyOptimizer(model.parameters()))

SUBMISSION = Submission(
    build_model=build_model,
    build_optimizer=build_optimizer,
    batch_size=512,       # optional; training
    eval_batch_size=1024, # optional; evaluation
    max_steps=20_000,     # optional; cannot exceed the evaluator ceiling
)
```

If omitted, `batch_size` and `max_steps` use the evaluator manifest defaults.
Evaluation uses `eval_batch_size` when provided, then an explicit participant
`batch_size`, then the evaluator manifest's evaluation batch size, and finally
the manifest's training batch size. A participant `max_steps` can end training
early. The evaluator's wall-clock deadline and absolute step ceiling always remain
enforced. An optional scheduler returned in `OptimizerBundle` is stepped after
every completed optimizer update.

`OptimizerBundle` can request 1–8 evaluator-owned forward/loss/backward
passes on the same batch before one optimizer update with
`backward_passes_per_step`. Gradients are cleared and clipped independently on
each pass. After each non-final pass,
`between_backward_passes(BackwardPassContext)` runs under `torch.no_grad()`
and may transform gradients, parameters, or optimizer state; a custom optimizer
can restore temporary perturbations when it performs the final update.

After an optimizer and scheduler update,
`should_reuse_batch(BatchReuseContext)` runs under `torch.no_grad()` and may
return `True` to request another update on that batch. The evaluator advances
after at most eight uses. The reuse callback is decision-only. Neither callback
may start nested model/loss calls, backward or autograd entry points, or
optimizer/scheduler steps.

Context pass and batch-use indexes are one-based; `completed_steps` counts
finished optimizer updates, and the reuse context's `loss` is a detached
Python `float`. One benchmark step remains one optimizer update, and all extra
passes and callbacks consume the same wall-clock budget.

Token tasks offer two mutually exclusive custom-loss callbacks. The legacy
`training_loss(logits, labels, auxiliary)` receives only valid tokens flattened
to `[valid_tokens, vocab_size]` and `[valid_tokens]`. For sequence-aware
losses, `token_training_loss(batch)` receives a `TokenLossBatch` whose
`logits`, `labels`, and boolean `valid_mask` retain
`[batch, target_length, ...]` boundaries. Its `target_positions` is present
for separate-output tasks and `None` for causal targets; invalid slots must be
ignored using `valid_mask`.

```python
import torch.nn.functional as F
from benchmark import TokenLossBatch

def token_training_loss(batch: TokenLossBatch):
    token_losses = F.cross_entropy(
        batch.logits.transpose(1, 2),
        batch.labels,
        ignore_index=-100,
        reduction="none",
    )
    target_counts = batch.valid_mask.sum(dim=1)
    sequence_losses = (
        (token_losses * batch.valid_mask).sum(dim=1)
        / target_counts.clamp_min(1)
    )
    return sequence_losses[target_counts > 0].mean()

SUBMISSION = Submission(
    build_model=build_model,
    build_optimizer=build_optimizer,
    token_training_loss=token_training_loss,
)
```

The website offers one basic, non-recurrent Transformer using `torch.optim.AdamW`. Its standalone `submission.py` lives under `submissions/baseline_adamw`.

### Compute tiers

The public Easy and Medium datasets provide separate prompt and output tensors.
The evaluator supplies a padding mask, not a causal mask, so models can attend
bidirectionally over the complete prompt in those practice tiers. Hard uses a
private hidden evaluator.

- **Easy:** datasets `e1`–`e5`, 60 training seconds, 60 accepted attempts per UTC day.
- **Medium:** datasets `m1`–`m5`, 600 training seconds, 6 accepted attempts per UTC day.
- **Hard:** dataset `h1`, 3,600 training seconds, 1 accepted attempt per UTC day.

Easy and Medium are practice tiers. The public leaderboard ranks only each participant's best successful Hard submission. Failed evaluations count after acceptance; authentication and validation rejections do not. Source and detailed results remain private.

Easy and Medium expose the same `Max T` and `OOD N Max T` fields as Hard, using the common T=1,2,4,8,16,32,64 ladder. Each profile remains specific to its dataset: Max T evaluates modulus identities used by the training dataset, while OOD N Max T evaluates unseen identities at nearby dataset-scale modulus sizes. These practice-tier profiles are diagnostic and do not change their exact-accuracy scores.

Hard ranking uses two certified depth values over private hidden profiles. **Max T** measures in-distribution problem families, while **OOD N Max T** measures out-of-distribution problem families. The evaluator details and data remain private.

A value is the largest T for which that rung and every lower rung have 100% exact-example accuracy. The leaderboard ranks by Max T, then OOD N Max T, then exact accuracy at the first uncertified rung in each profile. Earlier submission time is the final fallback. The public leaderboard shows each next-rung accuracy rounded to four decimal places while ranking uses the unrounded value. All other per-seed measurements and rung results remain private diagnostics.

## CLI

### Install the CLI

This installs the lightweight submission CLI only. To run evaluations locally, see [Local development](#local-development).

Install [uv](https://docs.astral.sh/uv/) and then install the command directly from GitHub:

```bash
uv tool install git+https://github.com/tilde-research/one-layer-deeper.git
one-layer --help
```

### Example workflow

```bash
one-layer login
one-layer validate submissions/baseline_adamw/submission.py
one-layer submit submissions/baseline_adamw/submission.py --tier easy --dataset e1 --wait
one-layer jobs
one-layer status <submission-id>
one-layer metrics <submission-id> --output metrics.jsonl
one-layer leaderboard
```

`one-layer login` opens GitHub authentication, receives a generated `old_…` API key through a temporary localhost callback, and saves it to `~/.config/one-layer/config.json` with user-only permissions. Signing in again rotates a lost key. The service stores the GitHub identity plus only the key's SHA-256 digest and short support prefix. By default, one evaluation may be queued or running per GitHub account.

`one-layer jobs` lists the signed-in participant's queued and running submissions,
including the submission IDs accepted by `one-layer status <submission-id>`. Use
`one-layer jobs --all` to include completed and failed submissions, or `--json`
for machine-readable output.

After a successful evaluation, `one-layer metrics <submission-id>` downloads a
bounded JSONL history containing evaluator-selected training, evaluation, and
summary metrics. Raw submission stdout, stderr, and exception text are not
included in participant-facing status responses or metric downloads, and are
deleted from the service database 24 hours after the run finishes.


## Local development

### Install locally

Clone the repository and install its dependencies:

```bash
git clone https://github.com/tilde-research/one-layer-deeper.git
cd one-layer-deeper
uv venv .venv
source .venv/bin/activate
uv sync
python -m unittest discover -s tests
```

### Example of running a submission locally

Modal is not required for local evaluation. The runner takes an evaluator-owned
manifest and one standalone submission file. Start with the short CPU smoke test:

```bash
python -m benchmark.runner \
  --manifest benchmark/manifests/smoke_cpu.json \
  --submission-file submissions/baseline_adamw/submission.py
```

The smoke manifest creates its small dataset automatically. Before running a
public Easy or Medium manifest, generate the full datasets referenced by those
manifests:

```bash
bash scripts/generate_datasets.sh
```

The script writes the datasets under `data/generated/`. You only need to run it
again if those generated files are removed. For a tier-faithful run on a local
H100, first find an idle GPU and expose only that device. The manifest's
`cuda:0` will then refer to the selected physical GPU:

```bash
CUDA_VISIBLE_DEVICES=0 python -m benchmark.runner \
  --manifest benchmark/manifests/h100_easy_e1.json \
  --submission-file submissions/baseline_adamw/submission.py
```

Hard evaluation is available only through hosted submission. The final `RESULT_JSON=...` line contains aggregate and split metrics.

## License

Licensed under the [Apache License 2.0](LICENSE).
