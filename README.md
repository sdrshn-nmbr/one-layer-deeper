# One Layer Deeper
An architecture-and-optimizer competition from **Core Automation × Tilde Research**.

Build the best function-composition model under a fixed persistent-state ceiling and H100 training-time budget. Participants control architecture, depth, optimizer, learning-rate schedule, and training loss. The evaluator controls data, the outer loop, and final evaluation.

> **Beta period:** July 31 through Sunday, August 2 at 10:00 PM PT.
>
> **Submission deadline:** Monday, August 31 at 10:00 PM PT. The service will not accept submissions after this time.

For competition updates, join [discord.gg/gpumode](https://discord.gg/gpumode) and follow the `#one-layer-deeper` channel.

We are grateful to [Modal](https://modal.com/) for supporting the GPU evaluation infrastructure and to [Northflank](https://northflank.com/) for supporting the competition service and leaderboard. Thank you both for helping make this research competition possible.


## Fork research log

This fork is a research workspace. The upstream project defines the competition. The work here asks a narrower question:

> Can a small learned model apply the update `x <- x^2 mod N` repeatedly, with one learned step that still works when it is run more times?

The goal is not to hide the arithmetic in Python. The model must learn the update from the evaluator's final answers. The model gets `x`, `N`, and `T`; it must return the value after `T` updates.

We started on Easy because its 60-second H100 budget makes bad ideas cheap to reject. Easy is still hard enough to expose the main problem: a model can get a few answers right without learning a reusable loop.

### How we run the research

The workflow is built around a few rules:

- Measure first. Keep score, exact accuracy, update count, wall-clock time, model size, and stage timing separate.
- Treat the evaluator as the source of truth. A local smoke test proves wiring, not H100 speed or competition score.
- Change one important thing at a time.
- Keep model construction behind typed experiment configs and factories so two experiments differ only where intended.
- Save the exact source, config, result, workload estimate, and checkpoint for each serious run.
- Use tests to catch prompt shortcuts, frozen state, incorrect per-example `T` masking, changed model state during evaluation, and broken gradients.
- Use internal measurements only when they can change the next decision.
- Keep score evidence and mechanism evidence separate. A higher score does not prove that the model learned repeated squaring.

The evidence ladder is:

1. Local semantic and gradient tests.
2. Competition validation of the exact standalone file.
3. Napkin math for parameters, activations, operations, and expected update count.
4. One clean H100 timing or training run.
5. Offline traces, state interventions, geometry, and Jacobian analysis on the saved checkpoint.
6. Transfer from fixed-modulus E1 to variable-modulus E5.
7. Medium only after the Easy mechanism test passes.

### What profiling changed

The first recurrent Transformer had about 202,000 model-state elements. At batch 512 and sequence length 12, one estimated training update was about 7.36 GFLOP. Attention was only about 1.5% of the estimated work.

The matched public recurrent-register model took a median 430.5 ms per measured H100 step. Batch fetch took 280.8 ms, forward and loss took 85.9 ms, and backward took 24.7 ms. The optimizer in an earlier profile was only about 3.2% of the full step.

These measurements changed the plan:

- Full attention was not the bottleneck.
- KDA, sliding-window attention, and sparse attention did not address the observed failure. The sequences are short and the problem is not long-context lookup.
- Custom attention kernels were not justified.
- A faster optimizer could not repair a model that learned the wrong computation.
- More updates were not automatically better. The 8.0% E1 model repeated the same score after 129, 186, and 231 updates.

The causal-state control was even smaller: 68,736 model-state elements. It completed 196 updates in 60.30 seconds. A rough estimate put the run near 23 GFLOP/s, which is consistent with a tiny recurrent workload dominated by framework and launch overhead. This is a directional estimate, not a formal roofline result.

### Experiment log

#### 1. Broad Transformer and register search

We first tested normal Transformer variants, tied recurrence, structured field/place features, continuous state, and hard discrete feedback.

The useful result was not a winning architecture. It was a clearer failure boundary:

| Model | E1 test | E1 OOD | E1 mean | Decision |
|---|---:|---:|---:|---|
| Continuous recurrent register | 4.67% | 3.00% | 3.83% | Stop |
| Discrete recurrent register | 2.00% | 1.00% | 1.50% | Stop |
| Matched public register model | 6.00% | 10.00% | 8.00% | Keep as reference |

The discrete model reached a lower validation loss but worse exact answers. This was an early warning that token loss and exact sequence accuracy can disagree.

We also audited a learned Square-then-Reduce design before building it. A public result had already tested the same basic idea at 1.67%, or 2.00% with Muon. We did not spend another H100 run repeating a known weak branch.

#### 2. Exact reproduction and isolated controls

We reproduced the public 8.0% register model. Parameter counts, initial logits, training logits, and loss matched numerically. The audit found a batch-dependent maximum readout that was easy to miss from the architecture description, so it became an explicit factory option.

The compact tracked ledger is in [`autoresearch/orchestrator-260804-2322/results.tsv`](autoresearch/orchestrator-260804-2322/results.tsv).

| Change from the 8.0% reference | E1 mean | Result |
|---|---:|---|
| Per-example final readout | 7.33% | Worse |
| Standard `0.02` initialization | 7.00% | Worse |
| Mask entropy only on valid tokens | 8.00% | No measured change |
| Backpropagate through every recurrent step | **8.50%** | Reproduced twice |

The 8.5% result is the best result in this fork so far. It is a reference score, not a mechanism result.

Transfer made that distinction clear:

| Dataset | Mean exact accuracy | Certified depth |
|---|---:|---:|
| E1 | 8.50% | None |
| E2 | 1.21% | None |
| E3 | 0.875% | None |
| E5 | 0.375% | None |

E1 uses a fixed modulus and is easy to fit with shortcuts. E5 varies the modulus and is the better early test of whether the model learned the operation.

#### 3. Looking inside the 8.5% model

We saved matched E1 and E5 checkpoints, then used several views of the same recurrent states:

- Per-step answer readouts.
- CKA, which compares the pairwise geometry of two state sets.
- Procrustes alignment, which asks whether two state sets match after a rotation and rescaling.
- A Jacobian lens fitted to this model's own state and output head.
- Direct state interventions: zero, freeze, swap, or remove a channel and rerun evaluation.

The E1 model reached 10.53% on the fixed `T=8` diagnostic after its first update and never improved after that. From step two onward, its states formed an exact two-step cycle. Adjacent states had CKA and Procrustes similarity near 1.0.

The Jacobian lens showed that the differentiable path from early recurrent state to the final state contracted almost to zero. The apparent cycle was caused by hard token switching, not by an unstable linear mode.

The decisive test was simpler: disabling the feedback state changed E1 test, E1 OOD, and the `T=8` diagnostic by exactly zero. Removing field/place features did cause a large collapse.

The plain conclusion is:

> The 8.5% model is a structured one-step classifier. It reads the prompt well, but it does not use its recurrent state to perform repeated squaring.

This is why CKA, Procrustes, and Jacobian lenses are supporting evidence rather than proof. Geometry can show that states are similar or that local influence contracts. Only interventions can show whether a state actually controls the answer.

#### 4. A model with no prompt-to-answer shortcut

The next architecture enforced this data flow:

```text
x -> encode once -> mutable residue state y0
N --------------> fixed modulus memory m
                  scratch state z0

(y1, z1) = tied_cell(y0, z0, m)
(y2, z2) = tied_cell(y1, z1, m)
...
answer = decode(yT)
```

The output head can read only the final residue state. The original `x` is not injected again. `N` stays available because every modular reduction needs it. A fixed GPU loop and a tensor mask select the requested `T` for each example.

Gate 0 passed 167 local contract tests, standalone loading, competition validation, exact-source materialization, checksum verification, and a real CPU forward/backward/update smoke test.

Its one clean E1 H100 run produced:

| Measurement | Result |
|---|---:|
| Updates | 196 |
| Training time | 60.30 s |
| Training loss | 2.860 -> 1.834 |
| Test exact accuracy | 3.33% |
| OOD exact accuracy | 0% |
| E1 mean | 1.67% |
| Certified depth | None |

The model optimized normally. Its poor exact accuracy was not evidence of an AdamW failure.

#### 5. Causal-state gate

The newer model passed one test that the 8.5% model failed: its propagated residue state really controls its predictions.

At `T=4`, freezing the residue after step 1 or step 2 changed 100% of predictions. At `T=2`, freezing after step 1 changed 92.1%. But the later updates made the result worse:

| Evaluation | Full model | Freeze after step 1 | Freeze after step 2 |
|---|---:|---:|---:|
| E1 fixed `T=4` | 0% | 7.89% | 7.89% |
| E1 held-out `T=6` | 0% | 6.00% | 9.00% |

We then compared each internal state with the mathematically correct answer for that step. These intermediate answers were used only after training as diagnostics. They never entered the loss.

| State at fixed `T=4` | Accuracy for the correct state at that step |
|---|---:|
| Encoded input `x` | 60.53% |
| After update 1 | 5.26% |
| After update 2 | 2.63% |
| After update 3 | 2.63% |
| After update 4 | 0% |

The state is causal, but it is not arithmetic. The model uses its notebook, but it writes changing guesses.

Scratch state also looked more like a shared clock or bias than useful per-example memory. Swapping scratch between examples changed only 2.63% of `T=4` predictions. Zeroing scratch changed every prediction and improved token loss.

The local state Jacobians did not show numerical explosion:

| Transition | Spectral radius | Median singular value |
|---|---:|---:|
| 0 -> 1 | 0.731 | 0.501 |
| 1 -> 2 | 0.718 | 0.482 |
| 2 -> 3 | 0.831 | 0.537 |
| 3 -> 4 | 0.937 | 0.651 |

The update is stable enough. It learned the wrong function. This closed the optimizer question for this checkpoint: Muon, SOAP, OKLS, or another AdamW tweak would optimize the wrong transition more efficiently.

### What changed our direction

Several attractive ideas did not survive contact with the measurements:

- **“A higher E1 score means learned recurrence.”** False. The 8.5% model ignored its feedback state.
- **“Low training loss means better exact answers.”** False for the discrete model and several controls.
- **“More updates should improve the result.”** False for the repeated 8.0% runs.
- **“The attention variant is the main architectural choice.”** Not at these sequence lengths and not for the failure we measured.
- **“A Jacobian lens reads the model's thoughts.”** No. It measures local first-order influence through a chosen path. It must be paired with finite interventions.
- **“This is mainly in-context learning.”** Not in the usual sense. The prompt has no demonstrations from which to infer a temporary rule. The weights learn one transition; the recurrent state must execute it.
- **“An advanced optimizer is the next lever.”** Not until a model learns a useful state transition and profiling shows an optimization problem.
- **“We should move to Medium after an E1 gain.”** No. E5 and the causal-state checks remain closed.

### Papers that changed the plan

Three papers produced concrete experiments rather than general inspiration:

- [Improving the Neural GPU Architecture for Algorithm Learning](https://huggingface.co/papers/1702.08727): use a tied recurrent digit grid with local, directional communication. This is the closest fit for products, carries, and digit-wise state.
- [Looped Transformers for Length Generalization](https://huggingface.co/papers/2409.15647): expose the same tied update to mixed depths. We borrow mixed-`T` training, but not repeated reinjection of the original prompt.
- [Less is More: Recursive Reasoning with Tiny Networks](https://huggingface.co/papers/2510.04871): keep static problem memory, mutable answer state, and scratch state separate. We borrow the state split, not its task or loss unchanged.

Tabular-model ideas also helped at the input boundary: `N`, `x`, and `T` are different fields, and decimal place matters. Field/place features improved E1. They are useful encodings, but they are not the arithmetic engine.

### Current decision

The current causal-state model should not receive more optimizer, kernel, or parameter tuning. It has independent per-digit GRU updates and communicates mostly through a mean-pooled scratch controller. Repeated squaring needs cross-digit products, carries, comparison, and reduction. Mean pooling is too weak and too lossy for that job.

The next architecture bet is a tied recurrent digit grid, closer to a Neural GPU:

- One lane per decimal position.
- Local left/right communication between neighboring digits.
- Static aligned modulus features.
- Continuous residue and scratch channels.
- A small global controller only if local communication learns useful state but cannot handle variable-modulus reduction.
- Final output decoded only from the last residue grid.
- AdamW first, with full backpropagation.

Promotion still requires more than a score increase:

- State zero/freeze/swap tests must change predictions.
- Correct intermediate-state accuracy must improve across useful steps.
- The state must avoid a fixed point or short cycle.
- E5 mean accuracy must at least double the 0.375% reference and have nonzero OOD-modulus accuracy.
- Medium remains blocked until the model shows both causal recurrence and useful arithmetic transfer.

### Code and evidence map

- [`submissions/recurrent/submission.py`](submissions/recurrent/submission.py) contains the typed experiment configs, recurrent models, and standalone materialization source.
- [`research/runner.py`](research/runner.py) runs parity checks, workload estimates, H100 experiments, and artifact receipts.
- [`research/step_benchmark.py`](research/step_benchmark.py) measures per-stage step time.
- [`research/interventions.py`](research/interventions.py) performs evaluation-only causal state tests.
- [`research/geometry.py`](research/geometry.py) records per-step readouts, CKA, Procrustes alignment, and cycle behavior.
- [`research/jacobian.py`](research/jacobian.py) fits the model-specific recurrent Jacobian lens and local transition spectra.
- [`autoresearch/orchestrator-260804-2322/results.tsv`](autoresearch/orchestrator-260804-2322/results.tsv) is the compact result ledger for the reproduced 8.5% reference and its controls.
- [`autoresearch/orchestrator-260804-2322/handoff.json`](autoresearch/orchestrator-260804-2322/handoff.json) records the closed mechanism verdict for that model.

Large checkpoints, profiler output, and detailed JSON traces stay under the ignored `.artifacts/` directory. They are local audit evidence and are not included in the public fork. The source, tests, compact result ledger, and final decisions are versioned here.

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
