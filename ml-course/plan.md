# Introductory Machine-Learning Lectures Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Split the overloaded introductory neural-network lecture into focused, visual decks covering supervised learning, optimization through Adam, dense neural networks, generalization and regularization, CNNs, and recurrent sequence models.

**Architecture:** Each topic has one owning deck. Slides use short bullets, equations, and one full-slide figure at a time. Numerical visuals come from small deterministic Python generators; conceptual visuals use TikZ.

**Tech Stack:** `ntuabeamer`, LaTeX, TikZ/PGFPlots, Python, NumPy, Matplotlib, `uv run --with`, Poppler.

**Spec:** Approved design and content requirements in this document under “Global Constraints” and “Deck Content”.

**Implementation priority:** Finish the two immediate introductory decks, then create the CNN and recurrent sequence-model decks before the remaining cleanup work.

## Global Constraints

- Preserve the existing conceptual order: fundamentals -> optimization -> dense neural networks -> regularization -> CNNs -> recurrent sequence models.
- Use `chapter*.pdf` only as a coverage reference. Do not copy its order, wording, figures, or code.
- Add no programming code, framework snippets, or screenshots of code to slides.
- Use short bullets. Use one sentence per bullet.
- Avoid prose paragraphs on slides.
- Prefer equations and visuals over narration.
- Give every figure its own slide.
- Use one visual argument per figure slide.
- Use takeaway titles, not titles such as “Visualization”.
- Keep labels and short annotations inside the figure.
- Use consistent notation, colors, axes, initial points, and optima across related figures.
- Prefer TikZ for conceptual diagrams.
- Use deterministic generators only when numerical experiments make the visual clearer.
- Put generators in the owning deck's `src/` directory.
- Put generated figures in the owning deck's `assets/generated/` directory.
- Compute output paths from `__file__`.
- Use explicit random seeds.
- Use `uv run --with`; do not add project dependencies.
- Add no abstraction shared between decks unless a second real use appears.
- Download an image only when it communicates better than TikZ or a generated figure.
- Record author, source URL, and license next to every downloaded asset.
- Do not copy anything from `../_archive` into the repository.
- Keep `TOPICS.toml` as the single topic-ownership registry.
- Keep every deck buildable on its own.
- Finish with `tools/build-all`; every target must report `ok`.

## Review Focus

- **Slide density:** No frame may contain a prose paragraph or a figure competing with long text.
- **Visual legibility:** Labels, legends, and equations must remain readable in the rendered slide.
- **Statistical correctness:** Validation data selects models; test data is used once for final evaluation.
- **Optimizer comparability:** Comparative trajectories must use the same objective, start point, budget, and axis limits.
- **Regularization interpretation:** The Gaussian-prior interpretation must state its likelihood/MAP assumptions and must not imply that the true weights are known to be zero.

## File Structure

### Create

- `presentations/supervised-learning/.latexmkrc`
- `presentations/supervised-learning/main.tex`
- `presentations/supervised-learning/src/make_supervised_figures.py`
- `presentations/supervised-learning/assets/generated/*.png`
- `presentations/optimization/.latexmkrc`
- `presentations/optimization/main.tex`
- `presentations/optimization/src/make_optimizer_figures.py`
- `presentations/optimization/assets/generated/*.png`
- `presentations/regularization/.latexmkrc`
- `presentations/regularization/main.tex`
- `presentations/regularization/src/make_regularization_figures.py`
- `presentations/regularization/assets/generated/*.png`
- `presentations/cnns/.latexmkrc`
- `presentations/cnns/main.tex`
- `presentations/rnns/.latexmkrc`
- `presentations/rnns/main.tex`

### Modify

- `presentations/neural-nets/main.tex`
- `TOPICS.toml`
- `ROADMAP.md`

### Remove after references are gone

- `presentations/neural-nets/assets/ai-ml-dl.png`
- `presentations/neural-nets/assets/cnn.png`
- `presentations/neural-nets/assets/deep-nn.png`
- `presentations/neural-nets/assets/deepnn.svg`
- `presentations/neural-nets/assets/gd.png`
- `presentations/neural-nets/assets/linear-regression-nn.png`
- `presentations/neural-nets/assets/linear-regression.png`
- `presentations/neural-nets/assets/linreg.svg`
- `presentations/neural-nets/assets/logistic-regression.png`
- `presentations/neural-nets/assets/nn.svg`
- `presentations/neural-nets/assets/rnn.png`
- `presentations/neural-nets/assets/rnn.svg`
- `presentations/neural-nets/assets/simple-nn.png`
- `presentations/neural-nets/src/skorch_example.py`

## Deck Content

### `supervised-learning`

- Machine learning as data, model, objective, and optimization.
- Features, targets, samples, and predictions.
- Regression versus classification.
- Training, validation, and test sets.
- Data leakage.
- Linear regression.
- Mean-squared error.
- Logistic regression.
- Sigmoid and binary cross-entropy.
- Loss versus evaluation metric.
- Feature normalization.
- Links to optimization and generalization.

### `optimization`

- Objective functions and parameter space.
- Gradients and descent directions.
- Vanilla gradient descent.
- Learning-rate behavior.
- Conditioning and feature scaling.
- Full-batch, stochastic, and mini-batch gradients.
- Epochs, updates, and fair comparisons.
- Momentum.
- AdaGrad as motivation for adaptive scaling.
- RMSProp.
- Adam first and second moments.
- Adam bias correction.
- Same-objective optimizer comparison.

### `neural-nets`

- A neuron as affine map plus activation.
- Dense layers and tensor dimensions.
- Forward propagation.
- Nonlinear activation functions.
- Regression, binary, and multiclass outputs.
- Computation graphs.
- Chain rule.
- Backpropagation.
- Forward -> loss -> backward -> optimizer.
- Initialization intuition.
- Signposts to regularization, CNNs, GNNs, sequence models, and generative models.

### `regularization`

- Generalization error.
- Underfitting and overfitting.
- Bias and variance definitions.
- Bias-variance decomposition.
- Model capacity and the bias-variance trade-off.
- Regularized objectives.
- L2 geometry and shrinkage.
- Gaussian zero-mean prior and MAP estimation.
- L1 geometry and sparsity.
- Laplace prior.
- Regularization strength `lambda`.
- Early stopping.
- Dropout mechanism.
- Inverted-dropout scaling.
- Dropout at training versus inference.
- Data augmentation.

Required wording:

- **Bias:** Systematic error that remains across possible training datasets.
- **High bias:** The model's average prediction is wrong.
- **Variance:** Sensitivity to which training samples were observed.
- **High variance:** The fitted prediction changes strongly when the dataset changes.
- **Irreducible noise:** Randomness that the model cannot remove.

Required equation under squared-error assumptions:

\[
\mathbb{E}\!\left[(Y-\hat f(X))^2\right]
= \operatorname{Bias}[\hat f(X)]^2
+ \operatorname{Var}[\hat f(X)]
+ \sigma^2.
\]

Required regularization explanation:

- Increasing regularization usually increases bias.
- Increasing regularization usually reduces variance.
- Select `lambda` with validation data.
- Do not select `lambda` with test data.

Required dropout sequence:

- **Slide 1 — “Dropout samples a smaller network”:** TikZ network with Bernoulli-masked units and connections.
- **Slide 2 — “Inverted dropout preserves the expected activation”:** show
  \(\tilde{\mathbf h}=\frac{\mathbf m}{1-p}\odot\mathbf h\),
  with \(m_i\sim\operatorname{Bernoulli}(1-p)\).
- **Slide 3 — “Dropout is disabled at inference”:** training and inference paths with identical expected scale.
- Keep Monte Carlo dropout outside the introductory core.

### `cnns`

- Spatial structure versus dense connectivity.
- Local receptive fields.
- Shared learned kernels.
- One convolution calculation.
- Feature maps.
- Input and output channels.
- Stride.
- Padding and output size.
- Pooling and strided convolution.
- Parameter-count comparison.
- Full CNN pipeline.
- Classification versus spatial field output.

### `rnns`

- Sequence inputs and shared parameters across time.
- Hidden state as a learned summary of the past.
- One recurrent update and one output update.
- Unrolling through time.
- Many-to-one and many-to-many outputs.
- Backpropagation through time.
- Vanishing and exploding gradients.
- Gradient clipping as protection against exploding gradients.
- LSTM gates and memory path.
- GRU as a simpler gated alternative.
- Signpost to attention and transformers.

---

### Task 1: Refocus the Dense Neural-Network Deck

**Files:**

- Modify: `presentations/neural-nets/main.tex`
- Reference temporarily: `presentations/neural-nets/assets/`

**Interfaces:**

- Consumes: optimization, generalization, and architecture concepts only through signpost frames.
- Produces: the owner deck for dense neural-network fundamentals and backpropagation.

- [ ] **Step 1: Record the baseline**

Run:

```bash
cd presentations/neural-nets && latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
```

Expected: `.build/main.pdf` builds successfully.

- [ ] **Step 2: Rewrite `main.tex` as the dense-network deck**

Use the `neural-nets` sequence in “Deck Content”.

Requirements:

- Remove ML taxonomy, linear/logistic regression, optimization, CNN, and RNN teaching frames.
- Keep only short signposts to owning decks.
- Replace raster network figures with TikZ.
- Give each network or computation-graph figure its own frame.
- Use short bullets and equations.
- Add no programming examples.

- [ ] **Step 3: Build and render the deck**

Run:

```bash
cd presentations/neural-nets && latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
pdftoppm -png -r 144 .build/main.pdf /tmp/neural-nets
```

Expected: build succeeds; rendered slides have no clipping, overlap, or unreadable labels.

- [ ] **Step 4: Commit the focused deck**

```bash
git add presentations/neural-nets/main.tex
git commit -m "refactor(neural-nets): focus the dense network lecture" -m "- keep forward propagation, backpropagation, and activation fundamentals" -m "- replace unrelated topic coverage with owning-deck signposts"
```

### Task 2: Create the Supervised-Learning Deck

**Files:**

- Create: `presentations/supervised-learning/.latexmkrc`
- Create: `presentations/supervised-learning/main.tex`
- Create: `presentations/supervised-learning/src/make_supervised_figures.py`
- Create: `presentations/supervised-learning/assets/generated/linear-regression.png`
- Create: `presentations/supervised-learning/assets/generated/logistic-regression.png`
- Create: `presentations/supervised-learning/assets/generated/normalization.png`

**Interfaces:**

- Consumes: no earlier deck.
- Produces: notation, losses, splits, and scaling assumed by later decks.

- [ ] **Step 1: Create the deck skeleton**

Copy the minimal `.latexmkrc` convention used by existing decks.

Use `\documentclass{ntuabeamer}` and `\graphicspath{{assets/}{assets/generated/}}`.

- [ ] **Step 2: Create deterministic supervised-learning figures**

Implement:

```python
def main() -> None:
    """Write the supervised-learning figures under assets/generated/."""
```

Requirements:

- Use NumPy and Matplotlib only.
- Use seed `20261029`.
- Derive the output directory from `Path(__file__)`.
- Generate one image per figure slide.
- Assert that every expected file exists and is non-empty.
- Do not expose source code in the slides.

- [ ] **Step 3: Run the generator**

Run:

```bash
uv run --with numpy --with matplotlib presentations/supervised-learning/src/make_supervised_figures.py
```

Expected: the three named PNG files are created.

- [ ] **Step 4: Write the supervised-learning slides**

Follow the approved deck content.

Visual-only frames:

- Regression fit.
- Logistic probability and decision boundary.
- Train/validation/test separation.
- Data leakage.
- Unnormalized versus normalized feature geometry.

- [ ] **Step 5: Build and visually inspect**

Run:

```bash
cd presentations/supervised-learning && latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
pdftoppm -png -r 144 .build/main.pdf /tmp/supervised-learning
```

Expected: all equations and figure labels are readable.

- [ ] **Step 6: Commit the deck**

```bash
git add presentations/supervised-learning
git commit -m "feat(supervised-learning): add the fundamentals lecture" -m "- separate regression, classification, data splits, and losses" -m "- add deterministic visual explanations without code examples"
```

### Task 3: Create the Optimization Deck Through Adam

**Files:**

- Create: `presentations/optimization/.latexmkrc`
- Create: `presentations/optimization/main.tex`
- Create: `presentations/optimization/src/make_optimizer_figures.py`
- Create: `presentations/optimization/assets/generated/objective-geometry.png`
- Create: `presentations/optimization/assets/generated/learning-rates.png`
- Create: `presentations/optimization/assets/generated/conditioning.png`
- Create: `presentations/optimization/assets/generated/gd-vs-sgd.png`
- Create: `presentations/optimization/assets/generated/momentum.png`
- Create: `presentations/optimization/assets/generated/rmsprop.png`
- Create: `presentations/optimization/assets/generated/optimizer-comparison.png`

**Interfaces:**

- Consumes: objectives, gradients, and feature scaling from `supervised-learning`.
- Produces: optimizer concepts referenced by `neural-nets`.

- [ ] **Step 1: Create the deck skeleton**

Use the existing class and build layout.

- [ ] **Step 2: Implement the deterministic optimizer experiment**

Implement:

```python
def main() -> None:
    """Run cheap optimizer experiments and write all approved figures."""
```

Requirements:

- Use one two-dimensional ill-conditioned quadratic objective.
- Use the same initial point and axis limits for comparisons.
- Use seed `20261029` for stochastic gradients.
- Compare equal gradient-evaluation budgets where methods are compared.
- Include vanilla GD, SGD, momentum, RMSProp, and Adam.
- Use AdaGrad only as a conceptual bridge in the slides.
- Assert that every method begins at the same point.
- Assert that every final objective is below its initial objective.
- Assert that every expected file exists and is non-empty.

- [ ] **Step 3: Run the optimizer generator**

Run:

```bash
uv run --with numpy --with matplotlib presentations/optimization/src/make_optimizer_figures.py
```

Expected: all seven named PNG files are created without assertion failures.

- [ ] **Step 4: Write the optimizer slides**

Use the approved optimization sequence.

Use TikZ for:

- Full-batch, stochastic, and mini-batch sampling.
- Momentum's moving-average update.
- Adam's first moment, second moment, bias correction, and update flow.
- Optimizer family tree.

Use generated figures for numerical trajectories.

- [ ] **Step 5: Build and visually inspect**

Run:

```bash
cd presentations/optimization && latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
pdftoppm -png -r 144 .build/main.pdf /tmp/optimization
```

Expected: trajectories remain distinguishable when projected and printed.

- [ ] **Step 6: Commit the deck**

```bash
git add presentations/optimization
git commit -m "feat(optimization): add gradient optimizers through Adam" -m "- compare learning rate, batching, momentum, RMSProp, and Adam" -m "- use reproducible trajectories on a shared objective"
```

### Task 4: Create the Generalization and Regularization Deck

**Files:**

- Create: `presentations/regularization/.latexmkrc`
- Create: `presentations/regularization/main.tex`
- Create: `presentations/regularization/src/make_regularization_figures.py`
- Create: `presentations/regularization/assets/generated/underfit-overfit.png`
- Create: `presentations/regularization/assets/generated/high-bias.png`
- Create: `presentations/regularization/assets/generated/high-variance.png`
- Create: `presentations/regularization/assets/generated/bias-variance-tradeoff.png`
- Create: `presentations/regularization/assets/generated/learning-curves.png`
- Create: `presentations/regularization/assets/generated/regularized-objective.png`
- Create: `presentations/regularization/assets/generated/lambda-path.png`
- Create: `presentations/regularization/assets/generated/gaussian-priors.png`
- Create: `presentations/regularization/assets/generated/early-stopping.png`

**Interfaces:**

- Consumes: model fitting from `supervised-learning` and dense networks from `neural-nets`.
- Produces: generalization and regularization concepts used by `cnns`.

- [ ] **Step 1: Create the deck skeleton**

Use the existing class and build layout.

- [ ] **Step 2: Implement deterministic resampling and regularization visuals**

Implement:

```python
def main() -> None:
    """Write the bias-variance and regularization figures."""
```

Requirements:

- Use seed `20261029`.
- Show repeated fits across resampled datasets.
- `high-bias.png`: fitted models remain similar but their mean misses the true function.
- `high-variance.png`: fitted models change strongly across datasets.
- `bias-variance-tradeoff.png`: bias squared, variance, noise, and total expected error versus capacity.
- `regularized-objective.png`: blue data-loss contours, orange L2-penalty contours, black total contours, and markers for zero, MLE, and MAP on the same axes.
- `lambda-path.png`: show the optimum moving toward zero as `lambda` increases.
- `gaussian-priors.png`: compare weak and strong zero-mean priors.
- Assert that increasing `lambda` monotonically decreases the L2 norm of the quadratic-model optimum.
- Assert that the selected early-stopping epoch is the validation-loss minimum.
- Assert that every expected file exists and is non-empty.

- [ ] **Step 3: Run the regularization generator**

Run:

```bash
uv run --with numpy --with matplotlib presentations/regularization/src/make_regularization_figures.py
```

Expected: all nine named PNG files are created without assertion failures.

- [ ] **Step 4: Write the bias and variance sequence**

Use separate frames for:

- Bias definition and high-bias visual.
- Variance definition and high-variance visual.
- Irreducible noise.
- The squared-error decomposition.
- Capacity versus bias and variance.
- Regularization as a controlled bias-for-variance trade.

Use the exact short definitions in “Deck Content”.

- [ ] **Step 5: Write the L1, L2, prior, and early-stopping sequence**

Use separate figure frames for:

- The combined regularized-objective contours.
- The optimum path as `lambda` increases.
- Gaussian priors.
- Gaussian likelihood + Gaussian prior -> L2 MAP estimate.
- L1 diamond versus L2 circle.
- Laplace prior versus Gaussian prior.
- Early stopping.

State the assumptions behind the MAP interpretation.

- [ ] **Step 6: Write the three dropout slides**

Use TikZ and the required sequence in “Deck Content”.

Acceptance checks:

- Training uses a sampled Bernoulli mask.
- Inverted dropout divides retained activations by `1-p`.
- Inference disables dropout.
- Expected activation scale matches between training and inference.
- Monte Carlo dropout is not taught in the core deck.

- [ ] **Step 7: Add one data-augmentation visual**

Show one source sample and permissible transformations.

State that transformations must preserve the target semantics.

- [ ] **Step 8: Build and visually inspect**

Run:

```bash
cd presentations/regularization && latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
pdftoppm -png -r 144 .build/main.pdf /tmp/regularization
```

Expected: contour labels, prior curves, masks, and decomposition terms are readable.

- [ ] **Step 9: Commit the deck**

```bash
git add presentations/regularization
git commit -m "feat(regularization): add generalization and dropout lectures" -m "- explain bias, variance, priors, L1, L2, and early stopping" -m "- add three visual dropout slides with correct train and inference behavior"
```

### Task 5: Create the CNN Deck

**Files:**

- Create: `presentations/cnns/.latexmkrc`
- Create: `presentations/cnns/main.tex`

**Interfaces:**

- Consumes: dense networks, optimization, and generalization.
- Produces: the owner deck for convolutional neural networks.

- [ ] **Step 1: Create the deck skeleton**

Use TikZ for all CNN mechanics.

Add no Python generator unless a concrete numerical visual cannot be expressed clearly in TikZ.

- [ ] **Step 2: Write the CNN slide sequence**

Give each item its own visual frame:

- Dense versus local connectivity.
- Sliding kernel and local receptive field.
- One dot-product calculation.
- Completed feature map.
- Multiple input and output channels.
- Stride.
- Padding.
- Max pooling.
- Parameter sharing and parameter counts.
- Full CNN architecture.
- Classification output.
- Spatial field output.

Required corrections:

- State that ML libraries commonly compute cross-correlation while using the name convolution.
- Do not present pooling as mandatory.
- Do not present a fully connected head as mandatory.
- Do not present CNNs as classification-only.

- [ ] **Step 3: Build and visually inspect**

Run:

```bash
cd presentations/cnns && latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
pdftoppm -png -r 144 .build/main.pdf /tmp/cnns
```

Expected: grid cells, kernels, channels, and dimension labels are readable.

- [ ] **Step 4: Commit the deck**

```bash
git add presentations/cnns
git commit -m "feat(cnns): add the introductory convolution lecture" -m "- explain kernels, channels, padding, pooling, and parameter sharing" -m "- replace the generic raster diagram with focused TikZ visuals"
```

### Task 6: Create the Recurrent Sequence-Model Deck

**Files:**

- Create: `presentations/rnns/.latexmkrc`
- Create: `presentations/rnns/main.tex`

**Interfaces:**

- Consumes: dense networks, optimization, and generalization.
- Produces: the owner deck for recurrent sequence models.

- [ ] **Step 1: Create the deck skeleton**

Use TikZ for recurrence, time unrolling, gradient flow, and gate diagrams.

Add no programming examples or framework screenshots.

- [ ] **Step 2: Write the recurrent-model sequence**

Give each mechanism its own visual frame:

- Recurrent state update.
- Unrolled shared cell.
- Many-to-one and many-to-many mappings.
- Backpropagation through time.
- Vanishing and exploding gradients.
- Gradient clipping.
- LSTM memory and gates.
- GRU as a compact gated alternative.
- Attention and transformer signpost.

Required corrections:

- Do not imply that hidden state stores the complete past.
- Do not claim that LSTMs eliminate all long-range gradient problems.
- Distinguish sequence length from batch size and feature dimension.
- Keep transformers outside the recurrent-model core.

- [ ] **Step 3: Build and visually inspect**

Run:

```bash
cd presentations/rnns && latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
pdftoppm -png -r 144 .build/main.pdf /tmp/rnns
```

Expected: time indices, shared parameters, arrows, and gate labels are readable.

- [ ] **Step 4: Commit the deck**

```bash
git add presentations/rnns
git commit -m "feat(rnns): add the introductory sequence-model lecture" -m "- explain recurrent state, unrolling, gradient flow, LSTMs, and GRUs" -m "- use focused TikZ visuals without programming examples"
```

### Task 7: Update Ownership and Remove Superseded Material

**Files:**

- Modify: `TOPICS.toml`
- Modify: `ROADMAP.md`
- Remove: superseded files listed under “File Structure”.

**Interfaces:**

- Consumes: all six completed decks.
- Produces: one canonical owner for every lecture topic.

- [ ] **Step 1: Split topic ownership**

Define owners for:

- `supervised-basics` -> `presentations/supervised-learning`
- `gradient-optimization` -> `presentations/optimization`
- `dense-neural-networks` -> `presentations/neural-nets`
- `generalization-regularization` -> `presentations/regularization`
- `convolutional-neural-networks` -> `presentations/cnns`
- `recurrent-neural-networks` -> `presentations/rnns`

Keep GNNs and generative modeling with their existing owners.

Point the LSTM mechanical-system work to the sequence-model owner.

- [ ] **Step 2: Update the roadmap in place**

- Remove completed neural-network signpost work.
- Remove the obsolete `skorch_example.py` generator check.
- Keep the future LSTM/sequence-model work once, without duplicating it.
- Preserve unrelated roadmap items.

- [ ] **Step 3: Prove old assets are unused**

Run:

```bash
rg -n 'ai-ml-dl|cnn\.png|deep-nn|deepnn|gd\.png|linear-regression-nn|linear-regression\.png|linreg|logistic-regression\.png|nn\.svg|rnn\.png|rnn\.svg|simple-nn|skorch_example' presentations TOPICS.toml ROADMAP.md
```

Expected: no live source reference remains.

- [ ] **Step 4: Remove superseded assets and script**

Delete only the exact files listed under “Remove after references are gone”.

- [ ] **Step 5: Commit ownership and cleanup**

```bash
git add -A presentations/neural-nets TOPICS.toml ROADMAP.md
git commit -m "chore(lectures): register split topic ownership" -m "- assign one owning deck to each introductory topic" -m "- remove superseded raster assets and the unused code example"
```

### Task 8: Full Verification

**Files:**

- Verify: all presentation sources and generated assets.

**Interfaces:**

- Consumes: the complete lecture split.
- Produces: a buildable, visually reviewed course sequence.

- [ ] **Step 1: Regenerate every new numerical visual**

Run:

```bash
uv run --with numpy --with matplotlib presentations/supervised-learning/src/make_supervised_figures.py
uv run --with numpy --with matplotlib presentations/optimization/src/make_optimizer_figures.py
uv run --with numpy --with matplotlib presentations/regularization/src/make_regularization_figures.py
```

Expected: all scripts exit successfully and all internal assertions pass.

- [ ] **Step 2: Build every repository target**

Run:

```bash
tools/build-all
```

Expected: every target reports `ok`.

- [ ] **Step 3: Render the six affected decks**

Run `pdftoppm -png -r 144` on:

- `presentations/supervised-learning/.build/main.pdf`
- `presentations/optimization/.build/main.pdf`
- `presentations/neural-nets/.build/main.pdf`
- `presentations/regularization/.build/main.pdf`
- `presentations/cnns/.build/main.pdf`
- `presentations/rnns/.build/main.pdf`

Expected:

- No clipping or overlap.
- No unreadable labels.
- No broken glyphs.
- No figure shares a frame with another distinct figure.
- No prose paragraph appears on a slide.
- Headers, footers, and page numbers match existing decks.

- [ ] **Step 4: Check repository hygiene**

Run:

```bash
git diff --check
git status --short
git ls-files | rg '/\.build/|\.pdf$|\.gif$|__pycache__'
```

Expected:

- `git diff --check` is clean.
- Only intended lecture sources and generated figures are changed.
- No built presentation, GIF, cache, or archive file is tracked.

- [ ] **Step 5: Final content audit**

Confirm:

- The six decks follow the approved course order.
- Each topic has one owner.
- Cross-references replace duplicated explanations.
- Test data is never used for tuning.
- Bias and variance are defined before the trade-off is discussed.
- The regularization contour slide shows data loss, penalty, and total on the same axes.
- Dropout training and inference behavior is correct.
- Adam includes bias correction.
- CNN pooling and dense heads are presented as optional design choices.
