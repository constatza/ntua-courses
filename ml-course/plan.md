# Machine-Learning Lecture Redesign Plan

## Goal

Build a coherent introductory university course that:

- follows the existing order in this repository;
- covers the conceptual content of the archived `chapter*.pdf` lectures;
- removes duplicated ownership between decks;
- keeps one notation and one visual language throughout;
- explains ideas intuitively and defines terms well, keeping derivations in appendices;
- uses equations and figures as the primary explanatory devices;
- contains no framework or programming-code slides.

The archive is a coverage checklist, not an ordering template. Its Python,
PyTorch, TensorBoard, and environment-setup material will not become lecture
content.

## Course-wide design rules

### Frame composition

- One principal idea per frame.
- Prefer a figure, diagram, or equation over descriptive prose.
- Give each substantial figure its own frame.
- Use short sentences when they explain *why* a result matters.
- Use bullets only for parallel ideas, comparisons, assumptions, or steps.
- Keep bullet lists concise; avoid both paragraph walls and bullet walls.
- Put secondary qualifications in a short footer, note, or later caveat frame.
- Do not include code listings or framework-specific APIs.

### Introductory level

The course is introductory. Intuition and good definitions come first; depth is optional.
"Simple" does not mean "minimal": a slide that shows a formula or a one-line bullet and
does not say what it means or why it matters teaches nothing.

- Explain, do not just state. A topic gets as many frames as it needs to be understood:
  motivation (what goes wrong without it), plain-language definition, a picture or small
  worked example with numbers, and the conclusion. One frame per topic is usually too few.
- Keep text per frame small but continue a topic across consecutive frames: each frame adds
  one step of the explanation (problem, idea, picture, example, conclusion). Little text per
  frame is a reason to add frames, not to drop explanation.
- Every frame ends with its takeaway in one plain sentence ("so ...", "this means ...").
  A frame the audience cannot summarize in a sentence is unfinished.
- Short sentences of real explanation are welcome; paragraphs of jargon and bare symbol
  strings are not. Prefer "what it is", "why it is needed", "how it works" over notation.
- Use elementary language. No probability jargon (conditioning, expectation over a mask,
  measurability) when a numeric example says the same: explain inverted dropout as
  "divide the kept activations by 1-p, because on average only a fraction 1-p survive,
  and the division restores the average size".

- Define every symbol in words on first use and say why it matters (for example, the
  Hessian is the curvature: how quickly the slope changes).
- Show the picture and the idea first; the formula is the compact summary, not the argument.
- Keep proofs and derivations out of the main flow: a one-line "why" at most. Derivations,
  convergence rates, variants and edge cases go to the deck's appendix.
- State every result with its scope ("for a quadratic", "near a minimum", "for linear
  models"). If it cannot be stated honestly in one line, drop it.
- An example must support its caption. Run it first; if it contradicts the claim, change the
  example or the claim, never ship it. No optimizer, model or method is shown as best unless
  the shown experiment supports exactly that.
- Core terms (bias, variance, gradient, learning rate, epoch, Hessian) get their own frame.
- Make topic changes obvious. Every deck opens (after the title) with a roadmap frame
  listing its parts in plain words (`\tableofcontents` over `\section{...}` is fine), and
  every part starts with `\section{...}` plus `\sectiondivider{...}` and ends with a
  one-sentence recap of what was learned and what comes next. A deck whose parts do not
  depend on each other is split into separate decks instead (update `TOPICS.toml`).
- Every deck ends with `\appendix` followed by `\sectiondivider{Appendix: optional depth}`.
  Aim for roughly 20--28 main frames.

### Mathematical conventions

- Samples: \(\mathcal D=\{(\mathbf x_i,y_i)\}_{i=1}^n\).
- Feature vectors are columns: \(\mathbf x\in\mathbb R^d\).
- Parameters: \(\boldsymbol\theta\); dense-layer weights: \(\mathbf W^{(\ell)}\).
- Predictions: \(\hat y=f_{\boldsymbol\theta}(\mathbf x)\).
- Per-sample loss: \(\ell(\hat y,y)\); empirical objective:
  \(L(\boldsymbol\theta)=n^{-1}\sum_i\ell_i(\boldsymbol\theta)\).
- Iterations use superscripts: \(\boldsymbol\theta^{(t)}\).
- Gradients are column vectors.
- State the convolution convention explicitly: slides use cross-correlation, as
  deep-learning libraries do.
- State optimizer recurrences before showing trajectories. Do not mix momentum
  conventions within the course.
- Every stochastic statement identifies its source of randomness.

### Visual conventions

- Reuse the present NTUA Beamer theme, typography, spacing, and color palette.
- Use the same colors for data, prediction, error, parameter, and gradient across decks.
- Use TikZ/PGFPlots for mathematical diagrams and plots when practical.
- Use a public image only when it teaches more clearly than a schematic; record its source.
- Check every TikZ arrow at rendered-slide scale, especially recurrent loops.
- Axes must show names, units or normalized scale, and the relevant optimum or boundary.
- Comparative plots use identical axes and budgets unless the difference is explicitly taught.

### Quality gates for every deck

1. Check statements and derivations independently from slide layout.
2. Check notation against this plan and earlier decks.
3. Compile with the deck-local build command.
4. Render every PDF page to images and inspect it visually.
5. Check for clipping, tiny text, collisions, degenerate arrows, and misleading axes.
6. Run `tools/build-all` after each implementation batch.
7. Keep generated PDFs and archive conversions untracked.

## Canonical order and topic ownership

1. `supervised-learning` — problem formulation and linear baselines.
2. `optimization` — first-order algorithms through Adam.
3. `neural-nets` — dense networks, backpropagation, and approximation.
4. `generalization` — bias, variance, penalties, early stopping, ensembles, dropout.
5. `geometric-deep-learning` — symmetry, invariance, and equivariance.
6. `cnns` — convolutional architectures in detail.
7. `rnns` — recurrent architectures in detail.
8. `transformers` — attention-based sequence models.
9. `gnns` — message passing on graphs.
10. `pinns-neural-operators` — learning solutions and solution operators for PDEs.
11. `diffusion-models` — score/noise learning and reverse diffusion.
12. `flow-matching` — continuous transport and conditional flow matching.

`pca-rom` and `pdf-distances` remain independent specialist decks. There will be
no generic “advanced architectures” survey: the geometric-deep-learning deck is
the shared conceptual bridge, while each mainstream architecture receives enough
space for motivation, intuition, mathematics, and limitations.

## Archive coverage map

| Archived material | Owning deck |
|---|---|
| ML definition, data, learning paradigms, tasks | `supervised-learning` |
| Linear and logistic models, splits, metrics | `supervised-learning` |
| Normal equations | `supervised-learning` |
| GD, SGD, mini-batches, momentum, RMSProp, Adam | `optimization` |
| Dense networks, forward propagation, activations | `neural-nets` |
| Computation graphs and backpropagation | `neural-nets` |
| Sine approximation and derivative-informed loss | `neural-nets` |
| Under/overfitting, early stopping, L1/L2 | `generalization` |
| Bias–variance, ensembles, dropout | `generalization` |
| Data augmentation | `geometric-deep-learning` |
| CNNs, filters, pooling, channels, padding | `cnns` |
| RNNs and sequence recurrence | `rnns` |
| Graph neural networks | `gnns` |
| Physics-inspired learning | `pinns-neural-operators` |
| Framework/code/setup demonstrations | deliberately excluded |

## Detailed presentation content

### 1. Supervised learning

**Purpose:** establish the prediction problem, evaluation discipline, and the
linear models used as baselines throughout the course.

1. **What is learned?** — data, hypothesis, loss, and prediction in one visual loop.
2. **Dataset notation** — samples, features, targets, dimensions, and the design matrix.
3. **Where supervised learning fits** — compact map of supervised, unsupervised,
   self-supervised, and reinforcement learning; only distinctions needed later.
4. **Regression or classification?** — paired visual examples and output spaces.
5. **Why start with simple models?** — interpretability, speed, and a diagnostic baseline.
6. **Generalization is the goal** — training performance versus unseen-data performance.
7. **Train, validation, test** — three non-overlapping roles shown as a data-flow diagram.
8. **Leakage** — one concrete normalization-before-split failure and its consequence.
9. **Affine linear prediction** — \(\hat y=\mathbf w^\top\mathbf x+b\), geometry included.
10. **Residuals** — vertical residuals on a scatter plot; sign and magnitude made explicit.
11. **MSE and RMSE** — equations, units, and sensitivity to large errors.
12. **Why squared error?** — Gaussian observation model leading to MSE likelihood.
13. **Matrix form** — augmented design matrix and \(\|\mathbf X\boldsymbol\beta-\mathbf y\|_2^2\).
14. **Normal equations** — gradient-to-zero derivation in a few aligned steps.
15. **Solving safely** — rank assumptions; solve/QR rather than explicit inversion.
16. **Binary probabilities** — why a raw affine output is not a probability.
17. **Logistic regression** — sigmoid, log-odds, and the linear decision boundary.
18. **Binary cross-entropy** — Bernoulli likelihood derivation and confident-error penalty.
19. **Thresholds encode costs** — move a threshold on overlapping class scores.
20. **Confusion matrix and metrics** — accuracy, precision, recall, specificity; no metric dump.
21. **Feature scaling** — same data before/after standardization; training statistics only.
22. **Scaling changes optimization geometry** — circular versus elongated contours.
23. **Summary** — formulate, split, fit, validate, test; bridge to optimization.

**Explicit boundary:** this deck defines objectives but does not teach iterative optimizers,
dense networks, or regularization.

### 2. Optimization: gradient descent through Adam

**Purpose:** explain how first-order methods move, why their behavior differs, and
how to compare them correctly, including on nonconvex landscapes.

**Flow:** landscape, gradient descent, step size and curvature, stochastic gradients,
beyond a single fixed step, comparison, bridge. Each section answers the question the
previous one raised; connections are made at section boundaries only.

1. **Landscape** — training as optimization; convexity and its consequence;
   nonconvex landscapes with several minima and saddles.
2. **Gradient descent** — directional derivative, steepest descent (Cauchy–Schwarz),
   the update, and the local linear model.
3. **Step size and curvature** — learning-rate behavior with the \(2/\lambda_{\max}\)
   limit; eigenvalue stability; the contraction factor and best constant step
   \(\rho^\star=(\kappa-1)/(\kappa+1)\); iteration count \(\approx(\kappa/2)\ln(1/\varepsilon)\);
   the same step on \(\kappa=1\) and \(\kappa=18\); standardization as removal of
   scale-induced ill-conditioning; the descent lemma and the \(1/T\) convex rate.
4. **Stochastic gradients** — batch estimates; unbiasedness and variance \(\sigma^2/b\);
   mini-batch noise floor on a finite-sum problem whose per-sample gradients do not
   vanish at the minimizer; epochs versus updates; schedules (decay with the
   Robbins–Monro condition, warm-up, cosine).
5. **Beyond a single fixed step** — Newton's method as the ideal curvature fix and why
   it is impractical (L-BFGS in one sentence); momentum recurrence
   \(\mathbf v_t=\beta\mathbf v_{t-1}+\mathbf g_t,\ \boldsymbol\theta_t=\boldsymbol\theta_{t-1}-\eta\mathbf v_t\);
   Nesterov; the \(\kappa\to\sqrt\kappa\) rate with Polyak's optimal \((\eta,\beta)\);
   AdaGrad; RMSProp (constant \(\eta\) leaves a neighborhood of size \(\propto\eta\));
   coordinate-wise scaling helps axis-aligned curvature only; Adam with moments,
   bias correction and defaults; gradient clipping and learning-rate search.
6. **Comparison** — declared protocol (same budget, tuned constant \(\eta\), momentum
   \(\beta\) searched, tail-gap scoring) with a generated results table; paths;
   objective gap and gradient norm; Rosenbrock valley; basin outcomes on a four-minimum
   landscape; saddle escape; selection table.
7. **Bridge** — gradients of large models come from backpropagation; start, step rule
   and schedule decide where nonconvex training ends; generalization is deferred.

**Correctness requirements (all enforced by `src/test_make_optimizer_figures.py`):**

- First three steps of momentum, AdaGrad, RMSProp, Adam, and Nesterov match hand values.
- Adam's first step equals the learning rate (bias correction).
- Gradients of every objective match centered differences.
- The mini-batch gradient is unbiased with variance \(\sigma^2/b\) (Monte Carlo).
- Measured contraction matches \((\kappa-1)/(\kappa+1)\), \(\sqrt\beta\) for tuned
  momentum, and \(1-1/\sqrt\kappa\) (within tolerance) for Nesterov; GD is stable
  exactly below \(2/\lambda_{\max}\); one Newton step solves a quadratic.
- The protocol table is generated, never typed; learning rates are tuned by search, not
  chosen per method by hand.
- Basin shares across optimizers agree, and saddle-escape order is GD > momentum > Adam.
- Fixed-step methods that do not reach the minimizer are labelled accurately.

### 3. Dense neural networks and backpropagation

**Purpose:** progress from an artificial neuron to multilayer approximation, with
enough calculus to derive backpropagation rather than present it as a recipe.

1. **Why leave linear models?** — one nonlinear decision boundary that a line cannot fit.
2. **Artificial neuron** — affine map followed by activation.
3. **Dense layer with shapes** — \(\mathbf z^{(\ell)}=\mathbf W^{(\ell)}\mathbf a^{(\ell-1)}+\mathbf b^{(\ell)}\).
4. **Composition creates depth** — network as nested functions.
5. **Why nonlinearity is necessary** — composition of affine maps collapses to one affine map.
6. **Activation functions** — ReLU, sigmoid, tanh plotted on common axes where meaningful.
7. **Derivative behavior** — saturation, dead regions, and gradient flow.
8. **Forward pass** — dimensions and data flow through a two-layer network.
9. **Worked forward example** — small numeric network, all intermediate values visible.
10. **Output layers match tasks** — linear, sigmoid, and softmax outputs with their losses.
11. **Computation graph** — local operations and dependencies.
12. **Chain rule as path multiplication** — scalar graph first.
13. **Jacobians, JVPs, and VJPs** — only the distinction needed for reverse mode.
14. **Why reverse mode fits scalar loss** — many parameters, one output.
15. **Backpropagation variables** — define \(\boldsymbol\delta^{(\ell)}=\partial L/\partial\mathbf z^{(\ell)}\).
16. **Backward recurrence** — derive
    \(\boldsymbol\delta^{(\ell)}=(\mathbf W^{(\ell+1)})^\top
    \boldsymbol\delta^{(\ell+1)}\odot\sigma'(\mathbf z^{(\ell)})\).
17. **Weight gradient** — outer product
    \(\partial L/\partial\mathbf W^{(\ell)}=\boldsymbol\delta^{(\ell)}(\mathbf a^{(\ell-1)})^\top\).
18. **Bias and activation gradients** — complete the local derivatives.
19. **Worked backward example** — reuse the forward example and compute every gradient.
20. **Gradient check** — central finite differences and expected truncation/roundoff behavior.
21. **Training loop as a mathematical cycle** — forward, loss, reverse, update; no code.
22. **Initialization and signal scale** — Xavier/He variance intuition.
23. **Features become hierarchical** — schematic, not an unsupported claim of semantics.
24. **Universal Approximation Theorem** — formal compact-domain statement with assumptions.
25. **What the theorem does not say** — no finite-width recipe, trainability, efficiency,
    or generalization guarantee.
26. **Proof intuition** — localized units or ridge functions combine to approximate a target.
27. **Sine approximation** — model capacity and training result on a controlled example.
28. **Sobolev/derivative-informed training** — function-plus-derivative loss and why it helps.
29. **Summary** — representation, differentiation, optimization, and approximation.

**Explicit boundary:** linear/logistic regression stays in `supervised-learning`;
optimizers stay in `optimization`; generalization techniques stay in `generalization`.

### 4. Generalization: bias, variance, and how to control them

**Purpose:** explain why low training error is insufficient, define bias and variance
in plain words, and organize the main tools for controlling generalization. Splits,
cross-validation and the train-versus-test picture are owned by `supervised-learning`
and only cross-referenced here. Augmentation is owned by `geometric-deep-learning`.

**Main flow (about 22 frames):**

1. **Error on new data** — what generalization means; cross-reference the splits.
2. **Underfit, good fit, overfit** — three fits to one dataset.
3. **Capacity curve** — training and validation error versus capacity.
4. **Bias** — the typical error of the fitted model, in plain words; why it matters.
5. **Variance** — how much the fit changes between training sets; why it matters.
6. **Bias/variance targets** — 2x2 picture of fitted weights over many training sets.
7. **Same idea in prediction space** — many fitted curves against the truth.
8. **Which do we prefer?** — rule of thumb: when in doubt, low variance and some bias.
9. **Diagnosis table** — high training error means bias; a large train-test gap means variance.
10. **Regularized objective** — data fit plus \(\lambda\Omega(\boldsymbol\theta)\); \(\lambda\) chosen on validation.
11. **L2** — contours and the shifted minimum; do not penalize the intercept.
12. **L1 versus L2 geometry** — exact zeros.
13. **Regularization path** — coefficient magnitude versus \(\lambda\).
14. **Early stopping** — validation curve and saved iterate.
15. **Ensembles and bagging** — averaging reduces variance.
16. **Dropout** — a new mask per training pass; full network at inference.
17. **Match the remedy to the failure** — summary.

**Appendix:** bias-variance decomposition and cross-term argument, noise term,
L2 as MAP and L1 as Laplace prior, prior-scale figure, averaging-variance formula,
inverted-dropout expectation, Monte Carlo dropout.

### 5. Geometric deep learning

**Purpose:** give a common language for architectures that respect structure and symmetry.

1. **Data live on domains** — grids, sequences, sets, graphs, manifolds.
2. **Transformations act on inputs** — group action \(g\cdot x\).
3. **Invariant maps** — \(f(g\cdot x)=f(x)\), with classification example.
4. **Equivariant maps** — \(f(g\cdot x)=g\cdot f(x)\), with segmentation example.
5. **Equivariance composes** — short formal proof for stacked equivariant layers.
6. **Three recurring biases** — locality, parameter sharing, and symmetry.
7. **Translations on a grid** — image and shifted image.
8. **Convolution is translation equivariant** — index-change proof.
9. **Where exact equivariance breaks** — boundaries, stride, discretization, padding.
10. **Pooling and invariance** — local stability versus loss of precise location.
11. **Sets have no canonical order** — permutation action and Deep Sets form.
12. **Graphs admit relabeling** — adjacency/features under a permutation matrix.
13. **Message passing is permutation equivariant** — aggregation proof sketch.
14. **Self-attention without positions** — permutation equivariance proof by matrix form.
15. **Position encodings add order** — why sequence models must break that symmetry.
16. **Recurrence respects causality, not permutation** — order is structural information.
17. **Augmentation learns approximate robustness** — finite transformed samples.
18. **Architectures encode exact structure** — within their mathematical assumptions.
19. **The target decides invariant or equivariant** — classification versus dense prediction.
20. **Wrong symmetries cause errors** — reflection, orientation, and boundary counterexamples.
21. **Architecture map** — CNN, RNN, Transformer, and GNN by domain and symmetry.
22. **Summary** — structure narrows the hypothesis class and improves data efficiency.

### 6. Convolutional neural networks

**Purpose:** develop CNN mechanics, geometry, and uses without treating convolution
as a mysterious feature detector.

1. **Why images challenge dense layers** — parameter count and ignored spatial locality.
2. **Local receptive fields** — compare dense and local connectivity.
3. **Cross-correlation convention** — 2-D index equation and kernel orientation.
4. **One verified calculation** — small image, kernel, and output with arithmetic.
5. **Sliding creates a feature map** — same weights at every location.
6. **Multiple input channels** — channel summation and tensor dimensions.
7. **Multiple output channels** — a learned filter bank.
8. **Parameter sharing** — parameter-count comparison with a dense alternative.
9. **Translation equivariance in practice** — shifted input and shifted activation.
10. **Nonlinearity builds hierarchy** — measured claim: receptive fields and composition grow.
11. **Receptive-field growth** — derive effective receptive field across layers.
12. **Stride** — spatial subsampling and its information cost.
13. **Padding** — valid, same, and boundary assumptions shown side by side.
14. **Dilation** — larger support without additional kernel parameters.
15. **Output-size formula** — kernel, padding, stride, and dilation in one equation.
16. **Pooling** — max/average behavior and the invariance-resolution trade-off.
17. **Strided convolution versus pooling** — learned versus fixed downsampling.
18. **Parameter accounting** — one realistic layer, including bias.
19. **What early filters respond to** — edges/textures shown as responses, not anthropomorphism.
20. **Classification architecture** — backbone, global pooling, head.
21. **Spatial prediction** — why segmentation retains or restores resolution.
22. **Encoder–decoder and skip connections** — U-Net-style diagram.
23. **Applications beyond photographs** — fields, spectra, and volumetric data.
24. **Limitations** — boundaries, aliasing, long-range context, and compute.
25. **Design checklist** — tensor shape, receptive field, symmetry, and task output.
26. **Summary** — local shared operators provide an image-specific inductive bias.

### 7. Recurrent neural networks

**Purpose:** explain recurrence, gradient propagation through time, and gated memory.

1. **Why sequences differ** — variable length, order, causality, and shared dynamics.
2. **Sequence tensor notation** — time, feature, batch axes.
3. **Vanilla recurrence** — \(\mathbf h_t=\phi(\mathbf W_x\mathbf x_t+\mathbf W_h\mathbf h_{t-1}+\mathbf b)\).
4. **The recursive cell** — redraw with a nondegenerate self-loop and clear state flow.
5. **Unrolling in time** — shared parameters made explicit.
6. **Input/output patterns** — many-to-one, one-to-many, and many-to-many.
7. **Hidden state as a bottleneck** — information must travel through repeated transitions.
8. **Backpropagation through time** — shared-parameter gradient as a sum over uses.
9. **Long products of Jacobians** — formal source of vanishing/exploding gradients.
10. **Spectral intuition** — repeated contraction or expansion.
11. **Gradient clipping** — what it controls and what it does not solve.
12. **Truncated BPTT** — compute/memory trade-off and deliberate gradient truncation.
13. **LSTM motivation** — an additive state path for longer credit assignment.
14. **LSTM gates** — forget, input, candidate, and output equations.
15. **Cell state versus hidden state** — distinct roles in a single diagram.
16. **Why the additive path helps** — derivative through the cell-state update.
17. **LSTM caveat** — gates mitigate; they do not guarantee unlimited memory.
18. **GRU** — one consistent convention and a compact comparison with LSTM.
19. **Bidirectional recurrence** — offline context and the causal-use limitation.
20. **Autoregressive decoding** — previous output as next input; teacher forcing mismatch.
21. **Sequential computation cost** — why time steps resist full parallelization.
22. **Where recurrence remains useful** — streaming, stateful control, compact models.
23. **Summary and attention bridge** — direct content-dependent access replaces one bottleneck.

### 8. Transformers

**Purpose:** derive attention and show how Transformer blocks model ordered data.

1. **Why attention?** — long paths and sequential computation in recurrence.
2. **Tokens and embeddings** — input matrix and dimensions.
3. **Queries, keys, values** — retrieval analogy tied to exact matrix definitions.
4. **Scaled dot-product attention** — full equation and row-wise softmax.
5. **Why divide by \(\sqrt{d_k}\)?** — dot-product variance argument.
6. **Small numerical example** — attention weights and weighted value combination.
7. **Self-attention shape flow** — \(\mathbf X\to\mathbf Q,\mathbf K,\mathbf V\to\mathbf Y\).
8. **Multi-head attention** — independent subspaces, concatenation, output projection.
9. **Attention alone has no order** — permutation-equivariance result from the geometry deck.
10. **Position information** — sinusoidal and learned encodings; no catalogue of variants.
11. **Causal mask** — triangular visibility matrix.
12. **Residual paths** — identity information and gradient flow.
13. **Layer normalization** — what is normalized and why pre-norm is common.
14. **Encoder block** — attention plus position-wise feed-forward layer.
15. **Decoder block** — masked self-attention and optional cross-attention.
16. **Cross-attention** — queries from one stream, keys/values from another.
17. **Language-model factorization** — \(p(x_{1:T})=\prod_t p(x_t\mid x_{<t})\).
18. **Training versus generation** — parallel teacher-forced training, sequential sampling.
19. **Quadratic attention cost** — memory/compute in sequence length.
20. **Other modalities** — patches, sets, and multimodal tokens.
21. **Interpretability caution** — attention weights are not automatically explanations.
22. **Failure modes** — context limits, data/computation, and positional extrapolation.
23. **Summary** — content-based interaction plus explicit structural information.

### 9. Graph neural networks

**Purpose:** derive permutation-equivariant message passing and explain its limits.

1. **Why graphs?** — irregular neighborhoods and relational data.
2. **Notation** — \(G=(V,E)\), node, edge, and graph features.
3. **Prediction levels** — node, edge, and graph tasks.
4. **Node labels are arbitrary** — same graph under two indexings.
5. **Required symmetry** — node outputs equivariant, graph outputs invariant.
6. **Message-passing template** — message, aggregate, update.
7. **Symmetric aggregation** — sum, mean, and max; ordering cannot matter.
8. **Permutation-equivariance proof** — relabel inputs and track one layer.
9. **Receptive fields** — \(k\) layers communicate across \(k\) hops.
10. **GCN layer** — normalized adjacency equation and self-loops.
11. **Why normalize?** — degree scale and stable aggregation.
12. **GraphSAGE** — neighborhood aggregation and inductive use.
13. **Graph attention** — learned neighbor weights and normalization scope.
14. **Edge features** — messages depend on relations, not only endpoint nodes.
15. **Global context** — graph-level state or virtual node.
16. **Invariant readout** — pool node states for graph prediction.
17. **Batching graphs** — disconnected union as a conceptual picture.
18. **Oversmoothing** — node representations become indistinguishable with depth.
19. **Expressivity** — intuitive connection to Weisfeiler–Lehman refinement.
20. **Oversquashing** — too much distant information through a small state.
21. **Mechanics and meshes** — physically meaningful edge and coordinate choices.
22. **Graph construction matters** — learned model cannot repair arbitrary relations reliably.
23. **Summary** — symmetry, locality, and relational inductive bias.

### 10. Physics-informed neural networks and neural operators

**Purpose:** replace the archive’s Hamiltonian/Lagrangian emphasis with two broadly
useful approaches to PDE learning.

1. **A differential-equation problem** — operator, domain, boundary/initial conditions.
2. **PINN objective** — one network represents one solution field.
3. **Automatic differentiation** — derivatives of the network with respect to coordinates.
4. **Collocation residual** — enforce the equation away from labeled observations.
5. **Composite loss** — data, PDE, boundary, and initial-condition terms.
6. **Units and loss scales** — nondimensionalization and weighting are mathematical issues.
7. **Worked 1-D boundary-value problem** — residual and exact solution on one example.
8. **Soft constraints** — penalty enforcement and residual trade-offs.
9. **Hard constraints** — trial solution that satisfies boundaries by construction.
10. **Inverse problem** — infer an unknown coefficient jointly with the field.
11. **Why PINNs can fail** — stiffness, spectral bias, optimization imbalance, poor sampling.
12. **Validation is not the training residual** — independent points and conserved quantities.
13. **From one solution to an operator** — map input functions/parameters to solution functions.
14. **Amortized solution** — expensive training, rapid inference over a distribution.
15. **Discretization and resolution** — what discretization-invariant claims require.
16. **DeepONet** — branch/trunk factorization and sensor/query roles.
17. **Fourier Neural Operator** — transform, mix retained modes, inverse transform.
18. **FNO layer equation** — spectral convolution plus local linear path and nonlinearity.
19. **Mode truncation** — efficiency and lost fine-scale information.
20. **Distribution shift** — geometry, coefficients, forcing, and resolution limits.
21. **PINN versus neural operator** — one-instance solve versus learned family of solves.
22. **Hybrid use** — data and physics can constrain either approach.
23. **Numerical baseline** — compare against established solvers on error and cost.
24. **Summary** — physics supplies structure, not immunity from optimization or validation.

### 11. Diffusion models

**Purpose:** explain noise-prediction training and reverse-time generation with a
consistent discrete and continuous view.

1. **Generative objective** — learn a distribution, not a single output.
2. **Forward noising chain** — gradually destroy structure with known Gaussian transitions.
3. **Noise schedule** — \(\beta_t\), \(\alpha_t\), and \(\bar\alpha_t\).
4. **Direct sampling at time \(t\)** —
   \(\mathbf x_t=\sqrt{\bar\alpha_t}\mathbf x_0+sqrt{1-\bar\alpha_t}\boldsymbol\epsilon\).
5. **Scores** — \(\nabla_{\mathbf x}\log p_t(\mathbf x)\) as direction toward higher density.
6. **Noise and score parameterizations** — exact Gaussian conditional relation.
7. **Training experiment** — sample data, time, and noise.
8. **Noise-prediction objective** — weighted MSE and what its optimum estimates.
9. **Conditional expectation result** — the MSE optimum is \(\mathbb E[\epsilon\mid x_t,t]\).
10. **Reverse posterior** — known form when conditioning on \(x_0\).
11. **Learned reverse transition** — replace unavailable clean-data information.
12. **Sampling trajectory** — noise to data through many denoising steps.
13. **DDPM and deterministic alternatives** — distinguish stochastic and implicit sampling.
14. **Classifier-free guidance** — conditional/unconditional score combination and trade-off.
15. **Continuous-time SDE view** — forward diffusion and reverse-time score term.
16. **Probability-flow ODE** — same marginals under an associated deterministic flow.
17. **Sampling cost** — network evaluations and solver/discretization error.
18. **Applications** — images, fields, audio, and inverse problems.
19. **Limitations and evaluation** — likelihood, sample metrics, memorization, conditioning.
20. **Summary** — learn how to reverse a known corruption process.

### 12. Flow matching

**Purpose:** formulate generation as learned continuous transport and connect it
precisely to continuous normalizing flows and diffusion.

1. **Transport view** — move a simple base distribution to the data distribution.
2. **ODE dynamics** — \(d\mathbf x_t/dt=\mathbf v_t(\mathbf x_t)\).
3. **Continuity equation** — conservation of probability under the velocity field.
4. **Continuous normalizing flow** — pushforward by the ODE solution.
5. **Change of log density** — divergence integral and its computational cost.
6. **Flow-matching objective** — regress a target velocity field.
7. **Conditional probability paths** — tractable paths conditioned on endpoints/data.
8. **Conditional flow matching theorem** — conditional regression yields the marginal field.
9. **A simple Gaussian path** — derive its target conditional velocity.
10. **Rectified flow** — straight interpolation and endpoint coupling.
11. **Coupling matters** — same marginals can induce very different path curvature.
12. **One-dimensional velocity field** — visual trajectories and field arrows.
13. **Sampling** — integrate the learned ODE from base to data.
14. **Model error versus solver error** — separate approximation from integration accuracy.
15. **Path curvature and step count** — why straighter paths can be easier to integrate.
16. **Flow matching versus diffusion** — target, dynamics, stochasticity, and sampling.
17. **Probability-flow bridge** — diffusion can induce a deterministic ODE transport.
18. **Applications** — high-dimensional generation and scientific fields.
19. **Limitations** — coupling bias, distribution shift, and numerical sensitivity.
20. **Summary** — directly learn a vector field whose flow transports probability.

## Implementation sequence

### Batch 1 — ownership and mathematical foundations

1. Update `TOPICS.toml` so every topic above has exactly one owner.
2. Update `ROADMAP.md` in place; remove the obsolete combined-deck plan.
3. Audit and correct optimizer recurrences and generated trajectories.
4. Add the hand-reference and convergence tests listed in the optimization section.
5. Rewrite `optimization`; compile, render, and inspect every frame.
6. Separate linear/logistic material from `neural-nets` and keep it only in
   `supervised-learning`.
7. Rewrite `supervised-learning`; compile, render, and inspect it.

This first batch produces the two immediately usable foundation presentations.

### Batch 2 — dense networks and generalization

1. Rewrite `neural-nets` around dense networks, reverse-mode differentiation,
   backpropagation, initialization, UAT, and the sine/Sobolev example.
2. Create `generalization` with bias–variance, L1/L2 priors and geometry, early
   stopping, ensembles, dropout, and visual augmentation.
3. Compile and inspect both before continuing.

### Batch 3 — symmetry and mainstream architectures

1. Create `geometric-deep-learning` first so later decks can reference its definitions.
2. Rewrite `cnns`, including the verified convolution example and equivariance caveats.
3. Rewrite `rnns`, replacing the malformed recursive TikZ arrow and checking BPTT math.
4. Create `transformers` with a verified numerical attention example.
5. Rewrite `gnns` around permutation-equivariant message passing.
6. Compile and inspect each deck as it is completed.

### Batch 4 — scientific and generative specialist decks

1. Create `pinns-neural-operators`; remove Hamiltonian/Lagrangian coverage from the plan.
2. Split the current combined generative material into `diffusion-models` and
   `flow-matching` without duplicating the probability-flow connection.
3. Compile, render, and inspect both specialist decks.

### Batch 5 — course-wide verification

1. Search for duplicate definitions, conflicting symbols, and stale cross-references.
2. Check every archive topic against the coverage map.
3. Check all equations, numerical examples, plot generators, captions, and sources.
4. Run all existing tests and `tools/build-all`.
5. Render every final PDF and perform a page-by-page visual review.
6. Confirm `.build/`, generated PDFs, and archive files remain untracked.

## Review standard

The course is ready only when a student can answer, from the slides:

- What problem does this method solve?
- What assumptions or symmetry does it encode?
- What is the governing equation or algorithm?
- Why should it work?
- What can fail, and how would we notice?
- How does it connect to the preceding and following lectures?

Formal material remains available even when an instructor may skip it live. Complexity
must earn its place by explaining a method, a guarantee, or a limitation; implementation
trivia and catalogues of variants do not.
