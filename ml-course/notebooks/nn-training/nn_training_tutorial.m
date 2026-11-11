%[text] # Training Neural Networks in MATLAB
%[text] **A surrogate-modelling tutorial with the Deep Learning Toolbox**
%[text] *Data-Driven Methods in Engineering, NTUA School of Civil Engineering*
%[text:tableOfContents]{"heading":"Contents"}
%[text] ## How to use this notebook
%[text] This is a MATLAB Live Script. Run it **section by section** (**Ctrl+Enter**) and look at each result before moving on.
%[text] - We build one neural-network surrogate and refine it step by step.
%[text] - **Try it** questions ask you to change one line, rerun a section, and explain what you see.
%[text] - You need the **Deep Learning Toolbox** ([`trainnet`](https://www.mathworks.com/help/deeplearning/ref/trainnet.html), [`trainingOptions`](https://www.mathworks.com/help/deeplearning/ref/trainingoptions.html)).
%[text] - Utilities (the data split, plotting, a one-line training call) are local functions at the end of the file. MATLAB requires them there; you do not need to read them to follow the text. \
%[text] ## Settings
%[text] All knobs live here. Change one, rerun the sections below, and compare.
%[text] - `SEED` fixes the random number generator ([`rng`](https://www.mathworks.com/help/matlab/ref/rng.html)), so every run is reproducible.
%[text] - `EPOCHS`, `BATCH_SIZE`, `LEARN_RATE` and `HIDDEN` define the reference training run.
%[text] - `BASE` collects the reference settings for the helper `trainSurrogate`. \
% Data
SEED = 20261111;
N_SAMPLES = 1000;
SPLIT = [0.6 0.2 0.2];           % train / validation / test fractions

% Reference training run
EPOCHS = 200;
BATCH_SIZE = 32;
LEARN_RATE = 1e-2;
HIDDEN = 8;                      % neurons in the hidden layer

MC_SAMPLES = 10000;              % fresh samples for the Monte Carlo comparison

% The physical problem: struct fields are read as COLUMN.L, INPUTS.EMean, ...
COLUMN = struct("L", 4, "I", 5e-6);                                  % length (m), second moment of area (m^4)
INPUTS = struct("EMean", 200e9, "EStd", 10e9, "KLimits", [0.6 1.0]); % E in Pa, K dimensionless

% Name-value pairs in a cell array; trainSurrogate(..., BASE{:}) expands them into arguments.
BASE = {"LearnRate", LEARN_RATE, "Epochs", EPOCHS, "BatchSize", BATCH_SIZE, "Seed", SEED};
%%
%[text] ## The problem: a buckling-load surrogate
%[text] A **surrogate** is a cheap learned imitation of an expensive model. Here the "expensive model" is a formula, so we always know the right answer.
%[text] ### The formula
%[text] A slender column under axial compression buckles at the Euler critical load
%[text]{"align":"center"} $P\_{cr} = \\frac{\\pi^2 E I}{(K L)^2}.$
%[text:table]
%[text] | Symbol | Meaning | Value |
%[text] | --- | --- | --- |
%[text] | $E$ | Young's modulus | uncertain |
%[text] | $K$ | effective-length factor | uncertain, between 0.6 and 1.0 |
%[text] | $L$ | length | 4 m |
%[text] | $I$ | second moment of area | $5\\times10^{-6}$ m$^4$ |
%[text:table]
%[text] `bucklingLoad` is the formula as an [anonymous function](https://www.mathworks.com/help/matlab/matlab_prog/anonymous-functions.html). It returns MN; the plots show the load against each input, with the other one fixed.
% @(E, K) names the inputs. COLUMN is copied into the function when this line runs:
% rerun it if you change COLUMN. The dotted operators .* ./ .^ work element by element,
% so E and K may be whole vectors.
bucklingLoad = @(E, K) pi^2 .* E .* COLUMN.I ./ (K .* COLUMN.L).^2 / 1e6;

% linspace gives a row; the transpose ' makes it a column (one value per row).
kLine = linspace(INPUTS.KLimits(1), INPUTS.KLimits(2), 100)';
eLine = linspace(INPUTS.EMean - 3 * INPUTS.EStd, INPUTS.EMean + 3 * INPUTS.EStd, 100)';

figure
tiledlayout(1, 2, TileSpacing="compact")

nexttile
    plot(kLine, bucklingLoad(INPUTS.EMean, kLine), LineWidth=1.5)
    xlabel("K")
    ylabel("P_{cr} (MN)")      % _{...} is a TeX subscript in labels
    title("E = 200 GPa")
    grid on

nexttile
    plot(eLine / 1e9, bucklingLoad(eLine, 0.8), LineWidth=1.5)   % E plotted in GPa
    xlabel("E (GPa)")
    ylabel("P_{cr} (MN)")
    title("K = 0.8")
    grid on
%%
%[text] - $K$ changes the load by a factor of almost 3, following $1/K^2$.
%[text] - $E$ changes it by a few percent, linearly. \
%[text] ### The uncertain inputs
%[text] - $K \\sim \\mathcal{U}(0.6,\\,1.0)$: real supports are only partially rigid.
%[text] - $E$ is lognormal with mean 200 GPa and standard deviation 10 GPa, so it is always positive. \
%[text] A lognormal $E = e^{Z}$ comes from a normal $Z \\sim \\mathcal{N}(m, s^2)$. Matching the mean and standard deviation of $E$ gives
%[text]{"align":"center"} $s = \\sqrt{\\ln\\left(1 + \\sigma\_E^2/\\mu\_E^2\\right)}, \\qquad m = \\ln \\mu\_E - s^2/2.$
%[text] `sampleInputs(n)` returns $n$ rows $[E\\;K]$: rows are observations, columns are features.
%[text] - $E$ = `exp` of a normal sample ([`randn`](https://www.mathworks.com/help/matlab/ref/double.randn.html)).
%[text] - $K$ = a uniform sample ([`rand`](https://www.mathworks.com/help/matlab/ref/double.rand.html)) stretched to $[0.6,\\,1.0]$. \
sLog = sqrt(log(1 + (INPUTS.EStd / INPUTS.EMean)^2));   % std of the underlying normal
mLog = log(INPUTS.EMean) - sLog^2 / 2;                   % mean of the underlying normal

% [a, b] puts two n-by-1 columns side by side: an n-by-2 matrix.
% randn and rand draw from the global generator, so rng(...) before a call fixes the result.
sampleInputs = @(n) [exp(mLog + sLog * randn(n, 1)), ...                     % lognormal E
                     INPUTS.KLimits(1) + diff(INPUTS.KLimits) * rand(n, 1)];  % uniform K

rng(SEED);
preview = sampleInputs(5000);

figure
tiledlayout(1, 2, TileSpacing="compact")

nexttile
    histogram(preview(:, 1) / 1e9, 40)   % column 1 is E; 40 bins
    xlabel("E (GPa)")
    ylabel("count")
    title("Lognormal E")

nexttile
    histogram(preview(:, 2), 40)         % column 2 is K
    xlabel("K")
    title("Uniform K")
%%
%[text] ## One input: every model is a curve
%[text] $E$ is fixed at its mean, so the network is a function of $K$ only and we can draw it on top of the true curve.
%[text] ### Generate and split the data
%[text] The rows are split **once** into three sets:
%[text:table]
%[text] | Set | Share | Used for |
%[text] | --- | --- | --- |
%[text] | training | 60% | fitting the weights |
%[text] | validation | 20% | comparing settings |
%[text] | test | 20% | one final error estimate, at the very end |
%[text:table]
rng(SEED);
X = sampleInputs(N_SAMPLES);
k = X(:, 2);                                 % keep only K: an N-by-1 input matrix
pcr = bucklingLoad(INPUTS.EMean, k);         % targets, with E fixed at its mean

% splitDataset (end of file) shuffles the rows with randperm and returns a struct
% with fields XTrain, YTrain, XVal, YVal, XTest, YTest.
data1 = splitDataset(k, pcr, SPLIT);

% Try it: keep only 15 training rows (uncomment), then rerun the sections below.
% data1.XTrain = data1.XTrain(1:15);
% data1.YTrain = data1.YTrain(1:15);

figure
    scatter(data1.XTrain, data1.YTrain, 8, "filled", DisplayName="training")   % DisplayName feeds the legend
    hold on                                                                    % keep adding to the same axes
    scatter(data1.XVal, data1.YVal, 8, "filled", DisplayName="validation")
    scatter(data1.XTest, data1.YTest, 8, "filled", DisplayName="test")
    hold off
    xlabel("K")
    ylabel("P_{cr} (MN)")
    legend
    grid on
%%
%[text] ### Define the network
%[text] A network with one hidden layer computes
%[text]{"align":"center"} $\\hat y = W\_2 \\tanh(W\_1 x' + b\_1) + b\_2.$
%[text] In MATLAB it is a column of layers:
%[text] - [`featureInputLayer`](https://www.mathworks.com/help/deeplearning/ref/nnet.cnn.layer.featureinputlayer.html)`(1, Normalization="zscore")`: one input feature, standardised to $x' = (x-\\mu)/\\sigma$ with training statistics.
%[text] - [`fullyConnectedLayer`](https://www.mathworks.com/help/deeplearning/ref/nnet.cnn.layer.fullyconnectedlayer.html)`(h)`: the affine map $Wx+b$ with $h$ outputs.
%[text] - [`tanhLayer`](https://www.mathworks.com/help/deeplearning/ref/nnet.cnn.layer.tanhlayer.html): the nonlinearity.
%[text] - A final `fullyConnectedLayer(1)` with no activation, so the output can take any value. \
% One layer per row. This only describes the network: the weights are created
% when trainnet runs, so the same layers1 can be trained again and again.
% No semicolon after ], so MATLAB prints the layer list.
layers1 = [
    featureInputLayer(1, Normalization="zscore")   % mu and sigma are computed from XTrain when training starts
    fullyConnectedLayer(HIDDEN)
    tanhLayer
    fullyConnectedLayer(1)                         % linear output
    ]
%%
%[text] ### Training options
%[text] Training minimises the mean squared error with **mini-batch gradient descent**: each update uses $b$ random rows, and one **epoch** is one pass over the training set.
%[text]{"align":"center"} $\\mathrm{MSE} = \\frac{1}{N}\\sum\_{i=1}^{N}(y\_i - \\hat y\_i)^2$
%[text] [`trainingOptions`](https://www.mathworks.com/help/deeplearning/ref/trainingoptions.html) collects the settings:
%[text:table]
%[text] | Option | Meaning |
%[text] | --- | --- |
%[text] | `"adam"` | the optimiser |
%[text] | `InitialLearnRate` | step size $\\eta$ |
%[text] | `MaxEpochs`, `MiniBatchSize` | training budget and rows per update |
%[text] | `L2Regularization=0` | no penalty: the loss is pure MSE |
%[text] | `ValidationData` | evaluated during training, never used for updates |
%[text] | `ValidationFrequency` | in iterations; one epoch here |
%[text] | `OutputNetwork="last-iteration"` | return the final network |
%[text] | `Plots="training-progress"` | live plot of the losses |
%[text:table]
% ValidationFrequency counts iterations (updates), not epochs:
% one epoch = ceil(rows / batch size) updates.
itersPerEpoch = ceil(size(data1.XTrain, 1) / BATCH_SIZE);

% ... continues a statement on the next line.
options1 = trainingOptions("adam", ...
    InitialLearnRate=LEARN_RATE, ...
    MaxEpochs=EPOCHS, ...
    MiniBatchSize=BATCH_SIZE, ...
    Shuffle="every-epoch", ...
    L2Regularization=0, ...                   % the default is 1e-4
    ValidationData={data1.XVal, data1.YVal}, ...
    ValidationFrequency=itersPerEpoch, ...
    OutputNetwork="last-iteration", ...       % the default returns the best-validation network
    ExecutionEnvironment="cpu", ...           % small networks are faster on the CPU
    Plots="training-progress", ...
    Verbose=false);
%%
%[text] ### Train
%[text] [`trainnet`](https://www.mathworks.com/help/deeplearning/ref/trainnet.html) initialises the weights, runs the optimiser and returns:
%[text] - the trained [`dlnetwork`](https://www.mathworks.com/help/deeplearning/ref/dlnetwork.html);
%[text] - a [`TrainingInfo`](https://www.mathworks.com/help/deeplearning/ref/deep.traininginfo.html) with the loss histories. \
% rng just before trainnet fixes the initial weights and the mini-batch shuffling.
rng(SEED);
[net1, info1] = trainnet(data1.XTrain, data1.YTrain, layers1, "mse", options1);

% Learnables is a table with one row per weight matrix or bias vector;
% cellfun applies numel to each and sum adds them up.
numParameters = sum(cellfun(@numel, net1.Learnables.Value))   % (1+1)*8 + (8+1)*1 = 25
%%
%[text] ### Loss history
%[text] - `info1.TrainingHistory`: one row per update, loss on the current mini-batch.
%[text] - `info1.ValidationHistory`: one row per validation check, loss on the whole validation set.
%[text] - `plotLoss` draws the mean training loss per epoch (solid) and the validation loss (dashed) on a **log axis**. \
%[text] How to read it:
%[text] - both still falling: train longer;
%[text] - both flat: converged;
%[text] - training falls, validation rises: **overfitting**;
%[text] - spikes or `NaN`: the learning rate is too large. \
head(info1.TrainingHistory)   % the first 8 rows of the table

figure
    plotLoss(info1, "reference")
    title("Loss history")
%%
%[text] ### Predict and compare with the formula
%[text] [`minibatchpredict`](https://www.mathworks.com/help/deeplearning/ref/minibatchpredict.html) evaluates the network. The curve extends beyond the sampled range (dotted lines), where the network **extrapolates**.
kGrid = linspace(0.5, 1.1, 300)';             % wider than the sampled 0.6-1.0

% Networks compute in single precision; double() converts back for plotting and arithmetic.
pNet = double(minibatchpredict(net1, kGrid));

figure
    plot(kGrid, bucklingLoad(INPUTS.EMean, kGrid), "k-", LineWidth=1.5, DisplayName="formula")
    hold on
    plot(kGrid, pNet, LineWidth=1.5, DisplayName="network")
    scatter(data1.XTrain, data1.YTrain, 10, "filled", MarkerFaceAlpha=0.3, DisplayName="training samples")
    xline(INPUTS.KLimits, ":", HandleVisibility="off")   % sampled range; kept out of the legend
    hold off
    xlabel("K")
    ylabel("P_{cr} (MN)")
    legend
    grid on
%[text] **Try it** (rerun from *Training options*):
%[text] - Set `LEARN_RATE = 1e-4` in Settings. What does the loss history look like after 200 epochs? And the curve?
%[text] - Set `LEARN_RATE = 1`. What happens to the loss?
%[text] - Keep only 15 training rows (*Generate and split the data*). Why does the fit get worse, and why does raising `EPOCHS` help only partly? \
%%
%[text] ### Underfitting and overfitting
%[text] Real data are noisy. We keep only **10** training rows, add 20% noise, and train two extreme networks for 1000 epochs:
%[text] - a **linear** model, with no hidden layer: too little capacity, so it is **biased** and cannot follow the curvature;
%[text] - a **large** network, two hidden layers of 64: enough capacity to chase the noise. \
%[text] The validation targets stay noise-free, so validation loss measures the error against the true curve.
%[text] From here on, `trainSurrogate` (end of file) does *Training options* plus *Train* in one line.
NOISE = 0.20;
N_NOISY = 10;

noisy = data1;                               % a struct copy: data1 is not changed
rng(SEED);
noisy.XTrain = data1.XTrain(1:N_NOISY);
noisy.YTrain = data1.YTrain(1:N_NOISY) .* (1 + NOISE * randn(N_NOISY, 1));   % multiplicative noise

layersLinear = [
    featureInputLayer(1, Normalization="zscore")
    fullyConnectedLayer(1)                   % no hidden layer: a straight line in K
    ];

layersLarge = [
    featureInputLayer(1, Normalization="zscore")
    fullyConnectedLayer(64)
    tanhLayer
    fullyConnectedLayer(64)
    tanhLayer
    fullyConnectedLayer(1)
    ];

% BASE{:} passes the reference settings; Epochs written after it overrides BASE's value.
% With 10 rows, one epoch is a single update, so 1000 epochs are only 1000 updates.
[netLinear, infoLinear] = trainSurrogate(noisy, layersLinear, BASE{:}, Epochs=1000);
[netLarge, infoLarge] = trainSurrogate(noisy, layersLarge, BASE{:}, Epochs=1000);

figure
tiledlayout(1, 3, TileSpacing="compact")

nexttile
    plotFit1D(netLinear, noisy, bucklingLoad, INPUTS)
    title("Linear: underfits")

nexttile
    plotFit1D(netLarge, noisy, bucklingLoad, INPUTS)
    title("64 + 64: overfits")

nexttile
    plotLoss(infoLinear, "linear")
    plotLoss(infoLarge, "64+64")
    title("Loss histories")
%[text] **Try it:**
%[text] - Which network has a large gap between training and validation loss? Which has none, but a high level?
%[text] - Find the epoch where the large network's validation loss is lowest. What would early stopping return?
%[text] - Train `layers1` (8 neurons) on `noisy` too. Where does it land between the two? \
%%
%[text] ## Two inputs: every model is a surface
%[text] Now both $E$ and $K$ vary, and the surrogate is a surface $\\hat P(E, K)$.
%[text] ### Generate the data
rng(SEED);
X2 = sampleInputs(N_SAMPLES);                % N-by-2: column 1 is E, column 2 is K
y2 = bucklingLoad(X2(:, 1), X2(:, 2));

figure
tiledlayout(1, 3, TileSpacing="compact")

nexttile
    scatter(X2(:, 1) / 1e9, X2(:, 2), 4, "filled")
    xlabel("E (GPa)")
    ylabel("K")
    title("Inputs")

nexttile
    scatter(X2(:, 2), y2, 4, "filled")
    xlabel("K")
    ylabel("P_{cr} (MN)")
    title("Load against K")

nexttile
    histogram(y2)
    xlabel("P_{cr} (MN)")
    title("Load distribution")
%%
%[text] ### The target in 3D
%[text] [`meshgrid`](https://www.mathworks.com/help/matlab/ref/meshgrid.html) builds a grid, [`surf`](https://www.mathworks.com/help/matlab/ref/surf.html) draws the formula, [`scatter3`](https://www.mathworks.com/help/matlab/ref/scatter3.html) adds the samples. Rotate it with the mouse.
% eGrid and kGrid2 are 100-by-100 matrices: every combination of eLine and kLine.
[eGrid, kGrid2] = meshgrid(eLine, kLine);

figure
    surf(eGrid / 1e9, kGrid2, bucklingLoad(eGrid, kGrid2), EdgeColor="none", FaceAlpha=0.8)
    hold on
    scatter3(X2(:, 1) / 1e9, X2(:, 2), y2, 6, "k", "filled")
    hold off
    xlabel("E (GPa)")
    ylabel("K")
    zlabel("P_{cr} (MN)")
    view(-35, 30)                            % azimuth, elevation in degrees
%%
%[text] ### Split and train the reference network
%[text] The same network as before, now with **two** input features.
data2 = splitDataset(X2, y2, SPLIT);

layers2 = [
    featureInputLayer(2, Normalization="zscore")   % Try it: Normalization="none"
    fullyConnectedLayer(HIDDEN)
    tanhLayer
    fullyConnectedLayer(1)
    ];

[net2, info2] = trainSurrogate(data2, layers2, BASE{:});

figure
    plotLoss(info2, "reference")
    title("Reference network")
%[text] **Try it:** set `Normalization="none"` and rerun this section. What happens to the loss, and why?
%[text] - Hint: $E \\approx 2\\times10^{11}$ while $K \\approx 0.8$.
%[text] - With raw inputs, $E$ dominates $W\_1 x$, `tanh` saturates at $\\pm 1$, and its gradient vanishes.
%[text] - `Normalization="zscore"` rescales each input with the **training** mean and standard deviation, stored in the layer. \
%%
%[text] ### Evaluate on the validation set
%[text] Three numbers, from `regressionMetrics`:
%[text]{"align":"center"} $\\mathrm{RMSE} = \\sqrt{\\mathrm{MSE}}, \\qquad R^2 = 1 - \\frac{\\sum\_i (y\_i-\\hat y\_i)^2}{\\sum\_i (y\_i-\\bar y)^2}$
%[text] - MSE in MN$^2$, RMSE in MN.
%[text] - $R^2 = 1$ is a perfect fit. \
%[text] Two plots, from `plotParity`:
%[text] - **Parity:** predicted against true; a perfect model lies on the diagonal.
%[text] - **Residuals:** $\\hat y - y$ against true; look for trends, not just size. \
yHat2 = double(minibatchpredict(net2, data2.XVal));

valMetrics = regressionMetrics(data2.YVal, yHat2)   % a struct with fields MSE, RMSE, R2
plotParity(data2.YVal, yHat2, "reference, validation");
%%
%[text] ### Refine: learning rate
%[text] Adam at a small, the reference and a large learning rate. Only `LearnRate` changes.
[netSlow, infoSlow] = trainSurrogate(data2, layers2, BASE{:}, LearnRate=1e-4);
[netFast, infoFast] = trainSurrogate(data2, layers2, BASE{:}, LearnRate=1);

% info2 (the reference, LearnRate=1e-2) is reused as the medium rate.
figure
    plotLoss(infoSlow, "\eta = 10^{-4}")
    plotLoss(info2, "\eta = 10^{-2}")
    plotLoss(infoFast, "\eta = 1")
    title("Adam: learning rate")
%[text] **Try it:**
%[text] - Which rate is too small, and how can you tell from the curve?
%[text] - What does the large rate do to the validation loss? \
%%
%[text] ### Refine: optimiser
%[text] Plain **SGD** takes the same step size for every weight:
%[text]{"align":"center"} $\\theta \\leftarrow \\theta - \\eta\\, \\nabla\_\\theta \\mathcal{L}.$
%[text] **Adam** adapts the step per weight, using running averages of the gradient and of its square. We train SGD at the same three rates.
% MATLAB has no plain "sgd" solver: "sgdm" with Momentum=0 (trainSurrogate's default) is plain SGD.
[netSgdSlow, infoSgdSlow] = trainSurrogate(data2, layers2, BASE{:}, Solver="sgdm", LearnRate=1e-4);
[netSgd, infoSgd] = trainSurrogate(data2, layers2, BASE{:}, Solver="sgdm", LearnRate=1e-2);
[netSgdFast, infoSgdFast] = trainSurrogate(data2, layers2, BASE{:}, Solver="sgdm", LearnRate=1);

figure
    plotLoss(infoSgdSlow, "SGD \eta = 10^{-4}")
    plotLoss(infoSgd, "SGD \eta = 10^{-2}")
    plotLoss(infoSgdFast, "SGD \eta = 1")
    title("SGD: learning rate")
%[text] **Try it:**
%[text] - Which optimiser is more sensitive to the learning rate?
%[text] - Add `Momentum=0.9` to the SGD calls. Does the medium rate improve? \
%%
%[text] ### Refine: a wider and deeper network
%[text] Two hidden layers of 16 neurons, Adam at the reference rate.
layersWide = [
    featureInputLayer(2, Normalization="zscore")
    fullyConnectedLayer(16)
    tanhLayer                                      % Try it: reluLayer
    fullyConnectedLayer(16)
    tanhLayer
    fullyConnectedLayer(1)
    ];

[netWide, infoWide] = trainSurrogate(data2, layersWide, BASE{:});

figure
    plotLoss(info2, "8")
    plotLoss(infoWide, "16-16")
    title("Network size")

yHatWide = double(minibatchpredict(netWide, data2.XVal));
wideMetrics = regressionMetrics(data2.YVal, yHatWide)
plotParity(data2.YVal, yHatWide, "16-16, validation");
%[text] **Try it:** did the wider network help? Compare its validation MSE with the reference, and count its parameters as for the first network. A bigger network has more parameters to fit, and is not automatically better.
%%
%[text] ### Choose a model
%[text] Pick the network with the lowest **validation** MSE above. The test set stays untouched until the next section.
chosen = net2;   % Try it: replace with your choice
%%
%[text] ### Final test, once
%[text] - The test set played no part in any choice, so its error is an honest estimate for new columns.
%[text] - If you change the model after seeing it, the test set becomes a second validation set. \
yTest = double(minibatchpredict(chosen, data2.XTest));

testMetrics = regressionMetrics(data2.YTest, yTest)
plotParity(data2.YTest, yTest, "chosen, test");
%%
%[text] ## Using the surrogate
%[text] ### Monte Carlo
%[text] Propagate the input uncertainty to the load with `MC_SAMPLES` fresh inputs:
%[text] - the same bin edges for formula and network, so the histograms compare;
%[text] - `Normalization="pdf"` so both have area one;
%[text] - the 95th percentile ([`prctile`](https://www.mathworks.com/help/matlab/ref/prctile.html)) describes the upper tail that matters for design. \
rng(SEED + 1000);                            % a different seed: these inputs are new
XMC = sampleInputs(MC_SAMPLES);
yTrue = bucklingLoad(XMC(:, 1), XMC(:, 2));
yNet = double(minibatchpredict(chosen, XMC));

edges = linspace(min(yTrue), max(yTrue), 50);   % 50 edges = 49 shared bins

figure
    histogram(yTrue, edges, Normalization="pdf", DisplayName="formula")
    hold on
    histogram(yNet, edges, Normalization="pdf", DisplayStyle="stairs", LineWidth=1.5, DisplayName="network")
    hold off
    xlabel("P_{cr} (MN)")
    ylabel("probability density")
    legend
    grid on

p95 = [prctile(yTrue, 95) prctile(yNet, 95)]   % [formula network]
%%
%[text] ### Response curves
%[text] Vary one input, hold the other at its mean. The network should reproduce a straight line in $E$ and a $1/K^2$ curve in $K$.
kMean = mean(INPUTS.KLimits);

% The network needs both inputs: pair each swept value with the fixed one.
% ones(size(v)) is a column of ones with the same size as v.
sweepE = [eLine, kMean * ones(size(eLine))];
sweepK = [INPUTS.EMean * ones(size(kLine)), kLine];

figure
tiledlayout(1, 2, TileSpacing="compact")

nexttile
    plot(eLine / 1e9, bucklingLoad(eLine, kMean), "k-", LineWidth=1.5, DisplayName="formula")
    hold on
    plot(eLine / 1e9, double(minibatchpredict(chosen, sweepE)), "--", LineWidth=1.5, DisplayName="network")
    hold off
    xlabel("E (GPa)")
    ylabel("P_{cr} (MN)")
    title("K = 0.8")
    legend
    grid on

nexttile
    plot(kLine, bucklingLoad(INPUTS.EMean, kLine), "k-", LineWidth=1.5, DisplayName="formula")
    hold on
    plot(kLine, double(minibatchpredict(chosen, sweepK)), "--", LineWidth=1.5, DisplayName="network")
    hold off
    xlabel("K")
    ylabel("P_{cr} (MN)")
    title("E = 200 GPa")
    legend
    grid on
%%
%[text] ## Takeaways
%[text] - Split once; choose by validation error; test once.
%[text] - Standardise the inputs with training statistics.
%[text] - Read loss histories on a log axis before reading final numbers.
%[text] - Too little capacity underfits (high, flat losses); too much overfits (a growing gap). Refine one setting at a time and look at parity and residual plots, not only at one number. \
%[text] ### Function cheat sheet
%[text:table]
%[text] | Task | MATLAB |
%[text] | --- | --- |
%[text] | layers | [`featureInputLayer`](https://www.mathworks.com/help/deeplearning/ref/nnet.cnn.layer.featureinputlayer.html), [`fullyConnectedLayer`](https://www.mathworks.com/help/deeplearning/ref/nnet.cnn.layer.fullyconnectedlayer.html), [`tanhLayer`](https://www.mathworks.com/help/deeplearning/ref/nnet.cnn.layer.tanhlayer.html), [`reluLayer`](https://www.mathworks.com/help/deeplearning/ref/nnet.cnn.layer.relulayer.html) |
%[text] | train | [`trainingOptions`](https://www.mathworks.com/help/deeplearning/ref/trainingoptions.html), [`trainnet`](https://www.mathworks.com/help/deeplearning/ref/trainnet.html), [`TrainingInfo`](https://www.mathworks.com/help/deeplearning/ref/deep.traininginfo.html) |
%[text] | predict | [`minibatchpredict`](https://www.mathworks.com/help/deeplearning/ref/minibatchpredict.html) |
%[text] | data | [`rng`](https://www.mathworks.com/help/matlab/ref/rng.html), [`randperm`](https://www.mathworks.com/help/matlab/ref/double.randperm.html), [`prctile`](https://www.mathworks.com/help/matlab/ref/prctile.html) |
%[text] | plot | [`semilogy`](https://www.mathworks.com/help/matlab/ref/semilogy.html), [`scatter`](https://www.mathworks.com/help/matlab/ref/scatter.html), [`histogram`](https://www.mathworks.com/help/matlab/ref/matlab.graphics.chart.primitive.histogram.html), [`surf`](https://www.mathworks.com/help/matlab/ref/surf.html) |
%[text:table]
%%
%[text] ## Local functions
function data = splitDataset(X, y, fractions)
% Random split of the rows into training, validation and test sets (uses the global rng).
    n = size(X, 1);
    order = randperm(n);                     % a random permutation of 1..n
    nTrain = round(fractions(1) * n);
    nVal = round(fractions(2) * n);

    train = order(1:nTrain);
    val = order(nTrain + 1:nTrain + nVal);
    test = order(nTrain + nVal + 1:end);

    data = struct("XTrain", X(train, :), "YTrain", y(train), ...
                  "XVal", X(val, :), "YVal", y(val), ...
                  "XTest", X(test, :), "YTest", y(test));
end

function [net, info] = trainSurrogate(data, layers, opts)
% trainingOptions + rng + trainnet, as in "Training options" and "Train". Never reads the test rows.
    arguments
        data (1,1) struct
        layers
        opts.Solver (1,1) string {mustBeMember(opts.Solver, ["adam" "sgdm"])} = "adam"
        opts.LearnRate (1,1) double {mustBePositive} = 1e-2
        opts.Momentum (1,1) double {mustBeInRange(opts.Momentum, 0, 1)} = 0
        opts.Epochs (1,1) double {mustBeInteger, mustBePositive} = 200
        opts.BatchSize (1,1) double {mustBeInteger, mustBePositive} = 32
        opts.Seed (1,1) double = 0
    end

    solverArgs = {};
    if opts.Solver == "sgdm"
        solverArgs = {"Momentum", opts.Momentum};   % Adam has no Momentum option
    end

    options = trainingOptions(opts.Solver, solverArgs{:}, ...
        InitialLearnRate=opts.LearnRate, ...
        MaxEpochs=opts.Epochs, ...
        MiniBatchSize=opts.BatchSize, ...
        Shuffle="every-epoch", ...
        L2Regularization=0, ...
        ValidationData={data.XVal, data.YVal}, ...
        ValidationFrequency=ceil(size(data.XTrain, 1) / opts.BatchSize), ...
        OutputNetwork="last-iteration", ...
        ExecutionEnvironment="cpu", ...
        Plots="none", ...
        Verbose=false);

    rng(opts.Seed);
    [net, info] = trainnet(data.XTrain, data.YTrain, layers, "mse", options);
end

function metrics = regressionMetrics(y, yHat)
    residual = y - yHat;
    metrics.MSE = mean(residual.^2);
    metrics.RMSE = sqrt(metrics.MSE);
    metrics.R2 = 1 - sum(residual.^2) / sum((y - mean(y)).^2);
end

function plotLoss(info, name)
% Adds to the current axes: mean mini-batch loss per epoch (solid) and validation loss (dashed), log y-axis.
    history = info.TrainingHistory;
    trainLoss = accumarray(history.Epoch, history.Loss, [], @mean);   % average the updates of each epoch
    itersPerEpoch = history.Iteration(end) / history.Epoch(end);

    p = semilogy(1:numel(trainLoss), trainLoss, "-", LineWidth=2, DisplayName=name + " train");
    hold on
    semilogy(info.ValidationHistory.Iteration / itersPerEpoch, info.ValidationHistory.Loss, "--", ...
        LineWidth=2, Color=p.Color, DisplayName=name + " val");
    set(gca, "YScale", "log");

    xlabel("Epoch")
    ylabel("MSE (MN^2)")
    legend(Location="best")
    grid on
end

function plotFit1D(net, data, bucklingLoad, inputs)
% Formula, network and training samples over K (E at its mean), in the current axes.
    kGrid = linspace(inputs.KLimits(1) - 0.1, inputs.KLimits(2) + 0.1, 300)';

    plot(kGrid, bucklingLoad(inputs.EMean, kGrid), "k-", LineWidth=1.5, DisplayName="formula")
    hold on
    plot(kGrid, double(minibatchpredict(net, kGrid)), LineWidth=1.5, DisplayName="network")
    scatter(data.XTrain, data.YTrain, 20, "filled", DisplayName="training samples")
    xline(inputs.KLimits, ":", HandleVisibility="off")
    hold off

    ylim([0 3])
    xlabel("K")
    ylabel("P_{cr} (MN)")
    grid on
end

function plotParity(y, yHat, name)
% New figure: predicted vs true (with the diagonal) and residuals vs true.
    figure
    tiledlayout(1, 2, TileSpacing="compact")

    nexttile
        scatter(y, yHat, 8, "filled", MarkerFaceAlpha=0.5)
        hold on
        plot([min(y) max(y)], [min(y) max(y)], "k--")   % the diagonal: perfect predictions
        hold off
        axis equal tight
        xlabel("true P_{cr} (MN)")
        ylabel("predicted (MN)")
        title("Parity: " + name)
        grid on

    nexttile
        scatter(y, yHat - y, 8, "filled", MarkerFaceAlpha=0.5)
        yline(0, "k--")
        xlabel("true P_{cr} (MN)")
        ylabel("residual (MN)")
        title("Residuals: " + name)
        grid on
end

%[appendix]{"version":"1.0"}
%---
%[metadata:view]
%   data: {"layout":"inline"}
%---
