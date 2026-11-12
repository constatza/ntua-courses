% Settings: change them here only
SEED = 0;
N_SAMPLES = 3000;
TRAIN_FRACTION = 0.6; VAL_FRACTION = 0.2;   % the rest is the test set
HIDDEN_LAYERS = [8];                        % hidden-layer widths only, e.g. [10 20 10]
ACTIVATION = @tanhLayer;                    % or @reluLayer
LEARNING_RATE = 1e-3;
BATCH_SIZE = 64;
EPOCHS = 1500;

% Standardize every subset with the TRAINING statistics
mu = mean(XTrain); sigma = std(XTrain, 1);
XTrainS = (XTrain - mu)./sigma;
XValS = (XVal - mu)./sigma;
XTestS = (XTest - mu)./sigma;

% Network: input variables X -> hidden layers -> 1 linear output
rng(SEED);
layers = featureInputLayer(4, Normalization="none");   % inputs are already standardized
for width = HIDDEN_LAYERS
    layers = [layers; fullyConnectedLayer(width); ACTIVATION()];
end
layers = [layers; fullyConnectedLayer(1)];

% 1 validation value per epoch; trainnet drops the last partial mini-batch, hence floor.
% Bonus: OutputNetwork="best-validation" returns the network of the epoch with the
% lowest validation loss (the default "auto" does the same when ValidationData is set).
stepsPerEpoch = floor(size(XTrainS, 1)/BATCH_SIZE);
opts = trainingOptions("adam", InitialLearnRate=LEARNING_RATE, MaxEpochs=EPOCHS, ...
    MiniBatchSize=BATCH_SIZE, Shuffle="every-epoch", ...
    ValidationData={XValS, yVal}, ValidationFrequency=stepsPerEpoch, ...
    OutputNetwork="last-iteration");                  % Bonus: "best-validation"
[net, info] = trainnet(XTrainS, yTrain, layers, "mse", opts);
% info.TrainingHistory (per batch) and info.ValidationHistory (per epoch) hold the losses.
