# ruff: noqa: F821  -- a fragment for the sheet (shown from line 2); the data arrays come from the student's own code
# Settings: change them here only
SEED = 0
N_SAMPLES = 3000
TRAIN_FRACTION, VAL_FRACTION = 0.6, 0.2  # the rest is the test set
HIDDEN_LAYERS = [8]                      # hidden-layer widths only, e.g. [10, 20, 10]
ACTIVATION = "tanh"
LEARNING_RATE = 1e-3
BATCH_SIZE = 64
EPOCHS = 1500

# Standardize every subset with the TRAINING statistics
mu, sigma = X_train.mean(axis=0), X_train.std(axis=0)
X_train_s, X_val_s, X_test_s = [(Z - mu) / sigma for Z in (X_train, X_val, X_test)]

# Network: input variables X -> hidden layers -> 1 linear output
keras.utils.set_random_seed(SEED)
model = keras.Sequential(
    [keras.Input(shape=(4,))]
    + [keras.layers.Dense(width, activation=ACTIVATION) for width in HIDDEN_LAYERS]
    + [keras.layers.Dense(1)]
)
model.compile(optimizer=keras.optimizers.Adam(LEARNING_RATE), loss="mse")

# Bonus only: return the weights of the epoch with the lowest validation loss
# (patience = EPOCHS, so it never stops training early)
keep_best = keras.callbacks.EarlyStopping(monitor="val_loss", patience=EPOCHS,
                                          restore_best_weights=True)

history = model.fit(X_train_s, y_train, batch_size=BATCH_SIZE, epochs=EPOCHS,
                    validation_data=(X_val_s, y_val), verbose=0,
                    callbacks=[])        # Bonus: callbacks=[keep_best]
train_mse = history.history["loss"]      # per epoch, averaged over its batches
val_mse = history.history["val_loss"]    # per epoch, on the whole validation set
