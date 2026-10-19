"""
=============================================================================
 HOMEWORK PART 2 -- BUILD YOUR OWN MODEL  (HWNet)
=============================================================================

`HWNet` plugs into the exact same training machinery as MixNet, EEGNet
and DeepConvNet, so the benchmark in Part 3 is a fair comparison: 
same folds, same subjects, same early stopping, same metrics.

WHAT IS ALREADY DONE FOR YOU (do not change unless you know why):
  * `train_step` / `val_step` / `test_step` / `pred_step` -- the custom
    TensorFlow loops that `mixnet.trainer.Trainer` calls every batch.
    NOTE: they are written for a SINGLE-TASK model -- one output, one loss
    (cross-entropy). If you go multi-task you MUST rewrite them; see
    "IF YOU GO MULTI-TASK" below.
  * inheritance from `BaseModel`, which gives you `.fit()`, `.evaluate()`
    and `.predict()` for free.

WHAT YOU MUST IMPLEMENT:
  1. `_config()` -- declare your architecture hyper-parameters.
  2. `build()`   -- return a `tf.keras.models.Model`.

-----------------------------------------------------------------------------
 CONTRACT YOUR `build()` MUST RESPECT
-----------------------------------------------------------------------------
  * Input : one tensor of shape `self.input_shape` (no batch axis).
  * Output: ONE tensor of shape (batch, num_class) holding probabilities,
            i.e. finish with `layers.Activation('softmax')`.
            Multi-output models take a different code path in
            `mixnet/models/base.py` and need extra work -- see
            "IF YOU GO MULTI-TASK" below before you try one.
  * Name  : pass `name=self.model_name`. Do NOT name it 'MixNet' or
            'MIN2Net' -- `base.py` branches on those names.
  * When `load_weights=True` you must call `model.load_weights(self.weights_dir)`
    (kept in the template below), otherwise `evaluate()` silently scores an
    untrained network.
-----------------------------------------------------------------------------

-----------------------------------------------------------------------------
 NOTE: IF YOU GO MULTI-TASK (optional, advanced -- read this before you start)
-----------------------------------------------------------------------------
 MixNet itself is multi-task: it optimises reconstruction (MSE) + deep metric
 learning (triplet) + classification (cross-entropy) at once, with adaptive
 loss weights. You are allowed to do the same, but the single-task contract
 above then no longer holds and FOUR things must change together:

 1. `build()` returns a LIST of outputs, with the classifier softmax LAST:
        Model(inputs=input1, outputs=[decoder_out, latent, softmax], ...)

 2. The config (`experiments/configs/HWNet.py`) must declare one entry per
    task, in the SAME order, in all three lists:
        'loss':         [MeanSquaredError(), triplet_loss(margin=1.0),
                         SparseCategoricalCrossentropy()],
        'loss_names':   ['mse', 'triplet', 'crossentropy'],
        'loss_weights': [1.0, 1.0, 1.0],
    Keep the name 'crossentropy' for the classification loss -- `Trainer`
    looks it up by that name to apply class balancing.

 3. `train_step` / `val_step` / `test_step` / `pred_step` below must compute
    every loss, combine them with the incoming `loss_weights`, and log one
    `*_<loss_name>_loss` entry per task. `mixnet/models/MixNet.py` is the
    reference implementation -- read its steps side by side with the
    single-task ones below. The shape of a multi-task step is:

        xr, z, y_logis = self.model(x, training=True)
        mse_loss   = self.loss.mse(x, xr)
        trp_loss   = self.loss.triplet(y, z)
        ce_loss    = self.loss.crossentropy(y, y_logis)
        losses     = [mse_loss, trp_loss, ce_loss]     # same order as loss_names
        train_loss = tf.reduce_sum(loss_weights * losses)
        ...
        return logs, (xr, z, y_logis)                  # outputs, classifier last

    The `loss_weights` argument is not decoration: `Trainer` passes the
    adaptive weights in through it every batch. Ignore it and adaptive
    gradient blending silently does nothing.

 4. `evaluate()` / `predict()` in `mixnet/models/base.py` unpack multiple
    outputs ONLY for models whose name starts with 'MixNet' or 'MIN2Net';
    every other model takes the single-output branch, `np.argmax(test_pred,
    axis=1)`, which produces garbage for a list of outputs. Do NOT rename
    your model to sneak into that branch, and do NOT edit `base.py`.
    Override `evaluate()` (and `predict()` if you use it) inside THIS class
    instead -- it is one of the two files you are allowed to change. Copy the
    MixNet branch of `base.py` as your starting point, remembering that
    `Trainer.testing()` hands you a tuple with the classifier output last.

 Multi-task is not required and earns no marks by itself. It earns marks only
 if your report shows what each auxiliary task contributed -- which is an
 ablation: train the same network with and without each extra loss.
-----------------------------------------------------------------------------

Minimal smoke-test architecture -- paste this into `build()` to verify the
plumbing end-to-end BEFORE you design anything clever, then replace it:

        input1  = layers.Input(shape=self.input_shape)
        x       = layers.Flatten()(input1)
        x       = layers.Dense(self.num_class)(x)
        softmax = layers.Activation('softmax', name='softmax')(x)
        model   = Model(inputs=input1, outputs=softmax, name=self.model_name)

Read `mixnet/models/EEGNet.py` and `mixnet/models/DeepConvNet.py` for two
complete, working examples of this same contract.
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Model

from mixnet import models
from mixnet.loss import *


def reservoir_weights(n_in, units, spectral_radius, input_scale, connectivity, seed):
    """Fixed random reservoir weights, x(t) = tanh(u(t) W_in + x(t-1) W).

    W is sparse and rescaled to the given spectral radius. Below 1 the state
    forgets its starting point (echo state property).
    """
    rng = np.random.RandomState(seed)
    w_in = rng.uniform(-input_scale, input_scale, size=(n_in, units))
    w = rng.uniform(-1.0, 1.0, size=(units, units))
    w *= rng.uniform(size=w.shape) < connectivity
    w *= spectral_radius / max(np.max(np.abs(np.linalg.eigvals(w))), 1e-12)
    return w_in, w


class ReservoirCell(layers.Layer):
    """Leaky reservoir cell, never trained:
    x(t) = (1 - leak) x(t-1) + leak tanh(u(t) W_in + x(t-1) W)."""

    def __init__(self, units, spectral_radius, leak, input_scale, connectivity, seed,
                 **kwargs):
        super().__init__(trainable=False, **kwargs)
        self.units = units
        self.state_size = units
        self.output_size = units
        self.spectral_radius = spectral_radius
        self.leak = leak
        self.input_scale = input_scale
        self.connectivity = connectivity
        self.seed = seed

    def build(self, input_shape):
        w_in, w = reservoir_weights(int(input_shape[-1]), self.units, self.spectral_radius,
                                    self.input_scale, self.connectivity, self.seed)
        self.w_in = self.add_weight(name='w_in', shape=w_in.shape, trainable=False,
                                    initializer=tf.constant_initializer(w_in))
        self.w = self.add_weight(name='w', shape=w.shape, trainable=False,
                                 initializer=tf.constant_initializer(w))
        self.built = True

    def get_config(self):
        # the checkpoint callback saves the whole model, so the layer must serialise
        config = super().get_config()
        config.pop('trainable', None)
        config.update(units=self.units, spectral_radius=self.spectral_radius, leak=self.leak,
                      input_scale=self.input_scale, connectivity=self.connectivity,
                      seed=self.seed)
        return config

    def call(self, inputs, states):
        prev = states[0]
        new = tf.tanh(tf.matmul(inputs, self.w_in) + tf.matmul(prev, self.w))
        out = (1.0 - self.leak) * prev + self.leak * new
        return out, [out]


class LogVariancePool(layers.Layer):
    """(batch, time, units) -> (batch, units): log variance over time after the washout."""

    def __init__(self, washout, **kwargs):
        super().__init__(**kwargs)
        self.washout = washout

    def get_config(self):
        config = super().get_config()
        config.update(washout=self.washout)
        return config

    def call(self, inputs):
        x = inputs[:, self.washout:, :]
        return tf.math.log(tf.math.reduce_variance(x, axis=1) + 1e-8)


class MeanPool(layers.Layer):
    """(batch, time, units) -> (batch, units): mean over time after the washout (ablation)."""

    def __init__(self, washout, **kwargs):
        super().__init__(**kwargs)
        self.washout = washout

    def get_config(self):
        config = super().get_config()
        config.update(washout=self.washout)
        return config

    def call(self, inputs):
        return tf.reduce_mean(inputs[:, self.washout:, :], axis=1)


class HWNet(models.base.BaseModel):

    def __init__(self,
                 optimizer,
                 input_shape=(400, 18),
                 num_class=2,
                 loss=[SparseCategoricalCrossentropy()],
                 loss_names=['crossentropy'],
                 loss_weights=[1.0],
                 model_name='HWNet',
                 data_format='channels_last',
                 **kwargs):
        super().__init__(num_class, loss, loss_names, loss_weights,
                         optimizer, data_format, **kwargs)
        self.input_shape = input_shape
        self.num_class = num_class
        self.model_name = model_name
        self._config(**kwargs)

    # =========================================================================
    #  Hyper-parameters
    # =========================================================================
    def _config(self, **kwargs):
        """Echo state network: fixed random reservoir + ridge-regression readout.

        input_shape = (n_samples, n_signals), time first as an RNN expects.
        """
        self.arch = 'esn'             # 'esn', or 'no_reservoir' (ablation)
        self.units = 54               # reservoir size N
        self.spectral_radius = 0.341
        self.leak = 0.752
        self.input_scale = 0.0124
        self.connectivity = 0.315     # fraction of non-zero recurrent weights
        self.reservoir_seed = 0       # the same reservoir for every subject and fold
        self.washout = 50             # samples (0.5 s at 100 Hz) dropped before pooling
        self.skip = False             # also pass the input signals to the pooling
        self.pool = 'logvar'          # 'logvar', or 'mean' (ablation)
        self.standardise = True       # standardise pooled features before the readout
        self.ridge_lambda = 64.49
        self.n_samples, self.n_signals = self.input_shape

        # Keep this last: it lets the experiment config override the defaults.
        for k in kwargs.keys():
            self.__setattr__(k, kwargs[k])

    # =========================================================================
    #  The network
    # =========================================================================
    def build(self, print_summary=True, load_weights=False):
        """Reservoir (fixed) -> log-variance over time -> linear readout -> softmax.

        The readout weights are set in closed form by fit(); the standardisation
        is folded into them, so the graph needs no separate layer for it.
        """
        input1 = layers.Input(shape=self.input_shape)

        if self.arch == 'esn':
            cell = ReservoirCell(self.units, self.spectral_radius, self.leak,
                                 self.input_scale, self.connectivity, self.reservoir_seed,
                                 name='reservoir_cell')
            x = layers.RNN(cell, return_sequences=True, unroll=True, trainable=False,
                           name='reservoir')(input1)
            if self.skip:
                x = layers.Concatenate(name='states_and_inputs')([x, input1])
        elif self.arch == 'no_reservoir':
            x = input1
        else:
            raise ValueError('Unknown arch: {}'.format(self.arch))

        if self.pool == 'logvar':
            x = LogVariancePool(self.washout, name='logvar_pool')(x)
        elif self.pool == 'mean':
            x = MeanPool(self.washout, name='mean_pool')(x)
        else:
            raise ValueError('Unknown pool: {}'.format(self.pool))

        x = layers.Dense(self.num_class, name='readout')(x)
        softmax = layers.Activation('softmax', name='softmax')(x)
        model = Model(inputs=input1, outputs=softmax, name=self.model_name)

        if print_summary:
            model.summary()
        if load_weights:
            print('loading weights from', self.weights_dir)
            model.load_weights(self.weights_dir)
        return model

    # =========================================================================
    #  Training: closed-form ridge readout instead of the gradient loop
    # =========================================================================
    def fit(self, X_train, y_train, X_val, y_val):
        """Fit the readout by ridge regression on the training fold.

        Targets are +1 for the true class and -1 otherwise; the bias is not
        penalised. Features are standardised with training-fold statistics and
        the standardisation is folded into the Dense weights. The validation
        fold is only scored; its mean squared error is logged as val_loss.
        """
        model = self.build(print_summary=self.print_summary)
        super().compile(model=model)
        pool_name = 'logvar_pool' if self.pool == 'logvar' else 'mean_pool'
        features = Model(model.input, model.get_layer(pool_name).output)
        f_tr = features.predict(X_train, batch_size=256).astype(np.float64)
        f_va = features.predict(X_val, batch_size=256).astype(np.float64)

        if self.standardise:
            mu, sd = f_tr.mean(axis=0), f_tr.std(axis=0) + 1e-8
        else:
            mu, sd = np.zeros(f_tr.shape[1]), np.ones(f_tr.shape[1])
        y_tr, y_va = np.asarray(y_train).astype(int), np.asarray(y_val).astype(int)
        targets = 2.0 * np.eye(self.num_class)[y_tr] - 1.0
        a = np.c_[(f_tr - mu) / sd, np.ones(len(f_tr))]
        penalty = self.ridge_lambda * np.eye(a.shape[1])
        penalty[-1, -1] = 0.0
        w = np.linalg.solve(a.T @ a + penalty, a.T @ targets)
        kernel = w[:-1] / sd[:, None]
        bias = w[-1] - (mu / sd) @ w[:-1]
        model.get_layer('readout').set_weights([kernel.astype(np.float32),
                                                bias.astype(np.float32)])

        scores_tr, scores_va = f_tr @ kernel + bias, f_va @ kernel + bias
        t_va = 2.0 * np.eye(self.num_class)[y_va] - 1.0
        log = {'epoch': 0,
               'train_acc': float((scores_tr.argmax(1) == y_tr).mean()),
               'val_acc': float((scores_va.argmax(1) == y_va).mean()),
               'val_loss': float(((scores_va - t_va) ** 2).mean()),
               'ridge_lambda': self.ridge_lambda}
        pd.DataFrame([log]).to_csv(self.csv_dir, index=False)
        model.save_weights(self.weights_dir)
        self.best_loss_weights = tf.constant([1.0])   # read by BaseModel.evaluate()
        print('ridge readout: train_acc {:.3f} val_acc {:.3f} val_mse {:.3f}'.format(
            log['train_acc'], log['val_acc'], log['val_loss']))

    # =========================================================================
    #  Provided for you -- custom training / evaluation steps
    # -------------------------------------------------------------------------
    #  SINGLE-TASK ONLY. Every step below assumes `self.model(x)` returns ONE
    #  tensor and that `self.loss` holds exactly one entry, `crossentropy`.
    #  NOTE: that `loss_weights` is accepted but unused here: with a single loss
    #  there is nothing to weight.
    # =========================================================================
    @tf.function
    def train_step(self, x, y, loss_weights):
        with tf.GradientTape() as tape:
            # NOTE: You may rewrite the following line to unpack multiple outputs if you go multi-task 
            # or use other loss functions. See MixNet for reference.
            y_logis = self.model(x, training=True)
            crossentropy_loss = self.loss.crossentropy(y, y_logis)
            losses = [crossentropy_loss]
            train_loss = tf.reduce_sum(crossentropy_loss)
            self.train_acc_metric.update_state(y, y_logis)
        grads = tape.gradient(train_loss, self.model.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_weights))
        logs = dict({'train_loss': train_loss})
        logs.update(dict(zip(['train_' + loss_name + '_loss'
                              for loss_name in self.loss_names], losses)))
        logs.update(dict({'train_acc': self.train_acc_metric.result()}))
        return logs, y_logis

    @tf.function
    def val_step(self, x, y, loss_weights):
        # NOTE: You may rewrite the following line to unpack multiple outputs if you go multi-task 
        # or use other loss functions. See MixNet for reference.
        y_logis = self.model(x, training=False)
        crossentropy_loss = self.loss.crossentropy(y, y_logis)
        losses = [crossentropy_loss]
        val_loss = tf.reduce_sum(crossentropy_loss)
        self.val_acc_metric.update_state(y, y_logis)
        logs = dict({'val_loss': val_loss})
        logs.update(dict(zip(['val_' + loss_name + '_loss'
                              for loss_name in self.loss_names], losses)))
        logs.update(dict({'val_acc': self.val_acc_metric.result()}))
        return logs, y_logis

    @tf.function
    def test_step(self, x, y, loss_weights):
        # NOTE: You may rewrite the following line to unpack multiple outputs if you go multi-task 
        # or use other loss functions. See MixNet for reference.
        y_logis = self.model(x, training=False)
        crossentropy_loss = self.loss.crossentropy(y, y_logis)
        losses = [crossentropy_loss]
        test_loss = tf.reduce_sum(crossentropy_loss)
        self.test_acc_metric.update_state(y, y_logis)
        logs = dict({'test_loss': test_loss})
        logs.update(dict(zip(['test_' + loss_name + '_loss'
                              for loss_name in self.loss_names], losses)))
        logs.update(dict({'test_acc': self.test_acc_metric.result()}))
        return logs, y_logis

    @tf.function
    def pred_step(self, x):
        # NOTE: You may rewrite the following line to unpack multiple outputs if you go multi-task 
        # or use other loss functions. See MixNet for reference.
        y_logis = self.model(x, training=False)
        return y_logis
