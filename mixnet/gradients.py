import tensorflow as tf
import numpy as np
from mixnet.utils import dotdict
from abc import abstractmethod

"""
class HistoricalTangentSlope(object):
    '''
    Huy Phan version
    '''
    def __init__(self, average_win=100, hist_size=100):
        self.train_loss = np.array([])
        self.valid_loss = np.array([])
        self.smoothed_train_loss = np.array([])
        self.smoothed_valid_loss = np.array([])
        self.average_win = average_win
        self.hist_size = hist_size
        self.prev_best_train_loss = None
        self.prev_best_valid_loss = None
        self.train_slope_ref = None
        self.valid_slope_ref = None

    def add_point(self, train_loss_val, valid_loss_val):
        self.train_loss = np.append(self.train_loss, [train_loss_val])
        self.valid_loss = np.append(self.valid_loss, [valid_loss_val])
        self._moving_average_smoothing(mode='both_loss')
        
    def add_point_train_loss(self, train_loss_val):
        self.train_loss = np.append(self.train_loss, [train_loss_val])
        self._moving_average_smoothing(mode='train_loss')
        
    def add_point_valid_loss(self, valid_loss_val):
        self.valid_loss = np.append(self.valid_loss, [valid_loss_val])
        self._moving_average_smoothing(mode='valid_loss')

    def _moving_average_smoothing(self, mode='both_loss'):
        if mode == 'both_loss' or mode == 'train_loss':
            size = len(self.train_loss)
            smoothed_val = np.mean(self.train_loss[np.max([0, (size - self.average_win)]):])
            self.smoothed_train_loss = np.append(self.smoothed_train_loss, [smoothed_val])
        if mode == 'both_loss' or mode == 'valid_loss':
            size = len(self.valid_loss)
            smoothed_val = np.mean(self.valid_loss[np.max([0, (size - self.average_win)]):])
            self.smoothed_valid_loss = np.append(self.smoothed_valid_loss, [smoothed_val])

    def _line_fit(self, train_loss, valid_loss):
        size = len(train_loss)
        train_val = train_loss[np.max([0, (size - self.hist_size)]):]
        t = np.arange(np.min([len(train_val), self.hist_size]))
        p_train = np.polyfit(t, train_val, 1)
        assert (len(p_train) == 2)
        p_train = p_train[0]

        size = len(valid_loss)
        valid_val = valid_loss[np.max([0, (size - self.hist_size)]):]
        t = np.arange(np.min([len(valid_val), self.hist_size]))
        p_valid = np.polyfit(t, valid_val, 1)
        assert (len(p_valid) == 2)
        p_valid = p_valid[0]

        return p_train, p_valid

    def compute_weight(self):
        N = len(self.smoothed_train_loss)
        N_valid = len(self.smoothed_valid_loss)
        if (self.train_slope_ref is None and self.valid_slope_ref is None):
            train_loss = self.smoothed_train_loss[max([0, N - self.hist_size - 1]): -1]
            valid_loss = self.smoothed_valid_loss[max([0, N_valid - self.hist_size - 1]): -1]
            self.train_slope_ref, self.valid_slope_ref = self._line_fit(train_loss, valid_loss)

        cur_train_loss = self.smoothed_train_loss[max([0, N - self.hist_size]):]
        cur_valid_loss = self.smoothed_valid_loss[max([0, N_valid - self.hist_size]):]
        train_slope, valid_slope = self._line_fit(cur_train_loss, cur_valid_loss)

        Ok = (valid_slope - train_slope) - (self.valid_slope_ref - self.train_slope_ref)
        Gk = valid_slope - self.valid_slope_ref

        w = Gk / (Ok * Ok + 1e-6)
        if (w < 0.):
            w = 0.

        # update references
        if (self.valid_slope_ref > valid_slope):
            self.train_slope_ref = train_slope
            self.valid_slope_ref = valid_slope

        return w, Gk, Ok
"""

class BaseGradientPolicy(object):
    def __init__(self, batch_size=100, valid_batch_size=100, overlap=0.25):
        self.train_loss = np.array([])
        self.valid_loss = np.array([])
        self.smoothed_train_loss = np.array([])
        self.smoothed_valid_loss = np.array([])
        self.batch_size = batch_size
        self.valid_batch_size = valid_batch_size
        self.hist_size = batch_size
        self.overlap = 1 + overlap

    def add_point(self, train_loss_val, valid_loss_val):
        self.train_loss = np.append(self.train_loss, [train_loss_val])
        self.valid_loss = np.append(self.valid_loss, [valid_loss_val])
        self._moving_average_smoothing(mode='both_loss')
        
    def add_point_train_loss(self, train_loss_val):
        self.train_loss = np.append(self.train_loss, [train_loss_val])
        self._moving_average_smoothing(mode='train_loss')
        
    def add_point_valid_loss(self, valid_loss_val):
        self.valid_loss = np.append(self.valid_loss, [valid_loss_val])
        self._moving_average_smoothing(mode='valid_loss')

    def _moving_average_smoothing(self, mode='both_loss'):
        if mode == 'both_loss' or mode == 'train_loss':
            smoothed_val = np.mean(self.train_loss[-int(self.batch_size*self.overlap):])
            self.smoothed_train_loss = np.append(self.smoothed_train_loss, [smoothed_val])
        if mode == 'both_loss' or mode == 'valid_loss':
            smoothed_val = np.mean(self.valid_loss[-int(self.valid_batch_size*self.overlap):])
            self.smoothed_valid_loss = np.append(self.smoothed_valid_loss, [smoothed_val])

    def _line_fit(self, train_loss, valid_loss):
        t = np.arange(len(train_loss))
        p_train = np.polyfit(t, train_loss, 1)
        assert (len(p_train) == 2)
        p_train = p_train[0]
        
        t = np.arange(len(valid_loss))
        p_valid = np.polyfit(t, valid_loss, 1)
        assert (len(p_valid) == 2)
        p_valid = p_valid[0]

        return p_train, p_valid

    @abstractmethod
    def compute_weight(self):
        '''compute and return w, Gk, Ok'''
        pass
        
class HistoricalTangentSlope(BaseGradientPolicy):
    '''
    Update from Huy Phan version
    '''
    def __init__(self, batch_size=100, valid_batch_size=100):
        super().__init__(batch_size=batch_size, valid_batch_size=valid_batch_size)
        self.train_slope_ref = None
        self.valid_slope_ref = None

    def compute_weight(self):
        if (self.train_slope_ref is None and self.valid_slope_ref is None):
            train_loss = self.smoothed_train_loss[-self.hist_size-1:-1]
            valid_loss = self.smoothed_valid_loss[-self.hist_size-1:-1]
            self.train_slope_ref, self.valid_slope_ref = self._line_fit(train_loss, valid_loss)

        cur_train_loss = self.smoothed_train_loss[-self.hist_size:]
        cur_valid_loss = self.smoothed_valid_loss[-self.hist_size:]
        train_slope, valid_slope = self._line_fit(cur_train_loss, cur_valid_loss)

        Ok = (valid_slope - train_slope) - (self.valid_slope_ref - self.train_slope_ref)
        Gk = valid_slope - self.valid_slope_ref

        w = Gk / (Ok * Ok + 1e-6)
        if (w < 0.):
            w = 0.

        # update references
        if (self.valid_slope_ref > valid_slope):
            self.train_slope_ref = train_slope
            self.valid_slope_ref = valid_slope

        return w, Gk, Ok

class TangentSlope(BaseGradientPolicy):
    '''
    Gradient slope
    '''
    def __init__(self, batch_size=100, valid_batch_size=100):
        super().__init__(batch_size=batch_size, valid_batch_size=valid_batch_size)

    def compute_weight(self):
        cur_train_loss = self.smoothed_train_loss[-self.hist_size:]
        cur_valid_loss = self.smoothed_valid_loss[-self.hist_size:]
        train_slope, valid_slope = self._line_fit(cur_train_loss, cur_valid_loss)

        Ok = valid_slope - train_slope
        Gk = valid_slope

        w = Gk / (Ok * Ok + 1e-6)
        if (w < 0.):
            w = 0.

        return w, Gk, Ok

class SecantSlope(BaseGradientPolicy):
    '''
    sec slope
    '''
    def __init__(self, batch_size=100, valid_batch_size=100):
        super().__init__(batch_size=batch_size, valid_batch_size=valid_batch_size)
        self.train_prev_point = None
        self.valid_prev_point = None
        
    def compute_weight(self):
        if (self.train_prev_point is None and self.valid_prev_point is None):
            train_loss = self.smoothed_train_loss[-self.hist_size-1:-1]
            valid_loss = self.smoothed_valid_loss[-self.hist_size-1:-1]
            self.train_prev_point, self.valid_prev_point = train_loss.mean(), valid_loss.mean()

        cur_train_loss = self.smoothed_train_loss[-self.hist_size:]
        cur_valid_loss = self.smoothed_valid_loss[-self.hist_size:]
        train_cur_point, valid_cur_point = cur_train_loss.mean(), cur_valid_loss.mean()

        Ok = (valid_cur_point - train_cur_point) - (self.valid_prev_point - self.train_prev_point)
        Gk = valid_cur_point - self.valid_prev_point

        w = Gk / (Ok * Ok + 1e-6)
        if (w < 0.):
            w = 0.

        # update references
        if (self.valid_prev_point > valid_cur_point):
            self.train_prev_point = train_cur_point
            self.valid_prev_point = valid_cur_point

        return w, Gk, Ok    
    
class Threshold(BaseGradientPolicy):
    '''
    loss threshold
    '''
    def __init__(self, batch_size=100, valid_batch_size=100):
        super().__init__(batch_size=batch_size, valid_batch_size=valid_batch_size)
        
    def compute_weight(self):
        train_loss = self.smoothed_train_loss[-self.hist_size:]
        valid_loss = self.smoothed_valid_loss[-self.hist_size:]

        Ok = valid_loss - train_loss
        Gk = valid_loss

        w = np.mean(Gk / (Ok * Ok + 1e-6))
        if (w < 0.):
            w = 0.

        return w, Gk.mean(), Ok.mean()

class BlendingRatio(BaseGradientPolicy):
    '''
    Scale to norm losses
    '''
    def __init__(self, batch_size=100, valid_batch_size=100):
        super().__init__(batch_size=batch_size, valid_batch_size=valid_batch_size)

    def compute_weight(self):
        # cur_train_loss = self.smoothed_train_loss[-self.hist_size:]
        cur_valid_loss = self.smoothed_valid_loss[-self.hist_size:]

        w = np.mean(cur_valid_loss)

        Ok = None
        Gk = None

        return w, Gk, Ok

class GradientBlending(object):
    def __init__(self, n=3, init_weights=None, adaptive_masked=True):
        self.n_gradient    = n if init_weights is None else len(init_weights)
        # self.init_weights = [1/n for i in range(n)] if init_weights is None else init_weights
        self.adaptive_masked = [False]*self.n_gradient if not adaptive_masked \
                            else [True]*self.n_gradient if adaptive_masked==True \
                            else adaptive_masked
        self.not_adapt_idx = [i for i, x in enumerate(self.adaptive_masked) if not x]
        self.n_adaptive    = n if adaptive_masked is None else sum(self.adaptive_masked)
        self.init_weights  = [1/self.n_adaptive if self.adaptive_masked[i] 
                             else 1. for i in range(self.n_gradient)] if init_weights is None else init_weights
        self.policy_map = dotdict({'HistoricalTangentSlope': HistoricalTangentSlope,
                                   'TangentSlope': TangentSlope,
                                   'SecantSlope': SecantSlope,
                                   'Threshold': Threshold,
                                   'BlendingRatio': BlendingRatio})
        
    def build(self, policy, batch_size, valid_batch_size):
        try:
            self.policy = self.policy_map[policy]
        except:
            self.policy = HistoricalTangentSlope
        print('... build adaptive policy:', self.policy)
        self.gradient = [self.policy(batch_size=batch_size, valid_batch_size=valid_batch_size) for i in range(self.n_gradient)]

    def add_point_train_loss(self, *args):
        if len(args) != self.n_gradient:
            raise Exception('len of args not match with `n`: ', len(args), '!=', self.n_gradient)
        for i, loss in enumerate(args):
            if self.adaptive_masked[i]:
                self.gradient[i].add_point_train_loss(loss)

    def add_point_valid_loss(self, *args):
        if len(args) != self.n_gradient:
            raise Exception('len of args not match with `n`: ', len(args), '!=', self.n_gradient)
        for i, loss in enumerate(args):
            if self.adaptive_masked[i]:
                self.gradient[i].add_point_valid_loss(loss)

    def compute_weight(self):
        w = []
        for i in range(self.n_gradient):
            if self.adaptive_masked[i]:
                w_, g_, o_ = self.gradient[i].compute_weight()
                w.append(w_)
        return w

    def compute_adaptive_weight(self, to_tensor=True):
        w_list = self.compute_weight()
        if self.policy.__name__ == 'BlendingRatio':
            w_min = np.min(w_list)
            if (w_min != 0.):
                zeros = [1e-5]*self.n_gradient
                weights = list(map(max, w_min/w_list, zeros))
                for idx in self.not_adapt_idx:
                    weights.insert(idx, self.init_weights[idx])
            else:
                weights = self.init_weights
        else:
            #Huy
            w_sum = np.sum(w_list)
            if (w_sum != 0.):
                zeros = [1e-5]*self.n_gradient
                weights = list(map(max, w_list/w_sum, zeros))
                for idx in self.not_adapt_idx:
                    weights.insert(idx, self.init_weights[idx])
            else:
                weights = self.init_weights
                
        if to_tensor:
            weights = tf.constant((weights), dtype='float32')
            
        return weights