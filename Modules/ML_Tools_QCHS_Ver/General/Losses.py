from __future__ import division
from keras import backend as K
from keras.layers import Input
from functools import partial, update_wrapper


def _wrapped_partial(func, *args, **kwargs):
    partial_func = partial(func, *args, **kwargs)
    update_wrapper(partial_func, func)
    return partial_func


def _patch_with_weights(func):
    weightsInput = Input(shape=(1,))
    func = _wrapped_partial(func, weights=weightsInput)
    func.weightsInput = Input(shape=(1,))
    return func


def patch_model_input(model, loss):
    weights = getattr(loss, 'weightsInput', None)
    if weights is not None:
        main_input = model.inputs[0]
        model.inputs = [main_input, weights]


def get_losses_dict(syst_factors, s_exp=None, b_exp=None):
    post_fix  = ('_S%.2e'%s_exp) if s_exp else ''
    post_fix += ('_B%.2e'%b_exp) if b_exp else ''
    for before, after in (('.', 'd'), ('+','p'), ('-','m')):
        post_fix = post_fix.replace(before, after)

    d = {
        'significanceLoss2Invert' + post_fix: significanceLoss2Invert(s_exp, b_exp),
        'sOverBLossInvert' + post_fix: sOverBLossInvert(s_exp, b_exp),
    }
    d.update({
        (
            ('asimovSignificanceLossInvert_Sys%s'%sf).replace('.','p')+post_fix,
                asimovSignificanceLossInvert(sf, s_exp, b_exp)
        )
        for sf in syst_factors
    })

    # for keras to find the find the functions on load, the need to be named
    for func_name, func in d.iteritems():
        func.__name__ = func_name

    return d


def significanceFull(s, b):
    pass


def asimovSignificanceFull(s, b, syst_factr):
    pass


def sOverBLossInvert(s_exp=None, b_exp=None):
    '''Define a loss function that calculates the significance based on fixed
    expected signal and expected background yields for a given batch size

    s_exp and b_exp are the number of recorded signal and background events,
    respectively (xsec*lumi).'''

    @_patch_with_weights
    def sOverBLossInvert(y_true, y_pred, weights):
        s, b = _get_s_b(s_exp, b_exp, y_true, y_pred, weights)
        return b/(s+K.epsilon()) #Add the epsilon to avoid dividing by 0

    return sOverBLossInvert


def significanceLoss2Invert(s_exp=None, b_exp=None):
    '''Define a loss function that calculates the significance based on fixed
    expected signal and expected background yields for a given batch size

    s_exp and b_exp are the number of recorded signal and background events,
    respectively (xsec*lumi).'''

    @_patch_with_weights
    def sigLoss2Invert(y_true, y_pred, weights):
        s, b = _get_s_b(s_exp, b_exp, y_true, y_pred, weights)
        return (s+b)/(s*s+K.epsilon()) #Add the epsilon to avoid dividing by 0

    return sigLoss2Invert


def asimovSignificanceLossInvert(syst_factr, s_exp=None, b_exp=None):
    '''Define a loss function that calculates the significance based on fixed
    expected signal and expected background yields for a given batch size

    s_exp and b_exp are the number of recorded signal and background events,
    respectively (xsec*lumi).'''

    @_patch_with_weights
    def asimovSigLossInvert(y_true, y_pred, weights):
        #Continuous version:
        #weights=y_true[:,1]
        #y_true=y_true[:,0]

        s, b = _get_s_b(s_exp, b_exp, y_true, y_pred, weights)
        eps = K.epsilon()  # Add the epsilon to avoid dividing by 0
        spb = s+b
        b2 = b*b
        syst = syst_factr * b
        syst2 = syst*syst + eps  # eps was needed almost always where syst2 was
        b2tsyst2 = b2*syst2
        bpb2tsyst2 = b+b2tsyst2

        return 0.5/(
            spb*K.log(spb*bpb2tsyst2/(b2+spb*b2tsyst2)+eps)
            -K.log(1+b/bpb2tsyst2*syst2*s)/syst2
        )

#        bpsyst2 = b+syst2
#
#        return 0.5/(
#            spb*K.log(spb*bpsyst2/(b2+spb*syst2)+eps)
#            -b2*K.log(1+syst2*s/(b*bpsyst2))/syst2
#        )

    return asimovSigLossInvert


def _get_s_b(s_exp, b_exp, y_true, y_pred, weights):
    #if not weights:
    #    import warnings
    #    warnings.warn('No weights. Loaded model from file?', RuntimeWarning)
    #    weights = 1.

    if '?' in str(weights):
        print('weights is Tensor-placeholder, continuing with weights=1.')
        weights = 1.

    s_weights = y_true * weights
    b_weights = (1-y_true) * weights

    s_exp_weight = s_exp/K.sum(s_weights) if s_exp else 1.
    b_exp_weight = b_exp/K.sum(b_weights) if b_exp else 1.

    s = s_exp_weight * K.sum(y_pred * s_weights)
    b = b_exp_weight * K.sum(y_pred * b_weights)
    return s, b
