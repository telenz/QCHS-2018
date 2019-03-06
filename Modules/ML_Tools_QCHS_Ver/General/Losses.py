from __future__ import division
from keras import backend as K
from keras.layers import Input
from functools import partial, update_wrapper


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


def sOverBLossInvert(s_exp=None, b_exp=None):
    '''Define a loss function that calculates the significance based on fixed
    expected signal and expected background yields for a given batch size

    s_exp and b_exp are the number of recorded signal and background events,
    respectively (xsec*lumi).'''

    #@_patch_with_weights
    def sOverBLossInvert(y_true, y_pred):
        s, b = _get_s_b(s_exp, b_exp, y_true, y_pred)
        return b/(s+K.epsilon()) #Add the epsilon to avoid dividing by 0

    return sOverBLossInvert


def significanceLoss2Invert(s_exp=None, b_exp=None):
    '''Define a loss function that calculates the significance based on fixed
    expected signal and expected background yields for a given batch size

    s_exp and b_exp are the number of recorded signal and background events,
    respectively (xsec*lumi).'''

    #@_patch_with_weights
    def sigLoss2Invert(y_true, y_pred):
        s, b = _get_s_b(s_exp, b_exp, y_true, y_pred)
        # s += K.epsilon()
        # b += K.epsilon()
        return (s+b)/(s*s+K.epsilon()) #Add the epsilon to avoid dividing by 0

    return sigLoss2Invert


def asimovSignificanceLossInvert(syst_factr, s_exp=None, b_exp=None):
    '''Define a loss function that calculates the significance based on fixed
    expected signal and expected background yields for a given batch size

    s_exp and b_exp are the number of recorded signal and background events,
    respectively (xsec*lumi).'''

    #@_patch_with_weights
    def asimovSigLossInvert(y_true, y_pred):        
        # The problem with inner_loss is that it's not monotonic. 
        # For a given s, there's a b value at which the function is maximal,
        # after that value the function is falling and thus promoting more
        # background events to smaller loss values.

        # coding dataflow according to:
        # https://stackoverflow.com/questions/33712178/tensorflow-nan-bug/42497444#42497444

        s, b = _get_s_b(s_exp, b_exp, y_true, y_pred)
        s += K.epsilon()
        b += K.epsilon()

        # minimal background value for inner_loss to be on the safe side.
        def max_bkg(s):
            return K.pow(s, .65)*1./K.pow(syst_factr, 0.65)
        
        b_max = max_bkg(s)
        s_b_ok = K.greater(b_max, b)
        s_b_ok = K.print_tensor(s_b_ok, 's_b_ok=')
        
        sb = K.stack([s,b])
        sb = K.print_tensor(sb, 'sb=')

        # Asimov loss function
        def inner_loss(sb):
            s, b = sb[0], sb[1]
            spb = s+b
            b2 = b*b
            syst = syst_factr * b
            syst2 = syst*syst
            bpsyst2 = b+syst2            
            eps = K.epsilon()
            return 0.5/(
                spb * K.log((spb*bpsyst2+eps)/(b2+spb*syst2+eps)+eps)
                -b2/(syst2+eps) * K.log(1+syst2*s/(b*bpsyst2+eps))
            )
        
        def sOverSqrtSpB(sb):
            s, b = sb[0], sb[1]
            return (s+b)/(s*s+K.epsilon())

        # s/sqrt(s+b) is safe, as it is monotonic in s and b
        def safe_inner_loss(sb):    
            s_b_max = K.stack([s,b_max])
            s_b_max0p95 = K.stack([s,b_max*0.95])
            s_b_max1p05 = K.stack([s,b_max*1.05])
            # the norm factor makes loss steady at the switching point 
            norm = inner_loss(s_b_max) / sOverSqrtSpB(s_b_max)
            return norm*sOverSqrtSpB(sb)
        
        #safe_sb = K.switch(s_b_ok, sb, K.ones(sb.shape))
        return K.switch(s_b_ok, inner_loss(sb), safe_inner_loss(sb))

    return asimovSigLossInvert


def _get_s_b(s_exp, b_exp, y_true, y_pred):

    # unfold encoded weights
    weights = K.abs(y_true)
    s_weights = (weights + y_true)/2.
    b_weights = (weights - y_true)/2.

    s_exp_weight = s_exp/(K.abs(K.sum(s_weights))+K.epsilon())
    b_exp_weight = b_exp/(K.abs(K.sum(b_weights))+K.epsilon())

    s = s_exp_weight * K.sum(y_pred * s_weights)
    b = b_exp_weight * K.sum(y_pred * b_weights)
    return s, b
