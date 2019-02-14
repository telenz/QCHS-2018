from __future__ import division
from keras import backend as K


def get_losses_dict(syst_factors, s_tot=None, b_tot=None, weights=1):
    post_fix  = ('_S%f'%s_tot).replace('.', 'p') if s_tot else ''
    post_fix += ('_B%f'%b_tot).replace('.', 'p') if b_tot else ''

    d = {
        'significanceLoss2Invert' + post_fix:
            significanceLoss2Invert(s_tot, b_tot, weights)
    }
    d.update({
        (
            ('asimovSignificanceLossInvert_Sys%s'%sf).replace('.','p')+post_fix,
                asimovSignificanceLossInvert(sf, s_tot, b_tot, weights)
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


def significanceLoss2Invert(s_tot=None, b_tot=None, weights=1):
    '''Define a loss function that calculates the significance based on fixed
    expected signal and expected background yields for a given batch size

    s_tot and b_tot are the number of recorded signal and background events,
    respectively (xsec*lumi).'''

    def sigLoss2Invert(y_true,y_pred):
        s, b = _get_s_b(s_tot, b_tot, y_true, y_pred, weights)
        return b/(s*s+K.epsilon()) #Add the epsilon to avoid dividing by 0

    return sigLoss2Invert


def asimovSignificanceLossInvert(syst_factr, s_tot=None, b_tot=None, weights=1):
    '''Define a loss function that calculates the significance based on fixed
    expected signal and expected background yields for a given batch size

    s_tot and b_tot are the number of recorded signal and background events,
    respectively (xsec*lumi).'''

    def asimovSigLossInvert(y_true, y_pred):
        #Continuous version:
        #weights=y_true[:,1]
        #y_true=y_true[:,0]

        s, b = _get_s_b(s_tot, b_tot, y_true, y_pred, weights)
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


def _get_s_b(s_tot, b_tot, y_true, y_pred, weights):
    s_weights = y_true * weights
    b_weights = (1-y_true) * weights

    s_tot_weight = s_tot/K.sum(s_weights) if s_tot else 1.
    b_tot_weight = b_tot/K.sum(b_weights) if b_tot else 1.

    s = s_tot_weight * K.sum(y_pred * s_weights)
    b = b_tot_weight * K.sum(y_pred * b_weights)
    return s, b
