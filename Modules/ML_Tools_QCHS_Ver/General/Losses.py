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


def asimovSignificanceLossInvert_OLD(syst_factr, s_exp=None, b_exp=None):
    '''Define a loss function that calculates the significance based on fixed
    expected signal and expected background yields for a given batch size

    s_exp and b_exp are the number of recorded signal and background events,
    respectively (xsec*lumi).'''

    #@_patch_with_weights
    def asimovSigLossInvert(y_true, y_pred):
        #Continuous version:
        #weights=y_true[:,1]
        #y_true=y_true[:,0]

        s, b = _get_s_b(s_exp, b_exp, y_true, y_pred)
        #s = K.abs(s - K.epsilon()) + K.epsilon()
        #b = K.abs(b - K.epsilon()) + K.epsilon()
        #s += K.epsilon()
        #b += K.epsilon()
        spb = s+b
        b2 = b*b
        syst = syst_factr * b
        syst2 = syst*syst
        bpsyst2 = b+syst2

        #return 0.5 / K.log(
        #     K.pow(spb*bpsyst2/(b2+spb*syst2), spb)
        #    *K.pow(1 + syst2*s/(b2+b*syst2),   -b2/(syst2+K.epsilon()))
        #)

        return 0.5/(
            spb * K.log((spb*bpsyst2+K.epsilon())/(b2+spb*syst2+K.epsilon())+K.epsilon())
            -b2/(syst2+K.epsilon()) * K.log(1+syst2*s/(b*bpsyst2+K.epsilon()))
        )

        #bOverS = b/(s+K.epsilon())

        #l_val *= K.relu(-K.relu(bOverS-1000)+1)
        #bOverS *= K.relu(-K.relu(-bOverS+1001)+1)  # on for x>a

        #return l_val #+ bOverS*100


#        return 0.5 / K.log(
#             K.pow(spb*bpsyst2/(b2+spb*syst2), spb)
#            *K.pow(1 + syst2*s/(b2+b*syst2),   -b2/(syst2+K.epsilon()))
#        )

#        return 0.5/(
#            spb * K.log(spb*bpsyst2/(b2+spb*syst2+K.epsilon())+K.epsilon())
#            -b2/(syst2+K.epsilon()) * K.log(1+syst2*s/(b*bpsyst2+K.epsilon()))
#        )

#        b2tsyst2 = b2*syst2
#        bpb2tsyst2 = b+b2tsyst2
#        return 0.5/(
#            spb*K.log(spb*bpb2tsyst2/(b2+spb*b2tsyst2+K.epsilon())+K.epsilon())
#            -b2*K.log(1+b/(bpb2tsyst2+K.epsilon())*syst2*s)/(syst2+K.epsilon())
#        )
#
#        return 0.5/(
#            spb*K.log(spb*(b+syst2)/(b2+spb*syst2+K.epsilon())+K.epsilon())
#            -b2*K.log(1+syst2*s/(b*(b+syst2)+K.epsilon()))/(syst2+K.epsilon())
#        )
#
#        sigB=syst_factr*b
#        return 0.5/(
#            (s+b)*K.log((s+b)*(b+sigB*sigB)/(b*b+(s+b)*sigB*sigB+K.epsilon())+K.epsilon())
#            -b*b*K.log(1+sigB*sigB*s/(b*(b+sigB*sigB)+K.epsilon()))/(sigB*sigB+K.epsilon())
#        )
#
#        return 1./(2*((s+b)*K.log((s+b)*(b+sigB*sigB)/(b*b+(s+b)*sigB*sigB+K.epsilon())+K.epsilon())-b*b*K.log(1+sigB*sigB*s/(b*(b+sigB*sigB)+K.epsilon()))/(sigB*sigB+K.epsilon()))) #Add the epsilon to avoid dividing by 0




    return asimovSigLossInvert

def asimovSignificanceLossInvert(syst_factr, s_exp=None, b_exp=None):
    '''Define a loss function that calculates the significance based on fixed
    expected signal and expected background yields for a given batch size

    s_exp and b_exp are the number of recorded signal and background events,
    respectively (xsec*lumi).'''

    #@_patch_with_weights
    def asimovSigLossInvert(y_true, y_pred):
        s, b = _get_s_b(s_exp, b_exp, y_true, y_pred)
        
        
        #s = K.print_tensor(s, 's=')
        #b = K.print_tensor(b, 'b=')

        # coding dataflow according to:
        # https://stackoverflow.com/questions/33712178/tensorflow-nan-bug/42497444#42497444

        # sanity table: 
        # s_min | b_max
        # 1     | 200
        # 10    | 1000
        # 100   | 4000
        # 1000  | 20000
        # 10000 | 100000
        
        s_b_ok = K.any(K.stack([
            K.all(K.stack([K.greater(200.,    b), K.greater(s, 1.)])),
            K.all(K.stack([K.greater(1000.,   b), K.greater(s, 10.)])),
            K.all(K.stack([K.greater(4000.,   b), K.greater(s, 100.)])),
            K.all(K.stack([K.greater(20000.,  b), K.greater(s, 1000.)])),
            K.all(K.stack([K.greater(100000., b), K.greater(s, 10000.)])),
        ]))
        s_b_ok = K.print_tensor(s_b_ok, 's_b_ok=')
        
        sb = K.stack([s,b])
        sb = K.print_tensor(sb, 'sb=')
        
        def inner_loss(sb):
            s, b = sb[0], sb[1]
            spb = s+b
            b2 = b*b
            syst = syst_factr * b
            syst2 = syst*syst
            bpsyst2 = b+syst2            
            return 0.5/(
                spb * K.log((spb*bpsyst2+K.epsilon())/(b2+spb*syst2+K.epsilon())+K.epsilon())
                -b2/(syst2+K.epsilon()) * K.log(1+syst2*s/(b*bpsyst2+K.epsilon()))
            )
        
            #return 0.5 / K.log(
            #     K.pow(spb*bpsyst2/(b2+spb*syst2), spb)
            #    *K.pow(1 + syst2*s/(b2+b*syst2),   -b2/(syst2+K.epsilon()))
            #)
        
        def safe_inner_loss(sb):
            s, b = sb[0], sb[1]
            return (s+b)/(s*s+K.epsilon())   # s/sqrt(s+b) is safe enough
        
        safe_sb = K.switch(s_b_ok, sb, K.ones(sb.shape))
        return K.switch(s_b_ok, inner_loss(safe_sb), safe_inner_loss(sb))
        
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
