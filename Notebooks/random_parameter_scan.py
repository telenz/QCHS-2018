
# coding: utf-8

# # Heiner classifier
# ReLU activation, 4 layers, 100 neurons per layer
# Validation score use ensemble of 10 models weighted by loss
#
# ### TODO:
#
# - What metrics to use during training?
# - Which plots are needed?
#

# ### Import modules

# In[1]:


from __future__ import division
import tensorflow as tf
import sys
import os
sys.path.append('../')
from Modules.Basics import *
from Modules.Class_Basics import *
from keras.models import load_model

dirLoc = '../Data/'
name = "weights/ReLU_Baseline_CLR_AsimovLoss"
#dirLoc = '../Data_3Fold/'
#name = "weights/ReLU_Baseline_CLR_AsimovLoss_3Fold"

name_pretrain = name.replace('weights/', 'weights/PRETRAIN_')

# ## Options

# In[2]:


with open(dirLoc + 'features.pkl', 'rb') as fin:
    classTrainFeatures = pickle.load(fin)

# In[33]:


nSplits = 10
patience = 50
maxEpochs = 200
preTrainMaxEpochs = 10

ensembleSize = 10
ensembleMode = 'loss'

compileArgs = {'loss':'binary_crossentropy', 'optimizer':'adam'}
trainParams = {'epochs' : 1, 'batch_size' : 256, 'verbose' : 0}
modelParams = {'version':'modelRelu', 'nIn':len(classTrainFeatures), 'compileArgs':compileArgs, 'mode':'classifier'}

plot_while_training=False

print ("\nTraining on", len(classTrainFeatures), "features:", [var for var in classTrainFeatures])


# b = 100000  # test
# t = 250000  # training
# v = 450000  # test

# sig weight sum:  1383.97719575 in 550000
# bkg weight sum:  821999.696026 in 550000

# expected signal and background numbers
s_exp = 691.988607714
b_exp = 410999.847322
print ('_S%.2e'%s_exp).replace('.', 'd').replace('+','p').replace('-','m')
print ('_B%.2e'%b_exp).replace('.', 'd').replace('+','p').replace('-','m')

new_loss_functions = get_losses_dict([0.001, 0.1, 0.3, 0.5], s_exp, b_exp)


# ## Import data

# In[4]:


trainData          = BatchYielder(h5py.File(dirLoc + 'train.hdf5', "r"))
trainDataTargetMod = BatchYielderTargetMod(h5py.File(dirLoc + 'train.hdf5', "r"))
nSplits = trainData.nFolds



# ## Train classifier



def _get_s_b(s_exp, b_exp, y_true, y_pred):

    # unfold encoded weights
    weights = K.abs(y_true)
    s_weights = (weights + y_true)/2.
    b_weights = (weights - y_true)/2.

    s_exp_weight = s_exp/(K.sum(s_weights)+K.epsilon())
    b_exp_weight = b_exp/(K.sum(b_weights)+K.epsilon())

    s = s_exp_weight * K.sum(y_pred * s_weights)
    b = b_exp_weight * K.sum(y_pred * b_weights)
    return s, b


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
        # sb = K.print_tensor(sb, 'sb=')

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

        #safe_sb = K.switch(s_b_ok, sb, K.ones(sb.shape))
        return K.switch(s_b_ok, inner_loss(sb), safe_inner_loss(sb))

    return asimovSigLossInvert


def get_losses_dict(syst_factors, s_exp=None, b_exp=None):
    post_fix  = ('_S%.2e'%s_exp) if s_exp else ''
    post_fix += ('_B%.2e'%b_exp) if b_exp else ''
    for before, after in (('.', 'd'), ('+','p'), ('-','m')):
        post_fix = post_fix.replace(before, after)

    d = {}
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


new_loss_functions.update(get_losses_dict([0.001, 0.1, 0.3, 0.5], s_exp, b_exp))



# recompile with new loss
def go_train(name, **compile_kws):
    newModelArgs = modelParams.copy()
    #loss = new_loss_functions['asimovSignificanceLossInvert_Sys0p5_S2d01ep03_B1d20ep06']
    loss = new_loss_functions['asimovSignificanceLossInvert_Sys0p5_S6d92ep02_B4d11ep05']
    #loss = new_loss_functions['sOverBLossInvert_S2d01ep03_B1d20ep06']
    #loss = new_loss_functions['sMinusBLossInvert_S2d01ep03_B1d20ep06']
    newModelArgs['compileArgs']['loss'] = loss
    newModelArgs['compileArgs'].update(compile_kws)
    trainParams['batch_size'] = 8192

    results, histories = batchTrainClassifier(
        trainDataTargetMod, nSplits, getModel, newModelArgs,
        trainParams, trainOnWeights=True, maxEpochs=maxEpochs,
        cosAnnealMult=2, reduxDecay=1,  # this line added
        patience=patience, verbose=1, amsSize=250000, binary=True,
        plotLR=False, # plotMomentum=plot_while_training,
        plot=False,
        stopIfStallingTest=30,
    )

    with open(name+'.pkl', 'w') as f:
        pickle.dump((results, histories), f)

    return np.median(list(h['val_loss'][-1] for h in histories))



# Random search go go go....
import scipy.stats
import random
import time


# specify parameters and distributions to sample from
rnd_do = scipy.stats.uniform(0., 0.4)
rnd_l2 = scipy.stats.uniform(0., 0.4)
rnd_lr = scipy.stats.uniform(1., 1.)
rnd_bn = ['none', 'pre', 'post']

if __name__ == '__main__':
    while True:
        compile_kws = dict(
            lr = np.power(10., -rnd_lr.rvs()),
            do = rnd_do.rvs(),
            l2 = rnd_l2.rvs(),
            #bn = random.choice(rnd_bn),
        )
        name = 'training '+ time.ctime() + '___ %s' % compile_kws
        print('='*50)
        print('starting training:')
        print(name)
        print('='*50)
        go_train(name, **compile_kws)
        print('='*50)
        print('done training:')
        print(name)
        print('='*50)
