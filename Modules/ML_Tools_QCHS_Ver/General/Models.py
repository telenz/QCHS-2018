from __future__ import division

from keras.models import Model, model_from_json, load_model
from keras.layers import Dense, Activation, AlphaDropout, Dropout, BatchNormalization, PReLU, Input
from keras.optimizers import *
from keras.regularizers import *
from keras.models import Sequential
from keras import backend as K

from Modules.ML_Tools_QCHS_Ver.General.Activations import *


def getOptimizer(compileArgs):
    if 'lr' not in compileArgs: compileArgs['lr'] = 0.001
    if compileArgs['optimizer'] == 'adam':
        if 'amsgrad' not in compileArgs: compileArgs['amsgrad'] = False
        if 'beta_1' not in compileArgs: compileArgs['beta_1'] = 0.9
        optimizer = Adam(lr=compileArgs['lr'], beta_1=compileArgs['beta_1'], beta_2=0.999, epsilon=1e-08, decay=0.0, amsgrad=compileArgs['amsgrad'])

    if compileArgs['optimizer'] == 'sgd':
        if 'momentum' not in compileArgs: compileArgs['momentum'] = 0.9
        if 'nesterov' not in compileArgs: compileArgs['nesterov'] = False
        optimizer = SGD(lr=compileArgs['lr'], momentum=compileArgs['momentum'], decay=0.0, nesterov=compileArgs['nesterov'])

    return optimizer


def getModel(version, nIn, compileArgs, mode, nOut=1):
    
    K.clear_session()
    
    #this is a hack postponing refactoring
    class MyMod(list):
        def add(self, obj):
            self.append(obj)
    model = MyMod()
    model.add(Input(shape=(nIn,)))

    depth = compileArgs.get('depth', 3)
    width = compileArgs.get('width', 100)
    do = compileArgs.get('do', False)
    bn = compileArgs.get('bn', False)
    reg = l2(compileArgs['l2']) if 'l2' in compileArgs else None

    if "modelRelu" in version:
        model.add(Dense(width, input_dim=nIn))
        if bn == 'pre': model.add(BatchNormalization())
        model.add(Activation('relu'))
        if bn == 'post': model.add(BatchNormalization())
        if do: model.add(Dropout(do))
        for i in range(depth):
            model.add(Dense(width))
            if bn == 'pre': model.add(BatchNormalization())
            model.add(Activation('relu'))
            if bn == 'post': model.add(BatchNormalization())
            if do: Dropout(do)

    if "modelPrelu" in version:
        model.add(Dense(width, input_dim=nIn, kernel_initializer='he_normal', kernel_regularizer=reg))
        if bn == 'pre': model.add(BatchNormalization())
        model.add(PReLU())
        if bn == 'post': model.add(BatchNormalization())
        if do: model.add(Dropout(do))
        for i in range(depth):
            model.add(Dense(width, kernel_initializer='he_normal', kernel_regularizer=reg))
            if bn == 'pre': model.add(BatchNormalization())
            model.add(PReLU())
            if bn == 'post': model.add(BatchNormalization())
            if do: Dropout(do)

    elif "modelSelu" in version:
        model.add(Dense(width, input_dim=nIn, kernel_initializer='lecun_normal', kernel_regularizer=reg))
        model.add(Activation('selu'))
        if do: model.add(AlphaDropout(do))
        for i in range(depth):
            model.add(Dense(width, kernel_initializer='lecun_normal', kernel_regularizer=reg))
            model.add(Activation('selu'))
            if do: model.add(AlphaDropout(do))

    elif "modelSwish" in version:
        model.add(Dense(width, input_dim=nIn, kernel_initializer='he_normal', kernel_regularizer=reg))
        if bn == 'pre': model.add(BatchNormalization())
        model.add(Activation(swish))
        if bn == 'post': model.add(BatchNormalization())
        if do: model.add(Dropout(do))
        for i in range(depth):
            model.add(Dense(width, kernel_initializer='he_normal', kernel_regularizer=reg))
            if bn == 'pre': model.add(BatchNormalization())
            model.add(Activation(swish))
            if bn == 'post': model.add(BatchNormalization())
            if do: Dropout(do)

    if 'class' in mode:
        if nOut == 1:
            model.add(Dense(1, activation='sigmoid', kernel_initializer='glorot_normal'))
        else:
            model.add(Dense(nOut, activation='softmax', kernel_initializer='glorot_normal'))

    elif 'regress' in mode:
        model.add(Dense(nOut, activation='linear', kernel_initializer='glorot_normal'))

    # instead of an actual model, 'model' is only a list so far
    # it's instantiated in a special way in order to propagate the weights to the loss function
    inputLayer = model.pop(0)
    lastLayer = inputLayer
    for l in model:
        lastLayer = l(lastLayer)

    # prep loss funtion to eat weights and make model
    loss_fnc = compileArgs['loss']
    weightsInput = getattr(loss_fnc, 'weightsInput', None)
    if weightsInput is not None:
        keras_model = Model(inputs=[inputLayer,weightsInput],outputs=lastLayer)

    else:
        keras_model = Model(inputs=[inputLayer],outputs=lastLayer)

    keras_model.compile(
        loss=loss_fnc,
        optimizer=getOptimizer(compileArgs),
        metrics=compileArgs.get('metrics')
    )
    return keras_model
