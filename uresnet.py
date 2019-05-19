from keras.layers import ZeroPadding2D, BatchNormalization, Conv2D, Activation, MaxPooling2D
from keras.layers import UpSampling2D, Input, Conv2D, Conv2DTranspose, Add, Concatenate
from keras.utils.data_utils import get_file
from keras.models import Model

def handle_block_names_old(stage):
    conv_name = 'decoder_stage{}_conv'.format(stage)
    bn_name = 'decoder_stage{}_bn'.format(stage)
    relu_name = 'decoder_stage{}_relu'.format(stage)
    up_name = 'decoder_stage{}_upsample'.format(stage)
    return conv_name, bn_name, relu_name, up_name


def up_block(filters, stage, kernel_size=(3,3), upsample_rate=(2,2),transpose_kernel_size=(4,4), batchnorm=False, skip=None):
    def layer(input_tensor):

        conv_name, bn_name, relu_name, up_name = handle_block_names_old(stage)

        x = Conv2DTranspose(filters, transpose_kernel_size, strides=upsample_rate,
                            padding='same', name=up_name)(input_tensor)
        if batchnorm:
            x = BatchNormalization(name=bn_name+'1')(x)
        x = Activation('relu', name=relu_name+'1')(x)

        if skip is not None:
            x = Concatenate()([x, skip])

        x = Conv2D(filters, kernel_size, padding='same', name=conv_name+'2')(x)
        if batchnorm:
            x = BatchNormalization(name=bn_name+'2')(x)
        x = Activation('relu', name=relu_name+'2')(x)

        return x
    return layer


def handle_block_names(stage, block):
    name_base = 'stage{}_unit{}_'.format(stage + 1, block + 1)
    conv_name = name_base + 'conv'
    bn_name = name_base + 'bn'
    relu_name = name_base + 'relu'
    sc_name = name_base + 'sc'
    return conv_name, bn_name, relu_name, sc_name

def identity_block(filters, stage, block):
    """The identity block is the block that has no conv layer at shortcut."""

    def layer(input_tensor):
        conv_params = { 'kernel_initializer': 'glorot_uniform', 'use_bias': False, 'padding': 'valid'}
        bn_params = {'axis': 3, 'momentum': 0.99, 'epsilon': 2e-5, 'center': True, 'scale': True}

        conv_name, bn_name, relu_name, sc_name = handle_block_names(stage, block)

        x = BatchNormalization(name=bn_name + '1', **bn_params)(input_tensor)
        x = Activation('relu', name=relu_name + '1')(x)
        x = ZeroPadding2D(padding=(1, 1))(x)
        x = Conv2D(filters, (3, 3), name=conv_name + '1', **conv_params)(x)

        x = BatchNormalization(name=bn_name + '2', **bn_params)(x)
        x = Activation('relu', name=relu_name + '2')(x)
        x = ZeroPadding2D(padding=(1, 1))(x)
        x = Conv2D(filters, (3, 3), name=conv_name + '2', **conv_params)(x)

        x = Add()([x, input_tensor])
        return x

    return layer


def conv_block(filters, stage, block, strides=(2, 2)):
    """The conv block is the block that has conv layer at shortcut."""

    def layer(input_tensor):
        conv_params = { 'kernel_initializer': 'glorot_uniform', 'use_bias': False, 'padding': 'valid'}
        bn_params = {'axis': 3, 'momentum': 0.99, 'epsilon': 2e-5, 'center': True, 'scale': True}
        conv_name, bn_name, relu_name, sc_name = handle_block_names(stage, block)

        x = BatchNormalization(name=bn_name + '1', **bn_params)(input_tensor)
        x = Activation('relu', name=relu_name + '1')(x)
        shortcut = x
        x = ZeroPadding2D(padding=(1, 1))(x)
        x = Conv2D(filters, (3, 3), strides=strides, name=conv_name + '1', **conv_params)(x)

        x = BatchNormalization(name=bn_name + '2', **bn_params)(x)
        x = Activation('relu', name=relu_name + '2')(x)
        x = ZeroPadding2D(padding=(1, 1))(x)
        x = Conv2D(filters, (3, 3), name=conv_name + '2', **conv_params)(x)


        shortcut = Conv2D(filters, (1, 1), name=sc_name, strides=strides, **conv_params)(shortcut)
        
        x = Add()([x, shortcut])
        return x

    return layer





def build_unet(backbone, classes, last_block_filters, skip_layers,
               n_upsample_blocks=5, upsample_rates=(2,2,2,2,2),activation='sigmoid'):

    input = backbone.input
    x = backbone.output

    # convert layer names to indices
    skip_layers = ([get_layer_number(backbone, l) if isinstance(l, str) else l for l in skip_layers])
    for i in range(n_upsample_blocks):

        # check if there is a skip connection
        if i < len(skip_layers):
            skip = backbone.layers[skip_layers[i]].output
        else:
            skip = None

        up_size = (upsample_rates[i], upsample_rates[i])
        filters = last_block_filters * 2**(n_upsample_blocks-(i+1))

        x = up_block(filters, i, upsample_rate=up_size, skip=skip)(x)

    if classes < 2:
        activation = 'sigmoid'

    x = Conv2D(classes, (3,3), padding='same', name='final_conv')(x)
    x = Activation(activation, name=activation)(x)

    model = Model(input, x)

    return model


def build_resnet(repetitions=(2, 2, 2, 2), input_shape=None):
    # Determine proper input shape
    input_shape = input_shape

    img_input = Input(shape=input_shape, name='data')
    
    # get parameters for model layers
    no_scale_bn_params = { 'axis': 3, 'momentum': 0.99, 'epsilon': 2e-5, 'center': True, 'scale': False}
    bn_params = { 'axis': 3, 'momentum': 0.99, 'epsilon': 2e-5, 'center': True, 'scale': True}
    conv_params = { 'kernel_initializer': 'glorot_uniform', 'use_bias': False, 'padding': 'valid'}

    init_filters = 64
    
    # resnet bottom
    x = BatchNormalization(name='bn_data', **no_scale_bn_params)(img_input)
    x = ZeroPadding2D(padding=(3, 3))(x)
    x = Conv2D(init_filters, (7, 7), strides=(2, 2), name='conv0', **conv_params)(x)
    x = BatchNormalization(name='bn0', **bn_params)(x)
    x = Activation('relu', name='relu0')(x)
    x = ZeroPadding2D(padding=(1, 1))(x)
    x = MaxPooling2D((3, 3), strides=(2, 2), padding='valid', name='pooling0')(x)
    
    # resnet body
    for stage, rep in enumerate(repetitions): #3,4,6,3
        for block in range(rep):

            filters = init_filters * (2**stage)
            
            # first block of first stage without strides because we have maxpooling before
            if block == 0 and stage == 0:
                x = conv_block(filters, stage, block, strides=(1, 1))(x)
                
            elif block == 0:
                x = conv_block(filters, stage, block, strides=(2, 2))(x)
                
            else:
                x = identity_block(filters, stage, block)(x)
                
    x = BatchNormalization(name='bn1', **bn_params)(x)
    x = Activation('relu', name='relu1')(x)

    # Create model.
    model = Model(img_input, x)

    return model

def ResNet34(input_shape, weights_info=None, resnet_untrainable=0):
    model = build_resnet(input_shape=input_shape, repetitions=(3, 4, 6, 3))
    model.name = 'resnet34'

    if weights_info:
        weights_path = get_file(weights_info['name'],weights_info['url'],cache_subdir='models',md5_hash=weights_info['md5']) 
        model.load_weights(weights_path)

    for layer in model.layers[:-1*resnet_untrainable]:
        layer.trainable = False
    return model


def UResNet34(input_shape=(None, None, 3), classes=1, decoder_filters=16, activation='sigmoid',resnet_untrainable = 0):
    weights={
        'url': 'https://github.com/qubvel/classification_models/releases/download/0.0.1/resnet34_imagenet_1000_no_top.h5',
        'name': 'resnet34_imagenet_1000_no_top.h5',
        'md5': '8caaa0ad39d927cb8ba5385bf945d582'
    }

    backbone = ResNet34(input_shape=input_shape, weights_info = weights, resnet_untrainable = resnet_untrainable)
    
    skip_connections = list([106,74,37,5])  # for resnet 34
    
    model = build_unet(backbone, classes, decoder_filters, skip_connections, activation=activation)

    model.name = 'u-resnet34'

    return model