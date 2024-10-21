from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, MaxPool2D, Conv2DTranspose, Concatenate, Input
from tensorflow.keras.models import Model


def conv_block(inputs, num_filters):

    x = Conv2D(num_filters, size=(3,3), padding="same")(inputs)
    x = BatchNormalization(x)
    x = Activation("relu")(x)

    x = Conv2D(num_filters, size=(3, 3), padding="same")(x)
    x = BatchNormalization(x)
    x = Activation("relu")(x)
    return x


def encoder_block(inputs, num_filters):
    """
    The encoder block of U-net
    :param inputs: The input
    :param num_filters:
    :return: x - output of the convolutional block and to be used for concatenation
             p - ouput of the maxpool2D block
    """

    x = conv_block(inputs, num_filters)
    p = MaxPool2D((2,2))(x)
    return x, p


def decoder_block(inputs, skip_features, num_filters):
    """
   The decoder block of the U-net
   :param inputs: The input
   :param skip_features:
   :param num_filters:
   :return:
    """
    x = Conv2DTranspose(num_filters, 2, strides = 2, padding = "same")(inputs)
    x = Concatenate()([x, skip_features])
    x = conv_block(x, num_filters)
    return x


def build_unet(input_shape):
    input = Input(input_shape)

    # Encoder block
    s1, p1 = encoder_block(input, 64)
    s2, p2 = encoder_block(p1, 128)
    s3, p3 = encoder_block(p2, 256)
    s4, p4 = encoder_block(p3, 512)

    # Base
    b1 = conv_block(p4, 1024)

    # Decoder Block
    d1 = decoder_block(b1, s4, 512)
    d2 = decoder_block(d1, s3, 256)
    d3 = decoder_block(d2, s2, 128)
    d4 = decoder_block(d3, s1, 64)

    outputs = Conv2D(1, size=1, padding="same", activation = "sigmoid")(d4)

    model = Model(input, outputs, name = "UNET")
    return model




