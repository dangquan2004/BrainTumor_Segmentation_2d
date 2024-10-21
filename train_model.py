import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import numpy as np

from glob import glob
from sklearn.utils import shuffle
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger, ReduceLROnPlateau, EarlyStopping, TensorBoard
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from unet_architecture import build_unet
from metrics import dice_loss, dice_coef


"""Global Value"""
height = 256
width = 256


def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


if __name__ == "__main__":
    # Seeding
    np.random.seed(42)
    tf.random.set_seed(42)

    """ Directory for storing file """
    create_dir("files")

