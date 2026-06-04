import numpy as np
import pandas as pd
import tensorflow as tf
import yaml

from pathlib import Path
import sys
from types import SimpleNamespace

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCVIS_LIB = PROJECT_ROOT / "scvis" / "lib"
CODE_DIR = PROJECT_ROOT / "Code"
SCVIS_CONFIG = SCVIS_LIB / "scvis" / "config" / "model_config.yaml"

tf.compat.v1.disable_v2_behavior()
tf.nn.dropout = tf.compat.v1.nn.dropout
for name in (
    "AUTO_REUSE",
    "Session",
    "get_variable",
    "placeholder",
    "random_normal",
    "reset_default_graph",
    "truncated_normal",
    "variable_scope",
):
    if not hasattr(tf, name):
        setattr(tf, name, getattr(tf.compat.v1, name))
if not hasattr(tf.train, "Saver"):
    tf.train.Saver = tf.compat.v1.train.Saver
if not hasattr(tf, "contrib"):
    tf.contrib = SimpleNamespace(
        layers=SimpleNamespace(
            xavier_initializer=tf.compat.v1.keras.initializers.glorot_uniform
        )
    )

sys.path.insert(0, str(SCVIS_LIB))
from scvis.vae import GaussianVAE

sys.path.insert(0, str(CODE_DIR))
from base import MLP, BatchManager


def load_aug(input_dim, model_file, feature_transform=None, shape=None):
    tf.reset_default_graph()

    # Model Configuration
    try:
        config_file_yaml = open(SCVIS_CONFIG, "r")
        config = yaml.load(config_file_yaml, Loader=yaml.FullLoader)
        config_file_yaml.close()
    except yaml.YAMLError as exc:
        print("Error in the configuration file: {}".format(exc))

    architecture = config["architecture"]
    architecture.update({"input_dimension": input_dim})

    # Setup our modified input to the model
    X = tf.placeholder(tf.float32, shape=[None, input_dim])
    D = tf.placeholder(tf.float32, shape=[1, input_dim])
    input = X + D

    # Perform any feature transformation specified
    if feature_transform is not None:
        matrix = np.float32(pd.read_csv(feature_transform, sep="\t", header=None).values)
        input = tf.matmul(input, matrix)

    # Compute the representation of our input
    vae = GaussianVAE(input, 1, architecture["inference"]["layer_size"], architecture["latent_dimension"],
                      decoder_layer_size=architecture["model"]["layer_size"])
    rep, _ = vae.encoder(prob=1.0)

    # Setup and restore the tf session
    sess = tf.Session()
    saver = tf.train.Saver()
    saver.restore(sess, str(model_file))

    return sess, rep, X, D
