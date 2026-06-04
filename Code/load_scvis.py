
import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import numpy as np
import pandas as pd
import tensorflow.compat.v1 as tf
import yaml

import sys
from pathlib import Path

ELDR_ROOT = Path(__file__).resolve().parents[1]
SCVIS_LIB = ELDR_ROOT / "scvis" / "lib"
SCVIS_CONFIG = SCVIS_LIB / "scvis" / "config" / "model_config.yaml"

sys.path.insert(0, str(SCVIS_LIB))
from scvis.vae import GaussianVAE

tf.disable_v2_behavior()

def load_vae(input_dim, model_file, feature_transform = None):

    tf.reset_default_graph()
    
    # Model Configuration
    try:
        with open(SCVIS_CONFIG, "r") as config_file_yaml:
            config = yaml.load(config_file_yaml, Loader = yaml.FullLoader)
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
        matrix =  np.float32(pd.read_csv(feature_transform, sep="\t", header = None).values)
        input = tf.matmul(input, matrix)

    # Compute the representation of our input
    vae = GaussianVAE(input, 1, architecture["inference"]["layer_size"], architecture["latent_dimension"], decoder_layer_size=architecture["model"]["layer_size"])
    rep, _ = vae.encoder(prob = 1.0)

    # Setup and restore the tf session
    sess = tf.Session()
    saver = tf.train.Saver()
    saver.restore(sess, model_file)

    return sess, rep, X, D
