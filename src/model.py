import tensorflow as tf
import numpy as np
import random
import os
from tensorflow import keras
from tensorflow.keras.layers import (
    Input, Embedding, Flatten, Multiply, Concatenate, Dense, Dropout, BatchNormalization
)

def generate_negative_samples(user_item_pairs, num_users, num_items, num_negatives=4):
    """Fast negative sampling using NumPy."""
    all_items = np.arange(num_items)
    user_negative_samples = []
    
    user_item_dict = {user: set() for user, _ in user_item_pairs}
    for user, item in user_item_pairs:
        user_item_dict[user].add(item)
    
    for user, pos_item in user_item_pairs:
        pos_set = user_item_dict[user]
        neg_samples = set()
        
        while len(neg_samples) < num_negatives:
            candidate_negatives = np.random.choice(all_items, num_negatives * 2)
            neg_samples.update(set(candidate_negatives) - pos_set)
            neg_samples = set(list(neg_samples)[:num_negatives])
        
        user_negative_samples.append((user, pos_item, 1))
        for neg_item in neg_samples:
            user_negative_samples.append((user, neg_item, 0))
    
    return user_negative_samples

def create_neucf_model(num_users, num_items, user_metadata_dim, item_metadata_dim, gmf_embedding_size=32, mlp_embedding_size=64):
    """Builds the NeuCF Model combining GMF, MLP components, and both user and product metadata."""

    # Inputs
    user_input = Input(shape=(1,), name="user_input")
    item_input = Input(shape=(1,), name="item_input")
    user_metadata_input = Input(shape=(user_metadata_dim,), name="user_metadata_input")
    item_metadata_input = Input(shape=(item_metadata_dim,), name="item_metadata_input")

    # GMF Pathway
    user_embedding_gmf = Embedding(input_dim=num_users, output_dim=gmf_embedding_size, name="user_embedding_gmf")(user_input)
    item_embedding_gmf = Embedding(input_dim=num_items, output_dim=gmf_embedding_size, name="item_embedding_gmf")(item_input)
    user_vec_gmf = Flatten()(user_embedding_gmf)
    item_vec_gmf = Flatten()(item_embedding_gmf)
    gmf_output = Multiply()([user_vec_gmf, item_vec_gmf])

    # MLP Pathway
    user_embedding_mlp = Embedding(input_dim=num_users, output_dim=mlp_embedding_size, name="user_embedding_mlp")(user_input)
    item_embedding_mlp = Embedding(input_dim=num_items, output_dim=mlp_embedding_size, name="item_embedding_mlp")(item_input)
    user_vec_mlp = Flatten()(user_embedding_mlp)
    item_vec_mlp = Flatten()(item_embedding_mlp)
    concat_vec = Concatenate()([user_vec_mlp, item_vec_mlp])

    # MLP Hidden Layers
    mlp_layer = Dense(128, activation="relu")(concat_vec)
    mlp_layer = Dropout(0.2)(mlp_layer)
    mlp_layer = Dense(64, activation="relu")(mlp_layer)
    mlp_layer = Dropout(0.2)(mlp_layer)
    mlp_layer = Dense(32, activation="relu")(mlp_layer)
    mlp_layer = BatchNormalization()(mlp_layer)
    mlp_output = Dense(16, activation="relu")(mlp_layer)

    # User Metadata Processing
    user_metadata_layer = Dense(16, activation="relu")(user_metadata_input)
    user_metadata_layer = Dense(8, activation="relu")(user_metadata_layer)

    # Item Metadata Processing
    item_metadata_layer = Dense(16, activation="relu")(item_metadata_input)
    item_metadata_layer = Dense(8, activation="relu")(item_metadata_layer)



    # Final Concatenation
    final_concat = Concatenate()([gmf_output, mlp_output, user_metadata_layer, item_metadata_layer])
    output = Dense(1, activation="sigmoid")(final_concat)

    # Model Compilation
    model = keras.Model(inputs=[user_input, item_input, user_metadata_input, item_metadata_input], outputs=output)
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001), loss="binary_crossentropy", metrics=["accuracy"])

    return model


def train_neucf(user_item_interactions, num_users, num_items, user_metadata, item_metadata, num_negatives=4, epochs=10, batch_size=32):
    """Handles data preparation, model creation, and training with user & item metadata"""

    # Ensure user_item_interactions is a list of (user, item) tuples
    user_item_pairs = user_item_interactions.iloc[:, :2].values.tolist()


    # Generate negative samples
    sampled_data = generate_negative_samples(user_item_pairs, num_users, num_items, num_negatives)

    # Unpack sampled data
    train_users, train_items, train_labels = zip(*sampled_data)
    train_users, train_items, train_labels = np.array(train_users), np.array(train_items), np.array(train_labels)

    # Get corresponding user & item metadata
    user_metadata_features = user_metadata[train_users]
    item_metadata_features = item_metadata[train_items]

    # Create and train model
    model = create_neucf_model(num_users, num_items, user_metadata.shape[1], item_metadata.shape[1])
    model.fit([train_users, train_items, user_metadata_features, item_metadata_features], train_labels, epochs=epochs, batch_size=batch_size)

    return model


def get_latest_pretrained_model(model_dir="models"):
    """Retrieve the latest pre-trained model file."""
    if not os.path.exists(model_dir):
        return None
    model_files = [f for f in os.listdir(model_dir) if f.startswith("neucf_pretrained") and f.endswith(".h5")]
    if not model_files:
        return None
    latest_model = max(model_files, key=lambda x: os.path.getctime(os.path.join(model_dir, x)))
    return os.path.join(model_dir, latest_model)
