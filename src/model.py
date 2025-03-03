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
    all_items = np.arange(num_items)  # Create an array of all item indices
    user_negative_samples = []

    user_item_dict = {}  # To store user interactions
    for user, item in user_item_pairs:
        if user not in user_item_dict:
            user_item_dict[user] = set()
        user_item_dict[user].add(item)

    for user, pos_item in user_item_pairs:
        pos_set = user_item_dict[user]  # Get all interacted items
        neg_samples = set()

        # Sample negatives efficiently
        while len(neg_samples) < num_negatives:
            candidate_negatives = np.random.choice(all_items, num_negatives * 2)  # Sample extra negatives
            neg_samples.update(set(candidate_negatives) - pos_set)
            neg_samples = set(list(neg_samples)[:num_negatives])  # Trim to exact count

        # Add positive and negative samples
        user_negative_samples.append((user, pos_item, 1))  # Positive sample
        for neg_item in neg_samples:
            user_negative_samples.append((user, neg_item, 0))  # Negative sample

    return user_negative_samples



def create_training_data(user_item_interactions, num_users, num_items, num_negatives=4):
    """Prepares training data with negative sampling"""
    sampled_data = generate_negative_samples(user_item_interactions, num_users, num_items, num_negatives)

    # Convert to NumPy arrays for training
    users, items, labels = zip(*sampled_data)
    return np.array(users), np.array(items), np.array(labels)


def create_neucf_model(num_users, num_items, gmf_embedding_size=32, mlp_embedding_size=64):
    """Builds the NeuCF Model combining GMF and MLP components."""
    user_input = Input(shape=(1,), name="user_input")
    item_input = Input(shape=(1,), name="item_input")

    # GMF Embeddings (Lower dimensional)
    user_embedding_gmf = Embedding(input_dim=num_users, output_dim=gmf_embedding_size, name="user_embedding_gmf")(user_input)
    item_embedding_gmf = Embedding(input_dim=num_items, output_dim=gmf_embedding_size, name="item_embedding_gmf")(item_input)

    # MLP Embeddings (Higher dimensional)
    user_embedding_mlp = Embedding(input_dim=num_users, output_dim=mlp_embedding_size, name="user_embedding_mlp")(user_input)
    item_embedding_mlp = Embedding(input_dim=num_items, output_dim=mlp_embedding_size, name="item_embedding_mlp")(item_input)

    # Flatten embeddings
    user_vec_gmf = Flatten()(user_embedding_gmf)
    item_vec_gmf = Flatten()(item_embedding_gmf)
    user_vec_mlp = Flatten()(user_embedding_mlp)
    item_vec_mlp = Flatten()(item_embedding_mlp)

    # GMF Component
    gmf_output = Multiply()([user_vec_gmf, item_vec_gmf])

    # MLP Component
    concat_vec = Concatenate()([user_vec_mlp, item_vec_mlp])
    mlp_layer = Dense(128, activation="relu")(concat_vec)
    mlp_layer = Dropout(0.2)(mlp_layer)  # Regularization
    mlp_layer = Dense(64, activation="relu")(mlp_layer)
    mlp_layer = Dropout(0.2)(mlp_layer)
    mlp_layer = Dense(32, activation="relu")(mlp_layer)
    mlp_layer = BatchNormalization()(mlp_layer)  # Stabilizing training
    mlp_output = Dense(16, activation="relu")(mlp_layer)

    # NeuMF: Combine GMF + MLP
    final_concat = Concatenate()([gmf_output, mlp_output])
    output = Dense(1, activation="sigmoid")(final_concat)

    # Define and compile model
    model = keras.Model(inputs=[user_input, item_input], outputs=output)
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001), loss="binary_crossentropy", metrics=["accuracy"])

    return model


def train_neucf(user_item_interactions, num_users, num_items, num_negatives=4, epochs=10, batch_size=32):
    """Handles data preparation, model creation, and training"""
    # Generate training data with negative samples
    train_users, train_items, train_labels = create_training_data(user_item_interactions, num_users, num_items, num_negatives)
    # Create model
    model = create_neucf_model(num_users, num_items)
    # Train the model
    model.fit([train_users, train_items], train_labels, epochs=epochs, batch_size=batch_size)

    return model


# Get the latest trained model
def get_latestpre_trained_model():
    model_dir = "models"
    models = sorted([f for f in os.listdir(model_dir) if f.startswith("neucf_pretrained_")], reverse=True)
    return os.path.join(model_dir, models[0]) if models else None