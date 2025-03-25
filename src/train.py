import numpy as np
import os
import datetime
import tensorflow as tf
from tensorflow.keras.models import save_model, load_model
from tensorflow.keras.optimizers import Adam
from data_loader import load_data
from model import train_neucf, get_latest_pretrained_model
from sklearn.model_selection import train_test_split

def train_model():
    interactions, user_encoder, item_encoder, user_metadata_np, product_metadata_np, category_encoder, product_data = load_data()

    model = train_neucf(
    user_item_interactions=interactions, 
    num_users=len(user_encoder.classes_), 
    num_items=len(item_encoder.classes_),
    user_metadata=user_metadata_np, 
    item_metadata=product_metadata_np,
    epochs=3, 
    batch_size=32
)
    model_dir = "models"
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, f"neucf_pretrained_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.h5")
    
    # Save the model
    model.save(model_path)
    print(f"✅ Model saved at {model_path}")

if __name__ == "__main__":
    train_model()
