import numpy as np
import os
import datetime
import tensorflow as tf
from tensorflow.keras.models import save_model, load_model
from tensorflow.keras.optimizers import Adam
from data_loader import load_data
from model import train_neucf, get_latestpre_trained_model
from sklearn.model_selection import train_test_split

def train_model():
    # Load data
    train, num_users, num_items = load_data()  

    # Prepare interactions as a list of (user, item) pairs
    user_item_interactions = list(zip(train['customer_id'], train['product_id']))

    # Convert Pandas Series to NumPy arrays before splitting
    customer_ids = train['customer_id'].values
    product_ids = train['product_id'].values
    ratings = train['rating'].values

    # Train-test split
    train_users, test_users, train_items, test_items, train_ratings, test_ratings = train_test_split(
        customer_ids, product_ids, ratings, test_size=0.1, random_state=42
    )

    # ✅ Step 1: Train NeuCF with negative sampling
    print("🚀 Pre-training the NeuCF model with negative sampling...")
    pre_trained_model = train_neucf(user_item_interactions, num_users, num_items, num_negatives=4, epochs=5, batch_size=256)

    # ✅ Save the pre-trained model with timestamp
    os.makedirs("models", exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    pre_trained_model_path = f"models/neucf_pretrained_{timestamp}.h5"
    save_model(pre_trained_model, pre_trained_model_path)
    print(f"✅ Pre-trained model saved at {pre_trained_model_path}")

    # ✅ Step 2: Load the latest pre-trained model and fine-tune on explicit ratings
    latest_pre_trained_model_path = get_latestpre_trained_model()
    if not latest_pre_trained_model_path:
        print("⚠️ No pre-trained model found. Exiting training process.")
        return

    print(f"🔥 Fine-tuning the NeuCF model using explicit ratings... (Using {latest_pre_trained_model_path})")
    model = load_model(latest_pre_trained_model_path)

    # ✅ Enable eager execution (fixes numpy() error)
    tf.config.run_functions_eagerly(True)

    # ✅ Explicitly compile the model (fixes missing metric issue)
    model.compile(optimizer=Adam(learning_rate=0.001), loss="mse", metrics=["mae"])

    # Fine-tune on explicit feedback (ratings)
    history = model.fit(
        [np.array(train_users), np.array(train_items)], np.array(train_ratings), 
        batch_size=256, epochs=5, 
        validation_data=([np.array(test_users), np.array(test_items)], np.array(test_ratings))
    )

    # ✅ Evaluate fine-tuned model
    loss, mae = model.evaluate([np.array(test_users), np.array(test_items)], np.array(test_ratings))
    print(f"📊 Fine-Tuned Model Evaluation: Loss = {loss:.4f}, MAE = {mae:.4f}")

    # ✅ Save the fine-tuned model
    fine_tuned_model_path = f"models/neucf_finetuned_{timestamp}.h5"
    save_model(model, fine_tuned_model_path)
    print(f"✅ Fine-tuned model saved to {fine_tuned_model_path}")

if __name__ == "__main__":
    train_model()
