import pandas as pd
import numpy as np
import pickle
import os
from sklearn.preprocessing import LabelEncoder


def load_data():
    """Load and preprocess datasets, updating encoders if new data exists."""
    print("🔄 Loading data...")

    # Load datasets
    orders = pd.read_csv("data/olist_orders_dataset.csv")
    order_items = pd.read_csv("data/olist_order_items_dataset.csv")
    customers = pd.read_csv("data/olist_customers_dataset.csv")

    # Check if required columns are present
    if 'customer_id' not in customers.columns:
        raise ValueError("Error: 'customer_id' column is missing in the customers dataset.")
    if 'order_id' not in orders.columns or 'customer_id' not in orders.columns:
        raise ValueError("Error: 'order_id' or 'customer_id' column is missing in the orders dataset.")
    if 'order_id' not in order_items.columns or 'product_id' not in order_items.columns:
        raise ValueError("Error: 'order_id' or 'product_id' column is missing in the order_items dataset.")

    # Merge datasets to get user-product interactions
    data = order_items.merge(orders, on="order_id").merge(customers, on="customer_id")

    # Remove rows where either customer_id or product_id is missing
    data = data.dropna(subset=['customer_id', 'product_id'])

    # Select the necessary columns
    data = data[['customer_id', 'product_id']]

    # Paths for encoder files
    user_encoder_path = "models/user_encoder.pkl"
    item_encoder_path = "models/item_encoder.pkl"

    # Check if encoders exist
    if os.path.exists(user_encoder_path) and os.path.exists(item_encoder_path):
        print("✅ Found pre-existing encoders, updating them...")

        # Load existing encoders
        with open(user_encoder_path, "rb") as f:
            user_encoder = pickle.load(f)
        with open(item_encoder_path, "rb") as f:
            item_encoder = pickle.load(f)

        # Merge old and new categories
        all_users = list(user_encoder.classes_) + list(data['customer_id'].unique())
        all_items = list(item_encoder.classes_) + list(data['product_id'].unique())

        # Refit encoders with all known IDs
        user_encoder = LabelEncoder().fit(all_users)
        item_encoder = LabelEncoder().fit(all_items)

    else:
        print("⚠️ Encoders not found! Creating new encoders...")
        user_encoder = LabelEncoder().fit(data['customer_id'])
        item_encoder = LabelEncoder().fit(data['product_id'])

    # Transform data using updated encoders
    data['customer_id'] = user_encoder.transform(data['customer_id'])
    data['product_id'] = item_encoder.transform(data['product_id'])

    # Save updated encoders
    os.makedirs("models", exist_ok=True)  # Ensure models directory exists
    with open(user_encoder_path, "wb") as f:
        pickle.dump(user_encoder, f)
    with open(item_encoder_path, "wb") as f:
        pickle.dump(item_encoder, f)

    print(f"✅ Total unique users: {len(user_encoder.classes_)}")
    print(f"✅ Total unique products: {len(item_encoder.classes_)}")

    # Implicit feedback: purchase = positive interaction
    data['rating'] = 1

    print(f"✅ Data Loaded: {len(data)} total samples")
    return data, len(user_encoder.classes_), len(item_encoder.classes_)


if __name__ == "__main__":
    train, num_users, num_items = load_data()
