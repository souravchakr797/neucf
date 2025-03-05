import pandas as pd
import numpy as np
import pickle
import os
from sklearn.preprocessing import LabelEncoder

def load_data():
    """Load and preprocess datasets, updating encoders if new data exists."""
    print("🔄 Loading data...")

    # Load datasets
    orders = pd.read_csv("data/olist_orders_dataset.csv", nrows=10000)
    order_items = pd.read_csv("data/olist_order_items_dataset.csv", nrows=5000)
    customers = pd.read_csv("data/olist_customers_dataset.csv", nrows=5000)
    products = pd.read_csv("data/olist_products_dataset.csv", nrows=5000)

    # Merge orders with customers to get user demographics
    user_data = orders.merge(customers, on="customer_id")[["customer_id", "customer_city", "customer_state"]]
    
    # Merge order_items with products to get product metadata
    product_data = order_items.merge(products, on="product_id")[["product_id", "product_category_name"]]
    
    # Drop duplicates to have a unique mapping
    user_data = user_data.drop_duplicates(subset=["customer_id"]).reset_index(drop=True)
    product_data = product_data.drop_duplicates(subset=["product_id"]).reset_index(drop=True)
   

    # Initialize encoders
    user_encoder = LabelEncoder()
    item_encoder = LabelEncoder()
    city_encoder = LabelEncoder()
    state_encoder = LabelEncoder()
    category_encoder = LabelEncoder()

    # Encode customer_id and product_id
    user_data["user_id"] = user_encoder.fit_transform(user_data["customer_id"])
    product_data["item_id"] = item_encoder.fit_transform(product_data["product_id"])

    # Encode customer city & state
    user_data["customer_city_encoded"] = city_encoder.fit_transform(user_data["customer_city"])
    user_data["customer_state_encoded"] = state_encoder.fit_transform(user_data["customer_state"])

    # Encode product category name
    product_data["product_category_encoded"] = category_encoder.fit_transform(product_data["product_category_name"])

    # Keep only required columns
    user_metadata = user_data[["user_id", "customer_city_encoded", "customer_state_encoded"]]
    product_metadata = product_data[["item_id", "product_category_encoded"]]

    # Create user-item interactions
    interactions = order_items[["order_id", "product_id"]].merge(orders[["order_id", "customer_id"]], on="order_id")
    interactions = interactions.merge(user_data[["customer_id", "user_id"]], on="customer_id")
    interactions = interactions.merge(product_data[["product_id", "item_id"]], on="product_id")
    interactions = interactions[["user_id", "item_id"]].drop_duplicates()

    # Add implicit feedback (1 for purchased items)
    interactions["label"] = 1

    # Convert metadata to numpy arrays for model training
    user_metadata_np = user_metadata.drop(columns=["user_id"]).to_numpy()
    product_metadata_np = product_metadata.drop(columns=["item_id"]).to_numpy()

    # Save encoders
    os.makedirs("models", exist_ok=True)
    with open("models/user_encoder.pkl", "wb") as f:
        pickle.dump(user_encoder, f)
    with open("models/item_encoder.pkl", "wb") as f:
        pickle.dump(item_encoder, f)
    with open("models/city_encoder.pkl", "wb") as f:
        pickle.dump(city_encoder, f)
    with open("models/state_encoder.pkl", "wb") as f:
        pickle.dump(state_encoder, f)
    with open("models/category_encoder.pkl", "wb") as f:
        pickle.dump(category_encoder, f)
    
    print("✅ Encoders saved successfully!")
    print("Processed data ready for training!")
    
    return interactions, user_encoder, item_encoder, user_metadata_np, product_metadata_np
