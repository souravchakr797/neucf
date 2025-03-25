import pandas as pd
import numpy as np
import pickle
import os
from sklearn.preprocessing import LabelEncoder

def load_encoder(encoder_path):
    """Load an encoder if it exists; otherwise, return a new LabelEncoder."""
    if os.path.exists(encoder_path):
        with open(encoder_path, "rb") as f:
            encoder = pickle.load(f)
            return encoder
    return LabelEncoder()

def update_encoder(encoder, new_values):
    """Update a LabelEncoder with new values while preserving previous mappings."""
    if hasattr(encoder, "classes_"):
        existing_classes = set(encoder.classes_)
    else:
        existing_classes = set()
    
    updated_classes = list(existing_classes | set(new_values))  # Merge old & new classes
    new_encoder = LabelEncoder()
    new_encoder.fit(updated_classes)
    return new_encoder

def load_data():
    """Load and preprocess datasets, ensuring encoders update dynamically."""
    print("🔄 Loading data...")

    # Load datasets
    orders = pd.read_csv("data/olist_orders_dataset.csv")
    order_items = pd.read_csv("data/olist_order_items_dataset.csv")
    customers = pd.read_csv("data/olist_customers_dataset.csv")
    products = pd.read_csv("data/olist_products_dataset.csv")

    # Merge orders with customers
    user_data = customers.merge(orders, on="customer_id", how="left")[["customer_id", "customer_city", "customer_state"]]
    
    # Merge order_items with products
    # Ensure all products are included, even if they haven't been ordered
    product_data = products.merge(order_items, on="product_id", how="left")[["product_id", "product_category_name"]]


    # Drop duplicates to have a unique mapping
    user_data = user_data.drop_duplicates(subset=["customer_id"]).reset_index(drop=True)
    product_data = product_data.drop_duplicates(subset=["product_id"]).reset_index(drop=True)

    # Handle missing category names
    product_data["product_category_name"] = product_data["product_category_name"].fillna("unknown").astype(str)

    # Load existing encoders (or create new ones)
    user_encoder = load_encoder("models/user_encoder.pkl")
    item_encoder = load_encoder("models/item_encoder.pkl")
    city_encoder = load_encoder("models/city_encoder.pkl")
    state_encoder = load_encoder("models/state_encoder.pkl")
    category_encoder = load_encoder("models/category_encoder.pkl")

    # Update encoders with new data
    user_encoder = update_encoder(user_encoder, user_data["customer_id"].unique())
    item_encoder = update_encoder(item_encoder, product_data["product_id"].unique())
    city_encoder = update_encoder(city_encoder, user_data["customer_city"].unique())
    state_encoder = update_encoder(state_encoder, user_data["customer_state"].unique())
    category_encoder = update_encoder(category_encoder, product_data["product_category_name"].unique())

    # Encode customer_id and product_id
    user_data["user_id"] = user_encoder.transform(user_data["customer_id"])
    product_data["item_id"] = item_encoder.transform(product_data["product_id"])

    # Encode customer city & state
    user_data["customer_city_encoded"] = city_encoder.transform(user_data["customer_city"])
    user_data["customer_state_encoded"] = state_encoder.transform(user_data["customer_state"])

    # Encode product category name
    product_data["product_category_encoded"] = category_encoder.transform(product_data["product_category_name"])

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

    # Save updated encoders
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

    print("✅ Encoders updated and saved successfully!")
    print("Processed data ready for recommendation!")

    return interactions, user_encoder, item_encoder, user_metadata_np, product_metadata_np, category_encoder, product_data
