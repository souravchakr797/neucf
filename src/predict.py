import numpy as np
import keras
from .model import get_latest_pretrained_model
from .data_loader import load_data

def recommend_top_5_products(user_id, model, product_categories):
    # Load model
    model = keras.models.load_model(model)

    # Load data encoders and metadata
    interactions, user_encoder, item_encoder, user_metadata_np, product_metadata_np, category_encoder, product_data = load_data()

    # Check if user_id exists
    if user_id not in user_encoder.classes_:
        print("❌ User ID not found in the dataset.")
        return []

    # Encode the user ID
    encoded_user_id = user_encoder.transform([user_id])[0]

    # Ensure product_categories is a list/array
    if isinstance(product_categories, str):  
        product_categories = [product_categories]  # Convert single category to a list

    # Encode product categories, ignoring ones not in the dataset
    valid_categories = [cat for cat in product_categories if cat in category_encoder.classes_]
    
    if not valid_categories:
        print("❌ None of the product categories were found in the dataset.")
        return []

    encoded_categories = category_encoder.transform(valid_categories)

    # Get product IDs belonging to these categories
    filtered_products = product_data[product_data["product_category_encoded"].isin(encoded_categories)]
    
    if filtered_products.empty:
        print("❌ No products found for the given categories.")
        return []

    # Get encoded product IDs
    encoded_product_ids = filtered_products["item_id"].values

    # Prepare user and item metadata
    user_metadata = np.tile(user_metadata_np[encoded_user_id], (len(encoded_product_ids), 1))
    item_metadata = product_metadata_np[encoded_product_ids]

    # Predict scores for filtered products
    user_inputs = np.full(len(encoded_product_ids), encoded_user_id)
    predictions = model.predict([user_inputs, encoded_product_ids, user_metadata, item_metadata]).flatten()

    # Get top 5 recommended products
    top_5_indices = np.argsort(predictions)[-5:][::-1]  # Get top 5 highest scores
    top_5_products = item_encoder.inverse_transform(encoded_product_ids[top_5_indices])

    return top_5_products
