import numpy as np
import keras as keras
from .model import get_latest_pretrained_model
from .data_loader import load_data
def recommend_top_5_products(user_id, model):
        
        model = keras.models.load_model(model)
        
        # Load data encoders and metadata
        interactions, user_encoder, item_encoder, user_metadata_np, product_metadata_np = load_data()
        
        # Check if user_id exists
        if user_id not in user_encoder.classes_:
            print("❌ User ID not found in the dataset.")
            return []

        # Encode the user ID
        encoded_user_id = user_encoder.transform([user_id])[0]
        
        # Get all product IDs
        all_product_ids = item_encoder.classes_
        encoded_product_ids = np.arange(len(all_product_ids))

        # Prepare user and item metadata
        user_metadata = np.tile(user_metadata_np[encoded_user_id], (len(encoded_product_ids), 1))
        item_metadata = product_metadata_np[encoded_product_ids]

        # Predict scores for all products
        user_inputs = np.full(len(encoded_product_ids), encoded_user_id)
        predictions = model.predict([user_inputs, encoded_product_ids, user_metadata, item_metadata]).flatten()
        
        # Get top 5 recommended products
        top_5_indices = np.argsort(predictions)[-5:][::-1]  # Get top 5 highest scores
        top_5_products = item_encoder.inverse_transform(top_5_indices)
        
        return top_5_products

