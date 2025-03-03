import numpy as np
import pickle
import os
from tensorflow.keras.models import load_model
from keras.models import load_model
import keras.losses

# Load encoders once
with open("models/user_encoder.pkl", "rb") as f:
    user_encoder = pickle.load(f)

with open("models/item_encoder.pkl", "rb") as f:
    item_encoder = pickle.load(f)

# Convert encoder classes to sets for efficient lookups
user_encoder_classes = {uid.strip().strip("'").strip('"') for uid in user_encoder.classes_}
item_encoder_classes = {item.strip().strip("'").strip('"') for item in item_encoder.classes_}

def recommend(user_id, item_ids, model_path):
    """Generate top 5 recommendations and return original product IDs."""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"❌ Model file {model_path} not found!")

    custom_objects = {"mse": keras.losses.mean_squared_error}

    # Load the model with the registered custom objects
    model = load_model(model_path, custom_objects=custom_objects)

    # Clean and check user ID
    user_id = str(user_id).strip().strip("'").strip('"')

    print(f"\n🔍 Checking User ID: {repr(user_id)}")
    print(f"✅ Total users in encoder: {len(user_encoder_classes)}")


    if user_id not in user_encoder_classes:
        print(f"❌ ERROR: User ID '{user_id}' NOT found in encoder!")
        return []

    encoded_user_id = user_encoder.transform([user_id])[0]
    print(f"✅ Encoded User ID: {encoded_user_id}")

    # Validate and encode item IDs
    item_ids_cleaned = [str(item).strip().strip("'").strip('"') for item in item_ids]
    valid_item_ids = [item for item in item_ids_cleaned if item in item_encoder_classes]

    if not valid_item_ids:
        print("❌ ERROR: None of the provided item IDs exist in the encoder!")
        return []

    encoded_item_ids = item_encoder.transform(valid_item_ids)
    print(f"✅ Encoded Item IDs: {encoded_item_ids}")

    # Prepare input for model prediction
    user_inputs = np.full(len(encoded_item_ids), encoded_user_id)
    item_inputs = np.array(encoded_item_ids)

    predictions = model.predict([user_inputs, item_inputs]).flatten()
    print("🔹 Raw Predictions:", predictions)

    # Get top 5 recommendations
    top_indices = np.argsort(predictions)[-5:][::-1]
    top_item_ids = item_encoder.inverse_transform(encoded_item_ids[top_indices])

    print("✅ Top Recommended Items:", top_item_ids)
    return top_item_ids.tolist()
