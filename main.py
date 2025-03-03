import json
import os
import pickle
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import urlparse, parse_qs
from src.predict import recommend

# Load the user encoder
with open("models/user_encoder.pkl", "rb") as f:
    user_encoder = pickle.load(f)

# Load the item encoder
with open("models/item_encoder.pkl", "rb") as f:
    item_encoder = pickle.load(f)    

# Get the latest trained model
def get_latest_model():
    model_dir = "models"
    models = sorted([f for f in os.listdir(model_dir) if f.startswith("neucf_finetuned_")], reverse=True)
    return os.path.join(model_dir, models[0]) if models else None

class RequestHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        parsed_path = urlparse(self.path)
        query_params = parse_qs(parsed_path.query)

        if parsed_path.path == "/recommend":
            user_id = query_params.get("user_id", [None])[0]

            if user_id is None:
                self.send_response(400)
                self.end_headers()
                self.wfile.write(b"Missing user_id parameter")
                return

            try:
                # Ensure user_id is a string
                user_id = str(user_id)

                # Encode user_id using user_encoder
                if user_id in user_encoder.classes_:
                    encoded_user_id = user_encoder.transform([user_id])[0]
                else:
                    raise ValueError(f"❌ User ID '{user_id}' not found in training data!")

                latest_model = get_latest_model()
                if not latest_model:
                    self.send_response(500)
                    self.end_headers()
                    self.wfile.write(b"No trained model found")
                    return

                # Define all_product_ids before using it
                all_product_ids = list(item_encoder.classes_)

                print(f"✅ latest_model: {latest_model}")
                print(f"✅ all_product_ids: {len(all_product_ids)} items")
                print(f"✅ Calling recommend() with user_id: {user_id}")

                # Use encoded user_id for recommendation
                recommendations = recommend(user_id, all_product_ids, latest_model)
                response = json.dumps({"user_id": user_id, "recommendations": recommendations})

                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(response.encode())

            except ValueError as e:
                self.send_response(400)
                self.end_headers()
                self.wfile.write(f"Invalid user_id: {str(e)}".encode())

        else:
            self.send_response(404)
            self.end_headers()
            self.wfile.write(b"Endpoint not found")

if __name__ == "__main__":
    server = HTTPServer(("0.0.0.0", 8080), RequestHandler)
    print("🚀 Serving API at http://localhost:8080")
    server.serve_forever()
