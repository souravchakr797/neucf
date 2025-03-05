import json
import os
import pickle
import numpy as np
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import urlparse, parse_qs
from src.predict import recommend_top_5_products

# Load the encoders
def load_pickle(file_path):
    with open(file_path, "rb") as f:
        return pickle.load(f)

user_encoder = load_pickle("models/user_encoder.pkl")
item_encoder = load_pickle("models/item_encoder.pkl")

# Get the latest trained model
def get_latest_model():
    model_dir = "models"
    models = sorted(
        [f for f in os.listdir(model_dir) if f.startswith("neucf_pretrained_")], 
        reverse=True
    )
    return os.path.join(model_dir, models[0]) if models else None

class RequestHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        parsed_path = urlparse(self.path)
        query_params = parse_qs(parsed_path.query)

        if parsed_path.path == "/neucf-recommend":
            user_id = query_params.get("user_id", [None])[0]

            if not user_id:
                self.send_response(400)
                self.end_headers()
                self.wfile.write(b"Missing user_id parameter")
                return

            try:
                # Ensure user_id is valid
                user_id = str(user_id)
                if user_id not in user_encoder.classes_:
                    raise ValueError(f"❌ User ID '{user_id}' not found in training data!")

                latest_model = get_latest_model()
                if not latest_model:
                    self.send_response(500)
                    self.end_headers()
                    self.wfile.write(b"No trained model found")
                    return

                all_product_ids = list(item_encoder.classes_)

                print(f"✅ latest_model: {latest_model}")
                print(f"✅ Calling recommend_top_5_products() for user_id: {user_id}")

                # Call recommend_top_5_products function
                top_5_recommendations = recommend_top_5_products(user_id, latest_model)
                if isinstance(top_5_recommendations, np.ndarray):
                    top_5_recommendations = top_5_recommendations.tolist()

                
                response = json.dumps({"user_id": user_id, "top_5_recommendations": top_5_recommendations})

                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(response.encode())

            except ValueError as e:
                self.send_response(400)
                self.end_headers()
                self.wfile.write(f"Invalid user_id: {str(e)}".encode())

            except Exception as e:
                self.send_response(500)
                self.end_headers()
                self.wfile.write(f"Error processing request: {str(e)}".encode())

        else:
            self.send_response(404)
            self.end_headers()
            self.wfile.write(b"Endpoint not found")

if __name__ == "__main__":
    server = HTTPServer(("0.0.0.0", 8080), RequestHandler)
    print("🚀 Serving API at http://localhost:8080")
    server.serve_forever()
