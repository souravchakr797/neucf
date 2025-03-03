# NeuCF Model for Collaborative Filtering

This project implements **Neural Collaborative Filtering (NeuCF)** for recommendation systems using TensorFlow and Keras. It combines **Generalized Matrix Factorization (GMF)** and **Multi-Layer Perceptron (MLP)** to enhance user-item interaction predictions.

## 📌 Features
- **Negative Sampling**: Efficiently generates negative samples for training.
- **Hybrid Model (GMF + MLP)**: Merges matrix factorization and deep learning-based approaches.
- **Batch Normalization & Dropout**: Helps improve model training stability and generalization.
- **Pretrained Model Loading**: Supports loading the latest trained NeuCF model.

## 📂 Project Structure
```
├── models/                   # Stores trained models
├── src/
│   ├── train.py              # Script to train the NeuCF model
│   ├── data_loader.py        # Handles dataset loading and preprocessing
│   ├── model.py              # Contains the NeuCF model definition
│   ├── predict.py             # To predicr
├── main.py                   # Starts the FastAPI server
├── requirements.txt          # Dependencies
└── README.md                 # Project documentation
```

## 🚀 How to Train the Model
Run the following command to train the NeuCF model:
```sh
python src/train.py
```
This will:
1. Load user-item interaction data.
2. Generate negative samples.
3. Train the NeuCF model.
4. Save the trained model in the `models/` directory.

## 🖥 Running the Server
To serve recommendations via an API, run:
```sh
python main.py
```
This will start a API server to make predictions using the trained NeuCF model.

## 📌 Model Architecture
The model combines:
- **GMF (Generalized Matrix Factorization)**: Uses embeddings and element-wise multiplication.
- **MLP (Multi-Layer Perceptron)**: Uses deep learning layers for non-linear feature interactions.
- **Final Layer**: Merges GMF & MLP outputs and predicts the probability of interaction.

## 📦 Dependencies
Ensure the required dependencies are installed:
```sh
pip install -r requirements.txt
```

## 📜 License
This project is open-source and available for use under the MIT License.

