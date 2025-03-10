from flask import Flask, request, render_template
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression

# ✅ Initialize Flask App
app = Flask(__name__)

# ✅ Load the dataset to preprocess new inputs
file_path = "final_data.csv"
df = pd.read_csv(file_path)

# ✅ Define feature columns
num_cols = ["pH", "N", "P", "K", "OC", "Particles", "Water_holding_content"]
cat_cols = ["Soil_type"]

# ✅ Create preprocessing pipeline
preprocessor = ColumnTransformer([
    ("num", StandardScaler(), num_cols),
    ("cat", OneHotEncoder(), cat_cols)
])

# ✅ Train Multiple Models
X = df.drop(columns=["crop_type"])
y = df["crop_type"]

# Encode target labels (crop_type) from strings to numeric values
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Preprocess the features (X)
X_processed = preprocessor.fit_transform(X)

# Define and train models
models = {
    "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42),
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric="mlogloss"),
    "SVM": SVC(kernel='rbf', probability=True),
    "DecisionTree": DecisionTreeClassifier(random_state=42),
    "LogisticRegression": LogisticRegression(max_iter=500)
}

# Train and save models
for name, model in models.items():
    model.fit(X_processed, y_encoded)
    with open(f"{name}_model.pkl", "wb") as model_file:
        pickle.dump(model, model_file)

# ✅ Home Route
@app.route('/')
def home():
    return render_template('index.html')

# ✅ Prediction Route
@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Get form data
        data = {
            "pH": [float(request.form["pH"])],
            "N": [float(request.form["N"])],
            "P": [float(request.form["P"])],
            "K": [float(request.form["K"])],
            "OC": [float(request.form["OC"])],
            "Particles": [float(request.form["Particles"])],
            "Water_holding_content": [float(request.form["Water_holding_content"])],
            "Soil_type": [request.form["Soil_type"]]
        }

        # Convert to DataFrame
        input_data = pd.DataFrame(data)

        # Preprocess input
        input_processed = preprocessor.transform(input_data)

        # Load the best model (change to preferred model)
        with open("XGBoost_model.pkl", "rb") as model_file:
            model = pickle.load(model_file)

        # Make prediction
        prediction = model.predict(input_processed)

        # Convert the numeric prediction back to the original label
        predicted_label = label_encoder.inverse_transform(prediction)

        return render_template('result.html', prediction=predicted_label[0])

# ✅ Run the Flask App
if __name__ == '__main__':
    app.run(debug=False, port=8000, use_reloader=False)

