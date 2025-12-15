import numpy as np
import cv2
import joblib
import base64
import io
from flask import Flask, request, render_template
from PIL import Image
from skimage.feature import local_binary_pattern

app = Flask(__name__)

try:
    model = joblib.load("svm_fruit_model.pkl")
    pca = joblib.load("pca_transformer.pkl")
except Exception as e:
    print(f"Error loading model: {e}")

CLASS_NAMES = [
    "Avocado",
    "Avocado ripe",
    "Banana Lady",
    "Banana Red",
    "Banana Yellow",
    "Carambula",
    "Cherimoya",
    "Dates",
    "Fig",
    "Guava",
    "Kaki",
    "Kiwi",
    "Lychee",
    "Mango",
    "Mango Red",
    "Mangostan",
    "Papaya",
    "Pineapple",
    "Pineapple Mini",
    "Pomegranate",
    "Quince",
    "Rambutan",
    "Salak",
]

def extract_color_histogram(img, bins=(8, 8, 8)):
    hist = cv2.calcHist([img], [0, 1, 2], None, bins, [0, 256, 0, 256, 0, 256])
    hist = cv2.normalize(hist, hist).flatten()

    return hist


def extract_lbp_features(gray_img, P=8, R=1, method="uniform"):
    lbp = local_binary_pattern(gray_img, P, R, method)
    hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, P + 3), range=(0, P + 2))
    hist = hist.astype("float")
    hist /= hist.sum() + 1e-6
    return hist


def preprocessing_pipeline(pil_img):
    img = np.array(pil_img.convert('RGB'))
    img = cv2.resize(img, (100, 100), interpolation=cv2.INTER_AREA)

    img_float = img.astype(np.float32) / 255.0
    img_uint8 = (img_float * 255).astype(np.uint8)

    feat_color = extract_color_histogram(img_uint8)
    img_gray = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2GRAY)
    feat_lbp = extract_lbp_features(img_gray)

    combined = np.hstack([feat_color, feat_lbp])
    combined = combined.reshape(1, -1)

    final_features = pca.transform(combined)

    return final_features


@app.route("/", methods=["GET", "POST"])
def index():
    prediction_text = None
    img_data = None
    confidence = None
    top_3_predictions = None

    if request.method == "POST":
        if "file" not in request.files:
            return render_template("index.html", msg="Tidak ada file")

        file = request.files["file"]

        if file.filename == "":
            return render_template("index.html", msg="Nama file kosong")

        if file:
            try:
                image = Image.open(file.stream)
                
                features = preprocessing_pipeline(image)
                
                preprocessed_img = cv2.resize(np.array(image.convert('RGB')), (100, 100), interpolation=cv2.INTER_AREA)
                preprocessed_pil = Image.fromarray(preprocessed_img)
                
                img_io = io.BytesIO()
                preprocessed_pil.save(img_io, "PNG")
                encoded_img = base64.b64encode(img_io.getvalue()).decode("ascii")
                img_data = f"data:image/png;base64, {encoded_img}"
                
                pred_index = model.predict(features)[0]
                prediction_text = CLASS_NAMES[int(pred_index)]
                
                if hasattr(model, 'predict_proba'):
                    probabilities = model.predict_proba(features)[0]
                    confidence = float(probabilities[int(pred_index)]) * 100
                    
                    top_3_indices = probabilities.argsort()[-3:][::-1]
                    top_3_predictions = [
                        {
                            'name': CLASS_NAMES[idx],
                            'probability': float(probabilities[idx]) * 100
                        }
                        for idx in top_3_indices
                    ]
                else:
                    confidence = None
                    top_3_predictions = None

            except Exception as e:
                prediction_text = f"Error: {str(e)}"
                confidence = None
                top_3_predictions = None

    return render_template("index.html", prediction=prediction_text, img_data=img_data, 
                         confidence=confidence, top_predictions=top_3_predictions)


if __name__ == "__main__":
    app.run(debug=True, port=7860)
