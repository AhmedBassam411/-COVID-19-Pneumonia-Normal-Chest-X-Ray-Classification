import os
import uuid
import numpy as np
import tensorflow as tf
from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
from PIL import Image
import matplotlib.cm as cm

# ===================== CONFIG =====================
IMG_SIZE = (224, 224)
MODEL_PATH = "cnn_model.keras"
UPLOAD_FOLDER = "static/uploads"
GRADCAM_FOLDER = "static/gradcam"

CLASS_NAMES = ["COVID19", "NORMAL", "PNEUMONIA"]

# ===================== APP =====================
app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(GRADCAM_FOLDER, exist_ok=True)

# ===================== LOAD MODEL =====================
model = tf.keras.models.load_model(MODEL_PATH, compile=False)

# ===================== IMAGE PREPROCESS =====================
def preprocess_image(image_path):
    img = Image.open(image_path).convert("RGB")
    img = img.resize(IMG_SIZE)
    img = np.array(img) / 255.0
    return np.expand_dims(img, axis=0)

# ===================== GRAD-CAM =====================
def generate_gradcam(model, img_array, class_index):
    # Find last convolutional layer
    last_conv_layer = None
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            last_conv_layer = layer.name
            break

    grad_model = tf.keras.models.Model(
        model.inputs,
        [model.get_layer(last_conv_layer).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_output, predictions = grad_model(img_array)
        loss = predictions[:, class_index]

    grads = tape.gradient(loss, conv_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_output = conv_output[0]
    heatmap = tf.reduce_sum(conv_output * pooled_grads, axis=-1)

    heatmap = tf.maximum(heatmap, 0)
    heatmap /= tf.reduce_max(heatmap) + 1e-8

    return heatmap.numpy()

def overlay_gradcam(image_path, heatmap):
    original_img = Image.open(image_path).convert("RGB")
    original_img = original_img.resize(IMG_SIZE)

    heatmap = Image.fromarray(np.uint8(255 * heatmap))
    heatmap = heatmap.resize(IMG_SIZE)

    colormap = cm.get_cmap("jet")
    heatmap = colormap(np.array(heatmap) / 255.0)
    heatmap = np.uint8(heatmap[:, :, :3] * 255)
    heatmap = Image.fromarray(heatmap)

    blended = Image.blend(original_img, heatmap, alpha=0.4)
    return blended

# ===================== ROUTES =====================
@app.route("/", methods=["GET", "POST"])
def index():
    result = {}

    if request.method == "POST":
        file = request.files.get("image")

        if not file or file.filename == "":
            result["error"] = "Please upload a valid image."
        else:
            filename = secure_filename(file.filename)
            unique_name = f"{uuid.uuid4().hex}_{filename}"
            image_path = os.path.join(UPLOAD_FOLDER, unique_name)
            file.save(image_path)

            img_array = preprocess_image(image_path)
            preds = model.predict(img_array, verbose=0)[0]

            class_idx = int(np.argmax(preds))

            heatmap = generate_gradcam(model, img_array, class_idx)
            gradcam_img = overlay_gradcam(image_path, heatmap)

            gradcam_name = f"gradcam_{unique_name}"
            gradcam_path = os.path.join(GRADCAM_FOLDER, gradcam_name)
            gradcam_img.save(gradcam_path)

            result = {
                "prediction": CLASS_NAMES[class_idx],
                "confidence": round(float(preds[class_idx]) * 100, 2),
                "probs": {
                    CLASS_NAMES[i]: round(float(preds[i]) * 100, 2)
                    for i in range(len(CLASS_NAMES))
                },
                "image_url": image_path,
                "gradcam_url": gradcam_path
            }

    return render_template("index.html", result=result)

# ===================== MAIN =====================
if __name__ == "__main__":
    app.run(debug=False)
