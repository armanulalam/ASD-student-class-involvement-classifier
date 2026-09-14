# Ensemble Image Classifier — Web App

A Flask web application built using the VGG16 + MobileNetV3Small ensemble model (`ensemble_vgg16_mobilenetv3.h5`). Users can upload an image through the browser, and the application will display the model's prediction and confidence.

## File Structure

```text
webapp/

├── app.py                      # Flask backend (model loading + /predict API)
├── convert_model.py            # ONE-TIME script: fixes the .h5 issue and creates a .keras file
├── requirements.txt            # Required packages
├── Procfile                    # Render/Railway/Heroku deployment
├── Dockerfile                  # Docker deployment
├── model/
│   ├── ensemble_vgg16_mobilenetv3.h5   # Original uploaded model
│   └── ensemble_fixed.keras             # Fixed version (already created)
├── templates/
│   └── index.html              # Main page (UI)
├── static/
│   ├── style.css               # Styling
│   └── script.js               # Upload + fetch("/predict") logic
└── README.md
```

## ⚠️ Important: The .h5 File Had a Keras 3 Bug (Fixed)

If the uploaded `.h5` file is loaded directly using `tf.keras.models.load_model()`, the following error occurs:

```text
AttributeError: 'list' object has no attribute 'shape' (Flatten.call)
```

This is an **actual Keras 3 bug**. When a nested Functional model such as VGG16 is placed inside a Sequential model, the legacy `.h5` loader does not handle it correctly.

The same issue occurs even with the latest TensorFlow/Keras versions (2.21 / Keras 3.15), so upgrading the versions will not fix it.

**Solution:** The `convert_model.py` script reconstructs the architecture using code, directly loads the weights (NumPy arrays) from the `.h5` file, assigns them to the newly created model, and then saves the model in the native `.keras` format. This format handles the nested model correctly.

**The fixed `.keras` file has already been included inside the zip file**, so you do not need to run anything separately. Simply run `app.py`, and the application should work.

If you want to convert a new `.h5` file in the future (for example, after retraining the model), run:

```bash
python convert_model.py
```

This will read:

```text
model/ensemble_vgg16_mobilenetv3.h5
```

and create:

```text
model/ensemble_fixed.keras
```

The script will also display a sanity-check prediction in the output.

## Model Information (Based on Inspection of the .h5 File)

The following information was found by inspecting the model:

* Framework: **Keras 3 (TensorFlow backend)**, `keras_version: 3.10.0`
* Input shape: `(224, 224, 3)`
* Architecture: A single input is passed into two branches:

  * **Branch 1:** VGG16 → Flatten → Dense(256, relu) → Dropout → Dense(2, softmax)
  * **Branch 2:** MobileNetV3Small → GlobalAveragePooling → Dense(128, relu) → Dropout → Dense(2, softmax)
  * The outputs of both branches are averaged using an **Average** layer → final output shape `(None, 2)`

Therefore, this is a **2-class (binary) classifier**.



## Running Locally

```bash
cd webapp

# Virtual environment (optional but recommended)
python3 -m venv venv

# Activate the virtual environment
source venv/bin/activate        # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Start the server
python app.py
```

Then open the following address in your browser:

**http://localhost:5000**



## Dockerfile (Optional)

You can use the following Dockerfile if you prefer Docker deployment:

```dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["gunicorn", "-w", "2", "-b", "0.0.0.0:8000", "--timeout", "120", "app:app"]
```

## API Endpoints

### `GET /`

Returns the main UI page.

### `POST /predict`

Accepts an image through `multipart/form-data` using the `file` field.

Example response:

```json
{
  "predicted_class": "Class_0",
  "confidence": 0.9231,
  "probabilities": {
    "Class_0": 0.9231,
    "Class_1": 0.0769
  }
}
```

### `GET /health`

Health-check endpoint:

```json
{
  "status": "ok"
}
```

## Common Problems and Solutions

| Problem                                                       | Cause / Solution                                                                                                                                                                                    |
| ------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Model takes a long time to load                               | This is normal — the model is an 89 MB TensorFlow model, so the initial loading may be slow.                                                                                                        |
| `ValueError: Unknown layer` or similar error                  | TensorFlow version mismatch — use `tensorflow==2.16.1` (Keras 3) as specified in `requirements.txt`.                                                                                                |
| The same class is predicted for every image                   | The preprocessing may be incorrect. Check the preprocessing notes above.                                                                                                                            |
| Out of memory                                                 | Use a server with at least 2 GB RAM and reduce the number of workers.                                                                                                                               |
| `FileNotFoundError: model/ensemble_fixed.keras was not found` | The `model/` folder may not have been extracted correctly from the zip file. Check whether `ensemble_fixed.keras` exists inside the `model/` folder. If it does not, run `python convert_model.py`. |
| `AttributeError: 'list' object has no attribute 'shape'`      | This occurs when trying to load the old `.h5` file directly. Check that `MODEL_PATH` in `app.py` points to the `.keras` file rather than the `.h5` file.                                            |
