# Ensemble Image Classifier — Web App

VGG16 + MobileNetV3Small ensemble মডেল (`ensemble_vgg16_mobilenetv3.h5`) দিয়ে বানানো একটি Flask ওয়েব অ্যাপ। ব্রাউজার থেকে ছবি আপলোড করলে মডেল প্রেডিকশন এবং কনফিডেন্স দেখাবে।

## ফাইল স্ট্রাকচার

```
webapp/
├── app.py                  # Flask ব্যাকএন্ড (মডেল লোড + /predict API)
├── convert_model.py        # ONE-TIME স্ক্রিপ্ট: .h5 বাগ ফিক্স করে .keras বানায়
├── requirements.txt        # প্রয়োজনীয় প্যাকেজ
├── Procfile                # Render/Railway/Heroku ডিপ্লয়মেন্ট
├── Dockerfile              # Docker ডিপ্লয়মেন্ট
├── model/
│   ├── ensemble_vgg16_mobilenetv3.h5   # আপনার অরিজিনাল আপলোড করা মডেল
│   └── ensemble_fixed.keras            # ফিক্সড ভার্সন (ইতিমধ্যে তৈরি করে দেওয়া আছে)
├── templates/
│   └── index.html          # মেইন পেজ (UI)
├── static/
│   ├── style.css           # স্টাইলিং
│   └── script.js           # আপলোড + fetch("/predict") লজিক
└── README.md
```

## ⚠️ গুরুত্বপূর্ণ: .h5 ফাইলে একটা Keras 3 bug ছিল (সমাধান করা হয়েছে)

আপনার আপলোড করা `.h5` ফাইলটা সরাসরি `tf.keras.models.load_model()` দিয়ে লোড
করলে এই এরর আসে:

```
AttributeError: 'list' object has no attribute 'shape'  (Flatten.call এ)
```

এটা **Keras 3-এর একটা আসল bug** — VGG16-এর মতো নেস্টেড Functional মডেল যখন
একটা Sequential মডেলের ভেতরে থাকে, legacy `.h5` ফরম্যাট লোডার সেটা ঠিকমতো
handle করতে পারে না। এটা লেটেস্ট TensorFlow/Keras ভার্সনেও (২.২১ / keras
৩.১৫) একইভাবে হয় — তাই ভার্সন আপগ্রেড করলে এটা ঠিক হবে না।

**সমাধান:** `convert_model.py` স্ক্রিপ্টটা আর্কিটেকচার কোড দিয়ে আবার বানায়,
`.h5` থেকে শুধু ওয়েট (numpy array) গুলো সরাসরি বসিয়ে দেয়, তারপর নতুন
native `.keras` ফরম্যাটে সেভ করে (এই ফরম্যাট নেস্টেড মডেল ঠিকমতো handle
করে)। **এই ফিক্সড `.keras` ফাইলটা ইতিমধ্যে zip-এর ভেতরে দিয়ে দেওয়া হয়েছে**
(`model/ensemble_fixed.keras`), তাই আপনার আলাদা করে কিছু চালানোর দরকার নেই
— `app.py` সরাসরি রান করলেই কাজ করবে।

যদি ভবিষ্যতে নতুন `.h5` ফাইল (রি-ট্রেইন করার পর) দিয়ে আবার কনভার্ট করতে
চান:
```bash
python convert_model.py
```
এটা `model/ensemble_vgg16_mobilenetv3.h5` থেকে পড়ে নতুন
`model/ensemble_fixed.keras` তৈরি করে দেবে (আউটপুটে একটা sanity-check
prediction ও দেখাবে)।

## মডেল সম্পর্কে (h5 ফাইল ইন্সপেক্ট করে যা পাওয়া গেছে)

- Framework: **Keras 3 (tensorflow backend)**, `keras_version: 3.10.0`
- Input shape: `(224, 224, 3)`
- Architecture: একটাই Input দুইটা branch-এ যায় —
  - Branch 1: VGG16 → Flatten → Dense(256, relu) → Dropout → Dense(2, softmax)
  - Branch 2: MobileNetV3Small → GlobalAveragePooling → Dense(128, relu) → Dropout → Dense(2, softmax)
  - দুই branch-এর আউটপুট `Average` layer দিয়ে গড় করা হয় → final output shape `(None, 2)`
- অর্থাৎ এটা একটা **২-ক্লাস (বাইনারি) ক্লাসিফায়ার**।

### ⚠️ দুটো জিনিস আপনাকে অবশ্যই ঠিক করতে হবে

1. **CLASS_NAMES** (`app.py`-এর উপরের দিকে):
   এই মডেলে ক্লাসের নাম সংরক্ষিত নেই (শুধু ইনডেক্স 0, 1)। আপনার training কোডে
   `train_generator.class_indices` প্রিন্ট করলে যে ম্যাপিং পাবেন, ঠিক সেই ক্রম অনুযায়ী
   `CLASS_NAMES = ["আপনার_ক্লাস_০", "আপনার_ক্লাস_১"]` বসিয়ে দিন। ভুল ক্রমে বসালে
   প্রেডিকশনের লেবেল উল্টে যাবে (probability ঠিক থাকবে, শুধু নামটা উল্টাবে)।

2. **Preprocessing** (`app.py`-এর `preprocess_image()` ফাংশন):
   মডেলের গঠন দেখে বোঝা যাচ্ছে VGG16 branch-এ কোনো built-in normalization layer
   নেই, কিন্তু MobileNetV3Small branch-এর ভেতরে একটা `Rescaling` layer আছে (যেটা
   raw 0-255 পিক্সেল আশা করে)। যেহেতু দুই branch একই input শেয়ার করছে, ধরে নেওয়া
   হয়েছে training-এর সময় raw pixel (0-255, float32) সরাসরি দেওয়া হয়েছিল — কোনো
   `/255` বা `vgg16.preprocess_input` করা হয়নি। এই অনুমান অনুযায়ী কোড লেখা হয়েছে।

   👉 **যদি প্রেডিকশন ভুল/এলোমেলো মনে হয়**, আপনার training notebook-এ
   `ImageDataGenerator(rescale=...)` বা `preprocess_input` ব্যবহার হয়েছিল কিনা চেক
   করুন, এবং `preprocess_image()` ফাংশনে সেই একই normalization বসিয়ে দিন। যেমন:
   ```python
   arr = np.array(img).astype("float32") / 255.0
   ```

## লোকালি রান করা

```bash
cd webapp

# ভার্চুয়াল এনভায়রনমেন্ট (ঐচ্ছিক কিন্তু ভালো অভ্যাস)
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# ডিপেন্ডেন্সি ইনস্টল
pip install -r requirements.txt

# সার্ভার চালু করুন
python app.py
```

তারপর ব্রাউজারে যান: **http://localhost:5000**

## প্রোডাকশন ডিপ্লয়মেন্ট

`debug=True` দিয়ে সরাসরি Flask সার্ভার প্রোডাকশনের জন্য উপযুক্ত না। Gunicorn দিয়ে চালান:

```bash
gunicorn -w 2 -b 0.0.0.0:8000 --timeout 120 app:app
```

- `-w 2` → 2টা worker (মডেল বড় হওয়ায় বেশি worker দিলে RAM বেশি লাগবে; সার্ভারের RAM
  অনুযায়ী সংখ্যা ঠিক করুন)
- `--timeout 120` → প্রথম রিকোয়েস্টে মডেল লোড/predict একটু সময় নিতে পারে

### কোথায় হোস্ট করবেন

- **Render / Railway**: `requirements.txt` + একটা `Procfile` লাগবে:
  ```
  web: gunicorn -w 2 -b 0.0.0.0:$PORT app:app
  ```
- **Hugging Face Spaces (Docker/Gradio SDK এর বদলে Docker টাইপ বেছে নিন)**: এই
  Flask স্ট্রাকচার সরাসরি কাজ করবে, শুধু একটা `Dockerfile` যোগ করতে হবে।
- **AWS EC2 / DigitalOcean VPS**: gunicorn + nginx reverse proxy দিয়ে চালানো
  যাবে (স্ট্যান্ডার্ড Flask deployment)।
- **⚠️ Vercel/Netlify (serverless) এড়িয়ে চলুন** — মডেল ফাইল ৮৯MB এবং TensorFlow
  লোড টাইম বেশি হওয়ায় সাধারণ serverless function limit-এ আটকে যাবে।

### Dockerfile (চাইলে ব্যবহার করতে পারেন)

```dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000
CMD ["gunicorn", "-w", "2", "-b", "0.0.0.0:8000", "--timeout", "120", "app:app"]
```

## API এন্ডপয়েন্ট

- `GET /` → মেইন UI পেজ
- `POST /predict` → `multipart/form-data` তে `file` ফিল্ডে ইমেজ পাঠাতে হবে
  ```json
  {
    "predicted_class": "Class_0",
    "confidence": 0.9231,
    "probabilities": { "Class_0": 0.9231, "Class_1": 0.0769 }
  }
  ```
- `GET /health` → হেলথ চেক (`{"status": "ok"}`)

## সাধারণ সমস্যা ও সমাধান

| সমস্যা | কারণ / সমাধান |
|---|---|
| মডেল লোড হতে অনেক সময় নেয় | স্বাভাবিক — 89MB TensorFlow মডেল, প্রথমবার লোড ধীর হবে |
| `ValueError: Unknown layer` জাতীয় এরর | TensorFlow ভার্সন মিসম্যাচ — `requirements.txt`-এ দেওয়া `tensorflow==2.16.1` (Keras 3) ভার্সনটাই ব্যবহার করুন |
| সব ছবিতেই একই ক্লাস আসছে | Preprocessing ভুল হতে পারে — উপরের "Preprocessing" নোট দেখুন |
| Out of memory | সার্ভারে কমপক্ষে 2GB RAM রাখুন, worker সংখ্যা কমান |
| `FileNotFoundError: model/ensemble_fixed.keras পাওয়া যায়নি` | zip থেকে `model/` ফোল্ডার ঠিকভাবে extract হয়নি — চেক করুন `ensemble_fixed.keras` ফাইলটা `model/` ফোল্ডারে আছে কিনা, না থাকলে `python convert_model.py` চালান |
| `AttributeError: 'list' object has no attribute 'shape'` | এটা পুরনো `.h5` ফাইল সরাসরি লোড করার চেষ্টা করলে হয় — `app.py` তে `MODEL_PATH` ঠিক আছে কিনা (`.keras` ফাইল, `.h5` না) চেক করুন |
