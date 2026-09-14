"""
convert_model.py
-----------------
এই স্ক্রিপ্টটা ONE-TIME চালাতে হবে (শুধু একবার)।

সমস্যা: `ensemble_vgg16_mobilenetv3.h5` ফাইলে VGG16 (একটা Functional মডেল) একটা
Sequential মডেলের ভেতরে নেস্টেড আছে। Keras 3-এর legacy .h5 লোডারে এই ধরনের
নেস্টেড স্ট্রাকচার লোড করার সময় একটা bug আছে:

    AttributeError: 'list' object has no attribute 'shape'  (Flatten.call এ)

এটা তাজা/লেটেস্ট TensorFlow-Keras (2.21 / keras 3.15) দিয়েও রিপ্রোডিউস হয় —
মানে ভার্সন আপগ্রেড করলেও এটা ঠিক হবে না।

সমাধান: আর্কিটেকচারটা কোড দিয়ে আবার বানিয়ে, .h5 ফাইল থেকে শুধু ওয়েট
(numpy array) গুলো সরাসরি h5py দিয়ে পড়ে প্রতিটা layer-এ বসিয়ে দিচ্ছি,
তারপর নতুন native `.keras` ফরম্যাটে সেভ করছি। .keras ফরম্যাট নেস্টেড মডেল
ঠিকমতো handle করতে পারে, তাই এরপর থেকে app.py তে .h5 লাগবে না,
শুধু `model/ensemble_fixed.keras` লাগবে।

চালানোর নিয়ম:
    python convert_model.py

এটা একবার চালালেই `model/ensemble_fixed.keras` তৈরি হয়ে যাবে, এবং
app.py সেই ফাইলটাই ব্যবহার করবে।
"""

import h5py
import numpy as np
from tensorflow.keras import layers, models, applications

SOURCE_H5 = "model/ensemble_vgg16_mobilenetv3.h5"
TARGET_KERAS = "model/ensemble_fixed.keras"
IMG_SIZE = (224, 224, 3)


def var_short_name(w) -> str:
    """একটা weight variable-এর short name বের করে, যেমন 'kernel' বা 'bias'."""
    n = w.name.split(":")[0]
    return n.split("/")[-1]


def load_weights_for_layer(layer, group: h5py.Group):
    """
    দেওয়া h5py group-এর datasets (kernel/bias/gamma/beta/...) layer-এর
    weight variable order অনুযায়ী সাজিয়ে layer.set_weights() কল করে।
    """
    if not layer.weights:
        return
    arrays = []
    for w in layer.weights:
        short = var_short_name(w)
        if short not in group:
            raise KeyError(
                f"'{short}' পাওয়া যায়নি layer '{layer.name}'-এর জন্য। "
                f"group-এ যা আছে: {list(group.keys())}"
            )
        arrays.append(group[short][()])
    layer.set_weights(arrays)


def main():
    print(f"'{SOURCE_H5}' থেকে ওয়েট পড়া হচ্ছে...")
    f = h5py.File(SOURCE_H5, "r")
    mw = f["model_weights"]

    # ---------------- Branch 1: VGG16 ----------------
    vgg_base = applications.VGG16(include_top=False, weights=None, input_shape=IMG_SIZE)
    branch1 = models.Sequential(
        [
            layers.Input(shape=IMG_SIZE),
            vgg_base,
            layers.Flatten(),
            layers.Dense(256, activation="relu", name="dense"),
            layers.Dropout(0.5),
            layers.Dense(2, activation="softmax", name="dense_1"),
        ],
        name="sequential",
    )

    g_seq = mw["sequential"]
    n1 = 0
    for layer in vgg_base.layers:
        if layer.weights:
            load_weights_for_layer(layer, g_seq[layer.name])
            n1 += 1
    load_weights_for_layer(branch1.get_layer("dense"), g_seq["sequential"]["dense"])
    load_weights_for_layer(branch1.get_layer("dense_1"), g_seq["sequential"]["dense_1"])
    print(f"  ✔ VGG16 branch: {n1}টা conv layer + dense + dense_1 লোড হয়েছে")

    # ---------------- Branch 2: MobileNetV3Small ----------------
    mnet_base = applications.MobileNetV3Small(
        include_top=False, weights=None, input_shape=IMG_SIZE, include_preprocessing=True
    )
    branch2 = models.Sequential(
        [
            layers.Input(shape=IMG_SIZE),
            mnet_base,
            layers.GlobalAveragePooling2D(),
            layers.Dense(128, activation="relu", name="dense_2"),
            layers.Dropout(0.5),
            layers.Dense(2, activation="softmax", name="dense_3"),
        ],
        name="sequential_1",
    )

    g_seq1 = mw["sequential_1"]
    n2 = 0
    missing = []
    for layer in mnet_base.layers:
        if layer.weights:
            if layer.name in g_seq1:
                load_weights_for_layer(layer, g_seq1[layer.name])
                n2 += 1
            else:
                missing.append(layer.name)
    load_weights_for_layer(branch2.get_layer("dense_2"), g_seq1["sequential_1"]["dense_2"])
    load_weights_for_layer(branch2.get_layer("dense_3"), g_seq1["sequential_1"]["dense_3"])
    print(f"  ✔ MobileNetV3Small branch: {n2}টা conv/bn layer + dense_2 + dense_3 লোড হয়েছে")
    if missing:
        print(f"  ⚠ সতর্কতা — এই layer গুলোর ওয়েট h5 তে পাওয়া যায়নি (random init থেকে গেছে): {missing}")

    # ---------------- Combine (ensemble average) ----------------
    inp = layers.Input(shape=IMG_SIZE, name="input_layer_7")
    out1 = branch1(inp)
    out2 = branch2(inp)
    avg = layers.Average(name="average_2")([out1, out2])
    model = models.Model(inputs=inp, outputs=avg)

    # স্যানিটি চেক — একটা ডামি ইনপুট দিয়ে prediction করে দেখা
    dummy = np.random.uniform(0, 255, size=(1, *IMG_SIZE)).astype("float32")
    pred = model.predict(dummy, verbose=0)
    print(f"\n  স্যানিটি চেক prediction (probabilities sum ~1 হওয়া উচিত): {pred}")

    model.save(TARGET_KERAS)
    print(f"\n✅ সফলভাবে সেভ হয়েছে: {TARGET_KERAS}")
    print("এখন থেকে app.py এই ফাইলটাই ব্যবহার করবে (.h5 আর দরকার নেই)।")


if __name__ == "__main__":
    main()
