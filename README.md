<h1 align="center">🧠 Skin Cancer Detection using CNN 🧬</h1>
<p align="center">
  A Deep Learning project to detect different types of skin cancer with just an image! 🌞🧑‍⚕️
</p>

---

## 🚀 Overview

Welcome to this cool project where **AI meets healthcare**!  
We’ve trained a powerful **Convolutional Neural Network (CNN)** using **TensorFlow & Keras** to detect types of **skin cancer** from medical images.

🩺 **Why this matters?**  
Early detection of skin cancer can save lives — and AI can help doctors detect it faster and more accurately 💡

---

## 📂 Dataset

We used the **Skin Cancer ISIC** dataset from The International Skin Imaging Collaboration 🌐

📁 Structure:

- Each folder contains subfolders like:
  - `melanoma/`
  - `nevus/`
  - `seborrheic_keratosis/`
- Each subfolder contains relevant images 🖼️

---

## 🏗️ Model Architecture

Our CNN includes:

🔸 `Rescaling` layer — normalizes pixel values  
🔸 `Conv2D` layers — extract features from images  
🔸 `MaxPooling2D` — downsample image representation  
🔸 `Dropout` — reduce overfitting  
🔸 `Dense` layers — classify the image  

✅ Output: `len(data_cat)` classes

```python
model = Sequential([
    layers.Rescaling(1./255),
    layers.Conv2D(16, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(32, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(64, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Flatten(),
    layers.Dropout(0.25),
    layers.Dense(128),
    layers.Dense(len(data_cat))
])
---

## 🧪 Training Details

📊 **Training Parameters**:

- 📐 Image size: `180 x 180`
- 📦 Batch size: `32`
- 🔁 Epochs: `15`
- 🧠 Loss: `SparseCategoricalCrossentropy`
- 🚀 Optimizer: `Adam`

📈 We plotted both **line** and **scatter** graphs to visualize model accuracy after each epoch!

---

## 🔍 Sample Prediction

Using this sample image:

🧾 **Prediction output:**

---

## 💾 Saved Model

After training, we saved the model as:

✅ You can now reuse it for future predictions or integrate it into a web/mobile app!

---

## 🛠️ How to Run It

1. 🍴 Clone the repository
2. 🔧 Install dependencies:
    ```bash
    pip install tensorflow numpy matplotlib
    ```
3. 🗂️ Set the correct dataset path in the code
4. 🧠 Train the model using:
    ```python
    model.fit(...)
    ```
5. 🔍 Test it with a single image using:
    ```python
    model.predict(...)
    ```

---

## 💡 Future Ideas

✨ Add a GUI using **Tkinter** or **Streamlit**  
📷 Real-time skin detection using **OpenCV**  
📈 Add more performance metrics: `Precision`, `Recall`, `Confusion Matrix`  
🧠 Try **Transfer Learning** with `MobileNet`, `ResNet`, etc.

---

## 🌐 Connect With Me

I'm always happy to connect, collaborate, or discuss ideas 💬

- 📧 Email: **nimmanirishik@gmail.com**
- 💼 [LinkedIn](https://linkedin.com/in/nimmani-rishik-66b632287)
- 📸 [Instagram](https://instagram.com/rishik_3142)

---

## ⭐ Support

If you liked this project or found it helpful, **please leave a star ⭐** — it means a lot and keeps me motivated! 🙏

<p align="center">
  Made with ❤️ by <b>Nimmani Rishik</b>
</p>
