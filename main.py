import sys
import os
from PyQt5 import QtWidgets, QtGui, QtCore
from PyQt5.QtWidgets import QLabel, QPushButton, QFileDialog, QVBoxLayout, QWidget, QComboBox
from keras.models import load_model
from keras.preprocessing import image
from keras.applications.densenet import DenseNet201, preprocess_input
import numpy as np
import pickle
from googletrans import Translator
import tensorflow as tf
import re
import argostranslate.package
import argostranslate.translate
# Load shared tokenizer and features
with open('./flick8r/saved_models/dense/tokenizer.pkl', 'rb') as f:
    tokenizer = pickle.load(f)

with open('./flick8r/saved_models/dense/features.pkl', 'rb') as f:
    all_features = pickle.load(f)
max_length = 34

# Load feature extractor
feature_extractor = DenseNet201(weights='imagenet', include_top=False, pooling='avg')

def extract_features_new_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    features = feature_extractor.predict(img_array)
    return np.reshape(features, features.shape[1])

def predict_caption(model, tokenizer, photo, max_length):
    in_text = 'startseq'
    for i in range(max_length):
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        sequence = np.pad(sequence, (0, max_length - len(sequence)), 'constant')
        sequence = np.expand_dims(sequence, axis=0)
        yhat = model.predict([np.expand_dims(photo, axis=0), sequence], verbose=0)
        yhat = np.argmax(yhat)
        word = tokenizer.index_word.get(yhat)
        if word is None:
            break
        in_text += ' ' + word
        if word == 'endseq':
            break
    return in_text


def clean_caption(caption):
    caption_clean = caption.replace('startseq', '').replace('endseq', '').strip()
    words = caption_clean.split()
    cleaned_words = [words[0]] + [words[i] for i in range(1, len(words)) if words[i] != words[i - 1]] if words else []
    return ' '.join(cleaned_words)

def translate_to_arabic(text):
    translator = Translator()
    translated = translator.translate(text, src='en', dest='ar')
    return translated.text

class CaptionApp(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()
        self.image_path = None
        self.model = None

    def initUI(self):
        self.setWindowTitle("Image Caption Generator & Translator")

        layout = QVBoxLayout()

        self.image_label = QLabel("No image uploaded")
        self.image_label.setAlignment(QtCore.Qt.AlignCenter)
        layout.addWidget(self.image_label)

        self.upload_btn = QPushButton("Upload Image")
        self.upload_btn.clicked.connect(self.upload_image)
        layout.addWidget(self.upload_btn)

        self.model_selector = QComboBox()
        self.model_selector.addItems(["cnn+lstm", "NIC", "SHOW AND TELL", "transformer"])
        layout.addWidget(self.model_selector)

        self.caption_btn = QPushButton("Generate Caption")
        self.caption_btn.clicked.connect(self.generate_caption)
        layout.addWidget(self.caption_btn)

        self.caption_label = QLabel("Caption: ")
        layout.addWidget(self.caption_label)

        self.translate_google_btn = QPushButton("Translate using Google")
        self.translate_google_btn.clicked.connect(self.translate_google)
        layout.addWidget(self.translate_google_btn)

        self.translate_model_btn = QPushButton("Translate using Model")
        self.translate_model_btn.clicked.connect(self.translate_model)
        layout.addWidget(self.translate_model_btn)

        self.translation_label = QLabel("Translation: ")
        layout.addWidget(self.translation_label)

        self.setLayout(layout)

    def upload_image(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Open Image", "", "Images (*.png *.xpm *.jpg *.jpeg)")
        if file_path:
            self.image_path = file_path
            pixmap = QtGui.QPixmap(file_path).scaled(300, 300)
            self.image_label.setPixmap(pixmap)

    def load_selected_model(self):
        selection = self.model_selector.currentText().lower()
        model_path = {
            "cnn+lstm": "model_dense.h5",
            "nic": "model_gru.h5",
            "show and tell": "model_bid.h5",
            "transformer": "model_transform.h5"
        }.get(selection, "model_dense.h5")
        return load_model(model_path)

    def generate_caption(self):
        if not self.image_path:
            self.caption_label.setText("Caption: No image selected")
            return

        image_name = os.path.basename(self.image_path)
        if image_name in all_features:
            features = all_features[image_name]
            if len(features.shape) > 1:
                features = np.reshape(features, (features.shape[1],))
        else:
            features = extract_features_new_image(self.image_path)

        model = self.load_selected_model()
        caption = predict_caption(model, tokenizer, features, max_length)
        cleaned_caption = clean_caption(caption)
        self.caption_label.setText(f"Caption: {cleaned_caption}")

    def translate_google(self):
        text = self.caption_label.text().replace("Caption: ", "")
        if text:
            translated = translate_to_arabic(text)
            self.translation_label.setText(f"Translation: {translated}")


    def translate_model(self):
    # Ensure English→Arabic package is installed
     argostranslate.package.update_package_index()
     available_packages = argostranslate.package.get_available_packages()
     package_to_install = next(
        (pkg for pkg in available_packages if pkg.from_code == "en" and pkg.to_code == "ar"),
        None
     )
     if package_to_install:
        downloaded_path = package_to_install.download()
        argostranslate.package.install_from_path(downloaded_path)
    # Get installed languages
     installed_languages = argostranslate.translate.get_installed_languages()
     from_lang = next((lang for lang in installed_languages if lang.code == "en"), None)
     to_lang = next((lang for lang in installed_languages if lang.code == "ar"), None)
     if from_lang and to_lang:
        translation = from_lang.get_translation(to_lang)
        text = self.caption_label.text().replace("Caption: ", "")
        translated_text = translation.translate(text)
        self.translation_label.setText(f"Translation: {translated_text}")
     else:
        self.translation_label.setText("Translation: Language pair not available")



if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    window = CaptionApp()
    window.show()
    sys.exit(app.exec_())
