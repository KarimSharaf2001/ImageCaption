import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.densenet import DenseNet201, preprocess_input
import pickle
from googletrans import Translator  # لازم تكون مثبت مكتبة googletrans==4.0.0-rc1

# تحميل الموديل والتوكنايزر والفيتشرز
model = load_model('trained_caption__Transformer_model.h5')  
tokenizer = pickle.load(open('tokenizer.pkl', 'rb'))  
all_features = pickle.load(open('features.pkl', 'rb'))  
max_length = 35  # عدل حسب مشروعك

# تحميل DenseNet201 لاستخراج الميزات
feature_extractor = DenseNet201(weights='imagenet', include_top=False, pooling='avg')

# دالة استخراج الفيتشر لصورة جديدة
def extract_features_new_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    features = feature_extractor.predict(img_array)
    features = np.reshape(features, features.shape[1])
    return features
# دالة التنبؤ بالكابشن
def predict_caption(model, tokenizer, photo, max_length):
    in_text = 'startseq'
    for i in range(max_length):
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        sequence = np.pad(sequence, (0, max_length-len(sequence)), 'constant')
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
# دالة توليد الكابشن للصورة
def generate_caption(image_path):
    image_name = image_path.split("/")[-1]  # لو على ويندوز ممكن تحتاج "\\" بدل "/"
    if image_name in all_features:
        features = all_features[image_name]
        if len(features.shape) > 1:
            features = np.reshape(features, (features.shape[1],))  # إعادة تشكيل الفيتشر إلى الشكل الصحيح
    else:
        features = extract_features_new_image(image_path)

    caption = predict_caption(model, tokenizer, features, max_length)
    return caption
# دالة الترجمة باستخدام Google Translate
def translate_to_arabic(text):
    translator = Translator()
    translated = translator.translate(text, src='en', dest='ar')
    return translated.text

# دالة فتح الصورة ومعالجة النتائج
def open_image():
    file_path = filedialog.askopenfilename()
    if file_path:
        img = Image.open(file_path)
        img = img.resize((300, 300))
        img = ImageTk.PhotoImage(img)
        panel.configure(image=img)
        panel.image = img

        # توليد الكابشن
        caption = generate_caption(file_path)
        caption_clean = caption.replace('startseq', '').replace('endseq', '').strip()
          # حذف الكلمات المتكررة المتتالية
        words = caption_clean.split()
        cleaned_words = [words[0]] + [words[i] for i in range(1, len(words)) if words[i] != words[i-1]]
        caption_clean = ' '.join(cleaned_words)

        # ترجمة الكابشن
        translated_caption = translate_to_arabic(caption_clean)

        # عرض النتائج
        caption_label.config(text=f"📝 Caption: {caption_clean}")
        translation_label.config(text=f"🌍 ترجمة: {translated_caption}")

# إنشاء نافذة
root = tk.Tk()
root.title("Image Caption Generator & Translator")
root.configure(bg="#F7F7F7")

# زرار اختيار الصورة
btn = tk.Button(root, text="📷 اختر صورة", command=open_image, bg="green", fg="white", font=("Helvetica", 16))
btn.pack(pady=10)
# مكان عرض الصورة
panel = tk.Label(root)
panel.pack(pady=10)

# مكان عرض الكابشن
caption_label = tk.Label(root, text="", bg="#F7F7F7", font=("Helvetica", 14), wraplength=500, justify="center")
caption_label.pack(pady=10)

# مكان عرض الترجمة
translation_label = tk.Label(root, text="", bg="#F7F7F7", font=("Helvetica", 14), fg="green", wraplength=500, justify="center")
translation_label.pack(pady=10)

root.mainloop()
