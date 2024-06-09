import spacy
from textblob import TextBlob
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import tkinter as tk
from tkinter import Text
from tkinter import PhotoImage
from tkinter import font
from PIL import Image, ImageTk
import requests
import os

# Load the English language model
nlp = spacy.load("en_core_web_sm")

# Function to extract nouns and keywords from a dream description
def extract_nouns_and_keywords(description):
    doc = nlp(description)
    
    # Extract nouns (tokens with part-of-speech tag "NOUN")
    nouns = [token.text for token in doc if token.pos_ == "NOUN"]
    
    words = description.lower().split()
    
    word_freq = {}
    for word in words:
        if word in word_freq:
            word_freq[word] += 1
        else:
            word_freq[word] = 1
    
    sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
    
    num_keywords = 5
    extracted_keywords = [word for word, freq in sorted_words[:num_keywords]]
    
    return nouns, extracted_keywords

# Function to extract named entities from a dream description
def extract_named_entities(description):
    doc = nlp(description)
    
    named_entities = [(ent.text, ent.label_) for ent in doc.ents]
    
    return named_entities

# Function to analyze sentiment of the dream description
def analyze_sentiment(description):
    blob = TextBlob(description)
    sentiment_score = blob.sentiment.polarity
    sentiment_label = "Positive" if sentiment_score > 0 else "Negative" if sentiment_score < 0 else "Neutral"
    return sentiment_score, sentiment_label

def process_dream():
    global image_visible
    window.geometry("800x600")
    # Get the dream description from the GUI
    dream_description = dream_entry.get("1.0", "end-1c")

    response = requests.get(f"https://image.pollinations.ai/prompt/{dream_description}")

    if response.status_code == 200:
        with open("rsc/generated_image.jpg", "wb") as image_file:
            image_file.write(response.content)

    if os.path.exists("rsc/generated_image.jpg"):
        img = Image.open("rsc/generated_image.jpg")
        img = img.resize((150, 150), Image.LANCZOS)
        photo = ImageTk.PhotoImage(img)
        image_label.config(image=photo)
        image_label.image = photo

    nouns, keywords = extract_nouns_and_keywords(dream_description)
    named_entities = extract_named_entities(dream_description)

    nouns_text.config(text="Nouns in the dream description:\n" + ", ".join(nouns))
    keywords_text.config(text="Extracted keywords:\n" + ", ".join(keywords))

    named_entities_label.config(text="Named Entities in the dream description:")
    named_entities_text.config(text="")

    # Update the formatted content
    for entity, label in named_entities:
        named_entities_text.config(text=named_entities_text.cget("text") + f"{entity} - {label}\n")
        
    sentiment_score, sentiment_label = analyze_sentiment(dream_description) 
    sentiment_score_text.config(text="Sentiment Score: " + str(sentiment_score))
    sentiment_label_text.config(text="Sentiment Label: " + sentiment_label)

def hide_welcome_page():
    welcome_label.pack_forget()
    img_label.pack_forget()
    intro_label.pack_forget()
    enter_button.pack_forget()

    dream_label.pack(pady=5)
    dream_entry.pack(pady=5)
    analyze_button.pack(pady=10)
    image_label.pack(pady=5)
    nouns_text.pack(pady=5)
    keywords_text.pack(pady=5)
    named_entities_label.pack(pady=5)
    named_entities_text.pack(pady=5)
    sentiment_score_text.pack(pady=5)
    sentiment_label_text.pack(pady=5)



# Create the GUI
window = tk.Tk()
window.title("Dream Analyzer")
window.geometry("800x300")
window.iconbitmap("rsc/icon.ico")

# Styling

# font
google_fonts_url = "https://fonts.googleapis.com/css2?family=Rethink+Sans:wght@400;700&display=swap"
font.nametofont("TkDefaultFont").actual()
custom_font = font.Font(family="Rethink Sans", size=12)

# welcome image

fore_img = Image.open("rsc/drm.png")
fore_img = fore_img.resize((100, 100), Image.LANCZOS)
fore_photo = ImageTk.PhotoImage(fore_img)

# Page 1
welcome_label = tk.Label(window, text="Welcome to the Dream Analyzer!", font=(custom_font, 12))
welcome_label.pack(anchor="center", pady=5)

img_label = tk.Label(window, image=fore_photo)
img_label.pack(anchor="center", pady=5)

intro_label = tk.Label(window, text="Created by:\n1. Fitra\n2. Raka\n3. Wilsent", font=(custom_font, 12))
intro_label.pack(anchor="center", pady=5)

enter_button = tk.Button(window, text="Enter", comman=hide_welcome_page, font=(custom_font, 12))
enter_button.pack(anchor="center", pady=5)

# Page 2
dream_label = tk.Label(window, text="Enter your dream description:", font=(custom_font, 12))

dream_entry = Text(window, width=30, height=5)

analyze_button = tk.Button(window, text = "Analyze Dream", command = process_dream, font=(custom_font, 12))

image_label = tk.Label(window, image = "")

nouns_text = tk.Label(window, text = "", font=(custom_font, 10))

keywords_text = tk.Label(window, text = "", font=(custom_font, 10))

named_entities_label = tk.Label(window, text="", font=(custom_font, 10))

named_entities_text = tk.Label(window, text="", font=(custom_font, 10))

sentiment_score_text = tk.Label(window, text = "", font=(custom_font, 10))

sentiment_label_text = tk.Label(window, text = "", font=(custom_font, 10))

window.mainloop()


# Positive dream
# I floated on clouds of cotton candy, basking in a golden sunset. Laughter echoed as friendly butterflies danced around me, 
# painting the sky with shimmering hues. It was a dream of pure joy, warmth, and endless serenity

# Negative dream
# In the cold morning fog, I stumbled through a desolate cemetery. Gravestones loomed like accusing shadows,
# and a chilling wind whispered tales of forgotten sorrow. It was a morning of haunting despair, where every step echoed 
# with the weight of a restless darkness
