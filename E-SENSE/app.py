import streamlit as st
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification, T5ForConditionalGeneration, T5Tokenizer
import torch
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
import time

#  Load classification model and tokenizer
model_path = 'distilbert_manipulation_model'
tokenizer = DistilBertTokenizerFast.from_pretrained(model_path)
model = DistilBertForSequenceClassification.from_pretrained(model_path)
model.eval()

# Load T5 rewriting model and tokenizer
t5_tokenizer = T5Tokenizer.from_pretrained("t5-small")
t5_model = T5ForConditionalGeneration.from_pretrained("t5-small")
t5_model.eval()

#  Function to predict manipulation
def predict_manipulation(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding='max_length', max_length=64)
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        prediction = torch.argmax(logits, dim=1).item()
        prob = torch.softmax(logits, dim=1).max().item()
    return prediction, prob

#  Ethical rewriting using T5
def ethical_rewrite_t5(text):
    input_text = f"rewrite: {text}"
    input_ids = t5_tokenizer(input_text, return_tensors="pt").input_ids
    with torch.no_grad():
        outputs = t5_model.generate(input_ids, max_length=128, num_beams=4, early_stopping=True)
    return t5_tokenizer.decode(outputs[0], skip_special_tokens=True)

# Ethical scoring function
def get_ethical_score(predicted_label):
    return 0.9 if predicted_label == 1 else 0.2

#  Selenium-based Amazon scraping
def scrape_description(url):
    options = Options()
    options.add_argument('--headless')
    options.add_argument('--no-sandbox')
    options.add_argument('--disable-dev-shm-usage')

    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)
    driver.get(url)
    time.sleep(3)  # Wait for page to load

    # Extract product title
    try:
        title = driver.find_element(By.ID, 'productTitle').text
    except:
        title = ''

    # Extract feature bullets
    try:
        bullets = driver.find_elements(By.CSS_SELECTOR, '#feature-bullets ul li span.a-list-item')
        bullet_text = ' '.join([b.text for b in bullets])
    except:
        bullet_text = ''

    # Extract product description
    try:
        desc = driver.find_element(By.ID, 'productDescription').text
    except:
        desc = ''

    driver.quit()

    # Combine all parts
    combined = ' '.join([title, bullet_text, desc]).strip()
    return combined if combined else "Description not found or unsupported URL structure."

#  Streamlit UI
st.title("🛒 E-Sense: Emotional Shopping Filter")

tab1, tab2 = st.tabs(["📝 Text Input", "🔗 URL Input"])

# Text Input Tab

with tab1:
    st.write("Enter a **product description** to analyze manipulative tactics and ethical score.")
    user_input = st.text_area("Product Description", "")

    if st.button("Analyze Description"):
        if user_input.strip() == "":
            st.warning("Please enter a product description.")
        else:
            pred_label, confidence = predict_manipulation(user_input)
            label_text = "Non-Manipulative ✅" if pred_label == 1 else "Manipulative ⚠️"
            ethical_score = get_ethical_score(pred_label)

            st.markdown(f"### 📝 **Prediction:** {label_text}")
            st.markdown(f"**Model Confidence:** {confidence:.2f}")
            st.markdown(f"**Ethical Score:** {ethical_score:.2f}")

            if pred_label == 0:
                st.error("⚠️ This product description uses manipulative tactics. Consider rewriting it ethically.")
                rewritten = ethical_rewrite_t5(user_input)
                st.markdown("🔎 **Ethical Rewrite Suggestion:**")
                st.success(rewritten)
            else:
                st.success("✅ This product description is ethical and non-manipulative.")

#  URL Input Tab
with tab2:
    st.write("Enter a **product URL (e.g., Amazon)** to fetch and analyze its description.")
    url_input = st.text_input("Product URL", "")

    if st.button("Analyze URL"):
        if url_input.strip() == "":
            st.warning("Please enter a product URL.")
        else:
            with st.spinner("Scraping product description... Please wait."):
                description = scrape_description(url_input)
            st.markdown(f"### 📝 **Scraped Description:**\n{description}")

            if description.startswith("Description not found"):
                st.error("⚠️ Unable to extract description from this URL. Try another product page.")
            else:
                pred_label, confidence = predict_manipulation(description)
                label_text = "Non-Manipulative ✅" if pred_label == 1 else "Manipulative ⚠️"
                ethical_score = get_ethical_score(pred_label)

                st.markdown(f"### 📝 **Prediction:** {label_text}")
                st.markdown(f"**Model Confidence:** {confidence:.2f}")
                st.markdown(f"**Ethical Score:** {ethical_score:.2f}")

                if pred_label == 0:
                    st.error("⚠️ This product description uses manipulative tactics. Consider rewriting it ethically.")
                    rewritten = ethical_rewrite_t5(description)
                    st.markdown("🔎 **Ethical Rewrite Suggestion:**")
                    st.success(rewritten)
                else:
                    st.success("✅ This product description is ethical and non-manipulative.")

# Footer
st.write("---")
st.caption("E-Sense Emotional Shopping Filter ")