import streamlit as st
import pandas as pd
import joblib
import os
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

st.set_page_config(
    page_title="Prismetric Service Chatbot",
    page_icon="🤖",
    layout="wide"
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

@st.cache_resource
def train_model():
    df = pd.read_csv(os.path.join(BASE_DIR, "chatbot_intents_dataset_v2.csv"))
    if "id" in df.columns:
        df = df.drop(columns=["id"])
    X = df["text"].astype(str)
    y = df["intent"].astype(str)
    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(
            ngram_range=(1, 2),
            max_features=5000,
            lowercase=True,
            strip_accents="unicode"
        )),
        ("clf", LogisticRegression(max_iter=1000))
    ])
    pipeline.fit(X, y)
    return pipeline

@st.cache_data
def load_data():
    services = pd.read_csv(os.path.join(BASE_DIR, "prismetric_services.csv"))
    pricing  = pd.read_csv(os.path.join(BASE_DIR, "prismetric_pricing_models.csv"))
    return services, pricing

model = train_model()
services_df, pricing_df = load_data()

CONFIDENCE_THRESHOLD = 0.30

def predict_intent(user_input):
    probabilities = model.predict_proba([user_input])[0]
    classes = model.classes_
    best_index = probabilities.argmax()
    intent = classes[best_index]
    confidence = probabilities[best_index]
    if confidence < CONFIDENCE_THRESHOLD:
        return "fallback_unknown", round(float(confidence), 4)
    return intent, round(float(confidence), 4)

def get_response(intent):
    if intent == "ask_service_list":
        categories = services_df["category"].unique()
        return "Prismetric offers services across: " + ", ".join(categories) + ". Ask me about any specific category."
    if intent == "ask_ai_services":
        ai = services_df[services_df["category"] == "AI Services"]["service_name"].tolist()
        return "Prismetric AI Services include: " + ", ".join(ai) + "."
    if intent == "ask_development_services":
        dev = services_df[services_df["category"] == "Development Services"]["service_name"].tolist()
        return "Prismetric Development Services include: " + ", ".join(dev) + "."
    if intent == "ask_industry_solutions":
        ind = services_df[services_df["category"] == "Industry Solutions"]["service_name"].tolist()
        return "Prismetric provides industry solutions for: " + ", ".join(ind) + "."
    if intent == "ask_service_details":
        return "Please mention the specific service you want details about. For example: AI Development, Chatbot Development, Mobile App Development."
    if intent == "ask_pricing_models":
        response = "Prismetric offers the following pricing models:\n"
        for _, row in pricing_df.iterrows():
            response += f"\n- {row['model_name']}: {row['short_description']}"
        return response
    if intent == "ask_cost_estimate":
        models_list = ", ".join(pricing_df["model_name"].tolist())
        return f"Cost depends on project scope and complexity. Prismetric uses: {models_list}. For an estimate visit prismetric.com."
    if intent == "ask_recommendation":
    return (
        "Based on your needs, here are some recommendations:\n\n"
        "🏥 **Healthcare** → Healthcare AI Solutions, ML Development, Data Engineering\n\n"
        "🛒 **Ecommerce** → AI Solutions for Ecommerce, Mobile App Development, SaaS App Development\n\n"
        "🏦 **Fintech** → AI Solutions for Fintech, AI Development, Data Engineering\n\n"
        "🚚 **Logistics** → AI Solutions for Logistics, AI Automation Agency, ML Development\n\n"
        "🎓 **Education** → AI Solutions for Education, SaaS App Development, NLP Services\n\n"
        "🏠 **Real Estate** → Real Estate AI Solutions, Mobile App Development, AI Development\n\n"
        "✈️ **Travel** → AI Solutions for Travel, Mobile App Development, AI Development\n\n"
        "🛍️ **Retail** → Retail AI Solutions, AI Development, ML Development\n\n"
        "Tell me your industry and I can give a more specific suggestion. "
        "Or visit prismetric.com for a free consultation."
    )
    if intent == "ask_contact_info":
        return "You can contact Prismetric at prismetric.com. Look for the Contact Us or Get a Free Quote option."
    return "I am not sure I understood that. You can ask me about services, pricing, cost estimates, or recommendations."

def chatbot(user_input):
    intent, confidence = predict_intent(user_input)
    response = get_response(intent)
    return intent, confidence, response

with st.sidebar:
    st.markdown("## 🏢 Prismetric")
    st.markdown("---")
    st.markdown("### 📋 Services We Offer")
    for category in services_df["category"].unique():
        st.markdown(f"**{category}**")
        services = services_df[services_df["category"] == category]["service_name"].tolist()
        for s in services:
            st.markdown(f"- {s}")
        st.markdown("")
    st.markdown("---")
    st.markdown("### 💰 Pricing Models")
    for _, row in pricing_df.iterrows():
        st.markdown(f"**{row['model_name']}**")
        st.markdown(f"{row['short_description']}")
        st.markdown("")
    st.markdown("---")
    st.markdown("🌐 [Visit Prismetric](https://www.prismetric.com)")

st.title("🤖 Prismetric Service Assistant")
st.markdown("Ask me about services, pricing, cost estimates, or get a recommendation.")
st.markdown("---")

if "messages" not in st.session_state:
    st.session_state.messages = []
    st.session_state.messages.append({
        "role": "assistant",
        "content": "Hello! I am the Prismetric Service Assistant. How can I help you today?"
    })

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Type your message here..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    intent, confidence, response = chatbot(prompt)
    with st.chat_message("assistant"):
        st.markdown(response)
        st.caption(f"Intent: `{intent}` | Confidence: `{confidence}`")
    st.session_state.messages.append({"role": "assistant", "content": response})
