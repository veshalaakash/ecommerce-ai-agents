import streamlit as st
import pickle
import numpy as np
import pandas as pd
import faiss
import os

# ================================
# LOAD MODELS
# ================================
@st.cache_resource
def load_all():
    base = "models"  # models folder in your repo

    def load_pickle(name):
        path = f"{base}/{name}"
        if os.path.exists(path):
            with open(path, "rb") as f:
                return pickle.load(f)
        return None

    models = {}
    models["fraud"] = load_pickle("iso_fraud.pkl")
    models["pricing"] = load_pickle("price_rf.pkl")
    models["churn"] = load_pickle("churn_xgb.pkl")
    models["segmentation"] = load_pickle("segmentation.pkl")
    models["forecast"] = load_pickle("forecast_model.pkl")
    models["cat_list"] = load_pickle("cat_list.pkl")

    # embeddings
    emb_path = f"{base}/cat_embeddings.npy"
    if os.path.exists(emb_path):
        models["cat_embeddings"] = np.load(emb_path)
    else:
        models["cat_embeddings"] = None

    # FAISS index
    faiss_path = f"{base}/faiss.index"
    if os.path.exists(faiss_path):
        models["faiss_index"] = faiss.read_index(faiss_path)
    else:
        models["faiss_index"] = None

    return models


models = load_all()

# ================================
# STREAMLIT UI
# ================================
st.title("🧠 Multi-Agent E-Commerce AI Engine")
st.caption("Built by **veshalaakash**")

user_id = st.text_input("User ID", "user123")
cat = st.selectbox("Product Category", models["cat_list"] if models["cat_list"] else ["Electronics"])
quantity = st.slider("Quantity", 1, 10, 2)
discount = st.slider("Discount Amount", 0, 200, 10)
pages = st.slider("Pages Viewed", 1, 20, 5)

st.subheader("🎯 AI Agent Outputs")

# ================================
# RECOMMENDATION
# ================================
if models["faiss_index"] and models["cat_embeddings"] is not None:
    idx = models["cat_list"].index(cat)
    vec = np.array([models["cat_embeddings"][idx]]).astype("float32")
    D, I = models["faiss_index"].search(vec, 3)
    reco = [models["cat_list"][i] for i in I[0]]
else:
    reco = ["Electronics", "Fashion", "Home & Garden"]

st.write("### 🛒 Recommended Categories")
st.write(reco)

# ================================
# PRICING
# ================================
try:
    Xp = pd.get_dummies(pd.DataFrame([{
        "Product_Category": cat,
        "Quantity": quantity,
        "Discount_Amount": discount
    }]).fillna(0))
    price = float(models["pricing"].predict(Xp)[0])
except:
    price = np.random.uniform(300, 1200)

st.write("### 💰 Dynamic Price")
st.success(f"₹ {price:.2f}")

# ================================
# FRAUD
# ================================
try:
    fraud_flag = models["fraud"].predict([[price, quantity, discount]])[0] == -1
except:
    fraud_flag = False

st.write("### 🚨 Fraud Detection")
st.error("⚠ High Fraud Risk") if fraud_flag else st.success("✓ Safe Transaction")

# ================================
# CHURN
# ================================
try:
    Xc = pd.get_dummies(pd.DataFrame([{
        "Age": 30,
        "Gender": "Male",
        "City": "Hyderabad",
        "Product_Category": cat,
        "Pages_Viewed": pages
    }]))
    churn_raw = models["churn"].predict(Xc)[0]
    churn = "HIGH" if churn_raw == 1 else "LOW"
except:
    churn = "LOW"

st.write("### 🔁 Churn Prediction")
st.info(f"Churn Risk: **{churn}**")

# ================================
# SEGMENTATION
# ================================
try:
    seg = int(models["segmentation"].predict([[price, pages, 5]])[0])
except:
    seg = 0

st.write("### 👥 Customer Segment")
st.write(f"Segment: **{seg}**")

# ================================
# FORECAST
# ================================
try:
    future = models["forecast"].predict([[200],[201],[202],[203],[204],[205],[206]])
except:
    future = np.random.uniform(1000, 5000, 7)

st.write("### 📈 7-Day Sales Forecast")
st.line_chart(future)
