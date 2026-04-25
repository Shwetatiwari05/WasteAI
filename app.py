# import streamlit as st
# import numpy as np
# import json
# from PIL import Image
# import tensorflow as tf
# import os

# st.set_page_config(
#     page_title = "WasteAI — Smart Waste Classifier",
#     page_icon  = "♻️",
#     layout     = "wide"
# )

# st.markdown("""
# <style>
# @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
# * { font-family: 'Inter', sans-serif; }
# #MainMenu, footer, header { visibility: hidden; }
# .stApp { background: #0a0a0f; }
# .block-container { padding: 2rem 3rem !important; }
# .stTabs [data-baseweb="tab-list"] {
#     background: #0d1117 !important;
#     border-radius: 10px; padding: 4px; gap: 4px;
#     border: 1px solid #1e2029; width: fit-content;
# }
# .stTabs [data-baseweb="tab"] {
#     background: transparent !important;
#     color: #475569 !important;
#     border-radius: 8px !important;
#     padding: 8px 20px !important;
#     font-size: 13px !important;
#     font-weight: 500 !important;
#     border: none !important;
# }
# .stTabs [aria-selected="true"] {
#     background: #1e2029 !important;
#     color: #f1f5f9 !important;
# }
# .stButton > button {
#     background: linear-gradient(135deg, #16a34a, #15803d) !important;
#     color: white !important; border: none !important;
#     border-radius: 10px !important; padding: 12px 32px !important;
#     font-size: 14px !important; font-weight: 600 !important;
#     box-shadow: 0 4px 15px rgba(22,163,74,0.3) !important;
#     width: 100% !important;
# }
# .stFileUploader > div {
#     background: #070709 !important;
#     border: 1.5px dashed #1e2029 !important;
#     border-radius: 12px !important;
# }
# .stImage img {
#     border-radius: 12px !important;
#     border: 1px solid #1e2029 !important;
# }
# [data-testid="stSidebar"] {
#     background: #070709 !important;
#     border-right: 1px solid #1e2029 !important;
# }
# </style>
# """, unsafe_allow_html=True)

# # ── Constants ──────────────────────────────────────────────────
# IMG_SIZE   = (224, 224)
# MODEL_PATH = "waste_model_fixed"
# IDX_PATH   = "class_indices_v2.json"

# ECO_INFO = {
#     "cardboard": {
#         "recyclable": True,  "emoji": "",
#         "tip": "Flatten boxes and keep them dry before placing in recycling.",
#         "color": "#f59e0b"
#     },
#     "glass": {
#         "recyclable": True,  "emoji": "",
#         "tip": "Rinse jars and bottles, remove lids before recycling.",
#         "color": "#3b82f6"
#     },
#     "metal": {
#         "recyclable": True,  "emoji": "",
#         "tip": "Rinse cans and crush them to save space in the bin.",
#         "color": "#6b7280"
#     },
#     "paper": {
#         "recyclable": True,  "emoji": "",
#         "tip": "Keep paper dry and free of grease for recycling.",
#         "color": "#eab308"
#     },
#     "plastic": {
#         "recyclable": True,  "emoji": "",
#         "tip": "Check resin code 1 or 2 — rinse before recycling.",
#         "color": "#22c55e"
#     },
#     "trash": {
#         "recyclable": False, "emoji": "",
#         "tip": "Dispose in general waste. Reduce single-use items.",
#         "color": "#ef4444"
#     }
# }

# # ── Model ──────────────────────────────────────────────────────
# @st.cache_resource(show_spinner=False)
# def load_model():
#     if not os.path.exists(MODEL_PATH):
#         return None, None
#     model = tf.keras.models.load_model(MODEL_PATH)
#     with open(IDX_PATH) as f:
#         idx_to_class = {int(k): v for k, v in json.load(f).items()}
#     return model, idx_to_class

# def preprocess_image(pil_image):
#     img = pil_image.convert("RGB").resize(IMG_SIZE)
#     arr = np.array(img, dtype=np.float32) / 255.0
#     return np.expand_dims(arr, axis=0)

# def predict(model, idx_to_class, pil_image):
#     processed  = preprocess_image(pil_image)
#     probs      = model.predict(processed, verbose=0)[0]
#     pred_idx   = int(np.argmax(probs))
#     pred_class = idx_to_class[pred_idx]
#     confidence = float(probs[pred_idx])
#     all_probs  = {idx_to_class[i]: float(probs[i]) for i in range(len(probs))}
#     return pred_class, confidence, all_probs

# # ── Result renderer ────────────────────────────────────────────
# def render_result(pred_class, confidence, all_probs):
#     info       = ECO_INFO.get(pred_class, {})
#     recyclable = info.get("recyclable", False)
#     emoji      = info.get("emoji", "❓")
#     tip        = info.get("tip", "")
#     conf_pct   = confidence * 100

#     if recyclable:
#         badge_bg     = "rgba(34,197,94,0.12)"
#         badge_border = "rgba(34,197,94,0.3)"
#         badge_color  = "#4ade80"
#         badge_text   = "♻ Recyclable"
#     else:
#         badge_bg     = "rgba(239,68,68,0.12)"
#         badge_border = "rgba(239,68,68,0.3)"
#         badge_color  = "#f87171"
#         badge_text   = "✕ Non-Recyclable"

#     if confidence < 0.60:
#         st.markdown("""
#         <div style="background:rgba(234,179,8,0.08);border:1px solid rgba(234,179,8,0.2);
#                     border-radius:10px;padding:12px 16px;margin-bottom:16px;">
#             <span style="color:#fbbf24;font-size:13px;font-weight:500;">
#                 ⚠ Low confidence — try clearer image, single item, plain background
#             </span>
#         </div>
#         """, unsafe_allow_html=True)

#     # Header card
#     st.markdown(f"""
#     <div style="background:#0d1117;border:1px solid #1e2029;border-radius:16px;
#                 padding:24px 28px;margin-bottom:16px;">
#         <div style="display:flex;align-items:center;
#                     justify-content:space-between;margin-bottom:20px;
#                     padding-bottom:20px;border-bottom:1px solid #1e2029;">
#             <div style="display:flex;align-items:center;gap:14px;">
#                 <span style="font-size:40px;line-height:1;">{emoji}</span>
#                 <div>
#                     <p style="font-size:11px;color:#475569;text-transform:uppercase;
#                                letter-spacing:1px;margin:0 0 6px;font-weight:600;">
#                         Detected Waste Type
#                     </p>
#                     <p style="font-size:28px;font-weight:700;
#                                color:#f1f5f9;margin:0;line-height:1;">
#                         {pred_class.capitalize()}
#                     </p>
#                 </div>
#             </div>
#             <div style="padding:7px 18px;border-radius:20px;font-size:12px;
#                          font-weight:600;letter-spacing:0.5px;
#                          background:{badge_bg};border:1px solid {badge_border};
#                          color:{badge_color};">
#                 {badge_text}
#             </div>
#         </div>
#         <p style="font-size:11px;color:#475569;text-transform:uppercase;
#                    letter-spacing:1px;margin:0 0 8px;font-weight:600;">
#             Confidence Score
#         </p>
#         <div style="display:flex;align-items:center;gap:14px;margin-bottom:20px;">
#             <div style="flex:1;background:#1e2029;border-radius:6px;
#                          height:8px;overflow:hidden;">
#                 <div style="width:{conf_pct:.1f}%;height:100%;border-radius:6px;
#                              background:linear-gradient(90deg,#16a34a,#4ade80);">
#                 </div>
#             </div>
#             <span style="font-size:20px;font-weight:700;
#                           color:#4ade80;min-width:56px;text-align:right;">
#                 {conf_pct:.1f}%
#             </span>
#         </div>
#         <div style="background:rgba(34,197,94,0.05);
#                     border:1px solid rgba(34,197,94,0.12);
#                     border-left:3px solid #4ade80;
#                     border-radius:10px;padding:14px 18px;">
#             <p style="font-size:10px;font-weight:600;color:#4ade80;
#                        text-transform:uppercase;letter-spacing:1px;margin:0 0 5px;">
#                 Eco Tip
#             </p>
#             <p style="font-size:14px;color:#94a3b8;margin:0;line-height:1.5;">
#                 {tip}
#             </p>
#         </div>
#     </div>
#     """, unsafe_allow_html=True)

#     # Probability bars card
#     st.markdown("""
#     <div style="background:#0d1117;border:1px solid #1e2029;
#                 border-radius:16px;padding:24px 28px;">
#         <p style="font-size:11px;color:#475569;text-transform:uppercase;
#                    letter-spacing:1px;margin:0 0 16px;font-weight:600;">
#             All Class Probabilities
#         </p>
#     """, unsafe_allow_html=True)

#     sorted_probs = sorted(all_probs.items(), key=lambda x: x[1], reverse=True)
#     top_class    = sorted_probs[0][0]

#     for cls, prob in sorted_probs:
#         is_top     = cls == top_class
#         bar_color  = "linear-gradient(90deg,#16a34a,#4ade80)" if is_top else "#1e2029"
#         name_color = "#f1f5f9" if is_top else "#64748b"
#         pct_color  = "#4ade80" if is_top else "#475569"
#         weight     = "600" if is_top else "400"
#         cls_emoji  = ECO_INFO.get(cls, {}).get("emoji", "")

#         st.markdown(f"""
#         <div style="display:flex;align-items:center;gap:10px;margin-bottom:10px;">
#             <span style="font-size:12px;color:{name_color};width:100px;
#                           font-weight:{weight};flex-shrink:0;">
#                 {cls_emoji} {cls.capitalize()}
#             </span>
#             <div style="flex:1;background:#1a1d27;border-radius:4px;
#                          height:5px;overflow:hidden;">
#                 <div style="width:{prob*100:.1f}%;height:100%;
#                              border-radius:4px;background:{bar_color};">
#                 </div>
#             </div>
#             <span style="font-size:11px;color:{pct_color};
#                           width:40px;text-align:right;font-weight:{weight};">
#                 {prob*100:.1f}%
#             </span>
#         </div>
#         """, unsafe_allow_html=True)

#     st.markdown("</div>", unsafe_allow_html=True)


# # ══════════════════════════════════════════════════════════════
# def main():
#     model, idx_to_class = load_model()

#     # ── Hero ──────────────────────────────────────────────────
#     st.markdown("""
#     <div style="background:linear-gradient(135deg,#0d1117 0%,#0a0a0f 100%);
#                 border-bottom:1px solid #1e2029;padding:36px 0 28px;margin-bottom:32px;">
#         <div style="display:inline-flex;align-items:center;gap:6px;
#                     background:rgba(34,197,94,0.1);border:1px solid rgba(34,197,94,0.2);
#                     color:#4ade80;padding:4px 12px;border-radius:20px;
#                     font-size:11px;font-weight:600;letter-spacing:0.5px;margin-bottom:14px;">
#             AI Powered · MobileNetV2
#         </div>
#         <h1 style="font-size:40px;font-weight:700;color:#f1f5f9;
#                     margin:0 0 8px;letter-spacing:-0.5px;">
#             Waste<span style="color:#4ade80;">AI</span>
#         </h1>
#         <p style="color:#475569;font-size:15px;margin:0;">
#             Real-time waste classification using deep learning.
#             Upload an image or use your webcam.
#         </p>
#         <div style="display:flex;gap:40px;margin-top:24px;">
#             <div>
#                 <p style="font-size:22px;font-weight:700;color:#f1f5f9;margin:0;">6</p>
#                 <p style="font-size:11px;color:#475569;text-transform:uppercase;
#                            letter-spacing:0.8px;margin:2px 0 0;">Waste Classes</p>
#             </div>
#             <div>
#                 <p style="font-size:22px;font-weight:700;color:#f1f5f9;margin:0;">224px</p>
#                 <p style="font-size:11px;color:#475569;text-transform:uppercase;
#                            letter-spacing:0.8px;margin:2px 0 0;">Input Size</p>
#             </div>
#             <div>
#                 <p style="font-size:22px;font-weight:700;color:#f1f5f9;margin:0;">~80%</p>
#                 <p style="font-size:11px;color:#475569;text-transform:uppercase;
#                            letter-spacing:0.8px;margin:2px 0 0;">Val Accuracy</p>
#             </div>
#             <div>
#                 <p style="font-size:22px;font-weight:700;color:#f1f5f9;margin:0;">MNV2</p>
#                 <p style="font-size:11px;color:#475569;text-transform:uppercase;
#                            letter-spacing:0.8px;margin:2px 0 0;">Architecture</p>
#             </div>
#         </div>
#     </div>
#     """, unsafe_allow_html=True)

#     if model is None:
#         st.error("Model not found. Place waste_model_fixed and class_indices_v2.json in project folder.")
#         st.stop()

#     # ── Sidebar ────────────────────────────────────────────────
#     with st.sidebar:
#         st.markdown("""
#         <p style="font-size:11px;font-weight:600;color:#475569;
#                    text-transform:uppercase;letter-spacing:1px;margin-bottom:14px;">
#             Waste Categories
#         </p>
#         """, unsafe_allow_html=True)

#         for cls, info in ECO_INFO.items():
#             badge = "✅" if info["recyclable"] else "❌"
#             st.markdown(f"""
#             <div style="display:flex;align-items:center;gap:8px;padding:8px 10px;
#                          border-radius:8px;margin-bottom:2px;">
#                 <span style="font-size:16px;">{info['emoji']}</span>
#                 <span style="font-size:13px;color:#94a3b8;font-weight:500;flex:1;">
#                     {cls.capitalize()}
#                 </span>
#                 <span style="font-size:12px;">{badge}</span>
#             </div>
#             """, unsafe_allow_html=True)

#         st.markdown("""
#         <div style="margin-top:20px;padding:14px;background:#0d1117;
#                     border:1px solid #1e2029;border-radius:10px;">
#             <p style="font-size:10px;color:#4ade80;margin:0 0 6px;
#                        font-weight:600;text-transform:uppercase;letter-spacing:1px;">
#                 Best Results
#             </p>
#             <p style="font-size:12px;color:#64748b;margin:0;line-height:1.6;">
#                 Single item on plain background. Avoid complex scenes.
#             </p>
#         </div>
#         """, unsafe_allow_html=True)

#     # ── Two column layout ──────────────────────────────────────
#     col1, col2 = st.columns([1, 1], gap="large")

#     with col1:
#         st.markdown("""
#         <p style="font-size:11px;font-weight:600;color:#475569;
#                    text-transform:uppercase;letter-spacing:1px;margin-bottom:12px;">
#             Input
#         </p>
#         """, unsafe_allow_html=True)

#         tab1, tab2 = st.tabs(["  Upload Image  ", "  Webcam  "])
#         pil_image  = None

#         with tab1:
#             uploaded = st.file_uploader(
#                 "image", type=["jpg", "jpeg", "png", "webp"],
#                 label_visibility="collapsed"
#             )
#             if uploaded:
#                 pil_image = Image.open(uploaded)
#                 st.image(pil_image, use_column_width=True)

#         with tab2:
#             camera = st.camera_input("photo", label_visibility="collapsed")
#             if camera:
#                 pil_image = Image.open(camera)

#         st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)

#         if pil_image:
#             classify_btn = st.button("⚡  Classify Waste", use_container_width=True)
#         else:
#             classify_btn = False
#             st.markdown("""
#             <div style="text-align:center;padding:40px 0;color:#1e2029;">
#                 <p style="font-size:13px;margin:8px 0 0;">
#                     Upload an image to get started
#                 </p>
#             </div>
#             """, unsafe_allow_html=True)

#     with col2:
#         st.markdown("""
#         <p style="font-size:11px;font-weight:600;color:#475569;
#                    text-transform:uppercase;letter-spacing:1px;margin-bottom:12px;">
#             Result
#         </p>
#         """, unsafe_allow_html=True)

#         if classify_btn and pil_image:
#             with st.spinner("Analysing…"):
#                 pred_class, conf, all_probs = predict(
#                     model, idx_to_class, pil_image
#                 )
#             render_result(pred_class, conf, all_probs)
#         else:
#             st.markdown("""
#             <div style="background:#0d1117;border:1px solid #1e2029;
#                         border-radius:16px;padding:80px 28px;text-align:center;">
#                 <div style="font-size:44px;opacity:0.15;margin-bottom:10px;">♻</div>
#                 <p style="font-size:13px;color:#1e2029;margin:0;">
#                     Result will appear here
#                 </p>
#             </div>
#             """, unsafe_allow_html=True)

#     # ── Footer ─────────────────────────────────────────────────
#     st.markdown("""
#     <div style="text-align:center;padding:32px 0 16px;margin-top:40px;
#                 border-top:1px solid #1e2029;">
#         <p style="font-size:12px;color:#1e2029;margin:0;">
#             Built with TensorFlow · MobileNetV2 · Streamlit · TrashNet Dataset
#         </p>
#     </div>
#     """, unsafe_allow_html=True)

# if __name__ == "__main__":
#     main()



import streamlit as st
import numpy as np
import json
from PIL import Image
import tensorflow as tf
import os

st.set_page_config(
    page_title = "WasteAI — Smart Waste Classifier",
    page_icon  = "♻️",
    layout     = "wide"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
* { font-family: 'Inter', sans-serif; }
#MainMenu, footer, header { visibility: hidden; }
.stApp { background: #0a0a0f; }
.block-container { padding: 2rem 3rem !important; }
.stTabs [data-baseweb="tab-list"] {
    background: #0d1117 !important;
    border-radius: 10px; padding: 4px; gap: 4px;
    border: 1px solid #1e2029; width: fit-content;
}
.stTabs [data-baseweb="tab"] {
    background: transparent !important;
    color: #475569 !important;
    border-radius: 8px !important;
    padding: 8px 20px !important;
    font-size: 13px !important;
    font-weight: 500 !important;
    border: none !important;
}
.stTabs [aria-selected="true"] {
    background: #1e2029 !important;
    color: #f1f5f9 !important;
}
.stButton > button {
    background: linear-gradient(135deg, #16a34a, #15803d) !important;
    color: white !important; border: none !important;
    border-radius: 10px !important; padding: 12px 32px !important;
    font-size: 14px !important; font-weight: 600 !important;
    box-shadow: 0 4px 15px rgba(22,163,74,0.3) !important;
    width: 100% !important;
}
.stFileUploader > div {
    background: #070709 !important;
    border: 1.5px dashed #1e2029 !important;
    border-radius: 12px !important;
}
.stImage img {
    border-radius: 12px !important;
    border: 1px solid #1e2029 !important;
}
[data-testid="stSidebar"] {
    background: #070709 !important;
    border-right: 1px solid #1e2029 !important;
}
</style>
""", unsafe_allow_html=True)

# ── Constants ──────────────────────────────────────────────────
IMG_SIZE   = (224, 224)
MODEL_PATH = "waste_model_v2.h5"
IDX_PATH   = "class_indices_v2.json"

ECO_INFO = {
    "cardboard": {
        "recyclable": True,  "emoji": "",
        "tip": "Flatten boxes and keep them dry before placing in recycling.",
        "color": "#f59e0b"
    },
    "glass": {
        "recyclable": True,  "emoji": "",
        "tip": "Rinse jars and bottles, remove lids before recycling.",
        "color": "#3b82f6"
    },
    "metal": {
        "recyclable": True,  "emoji": "",
        "tip": "Rinse cans and crush them to save space in the bin.",
        "color": "#6b7280"
    },
    "paper": {
        "recyclable": True,  "emoji": "",
        "tip": "Keep paper dry and free of grease for recycling.",
        "color": "#eab308"
    },
    "plastic": {
        "recyclable": True,  "emoji": "",
        "tip": "Check resin code 1 or 2 — rinse before recycling.",
        "color": "#22c55e"
    },
    "trash": {
        "recyclable": False, "emoji": "",
        "tip": "Dispose in general waste. Reduce single-use items.",
        "color": "#ef4444"
    }
}

# ── Model ──────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_model():
    if not os.path.exists(MODEL_PATH):
        return None, None
    import keras
    model = keras.models.load_model(MODEL_PATH)
    with open(IDX_PATH) as f:
        idx_to_class = {int(k): v for k, v in json.load(f).items()}
    return model, idx_to_class

def preprocess_image(pil_image):
    img = pil_image.convert("RGB").resize(IMG_SIZE)
    arr = np.array(img, dtype=np.float32) / 255.0
    return np.expand_dims(arr, axis=0)

def predict(model, idx_to_class, pil_image):
    processed  = preprocess_image(pil_image)
    probs      = model.predict(processed, verbose=0)[0]
    pred_idx   = int(np.argmax(probs))
    pred_class = idx_to_class[pred_idx]
    confidence = float(probs[pred_idx])
    all_probs  = {idx_to_class[i]: float(probs[i]) for i in range(len(probs))}
    return pred_class, confidence, all_probs

# ── Result renderer ────────────────────────────────────────────
def render_result(pred_class, confidence, all_probs):
    info       = ECO_INFO.get(pred_class, {})
    recyclable = info.get("recyclable", False)
    emoji      = info.get("emoji", "❓")
    tip        = info.get("tip", "")
    conf_pct   = confidence * 100

    if recyclable:
        badge_bg     = "rgba(34,197,94,0.12)"
        badge_border = "rgba(34,197,94,0.3)"
        badge_color  = "#4ade80"
        badge_text   = "♻ Recyclable"
    else:
        badge_bg     = "rgba(239,68,68,0.12)"
        badge_border = "rgba(239,68,68,0.3)"
        badge_color  = "#f87171"
        badge_text   = "✕ Non-Recyclable"

    if confidence < 0.60:
        st.markdown("""
        <div style="background:rgba(234,179,8,0.08);border:1px solid rgba(234,179,8,0.2);
                    border-radius:10px;padding:12px 16px;margin-bottom:16px;">
            <span style="color:#fbbf24;font-size:13px;font-weight:500;">
                ⚠ Low confidence — try clearer image, single item, plain background
            </span>
        </div>
        """, unsafe_allow_html=True)

    # Header card
    st.markdown(f"""
    <div style="background:#0d1117;border:1px solid #1e2029;border-radius:16px;
                padding:24px 28px;margin-bottom:16px;">
        <div style="display:flex;align-items:center;
                    justify-content:space-between;margin-bottom:20px;
                    padding-bottom:20px;border-bottom:1px solid #1e2029;">
            <div style="display:flex;align-items:center;gap:14px;">
                <span style="font-size:40px;line-height:1;">{emoji}</span>
                <div>
                    <p style="font-size:11px;color:#475569;text-transform:uppercase;
                               letter-spacing:1px;margin:0 0 6px;font-weight:600;">
                        Detected Waste Type
                    </p>
                    <p style="font-size:28px;font-weight:700;
                               color:#f1f5f9;margin:0;line-height:1;">
                        {pred_class.capitalize()}
                    </p>
                </div>
            </div>
            <div style="padding:7px 18px;border-radius:20px;font-size:12px;
                         font-weight:600;letter-spacing:0.5px;
                         background:{badge_bg};border:1px solid {badge_border};
                         color:{badge_color};">
                {badge_text}
            </div>
        </div>
        <p style="font-size:11px;color:#475569;text-transform:uppercase;
                   letter-spacing:1px;margin:0 0 8px;font-weight:600;">
            Confidence Score
        </p>
        <div style="display:flex;align-items:center;gap:14px;margin-bottom:20px;">
            <div style="flex:1;background:#1e2029;border-radius:6px;
                         height:8px;overflow:hidden;">
                <div style="width:{conf_pct:.1f}%;height:100%;border-radius:6px;
                             background:linear-gradient(90deg,#16a34a,#4ade80);">
                </div>
            </div>
            <span style="font-size:20px;font-weight:700;
                          color:#4ade80;min-width:56px;text-align:right;">
                {conf_pct:.1f}%
            </span>
        </div>
        <div style="background:rgba(34,197,94,0.05);
                    border:1px solid rgba(34,197,94,0.12);
                    border-left:3px solid #4ade80;
                    border-radius:10px;padding:14px 18px;">
            <p style="font-size:10px;font-weight:600;color:#4ade80;
                       text-transform:uppercase;letter-spacing:1px;margin:0 0 5px;">
                Eco Tip
            </p>
            <p style="font-size:14px;color:#94a3b8;margin:0;line-height:1.5;">
                {tip}
            </p>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Probability bars card
    st.markdown("""
    <div style="background:#0d1117;border:1px solid #1e2029;
                border-radius:16px;padding:24px 28px;">
        <p style="font-size:11px;color:#475569;text-transform:uppercase;
                   letter-spacing:1px;margin:0 0 16px;font-weight:600;">
            All Class Probabilities
        </p>
    """, unsafe_allow_html=True)

    sorted_probs = sorted(all_probs.items(), key=lambda x: x[1], reverse=True)
    top_class    = sorted_probs[0][0]

    for cls, prob in sorted_probs:
        is_top     = cls == top_class
        bar_color  = "linear-gradient(90deg,#16a34a,#4ade80)" if is_top else "#1e2029"
        name_color = "#f1f5f9" if is_top else "#64748b"
        pct_color  = "#4ade80" if is_top else "#475569"
        weight     = "600" if is_top else "400"
        cls_emoji  = ECO_INFO.get(cls, {}).get("emoji", "")

        st.markdown(f"""
        <div style="display:flex;align-items:center;gap:10px;margin-bottom:10px;">
            <span style="font-size:12px;color:{name_color};width:100px;
                          font-weight:{weight};flex-shrink:0;">
                {cls_emoji} {cls.capitalize()}
            </span>
            <div style="flex:1;background:#1a1d27;border-radius:4px;
                         height:5px;overflow:hidden;">
                <div style="width:{prob*100:.1f}%;height:100%;
                             border-radius:4px;background:{bar_color};">
                </div>
            </div>
            <span style="font-size:11px;color:{pct_color};
                          width:40px;text-align:right;font-weight:{weight};">
                {prob*100:.1f}%
            </span>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
def main():
    model, idx_to_class = load_model()

    # ── Hero ──────────────────────────────────────────────────
    st.markdown("""
    <div style="background:linear-gradient(135deg,#0d1117 0%,#0a0a0f 100%);
                border-bottom:1px solid #1e2029;padding:36px 0 28px;margin-bottom:32px;">
        <div style="display:inline-flex;align-items:center;gap:6px;
                    background:rgba(34,197,94,0.1);border:1px solid rgba(34,197,94,0.2);
                    color:#4ade80;padding:4px 12px;border-radius:20px;
                    font-size:11px;font-weight:600;letter-spacing:0.5px;margin-bottom:14px;">
            AI Powered · MobileNetV2
        </div>
        <h1 style="font-size:40px;font-weight:700;color:#f1f5f9;
                    margin:0 0 8px;letter-spacing:-0.5px;">
            Waste<span style="color:#4ade80;">AI</span>
        </h1>
        <p style="color:#475569;font-size:15px;margin:0;">
            Real-time waste classification using deep learning.
            Upload an image or use your webcam.
        </p>
        <div style="display:flex;gap:40px;margin-top:24px;">
            <div>
                <p style="font-size:22px;font-weight:700;color:#f1f5f9;margin:0;">6</p>
                <p style="font-size:11px;color:#475569;text-transform:uppercase;
                           letter-spacing:0.8px;margin:2px 0 0;">Waste Classes</p>
            </div>
            <div>
                <p style="font-size:22px;font-weight:700;color:#f1f5f9;margin:0;">224px</p>
                <p style="font-size:11px;color:#475569;text-transform:uppercase;
                           letter-spacing:0.8px;margin:2px 0 0;">Input Size</p>
            </div>
            <div>
                <p style="font-size:22px;font-weight:700;color:#f1f5f9;margin:0;">~80%</p>
                <p style="font-size:11px;color:#475569;text-transform:uppercase;
                           letter-spacing:0.8px;margin:2px 0 0;">Val Accuracy</p>
            </div>
            <div>
                <p style="font-size:22px;font-weight:700;color:#f1f5f9;margin:0;">MNV2</p>
                <p style="font-size:11px;color:#475569;text-transform:uppercase;
                           letter-spacing:0.8px;margin:2px 0 0;">Architecture</p>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    if model is None:
        st.error("Model not found. Place waste_model_fixed and class_indices_v2.json in project folder.")
        st.stop()

    # ── Sidebar ────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("""
        <p style="font-size:11px;font-weight:600;color:#475569;
                   text-transform:uppercase;letter-spacing:1px;margin-bottom:14px;">
            Waste Categories
        </p>
        """, unsafe_allow_html=True)

        for cls, info in ECO_INFO.items():
            badge = "✅" if info["recyclable"] else "❌"
            st.markdown(f"""
            <div style="display:flex;align-items:center;gap:8px;padding:8px 10px;
                         border-radius:8px;margin-bottom:2px;">
                <span style="font-size:16px;">{info['emoji']}</span>
                <span style="font-size:13px;color:#94a3b8;font-weight:500;flex:1;">
                    {cls.capitalize()}
                </span>
                <span style="font-size:12px;">{badge}</span>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("""
        <div style="margin-top:20px;padding:14px;background:#0d1117;
                    border:1px solid #1e2029;border-radius:10px;">
            <p style="font-size:10px;color:#4ade80;margin:0 0 6px;
                       font-weight:600;text-transform:uppercase;letter-spacing:1px;">
                Best Results
            </p>
            <p style="font-size:12px;color:#64748b;margin:0;line-height:1.6;">
                Single item on plain background. Avoid complex scenes.
            </p>
        </div>
        """, unsafe_allow_html=True)

    # ── Two column layout ──────────────────────────────────────
    col1, col2 = st.columns([1, 1], gap="large")

    with col1:
        st.markdown("""
        <p style="font-size:11px;font-weight:600;color:#475569;
                   text-transform:uppercase;letter-spacing:1px;margin-bottom:12px;">
            Input
        </p>
        """, unsafe_allow_html=True)

        tab1, tab2 = st.tabs(["  Upload Image  ", "  Webcam  "])
        pil_image  = None

        with tab1:
            uploaded = st.file_uploader(
                "image", type=["jpg", "jpeg", "png", "webp"],
                label_visibility="collapsed"
            )
            if uploaded:
                pil_image = Image.open(uploaded)
                st.image(pil_image, use_container_width=True)

        with tab2:
            camera = st.camera_input("photo", label_visibility="collapsed")
            if camera:
                pil_image = Image.open(camera)

        st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)

        if pil_image:
            classify_btn = st.button("Classify Waste", use_container_width=True)
        else:
            classify_btn = False
            st.markdown("""
            <div style="text-align:center;padding:40px 0;color:#1e2029;">
                <p style="font-size:13px;margin:8px 0 0;">
                    Upload an image to get started
                </p>
            </div>
            """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <p style="font-size:11px;font-weight:600;color:#475569;
                   text-transform:uppercase;letter-spacing:1px;margin-bottom:12px;">
            Result
        </p>
        """, unsafe_allow_html=True)

        if classify_btn and pil_image:
            with st.spinner("Analysing…"):
                pred_class, conf, all_probs = predict(
                    model, idx_to_class, pil_image
                )
            render_result(pred_class, conf, all_probs)
        else:
            st.markdown("""
            <div style="background:#0d1117;border:1px solid #1e2029;
                        border-radius:16px;padding:80px 28px;text-align:center;">
                <div style="font-size:44px;opacity:0.15;margin-bottom:10px;">♻</div>
                <p style="font-size:13px;color:#1e2029;margin:0;">
                    Result will appear here
                </p>
            </div>
            """, unsafe_allow_html=True)

    # ── Footer ─────────────────────────────────────────────────
    st.markdown("""
    <div style="text-align:center;padding:32px 0 16px;margin-top:40px;
                border-top:1px solid #1e2029;">
        <p style="font-size:12px;color:#1e2029;margin:0;">
            Built with TensorFlow · MobileNetV2 · Streamlit · TrashNet Dataset
        </p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()