import io
import json
import altair as alt
import pandas as pd
import streamlit as st
import torch
import torch.nn as nn
import torchvision.transforms as T
from PIL import Image
from torchvision import models

st.set_page_config(page_title="Cervix transformation-zone classifier", layout="wide")

st.markdown("""
<style>
.block-container {padding-top: 2rem;}
img {border-radius: 8px;}
</style>
""", unsafe_allow_html=True)

WEIGHTS_DIR = "weights"
PRE_SIZE = 384   # training images were shrunk to a longest side of 384 px
NORM = T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
ARCH = {"efficientnet_b0": models.efficientnet_b0, "efficientnet_b2": models.efficientnet_b2}
THRESHOLD = 0.60  # placeholder, to be chosen from the validation set
MAX_BYTES = 10 * 1024 * 1024  # 10 MB upload limit


class PadSquare:
    """Pad the image with black to a square, so nothing is cropped."""
    def __call__(self, im):
        w, h = im.size
        s = max(w, h)
        canvas = Image.new("RGB", (s, s), (0, 0, 0))
        canvas.paste(im, ((s - w) // 2, (s - h) // 2))
        return canvas


def build(arch):
    m = ARCH[arch](weights=None)
    m.classifier[1] = nn.Linear(m.classifier[1].in_features, 3)
    return m


@st.cache_resource
def load_pipeline():
    """Read final_pipeline.json and load the three trained models (cached)."""
    with open(f"{WEIGHTS_DIR}/final_pipeline.json") as f:
        cfg = json.load(f)
    nets = []
    for s in cfg["models"]:
        m = build(s["arch"])
        m.load_state_dict(torch.load(f"{WEIGHTS_DIR}/{s['weights']}", map_location="cpu"))
        m.eval()
        nets.append((m, s["size"]))
    return cfg, nets


cfg, nets = load_pipeline()
CLASSES = [c.replace("_", " ") for c in cfg["classes"]]


def preprocess(im, size):
    im = im.convert("RGB").copy()
    im.thumbnail((PRE_SIZE, PRE_SIZE))
    im = PadSquare()(im)
    im = T.Resize((size, size))(im)
    x = NORM(T.ToTensor()(im))
    return x.unsqueeze(0)


@torch.no_grad()
def predict(im):
    probs = []
    for model, size in nets:
        probs.append(torch.softmax(model(preprocess(im, size)), dim=1))
    p = torch.stack(probs).mean(0)
    logits = torch.log(p.clamp_min(1e-12)) / cfg["temperature"]
    return torch.softmax(logits, dim=1)[0].numpy()


def gradcam(model, size, im, cls):
    store = {}
    layer = model.features[-1]  # last convolution block
    def fwd(m, i, o):
        store["a"] = o
        o.register_hook(lambda g: store.update(g=g))
    h = layer.register_forward_hook(fwd)
    with torch.enable_grad():
        out = model(preprocess(im, size))
        model.zero_grad()
        out[0, cls].backward()
    h.remove()
    w = store["g"].mean(dim=(2, 3), keepdim=True)
    cam = torch.relu((w * store["a"]).sum(1))[0]
    cam = (cam / (cam.max() + 1e-8)).detach().numpy()
    base = T.Resize((size, size))(PadSquare()(im.copy().convert("RGB")))
    heat = Image.fromarray((cam * 255).astype("uint8")).resize((size, size), Image.BILINEAR)
    red = Image.merge("RGB", (heat, Image.new("L", (size, size)), Image.new("L", (size, size))))
    return Image.blend(base, red, 0.45)


def load_image(uploaded):
    """Open the upload safely. Returns (image, None) or (None, error message)."""
    if uploaded.size > MAX_BYTES:
        return None, "File is larger than 10 MB. Please upload a smaller image."
    try:
        im = Image.open(uploaded)
        im.load()
        return im.convert("RGB"), None
    except Exception:
        return None, "Could not read this file as an image. Please use a JPG or PNG."


@st.cache_data(show_spinner=False)
def analyse(file_bytes):
    """Run the ensemble and Grad-CAM once per file, then reuse the result."""
    im = Image.open(io.BytesIO(file_bytes)).convert("RGB")
    p = predict(im)
    model_c, size_c = nets[2]
    cam_img = gradcam(model_c, size_c, im, int(p.argmax()))
    return p, cam_img


# ---------- sidebar ----------
with st.sidebar:
    st.header("About")
    st.write("Classifies the cervical transformation zone (Type 1, 2 or 3) "
             "from a colposcopy image, using an ensemble of three EfficientNets.")
    st.write("Trained on the Kaggle Intel & MobileODT dataset. "
             "Held-out test split (222 images): accuracy 0.752, log loss 0.560.")
    with st.expander("Limitations"):
        st.write("- Research demo, not a medical device.")
        st.write("- Results come from one dataset and one split. Performance on other cameras or clinics is unknown.")
        st.write("- Some test images are near-duplicates of training images. In a second check on 512 competition images, "
                 "those close to training images scored higher (about 0.93 accuracy) than the rest (about 0.74), "
                 "so headline scores may be optimistic.")
        st.write("- The dataset has no patient IDs.")
        st.write(f"- The review threshold ({THRESHOLD:.2f}) is a placeholder, not yet validated.")
        st.write("- Grad-CAM shows where the model looked, not why.")
        st.write("- The model cannot tell whether an image shows a cervix: any image gets a prediction.")

# ---------- main page ----------
st.title("Cervix transformation-zone classifier")
st.caption("Upload a colposcopy image to see the predicted type, the class probabilities and a Grad-CAM heatmap.")
st.warning("Research demo only. Not a medical device and not for clinical use.")

# ---------- project introduction ----------
st.subheader("About this project")
st.write(
    "This tool looks at a colposcopy photo of the cervix and estimates which **type of "
    "transformation zone** it shows. Three trained neural networks (EfficientNets) each "
    "give an opinion, their answers are averaged, and the result is calibrated to make the "
    "percentages more reliable. A Grad-CAM heatmap shows which part of the image "
    "influenced the answer."
)

st.markdown("##### What is the transformation zone?")
st.write(
    "The transformation zone is the area of the cervix where cell changes, including "
    "precancer, most often develop. Doctors classify it into three types depending on "
    "where it sits and whether it can be seen completely during colposcopy "
    "(IFCPC 2011 classification)."
)

t1, t2, t3 = st.columns(3)
with t1:
    with st.container(border=True):
        st.markdown(":green[**Type 1**]")
        st.write("The zone lies entirely on the **outer surface** of the cervix and is "
                 "fully visible.")
        st.caption("Easiest to examine.")
with t2:
    with st.container(border=True):
        st.markdown(":orange[**Type 2**]")
        st.write("The zone reaches **into the cervical canal**, but its upper edge is "
                 "still fully visible.")
        st.caption("May need a speculum or instrument to see it.")
with t3:
    with st.container(border=True):
        st.markdown(":red[**Type 3**]")
        st.write("The zone reaches **into the canal** and its upper edge **cannot be "
                 "fully seen**.")
        st.caption("Hardest to examine and treat.")

st.write(
    "**Why it matters:** the type affects how the examination is done and how any "
    "treatment is planned, so judging it consistently is useful."
)

st.markdown("##### How this app works")
h1, h2, h3, h4 = st.columns(4)
h1.markdown("**1. Upload**  \nA colposcopy image (JPG or PNG)")
h2.markdown("**2. Predict**  \n3 EfficientNet models vote")
h3.markdown("**3. Explain**  \nGrad-CAM heatmap and probabilities")
h4.markdown("**4. Flag**  \nLow-confidence cases are marked for human review")

k1, k2, k3 = st.columns(3)
k1.metric("Models in ensemble", "3")
k2.metric("Test accuracy", "0.752")
k3.metric("Held-out test images", "222")
st.caption("Source of the type definitions: IFCPC 2011 colposcopic terminology. "
           "Data: Kaggle Intel & MobileODT Cervical Cancer Screening.")
st.divider()

up = st.file_uploader("Upload a colposcopy image", type=["jpg", "jpeg", "png"])
st.caption("Use only public or de-identified images. Do not upload patient data.")

if up:
    img, err = load_image(up)
    if err:
        st.error(err)
        st.stop()
    with st.spinner("Running 3 models..."):
        p, cam_img = analyse(up.getvalue())
        top = int(p.argmax())

    col1, col2, col3 = st.columns([1, 1, 1.3])

    with col1:
        st.subheader("Uploaded image")
        st.image(img, width=320)

    with col2:
        st.subheader("Grad-CAM (model C)")
        st.image(cam_img, width=320)
        st.caption("Red = regions that raised the predicted class score.")

    with col3:
        st.subheader("Result")
        m1, m2 = st.columns(2)
        m1.metric("Predicted class", CLASSES[top])
        m2.metric("Confidence", f"{p[top]:.0%}")
        low_conf = bool(p[top] < THRESHOLD)
        if low_conf:
            st.error("Low confidence: needs human review.")
        else:
            st.success("Confidence above the review threshold.")

        df = pd.DataFrame({"class": CLASSES, "probability": p})
        chart = alt.Chart(df).mark_bar().encode(

            x=alt.X("class:N", sort=None, title=None, axis=alt.Axis(labelAngle=0)),
            y=alt.Y("probability:Q", scale=alt.Scale(domain=[0, 1])),
            tooltip=["class", alt.Tooltip("probability:Q", format=".3f")],
        ).properties(height=250)
        st.altair_chart(chart, width="stretch")

        for c, v in zip(CLASSES, p):
            st.write(f"**{c}**: {v:.2f}")

st.divider()
st.caption(Prajakta
    "Research demo, not a medical device. Project by  More, June 2026. "
    "[Code on GitHub](https://github.com/Prajakta-More05/cervical-transformation-zone-classification)"
)