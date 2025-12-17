import streamlit as st
from PIL import Image
from generate_report import generate_report
import tempfile
import os
import uuid

st.set_page_config(page_title="Radiology Report Generator", layout="wide")
st.title("🩺 Radiology Report Generator")

# ---------------- STATE ----------------
if "report" not in st.session_state:
    st.session_state.report = None

# ---------------- UPLOAD ----------------
uploaded_file = st.file_uploader(
    "Upload chest X-ray image",
    type=["png", "jpg", "jpeg"]
)

if uploaded_file:
    image = Image.open(uploaded_file)

    st.image(
        image,
        caption="Uploaded X-ray",
        use_container_width=True
    )

    if st.button("Generate Report"):
        with st.spinner("Generating radiology report..."):
            temp_path = os.path.join(
                tempfile.gettempdir(),
                f"xray_{uuid.uuid4().hex}.png"
            )

            image.save(temp_path)
            st.session_state.report = generate_report(temp_path)

            try:
                os.remove(temp_path)
            except:
                pass

# ---------------- OUTPUT ----------------
if st.session_state.report:
    st.subheader("📄 Generated Radiology Report")
    st.markdown(
        f"""
        <div style="
            background-color:#1e1e1e;
            padding:20px;
            border-radius:10px;
            font-size:16px;
            line-height:1.6;
        ">
        {st.session_state.report}
        </div>
        """,
        unsafe_allow_html=True
    )
