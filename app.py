import streamlit as st 
import cv2
from PIL import Image
import numpy as np
from predict import predict
import io

st.title("COVID Detector AI")
st.subheader("A Deep Learning Based Xray Diagnosis App")
st.markdown("> Disclaimer : I do not claim this application as a highly accurate COVID Diagnosis Tool. This Application has not been professionally or academically Vetted. This is purely for Educational Purpose to demonstrate the Potential of AI's help in Medicine.")
st.markdown("Developed by [Arun Pandian R](https://arunrk7codie.github.io)")

st.markdown("**Note:** You should upload atmost one Chest Xray Image of either class (COVID19 Infected or Normal). Since this application is a Classification Task not a Segmentation.")

uploaded_file = st.file_uploader("Choose an Chest Xray Image...", type=("jpg", "png", "jpeg"))
if uploaded_file is not None:
    st.image(
        uploaded_file,
        caption="Uploaded CT Scan",
        use_column_width=True,    )
    image = Image.open(uploaded_file)
    pred = predict(np.array(image))
    result = "COVID Positive" if pred=="covid19" else "COVID Negative"
    st.markdown("## **Diagnosed Result:**"  + result)

st.markdown("### Check out the [GitHub Repository](https://github.com/ArunRK7Codie/covid-19-detection)")
st.markdown("> Find more info about the CoronaVirus  on [who.int](https://www.who.int/news-room/q-a-detail/q-a-coronaviruses#:~:text=symptoms) and [mohfw.gov](https://www.mohfw.gov.in/)")

