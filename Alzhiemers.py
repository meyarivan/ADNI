import cv2
import numpy as np
import keras
from keras.models import load_model
import matplotlib.pyplot as plt
import streamlit as st
from PIL import Image

choice=st.sidebar.radio("Select one of the Below ",["Home","Model Insight","Accuracy and Loss Graphs"])

if choice=="Home":
    st.title("Alzhiemers Prediction Model using CNN and data correction using SMOTE")
    Labels=['Late mild cognitive impairment','Alzheimer_s disease','Cognitively normal','Early mild cognitive impairment']
    image = st.file_uploader("Upload the Brain MRI")
    model = load_model('C:\\Users\\Krishang Virmani\\Desktop\\Python CB\\ADNI\\CNN_model.h5')  
    def preprocess_image(image):
        image=np.array(Image.open(image))
        image=cv2.resize(image,(176,176))
        image_with_channels=np.expand_dims(image,axis=-1)
        image_with_channels=np.concatenate([image_with_channels]*3,axis=-1)
        image_with_batch = np.expand_dims(image_with_channels, axis=0)
        return image_with_batch    
    def classify(image):
        fig,ax=plt.subplots()
        ax.imshow(image[0])
        predictions = model.predict(image)
        predicted_class_index = np.argmax(predictions)
        ax.set_title(Labels[predicted_class_index],fontdict={"fontsize":25},loc="center")
        st.pyplot(fig)    
    if image is not None:
        image=preprocess_image(image)
        st.image(image,use_column_width=True,caption="Uploaded Image")
        if st.button("Classify"):
            with st.spinner("Classifying..."):
                classify(image)
            st.success("Classifcation Done!")

if choice=="Model Insight":
    st.title("Here is a look at the Model source code for the project")
    st.image("Model_VS.png",caption="This is the Model",use_column_width=True)
    st.text("""
Model: "sequential"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 conv2d (Conv2D)             (None, 88, 88, 32)        896       
                                                                 
 max_pooling2d (MaxPooling2  (None, 44, 44, 32)        0         
 D)                                                              
                                                                 
 conv2d_1 (Conv2D)           (None, 22, 22, 64)        18496     
                                                                 
 max_pooling2d_1 (MaxPoolin  (None, 11, 11, 64)        0         
 g2D)                                                            
                                                                 
 conv2d_2 (Conv2D)           (None, 6, 6, 128)         73856     
                                                                 
 max_pooling2d_2 (MaxPoolin  (None, 3, 3, 128)         0         
 g2D)                                                            
                                                                 
 flatten (Flatten)           (None, 1152)              0         
                                                                 
 dense (Dense)               (None, 1024)              1180672   
                                                                 
 dropout (Dropout)           (None, 1024)              0         
                                                                 
...
Total params: 1278020 (4.88 MB)
Trainable params: 1278020 (4.88 MB)
Non-trainable params: 0 (0.00 Byte)
_________________________________________________________________""")
    
    
if choice=="Accuracy and Loss Graphs":
    st.image("Loss.png",use_column_width=True)
    st.image("Accuracy.png",use_column_width=True)