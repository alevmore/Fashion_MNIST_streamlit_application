import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import pickle

#  Download models and their training histories


# Convolutional Neural Network model
model_conv_path = "C:\\Users\\DELL\\Desktop\\data_analityka\\strlit_hw_16\\fashion_mnist_model_conv.h5"
model_conv = tf.keras.models.load_model(model_conv_path)

history_conv_path = "C:\\Users\\DELL\\Desktop\\data_analityka\\strlit_hw_16\\fashion_mnist_model_conv_history.pkl"
with open(history_conv_path, 'rb') as file:
    history_conv = pickle.load(file)

# VGG16 model
from tensorflow.keras.applications import VGG16

conv_base = VGG16(weights="imagenet", include_top=False, input_shape=(32, 32, 3))
conv_base.trainable = False

model_vgg = tf.keras.models.Sequential([
    conv_base,
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(256, activation="relu"),
    tf.keras.layers.Dense(10, activation="softmax")
])

model_vgg.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

history_vgg16_path = "C:\\Users\\DELL\\Desktop\\data_analityka\\strlit_hw_16\\fashion_mnist_model_VGG16_history.pkl"
with open(history_vgg16_path, 'rb') as file:
    history_vgg16 = pickle.load(file)

# Fuctions for images classification 

# Function for image classification from the Convolutional Neural Network model

def classify_image_conv(image, model):
    img = image.resize((28, 28))
    img = img.convert('L')
    img = np.array(img)
    img = img.reshape(1, 28, 28, 1)
    img = img.astype('float32') / 255.0

    prediction = model.predict(img)

    return prediction, np.argmax(prediction)

# Function for image classification from model VGG16

def classify_image_vgg16(image, model):
    img = image.resize((32, 32))  
    img = np.array(img)
    img = img.reshape(1, 32, 32, 3)
    img = img.astype('float32') / 255.0

    prediction = model.predict(img)

    return prediction, np.argmax(prediction)

# The main part of Streamlit application
def main():
    st.title('Fashion MNIST Image Classifier')

    # Selecting a model using the radio buttons
    model_choice = st.radio("Select the model for image classification:",
                            ('Convolutional Neural Network', 'VGG16'))

    st.write('Download an image for classification')

    uploaded_file = st.file_uploader("Select an image...", type=['jpg', 'jpeg', 'png'])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image.', use_column_width=True)

        if st.button('CLASSIFY'):
            if model_choice == 'Convolutional Neural Network':
                probabilities, predicted_class = classify_image_conv(image, model_conv)
                history = history_conv
            elif model_choice == 'VGG16':
                probabilities, predicted_class = classify_image_vgg16(image, model_vgg)
                history = history_vgg16

            st.write(f'Probabilitiesі: {probabilities.flatten()}')
            st.write(f'The intended class: {predicted_class}')

            fig, ax = plt.subplots()
            ax.bar(np.arange(10), probabilities.flatten(), align='center', alpha=0.5)
            ax.set_xticks(np.arange(10))
            ax.set_xticklabels(['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                                'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot'], rotation=45)
            ax.set_ylabel('Probability')
            ax.set_title('Probabilities for each class')
            st.pyplot(fig)

            # Display the graphs of model losses and accuracy
            st.subheader(f'Graphs of model losses and accuracy {model_choice}')
            st.line_chart(history['loss'], use_container_width=True)
            st.line_chart(history['accuracy'], use_container_width=True)

# Calling the main function 
if __name__ == '__main__':
    main()
