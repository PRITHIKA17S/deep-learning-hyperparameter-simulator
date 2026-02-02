import streamlit as st
import tensorflow as tf
import matplotlib.pyplot as plt

st.title("Deep Learning Hyperparameter Tuning Simulator")

lr = st.slider("Learning Rate", 0.0001, 0.1, 0.001)
epochs = st.slider("Epochs", 1, 15, 5)
batch = st.selectbox("Batch Size", [16, 32, 64])

(x_train,y_train),(x_test,y_test)=tf.keras.datasets.mnist.load_data()
x_train,x_test = x_train/255.0, x_test/255.0

model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28,28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

history = model.fit(
    x_train, y_train,
    epochs=epochs,
    batch_size=batch,
    validation_data=(x_test,y_test),
    verbose=0
)

fig, ax = plt.subplots()
ax.plot(history.history['loss'], label="Training Loss")
ax.plot(history.history['val_loss'], label="Validation Loss")
ax.legend()
st.pyplot(fig)

st.subheader("Explanation")

if lr > 0.05:
    st.warning("High learning rate → model not converging")
elif lr < 0.001:
    st.info("Low learning rate → slow learning")

if epochs < 3:
    st.warning("Too few epochs → underfitting")
elif epochs > 10:
    st.info("Too many epochs → overfitting possible")