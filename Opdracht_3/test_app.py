import streamlit as st

# Title
st.title("Welcome to Streamlit! 🎉")

# Text
st.write("This is a simple app to test how Streamlit works.")

# Slider
number = st.slider("Pick a number", 0, 100)

# Display the chosen number
st.write(f"You picked: {number}")
