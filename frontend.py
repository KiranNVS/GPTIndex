import requests
import streamlit as st
from openai_utils import request_api, parse_api_result

st.title("Frontend Demo")

# Create text input box for user input
user_input = st.text_area('Text to complement', 'It was the best of times, it was the worst of times, ')

# Handle user input and API call
if st.button("Submit"):
    if not user_input:
        st.error("Please enter some text.")
    else:
        api_response = parse_api_result(request_api(user_input, engine="text-davinci-002"))
        st.write("OpenAI API response:")
        st.write(api_response[0].strip())
