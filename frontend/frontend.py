import requests
import streamlit as st

from openai_utils import request_api, parse_api_result


parameters = {'api_key': '',
              'model': 'text-davinci-002',
              'temperature': 0.0,
              'max_length': 64,
              'top_p': 1.0,
              'best_of': 1,
              # 'logprobs': 10,
              }

with st.sidebar:
    parameters['api_key'] = st.text_input('OpenAI API Token',
                                     type='password',
                                     placeholder='Enter your API key',
                                     )

    parameters['model'] = st.selectbox(
                                    "Model",
                                    options = ('gpt-4',
                                              'text-davinci-003',
                                              parameters['model'],
                                              'code-davinci-002',
                                              ),
                                    index = 2
                                    )

    parameters['temperature'] = st.slider('Temperature',
                                    min_value=0.0,
                                    max_value=1.0,
                                    value=parameters['temperature'],
                                    step=0.01,
                                    help='Controls randomness: Lowering results in less random completions. As the temperature approaches zero, the model will become deterministic and repetitive.'
                                    )
    
    parameters['max_length'] = st.slider('Maximum Length',
                                    min_value=1,
                                    max_value=100, #4000
                                    value=parameters['max_length'],
                                    step=1,
                                    help='The maximum number of tokens to generate. Requests can use up to 100 tokens. The exact limit varies by model. (One token is roughly 4 characters for normal English text)'
                                    )
    
    parameters['top_p'] = st.slider('Top P',
                                    min_value=0.0,
                                    max_value=1.0,
                                    value=parameters['top_p'],
                                    step=0.01,
                                    help='Controls diversity via nucleus sampling: 0.5 means half of all likelihood-weighted options are considered.'
                                    )
    parameters['best_of'] = st.slider('Best of',
                                    min_value=1,
                                    max_value=10,
                                    value=parameters['best_of'],
                                    step=1,
                                    help='Generates multiple completions server-side, and displays only the best. Streaming only works when set to 1. Since it acts as a multiplier on the number of completions, this parameters can eat into your token quota very quickly \u2013 use caution!'
                                    )

st.title("CKIDS Event Forecasting Demo")

# Create text input box for user input
user_input = st.text_area('Text to complement', 
                          value='It was the best of times, it was the worst of times, ',
                          placeholder='',
                          )

# Handle user input and API call
if st.button("Submit"):
    if not user_input:
        st.error("Please enter some text.")
    if not parameters['api_key']:
        st.error("Please enter your OpenAI API Key")
    else:
        # response = request_api(user_input, parameters)
        query = {'query': user_input}
        print(f'calling backend with query:, {query}')
        response = requests.post('http://127.0.0.1:9000/query', json=query)
        print('response:', response.text)
        # parsed_response = parse_api_result(response, parameters)
        
        st.info("OpenAI API response:")
        st.write(response.json())