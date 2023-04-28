import requests
import streamlit as st

parameters = {'api_key': '*',  # placeholder
              'model': 'text-davinci-002',
              'temperature': 0.0,
              'max_length': 64,
              'top_p': 1.0,
              'best_of': 1,
              # 'logprobs': 10,
              'test_mode': False
              }

with st.sidebar:
    parameters['model'] = st.selectbox(
                                    "Model",
                                    options = ('GPT-4',
                                              'text-davinci-003',
                                              'alpaca',
                                              ),
                                    index = 2
                                    )

    if parameters['model'] != 'alpaca':
        parameters['api_key'] = st.text_input('OpenAI API Token',
                                    type='password',
                                    placeholder='Enter your API key',
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
    if parameters['model'] == 'alpaca':
        parameters['test_mode'] = st.checkbox('Test Mode',
                                value=parameters['test_mode'],
                                help='Pass minimal prompt to the model to reduce response time.'
                                )
if __name__ == "__main__":
    st.title("CKIDS Event Forecasting Demo")

    # Create text input box for user input
    user_input = st.text_area('Text to complement', 
                            value='It was the best of times, it was the worst of times, ',
                            placeholder='',
                            )

    # Handle user input and API call
    if st.button('Submit'):
        if not user_input:
            st.error("Please enter some text.")
        if parameters['model'] != 'alpaca' and not parameters['api_key']:
            st.error("Please enter your OpenAI API Key")
        
        payload = {'params': parameters,
                    'query': user_input,
                    }
        
        response = requests.post('http://127.0.0.1:9002/query', json=payload)
        
        st.info("'{}' generated output:".format(parameters['model']))
        
        st.write("Answer:")
        st.write(response.json()['response'])
        st.write("Context:")
        st.write(response.json()['context'])
