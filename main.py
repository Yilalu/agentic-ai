import streamlit as st
from src.model import create_model, create_embedding_model

# Configure the page
st.set_page_config(
    page_title="Agentic AI",
    page_icon="🤖",
    layout="wide"
)

# Title
st.title("Agentic AI")

st.markdown("""
Welcome to the Agentic AI application!

This app demonstrates the capabilities of an autonomous AI agent that can handle various banking-related tasks. The agent can process support cases, triage issues, and provide information based on a knowledge base of banking policies.
""")

# Create the model
model = create_model()

st.success("Model created successfully. You can now interact with the agent.")

# User input
user_query = st.text_input(
    "Enter your query here:",
    key="user_query"
)

# invoke the model if the user entered something
if user_query:
    st.write("**You entered:**", user_query)

    try:
        model_response = model.invoke(user_query)

        # Display the response based on its structure
        if hasattr(model_response, "content"):
            if isinstance(model_response.content, str):
                st.write("**Model response:**", model_response.content)
            elif isinstance(model_response.content, list):
                text = model_response.content[0].get("text", "")
                st.write("**Model response:**", text)
            else:
                st.write("**Model response:**", model_response.content)
        else:
            st.write("**Model response:**", model_response)

    except Exception as e:
        st.error(f"Error invoking model: {e}")

