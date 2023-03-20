import streamlit as st
st.set_page_config(page_title="MNIST & Streamlit 👋", layout="wide")

st.sidebar.markdown("# Main page 🎈")

st.title("Welcome to MNIST Streamlit App👋")
st.info("This Streamlit app is designed to implement CNN deep learning using PyTorch with the MNIST dataset. It allows users to conduct training, evaluation, and inference.")
st.caption('23.03.20 uploaded')

st.markdown(
    """
    ## Training Page
    ✅ Train the model after setting hyperparameters

    ✅ Shows the history of loss and accuracy as a plot and dataframe 📊📈
    
    ✅ Implement sample visualization per epoch 🖼️
    
    ✅ Implement plot and dataframe save function after completion of learning 💾💻


    ## Evaluation Page (Under Development)
    ❎ Display saved model list as a dataframe and select the model 📋🔎
    
    ❎ Display the results of the metrics specified for the test set 📊👀

    
    ## Inference Page (Under Development)
    ❎ Display saved model list as a dataframe and select the model 📋🔎
    
    ❎ Show the results by inference of the selected model and the selected sample and save the results 💾💻

    """ 
)

st.balloons()
container = st.container()
with container:
    st.markdown("""
    ## 🦆 duc-ke (developer) info.
    - **Github**: [https://github.com/duc-ke](https://github.com/duc-ke)
    - **Email**: [applekey2722@gmail.com](applekey2722@gmail.com)
    """)