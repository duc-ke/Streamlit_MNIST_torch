import streamlit as st
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import pandas as pd
import altair as alt
import matplotlib.pyplot as plt
import base64
import io
import copy
import numpy as np
import torch.backends.cudnn as cudnn
import random
# from streamlit.report_thread import add_report_ctx, get_report_ctx

seed = 42



# Define the CNN architecture
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout2d(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = nn.functional.relu(x)
        x = self.conv2(x)
        x = nn.functional.relu(x)
        x = nn.functional.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = nn.functional.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = nn.functional.log_softmax(x, dim=1)
        return output

# Load MNIST data
def load_data(batch_size):
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('./data', train=False, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader

# Train the model
# ... [same as before]

# Main function
def main():

        ## [ENV FIX] ##
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if use multi-GPU
    cudnn.benchmark = False
    cudnn.deterministic = True
    random.seed(seed)
    
    # Setup Streamlit
    # 사용자 정의 페이지 제목, 레이아웃, 초기사이드바 설정
    st.set_page_config(page_title="MNIST Trainer", layout="wide", initial_sidebar_state="collapsed")
    st.title("PyTorch Deep Learning Training with Streamlit")
    st.write("This app trains a simple CNN on the MNIST dataset.")

    # Set hyperparameters
    ## 사이드바 헤더 설정
    st.sidebar.header("Hyperparameters")
    ## input을 딱 잡는게 아니라 사이드바로 구성
    batch_size = st.sidebar.slider("Batch size", min_value=1, max_value=1000, value=64, step=1)
    epochs = st.sidebar.slider("Epochs", min_value=1, max_value=100, value=5, step=1)
    learning_rate = st.sidebar.slider("Learning rate", min_value=0.0001, max_value=1.0, value=0.01, step=0.0001, format="%.4f")

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    st.write(f"Device: {device}")

    # Load data
    train_loader, test_loader = load_data(batch_size)

    # Initialize the model, optimizer, and train
    model = Net().to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)


    # Check if the training has been completed
    if "trained" not in st.session_state:
        st.session_state.trained = False

    # Add a variable to the session_state to track the state of the Train button
    if "train_button_clicked" not in st.session_state:
        st.session_state.train_button_clicked = False

    # Declare the figure outside the loop
    train_button = st.button("Train")

    if train_button:
        st.session_state.train_button_clicked = not st.session_state.train_button_clicked

    fig_result = None
    csv_result = None
    # Create two columns for DataFrame and chart
    col1, col2 = st.columns(2)
    # Create a container for the line chart
    chart_container = col1.empty()
    # Create a container for the results
    results_container = col2.empty()

    if st.session_state.train_button_clicked and not st.session_state.trained:
        with st.spinner("Training the model..."):
            # Create a container for the results
            results = pd.DataFrame(columns=['Epoch', 'Train Loss', 'Test Loss', 'Accuracy'])
            progress_bar = st.progress(0)
            progress_text = st.empty()


            for epoch in range(1, epochs + 1):
                train_loss = train(model, device, train_loader, optimizer, epoch, progress_bar, progress_text)
                test_loss, accuracy = test(model, device, test_loader)
                # Append the results to the DataFrame
                results = results.append({'Epoch': epoch, 'Train Loss': train_loss, 'Test Loss': test_loss, 'Accuracy': accuracy}, ignore_index=True)

                # Display the DataFrame in a scrollable table
                # results_container.write(
                #     f'<style>.scrollable {{max-height: 150px; overflow-y: scroll;}}</style>'
                #     f'<div class="scrollable">{results.to_html(index=False)}</div>',
                #     unsafe_allow_html=True,
                # )
                # Display the DataFrame in a scrollable table
                with results_container:
                    st.dataframe(results)

                # Update the line chart 3(matplotlib 사용)
                fig, ax1 = plt.subplots(figsize=(12, 6))
                ax1.clear()
                ax1.set_xlabel('Epoch')
                ax1.set_ylabel('Loss', color='blue')
                ax1.plot(results['Epoch'], results['Train Loss'], label='Train Loss', color='blue')
                ax1.plot(results['Epoch'], results['Test Loss'], label='Test Loss', color='green')
                ax1.tick_params(axis='y', labelcolor='blue')
                ax1.legend(loc='upper left')

                ax2 = ax1.twinx()
                ax2.set_ylabel('Accuracy', color='red')
                ax2.plot(results['Epoch'], results['Accuracy'], label='Accuracy', color='red')
                ax2.tick_params(axis='y', labelcolor='red')
                ax2.legend(loc='upper right')

                fig.tight_layout()

                st.session_state.fig = copy.deepcopy(fig)
                chart_container.pyplot(fig, clear_figure=True)

            st.session_state.results = results
            st.session_state.trained = True
            st.success("Training and evaluation completed!")

    # Display the download buttons if the training has been completed
    if st.session_state.trained:
        # # Display the download buttons for the DataFrame and the plot
        results_container.dataframe(st.session_state.results)
        chart_container.pyplot(st.session_state.fig, clear_figure=False)

        # Display the download buttons for the DataFrame and the plot
        st.write("Download the results as a CSV file:")
        dataframe_csv_buffer = dataframe_to_csv_buffer(st.session_state.results)
        st.download_button("Download CSV", dataframe_csv_buffer, "results.csv", "text/csv")

        st.write("Download the plot as an image:")
        fig_image_buffer = fig_to_image_buffer(st.session_state.fig)
        st.download_button("Download Image", fig_image_buffer, "plot.png", "image/png")

        
    # Add a refresh button
    if st.button("Refresh"):
        # Clear GPU memory cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Reset session state
        reset_session_state()

        # Refresh the UI by updating containers
        results_container.empty()
        chart_container.empty()

        # Reset the session state and re-run the main function
        st.experimental_rerun()



def train(model, device, train_loader, optimizer, epoch, progress_bar, progress_text):
    model.train()
    train_loss = 0
    total_samples = 0

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = nn.functional.nll_loss(output, target)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        total_samples += len(data)

        progress_bar.progress((batch_idx + 1) / len(train_loader))
        progress_text.text(f"Epoch {epoch} - Batch {batch_idx}/{len(train_loader)} - Loss: {loss.item():.6f}")

    avg_train_loss = train_loss / total_samples
    return avg_train_loss

# Test the model
def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += nn.functional.nll_loss(output, target, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)

    return test_loss, accuracy


def dataframe_to_csv_download_link(df, filename="results.csv"):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    return f'<a href="data:file/csv;base64,{b64}" download="{filename}">Download CSV</a>'

def fig_to_image_download_link(fig, filename="plot.png"):
    buffer = io.BytesIO()
    fig.savefig(buffer, format='png')
    buffer.seek(0)
    b64 = base64.b64encode(buffer.getvalue()).decode()
    return f'<a href="data:image/png;base64,{b64}" download="{filename}">Download Plot</a>'


def dataframe_to_csv_buffer(dataframe):
    buffer = io.BytesIO()
    dataframe.to_csv(buffer, index=False, encoding='utf-8')
    buffer.seek(0)
    return buffer

def reset_session_state():
    st.session_state.train_button_clicked = False
    st.session_state.trained = False
    st.session_state.results = None
    st.session_state.fig = None


from matplotlib.backends.backend_agg import FigureCanvasAgg

def fig_to_image_buffer(fig):
    buf = io.BytesIO()
    FigureCanvasAgg(fig).print_png(buf)
    buf.seek(0)
    return buf




# Run the app
if __name__ == "__main__":
    main()
