import streamlit as st
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

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

    if st.button("Train"):
        with st.spinner("Training the model..."):
            ## 프로그래스바 설정, 교육 진행률 표시하고 진행률 텍스트를 업데이트
            progress_bar = st.progress(0)
            progress_text = st.empty()
            for epoch in range(1, epochs + 1):
                train(model, device, train_loader, optimizer, epoch, progress_bar, progress_text)
            st.success("Training completed!")

# Train the model (updated)
def train(model, device, train_loader, optimizer, epoch, progress_bar, progress_text):
    model.train()
    total_batches = len(train_loader)
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = nn.functional.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 10 == 0:
            progress_bar.progress(batch_idx / total_batches)
            progress_text.text(f"Epoch {epoch} - Batch {batch_idx}/{total_batches} - Loss: {loss.item():.6f}")

# Run the app
if __name__ == "__main__":
    main()
