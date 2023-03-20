import streamlit as st
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import pandas as pd
import altair as alt
import matplotlib.pyplot as plt

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
            # Create a container for the results
            results = pd.DataFrame(columns=['Epoch', 'Train Loss', 'Test Loss', 'Accuracy'])
            progress_bar = st.progress(0)
            progress_text = st.empty()


            # Create a container for the line chart
            chart_container = st.empty()
            
            # Create a container for the results
            results_container = st.empty()

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

                # Update the line chart (accuracy, train loss, test loss가 각각 나옴)
                """
                chart_data = results.melt('Epoch', var_name='a', value_name='b')
                line_chart = alt.Chart(chart_data).mark_line().encode(
                    x='Epoch:Q',
                    y=alt.Y('b:Q', axis=alt.Axis(title='Value')),
                    color='a:N'
                ).facet(
                    column='a:N'
                ).resolve_scale(
                    y='independent'
                ).interactive()
                chart_container.altair_chart(line_chart, use_container_width=True)
                """

                # Update the line chart 2(loss끼리 묶고 accuracy 따로 있음)
                """
                train_test_loss_chart = alt.Chart(results).mark_line().encode(
                    x='Epoch:Q',
                    y=alt.Y('Train Loss:Q', axis=alt.Axis(title='Train Loss')),
                    color=alt.value('blue')
                ) + alt.Chart(results).mark_line().encode(
                    x='Epoch:Q',
                    y=alt.Y('Test Loss:Q', axis=alt.Axis(title='Test Loss')),
                    color=alt.value('green')
                )

                accuracy_chart = alt.Chart(results).mark_line().encode(
                    x='Epoch:Q',
                    y=alt.Y('Accuracy:Q', axis=alt.Axis(title='Accuracy')),
                    color=alt.value('red')
                )

                train_test_loss_chart = train_test_loss_chart.interactive()
                accuracy_chart = accuracy_chart.interactive()

                chart = alt.vconcat(train_test_loss_chart, accuracy_chart).resolve_scale(y='independent')
                chart_container.altair_chart(chart, use_container_width=True)
                """

                # Update the line chart 3(matplotlib 사용)
                fig, ax1 = plt.subplots(figsize=(12, 6))
                ax1.set_xlabel('Epoch')
                ax1.set_ylabel('Loss', color='blue')
                ax1.plot(results['Epoch'], results['Train Loss'], label='Train Loss', color='blue')
                ax1.plot(results['Epoch'], results['Test Loss'], label='Test Loss', color='green')
                ax1.tick_params(axis='y', labelcolor='blue')

                ax2 = ax1.twinx()
                ax2.set_ylabel('Accuracy', color='red')
                ax2.plot(results['Epoch'], results['Accuracy'], label='Accuracy', color='red')
                ax2.tick_params(axis='y', labelcolor='red')

                fig.tight_layout()

                chart_container.pyplot(fig, clear_figure=True)

            st.success("Training and evaluation completed!")


# Train the model (updated)
def __train(model, device, train_loader, optimizer, epoch, progress_bar, progress_text):
    model.train()
    train_loss = 0
    total_batches = len(train_loader)
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = nn.functional.nll_loss(output, target)
        train_loss += loss
        loss.backward()
        optimizer.step()
        if batch_idx % 10 == 0:
            progress_bar.progress(batch_idx / total_batches)
            progress_text.text(f"Epoch {epoch} - Batch {batch_idx}/{total_batches} - Loss: {loss.item():.6f}")
            # st.write(f"Epoch {epoch} - Batch {batch_idx}/{total_batches} - Loss: {loss.item():.6f}")
    
    train_loss /= len(train_loader.dataset)
    return train_loss

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


# Run the app
if __name__ == "__main__":
    main()
