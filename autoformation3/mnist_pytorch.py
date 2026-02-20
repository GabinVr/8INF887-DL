import marimo

__generated_with = "0.19.10"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import torchvision
    import torch.nn as nn
    import torch
    import torch.optim as optim
    from torch.utils.data import DataLoader
    from torchvision.datasets import MNIST
    from torchvision import transforms
    import matplotlib.pyplot as plt

    return DataLoader, MNIST, mo, nn, optim, torch, transforms


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # 8INF887 - Deep learning
    ## Autoformation 3: *vous devez créer votre propre CNN pour le problème MNIST ou un autre dataset. Votre CNN doit minimalement avoir 4 couches de convolution, une couche de dropout, une couche de pooling et terminer sur un softmax.*
    ### Gabin VRILLAULT - Février 2026
    """)
    return


@app.cell
def _(torch):
    device = 'cpu'
    if torch.cuda.is_available():
        device = 'gpu'
    elif torch.backends.mps.is_available():
        device = 'mps'

    print(f"On utilise {device} ")
    return (device,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Voici mon modèle composé de 3 couches convolutions 1 couche de max-pooling, et deux couches dense séparé d'une couche de dropout puis un softmax.
    """)
    return


@app.cell(hide_code=True)
def _(DataLoader, MNIST, transforms):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    train_dataset = MNIST(root='./data', train=True, transform=transform, download=True)
    test_dataset = MNIST(root='./data', train=False, transform=transform, download=True)

    print(f"{train_dataset[0][0].shape}")

    train_loader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=64, shuffle=False)
    return test_loader, train_loader


@app.cell
def _(mo, nn, torch):
    class ConvBlock(torch.nn.Module):
        def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, stride=2):
            super(ConvBlock, self).__init__()
            self.conv = torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=padding)
            self.relu = torch.nn.ReLU()

        def forward(self, x):
            x = self.conv(x)
            x = self.relu(x)

            return x

    class CNN_MNIST_torch(nn.Module):
        def __init__ (self):
            super(CNN_MNIST_torch, self).__init__()
            self.conv1 = ConvBlock(1, 32)
            self.conv2 = ConvBlock(32, 64)
            self.conv3 = torch.nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1) # J'ai mis stride à 1 car il y'a le pooling après.
            self.relu = torch.nn.ReLU()
            self.pool = torch.nn.MaxPool2d(kernel_size=2, stride=2)
            self.fc1 = nn.Linear(3*3*128,128)
            self.dropout = nn.Dropout(0.5)
            self.fc2 = nn.Linear(128,10)


        def forward(self, x):
            x = self.conv1(x)
            x = self.conv2(x)
            x = self.conv3(x)
            x = self.relu(x)
            x = self.pool(x)
            x = self.fc1(x.flatten(start_dim=1))
            x = self.relu(x)
            x = self.dropout(x)
            x = self.fc2(x)
            return nn.Softmax(dim=1)(x)
    mo.show_code()
    return (CNN_MNIST_torch,)


@app.cell
def _(CNN_MNIST_torch, train_loader):
    it = iter(train_loader)
    X_batch, y_batch = next(it)
    cnn = CNN_MNIST_torch()
    print(f"{cnn.forward(X_batch).shape}")
    return (cnn,)


@app.cell
def _(cnn, device, mo, nn, optim, train_loader):
    cnn.to(device)
    # Normal gradient descent
    criterion = nn.MSELoss()
    optimizer = optim.SGD(cnn.parameters(), lr=0.01, momentum=0.9)
    def train():
        for epoch in range(5):
            cnn.train()
            for batch_idx, (images, labels) in enumerate(train_loader):
                images, labels = images.to(device), labels.to(device)
    
                # Forward pass
                outputs = cnn(images)
                loss = criterion(outputs, nn.functional.one_hot(labels, num_classes=10).float())
    
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
    
            print(f"Epoch [{epoch + 1}/{5}], Loss: {loss.item():.4f}")
    with mo.redirect_stdout():
        mo.show_code(train())
    return


@app.cell
def _(cnn, device, test_loader, torch):
    def _():
        cnn.eval()  # Mode évaluation
        with torch.no_grad():
            correct = 0
            total = 0
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = cnn(images)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        print(f"Précision sur l'ensemble de test : {100 * correct / total:.2f}%")
        return 100 * correct / total
    precision = _()
    return (precision,)


@app.cell(hide_code=True)
def _(mo, precision):
    mo.md(f"""
    Sur l'ensemble de test mon CNN a eu {precision:.2f}% de réussite
    """)
    return


if __name__ == "__main__":
    app.run()
