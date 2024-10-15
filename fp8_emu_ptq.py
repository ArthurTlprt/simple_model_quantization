import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
from mpemu import mpt_emu


class Classifier(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(1, 8, 3, stride=2),
            nn.ReLU(),
            nn.Conv2d(8, 8*8, 3, stride=2),
            nn.ReLU(),
            nn.AvgPool2d(3, 2),
            nn.Flatten(),
            nn.Dropout(),
            nn.Linear(256, 128), 
            nn.Dropout(),
            nn.Linear(128, 10), 
            nn.Softmax()
        )

    def forward(self, x):
        return self.model(x)
    

def train(dataloader, model, loss_fn, optimizer):
    model.to("cuda")
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to("cuda"), y.to("cuda")

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def test(dataloader, model, loss_fn, backend="cpu"):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    model.to(backend)
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(backend), y.to(backend)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")


if __name__ == "__main__":

    model = Classifier()

    training_data = datasets.FashionMNIST(
        root="data",
        train=True,
        download=True,
        transform=ToTensor(),
    )

    # Download test data from open datasets.
    test_data = datasets.FashionMNIST(
        root="data",
        train=False,
        download=True,
        transform=ToTensor(),
    )

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

    batch_size = 64

    # Create data loaders.
    train_dataloader = DataLoader(training_data, batch_size=batch_size)
    test_dataloader = DataLoader(test_data, batch_size=batch_size)

    model, emulator = mpt_emu.quantize_model(model, optimizer=optimizer, dtype="e4m3")
    emulator.set_default_inference_qconfig()

    for i in range(1):
        train(train_dataloader, model, loss_fn, optimizer)
        test(test_dataloader, model, loss_fn)

    eval_model = emulator.fuse_bnlayers_and_quantize_model(model)
    
    
    
    
    print(emulator.emulator.model_qconfig_dict)
    print(emulator.emulator.mod_qconfig)
    for name, param in emulator.emulator.model.named_parameters():
        print(param.dtype)
    print(eval_model)

    torch.set_printoptions(precision=10)
    print(model.eval().model[0].weight[0, :])
    print(eval_model.eval().model[0].weight[0, :])

    test(test_dataloader, eval_model, loss_fn)

    for name, param in eval_model.named_parameters():
        print(param.dtype)