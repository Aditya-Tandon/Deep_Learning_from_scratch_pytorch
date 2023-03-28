import torch


class Neuron:
    def __init__(self, num_inputs, num_outputs):
        self.w = torch.randn(
            num_inputs, num_outputs, requires_grad=True, dtype=torch.float32
        )
        self.b = torch.randn(
            num_outputs, 1, requires_grad=True, dtype=torch.float32
        )

    def forward(self, inputs):
        self.out = (inputs @ self.w + self.b).relu()
        return self.out

    def train(self, inputs, outputs, epochs, learning_rate):
        for _ in range(epochs):
            loss = ((outputs - self.forward(inputs)) ** 2).flatten().sum()
            loss.backward()
            print(f"loss: {loss.data}")
            self.w.data = self.w.data - self.w.grad * learning_rate
            self.b.data = self.b.data - self.b.grad * learning_rate
            self.w.grad = None
            self.b.grad = None
