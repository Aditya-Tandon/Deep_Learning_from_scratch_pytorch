import torch


class Layer:
    def __init__(self, num_neurons, inputs_per_neuron):
        self.num_neurons = num_neurons
        self.w = torch.randn(
            num_neurons,
            inputs_per_neuron,
            requires_grad=True,
            dtype=torch.float32,
        )
        self.b = torch.randn(
            num_neurons, 1, requires_grad=True, dtype=torch.float32
        )

    def forward(self, inputs):
        out = ((self.w @ inputs) + self.b).relu()
        return out


class MLP:
    def __init__(
        self, num_layers, num_inputs, num_neurons_per_layer, num_final_out
    ):
        self.num_layers = num_layers
        self.layers = [Layer(num_neurons_per_layer, num_inputs)]

        for _ in range(num_layers):
            layer = Layer(num_neurons_per_layer, num_neurons_per_layer)
            self.layers.append(layer)

        layer_out = Layer(num_final_out, num_neurons_per_layer)
        self.layers.append(layer_out)

    def forward(self, input):
        out_prev = input
        for i in range(len(self.layers)):
            out_prev = self.layers[i].forward(out_prev)
        return out_prev

    def train(self, epochs, learning_rate, input, output):
        for _ in range(epochs):
            loss = ((output - self.forward(input)) ** 2).flatten().sum()
            print(f"Loss: {loss}")
            loss.backward()
            for layer in self.layers:
                layer.w.data -= learning_rate * layer.w.grad
                layer.b.data -= learning_rate * layer.b.grad
                layer.w.grad = None
                layer.b.grad = None
