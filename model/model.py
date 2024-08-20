import torch.nn as nn


class NeRFModel(nn.Module):
    def __init__(
        self,
        num_hidden_layers,
        hidden_dim,
    ):
        super().__init__()

        self.num_layers = 3  # variable, [2-4]
        self.input_dim = 3  # constant, r,g,b,theta,phi
        self.output_dim = 4  # constant, 3 for rgb and 1 for sigma

        self.input_layer = nn.Linear(self.input_dim, hidden_dim)
        self.hidden_layers = nn.Sequential(
            *[nn.Linear(hidden_dim, hidden_dim) for _ in range(num_hidden_layers)]
        )
        self.output_layer = nn.Linear(hidden_dim, self.output_dim)

    def forward(self, x):
        x = self.input_layer(x)
        x = self.hidden_layers(x)
        return self.output_layer(x)


nerf_model = None


def get_nerf_model():
    global nerf_model
    if nerf_model:
        return nerf_model

    device = get_cuda_if_available()
    num_hidden_layers = 3
    hidden_dim = 4
    nerf_model = NeRFModel(num_hidden_layers, hidden_dim).to(device)
    return nerf_model


def get_cuda_if_available():
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"
