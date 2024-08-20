from dataclasses import dataclass
import torch


@dataclass(slots=True)
class Camera:
    position: torch.Tensor
    orientation: torch.Tensor
    image_width: int
    image_height: int
    fov: float = 90.0
    near: float = 0.0
    far: float = 1.0
    # assuming forward direction is (0, 0, -1)
    forward_vec = torch.tensor([0.0, 0.0, -1.0])
    # assuming right direction is (1, 0, 0)
    right_vec = torch.Tensor([1.0, 0.0, 0.0])


@dataclass(slots=True)
class Ray:
    origin: torch.Tensor
    direction: torch.Tensor

    def to_input_tensor(self, device="cpu"):
        """
        Normalize and concatenate origin and direction tensors into a single input tensor.

        Args:
            device (str, optional): Device to move the tensor to (default: "cpu")

        Returns:
            torch.Tensor: Input tensor of shape (6,) with normalized origin and direction
        """
        # Normalize origin vector
        self.origin /= torch.norm(self.origin)

        # Normalize direction vector
        self.direction /= torch.norm(self.direction)

        # Concatenate origin and direction tensors
        input_tensor = torch.cat(
            [self.origin, self.direction], dim=0, dtype=torch.float32
        )

        # Move the tensor to the specified device
        input_tensor = input_tensor.to(device)

        return input_tensor


@dataclass(slots=True)
class Volume:
    origin: torch.Tensor = torch.tensor([0.0, 0.0, 0.0])
    dimensions: torch.Tensor = torch.tensor([1.0, 1.0, 1.0])
    resolution: torch.Tensor = torch.tensor([128, 128, 1])


@dataclass()
class Point:
    position: torch.Tensor
    view_direction: torch.Tensor

    color: torch.Tensor
    density: float
