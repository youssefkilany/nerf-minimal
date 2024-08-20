import torch
from .nerf_units import Ray, Camera, Volume, Point
from .visualization import visualize_rays


def generate_rays(camera: Camera) -> list[Ray]:
    """
    Generate rays that go through the image plane given the camera position and view direction.

    Args:
        camera: Camera object

    Returns:
        A list of Ray objects
    """
    # Normalize the view direction
    view_direction = torch.matmul(camera.orientation, camera.forward_vec)
    view_direction /= torch.linalg.norm(view_direction)

    # Calculate the right vector (assuming up vector is (0, 1, 0))
    right_vector = torch.cross(view_direction, camera.right_vec, dim=0)
    right_vector /= torch.linalg.norm(right_vector)

    # calculate up vector
    up_vector = torch.cross(view_direction, right_vector, dim=0)
    up_vector /= torch.linalg.norm(up_vector)

    # calculate aspect ratio
    aspect_ratio = camera.image_width / camera.image_height

    # calculate scale
    scale = torch.tan(torch.deg2rad(torch.tensor(camera.fov / 2)))

    # calculate image plane coordinates
    image_plane_coords = torch.meshgrid(
        torch.linspace(-scale * aspect_ratio, scale * aspect_ratio, camera.image_width),
        torch.linspace(-scale, scale, camera.image_height),
        indexing="ij",
    )

    # calculate ray origins and directions
    ray_origins = camera.position + view_direction * camera.near
    ray_origins = ray_origins.repeat(
        camera.image_width * camera.image_height, 1
    ).reshape(-1, 3)

    ray_directions = (
        image_plane_coords[0].flatten()[:, None] * right_vector[None, :]
        + image_plane_coords[1].flatten()[:, None] * up_vector[None, :]
        + view_direction[None, :]
    )

    return [
        Ray(origin, direction) for origin, direction in zip(ray_origins, ray_directions)
    ]


def sample_points_along_ray(
    ray: Ray, volume: Volume, n: int = 100, device="cpu"
) -> list[Point]:
    """
    Sample `n` points along a ray within a volume.

    Args:
        ray: The ray to sample points from.
        volume: The volume to sample points within.
        n: The number of points to sample.
        device: The device to perform the computation on (default: "cpu").

    Returns:
        A tensor of shape `(n, 3)` containing the sampled points in world coordinates.
    """

    volume_end = volume.origin + volume.dimensions
    # Sample points along the ray
    sampled_points = []
    for i in torch.linspace(0, 1, n):
        sampled_point = ray.origin + i * ray.direction
        if all(volume.origin <= sampled_point) and all(sampled_point < volume_end):
            sampled_points.append(Point(position=sampled_point))

    return sampled_points


def sample_points_within_camera(
    camera: Camera, volume: Volume, n: int = 100, device="cpu"
) -> list[Point]:
    return [
        sample_points_along_ray(ray, volume, n, device) for ray in generate_rays(camera)
    ]


if __name__ == "__main__":
    from visualization import visualize_rays

    camera = Camera(
        position=torch.tensor([0, 0, 0]),
        orientation=torch.eye(3),
        image_width=3,
        image_height=3,
        fov=90.0,
        near=0.0,
        far=1.0,
    )
    rays = generate_rays(camera)

    volume = Volume(
        origin=torch.tensor([-1, -1, -1]),
        dimensions=torch.tensor([2, 2, 2]),
    )

    visualize_rays(camera, rays, visualize_samples=True, num_smples=50, volume=volume)
