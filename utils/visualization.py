from .render_units import Ray, Camera, Volume


def visualize_rays_samples(
    rays: list[Ray],
    num_smples: int = 0,
    volume: Volume = None,
    ax=None,
):
    from utils import sample_points_along_ray

    for ray in rays:
        samples = sample_points_along_ray(ray, volume, num_smples)
        ax.plot3D(*zip(*samples), "ko")


def visualize_rays(
    camera: Camera,
    rays: list[Ray],
    ax=None,
):
    """
    Visualize rays in 3D space given the camera.

    Args:
        camera: Camera object
        rays: List of Ray objects
        ax: Existing 3D axis to plot on (optional)
        kwargs: Additional keyword arguments for seaborn plot (e.g., color, marker, etc.)

    Returns:
        3D axis object
    """
    import matplotlib.pyplot as plt

    if ax is None:
        plt.close("all")
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        ax.set_aspect("equal")

    # Plot camera position and orientation
    ax.scatter(
        camera.position[0],
        camera.position[1],
        camera.position[2],
        c="b",
        marker="o",
        label="Camera",
    )

    # Plot rays
    for ray in rays:
        origin = ray.origin.numpy()
        direction = ray.direction.numpy()
        points = [origin + direction * camera.near, origin + direction * camera.far]
        ax.plot3D(*zip(*points), c="r", lw=0.5)

    # Set axis labels and title
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("Rays in 3D Space")

    plt.show()

    return ax


def visualize_rays_mayavi(rays: list[Ray]):
    import torch
    from mayavi import mlab

    ray_origins = torch.cat([ray.origin[:, None] for ray in rays]).reshape(-1, 3)
    ray_directions = torch.cat([ray.direction for ray in rays]).reshape(-1, 3)
    mlab.quiver3d(
        ray_origins[:, 0],
        ray_origins[:, 1],
        ray_origins[:, 2],
        ray_directions[:, 0],
        ray_directions[:, 1],
        ray_directions[:, 2],
        color=(0, 1, 1),
        scale_mode="vector",
        line_width=0.1,
        resolution=4,
        scale_factor=0.1,
    )

    mlab.show()


def visualize_image(tensor_image):
    import matplotlib.pyplot as plt

    plt.imshow(tensor_image.permute(1, 2, 0).numpy())
    plt.axis("off")
    plt.show()
