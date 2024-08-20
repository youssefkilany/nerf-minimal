def test_load_images():
    from data import dataset_base_path, load_dataset_data
    from utils import visualize_image
    from random import randint

    imgs, poses, render_poses, [H, W, focal], splits = load_dataset_data(
        dataset_base_path, testskip=10
    )
    print(f"{imgs.shape = }")
    print(f"{poses.shape = }")
    [print(f"split[{idx}].shape = {split.shape}") for idx, split in enumerate(splits)]
    print(f"{render_poses.shape = }")
    print(f"{H = }", f", {W = }", f", {focal = :.3f}")
    visualize_image(imgs[randint(0, imgs.shape[0])])


def test_render_utils():
    from utils import Camera, Volume, visualize_rays, generate_rays
    import torch

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
