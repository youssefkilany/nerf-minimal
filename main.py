import argparse

if __name__ == '__main__':

    # parse args for rendering an image
    parser = argparse.ArgumentParser(description='Render an image using NeRF')

    parser.add_argument('--image_path', type=str, required=True, help='Path to the image to render')
    parser.add_argument('--num_samples', type=int, default=64, help='Number of samples to use for rendering')
    parser.add_argument('--num_rays', type=int, default=1024, help='Number of rays to use for rendering')
    parser.add_argument('--render_size', type=int, default=512, help='Size of the rendered image')
    parser.add_argument('--fov', type=float, default=30.0, help='Field of view for rendering')

    args = parser.parse_args()

    print(f'{args = }')
