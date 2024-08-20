def setup_parser():
    import argparse
    import re

    def render_size_type(arg_value, pat=re.compile(r"^(\d+)x(\d+)$")):
        if not pat.match(arg_value):
            raise argparse.ArgumentTypeError("invalid value")
        h, w = pat.match(arg_value).groups()
        return h, w

    # parse args for rendering an image
    parser = argparse.ArgumentParser(
        description="Render the scene from (one or more) (given or random) angles using NeRF"
    )

    parser.add_argument(
        "--num_samples",
        type=int,
        default=1,
        help="Number of rendered samples",
    )

    parser.add_argument(
        "--num_rays", type=int, default=1024, help="Number of rays to use for rendering"
    )

    parser.add_argument(
        "--render_size",
        type=render_size_type,
        default="128x128",
        help="Size of the rendered image, in format: hxw, eg: 128x128",
    )

    args = parser.parse_args()

    return parser, args


if __name__ == "__main__":
    from utils.test import test_model_train_loop
    from utils import seed_everything

    parser, args = setup_parser()

    seed_everything(12321)

    test_model_train_loop()
