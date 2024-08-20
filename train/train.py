from .optim import get_optimizer, get_loss_function, update_lr
from .utils import render
import torch
from torch.utils.data import DataLoader


def train_nerf(
    model,
    train_dataloader: DataLoader,
    val_dataloader: DataLoader,
    device: str,
    optimizer="Adam",
    loss_function="img2mse",
    train_settings: dict = {},
    log_settings: dict = {},
):
    if isinstance(optimizer, str):
        optimizer = get_optimizer(optimizer, model.parameters())

    if isinstance(loss_function, str):
        loss_function = get_loss_function(loss_function)

    h, w, focal = train_settings["hwf"]
    epochs_num = train_settings["epochs_num"]
    chunk = train_settings["chunk"]
    near, far = train_settings["near"], train_settings["far"]
    intrinsic_mat = torch.tensor(
        [
            [focal, 0, 0.5 * w],
            [0, focal, 0.5 * h],
            [0, 0, 1],
        ]
    ).to(device)

    for epoch in range(1, epochs_num + 1):
        # training
        tot_train_loss = 0.0
        for image, c2w_mat, *_ in train_dataloader:
            image, c2w_mat = image[0], c2w_mat[0]
            image = image.to(device)
            c2w_mat = c2w_mat.to(device)
            rgb = render(
                model, h, w, intrinsic_mat, chunk, device, None, c2w_mat, near, far
            )

            optimizer.zero_grad()
            train_loss = loss_function(rgb, image.permute(1, 2, 0))
            tot_train_loss += train_loss.item()
            train_loss.backward()
            optimizer.step()

            update_lr(optimizer, epoch)

        avg_train_loss = tot_train_loss / len(train_dataloader)

        # validation
        tot_val_loss = 0.0
        with torch.no_grad():
            for image, c2w_mat, *_ in val_dataloader:
                rgb = render(
                    model, h, w, intrinsic_mat, chunk, device, None, c2w_mat, near, far
                )
                val_loss = loss_function(rgb, image)
                tot_val_loss += val_loss.item()
            avg_val_loss = tot_val_loss / len(val_dataloader)

        save_every = log_settings["save_every"]
        initial_epochs = log_settings["initial_epochs"]
        initial_save_every = log_settings["initial_save_every"]

        # save checkpoint
        if epoch % save_every == 0 or (
            epoch < initial_epochs and epoch % initial_save_every == 0
        ):
            with open(f"checkpoint_{epoch}.pt", "wb") as f:
                f.write(
                    {
                        "metrics": {
                            "train_loss": avg_train_loss,
                            "val_loss": avg_val_loss,
                        },
                        "model_weights": model.state_dict(),
                        "optimizer_weights": optimizer.state_dict(),
                    }
                )
