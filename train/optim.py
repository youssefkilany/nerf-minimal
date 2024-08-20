from torch import optim, nn


lr = 5e-4
decay_rate = 0.1
decay_steps = 20 * 1000


def get_optimizer(optimizer_name, model_parameters):
    match optimizer_name:
        case "Adam":
            return optim.Adam(
                model_parameters,
                lr=5e-4,
                betas=(0.9, 0.999),
                eps=1e-07,
                weight_decay=0,
                amsgrad=False,
            )

        case _:
            raise TypeError("Unknown optimizer name")


def get_loss_function(loss_function_name):
    def img2mse(pred_img, gt_img):
        return (pred_img - gt_img).pow(2).mean()

    def l2_loss(pred, gt):
        return (pred - gt).pow(2).sum()

    match loss_function_name:
        case "img2mse":
            return img2mse
        case "L2":
            return l2_loss
        case "MSE":
            return nn.MSELoss()

        case _:
            raise TypeError("Unknown loss function name")


def update_lr(optimizer, step):
    new_lr = lr * (decay_rate ** (step / decay_steps))
    for param_group in optimizer.param_groups:
        param_group["lr"] = new_lr
