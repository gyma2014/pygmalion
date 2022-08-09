import torch
import config
from torchvision.utils import save_image, make_grid
import numpy as np


def dumpObj(fake, objFile):
    points = []
    for i in range(len(fake)):
        for j in range(len(fake[i])):
            if fake[i][j][0] == 0:
                continue
            points.append(f'v {j} {500-i} {fake[i][j][0]}')

    with open(objFile, 'w') as fp:
        fp.write('\n'.join(points))

def tensor2np(tensor):
    grid = make_grid(tensor)
    # Add 0.5 after unnormalizing to [0, 255] to round to nearest integer
    ndarr = grid.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to("cpu", torch.uint8).numpy()
    return ndarr

def save_some_examples(gen, val_loader, epoch, folder):
    x, y = next(iter(val_loader))
    x, y = x.to(config.DEVICE), y.to(config.DEVICE)
    gen.eval()

    vis_x = tensor2np(x*0.5 + 0.5)
    vis_y = tensor2np(y*0.5 + 0.5)

    with torch.no_grad():
        y_fake = gen(x)
        y_fake = y_fake * 0.5 + 0.5  # remove normalization#
        #save_image(y_fake, folder + f"/y_gen_{epoch}.png")
        #save_image(x * 0.5 + 0.5, folder + f"/input_{epoch}.png")
        #if epoch == 1:
        #    save_image(y * 0.5 + 0.5, folder + f"/label_{epoch}.png")
        
        vis_y_fake = tensor2np(y_fake)
        
        print(vis_y_fake[0][0])

        # dump x, y and y_fake together
        output_y = np.vstack((vis_y, vis_y_fake))
        output = np.vstack((vis_x, output_y))

        #save_image(output, folder + f"/gen_{epoch}.png")
        from PIL import Image
        plimage = Image.fromarray(output)
        plimage.save(folder + f"/gen_{epoch}.png")

        # dumpObj(output_y, folder + f"/gen_{epoch}.obj")

    gen.train()


def save_checkpoint(model, optimizer, filename="my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    checkpoint = {
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    torch.save(checkpoint, filename)


def load_checkpoint(checkpoint_file, model, optimizer, lr):
    print("=> Loading checkpoint")
    checkpoint = torch.load(checkpoint_file, map_location=config.DEVICE)
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])

    # If we don't do this then it will just have learning rate of old checkpoint
    # and it will lead to many hours of debugging \:
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr

