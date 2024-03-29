from typing import Callable, Optional
import math
import numpy as np
import torch
import torchvision


def events_to_frames(frames, polarity: bool = False):
    if len(frames.shape) == 3:
        frames = frames.unsqueeze(-1).repeat(1, 1, 1, 3)
    else:
        if not polarity:
            frames = frames.abs().sum(-1)
        elif polarity:
            frames = torch.concat(
                [frames, torch.zeros(*frames.shape[:-1], 1, device=frames.device)],
                dim=-1,
            )
    frames = ((frames / frames.max()) * 255).int()
    return frames


def rotate_tensor(input, x):
    rotated_input = torchvision.transforms.functional.rotate(
        torch.unsqueeze(input, dim=0),
        x,
        expand=True,
        fill=0,
        interpolation=torchvision.transforms.InterpolationMode.NEAREST,
    )
    return rotated_input[0]


def skew_tensor(image, shear_angle, shear):
    pad = int(image.size()[0] * 2)

    y_shear = shear * math.sin(math.pi * shear * shear_angle / 180)
    x_shear = shear * math.cos(math.pi * shear * shear_angle / 180)

    # define the transformation to be applied to images
    transform = torchvision.transforms.Compose(
        [
            torchvision.transforms.Pad(pad),
            torchvision.transforms.RandomAffine(
                degrees=0,
                shear=[x_shear, x_shear + 0.01, y_shear, y_shear + 0.01],
                fill=0,
                interpolation=torchvision.transforms.InterpolationMode.NEAREST,
            ),
        ]
    )

    skew_image = transform(torch.unsqueeze(image, dim=0))
    skew_image = skew_image[0]

    non_zero_indices = torch.nonzero(skew_image)
    if non_zero_indices.shape[0] == 0:  # Allow for zero shear
        return image

    x_min = torch.min(non_zero_indices[:, 0])
    y_min = torch.min(non_zero_indices[:, 1])
    x_max = torch.max(non_zero_indices[:, 0])
    y_max = torch.max(non_zero_indices[:, 1])

    return skew_image[x_min:x_max, y_min:y_max]


# improved shape rendering function
def render_shape_improved(
    shape_fn: Callable[[int, float, str], torch.Tensor],
    len: int,
    resolution: torch.Size,
    device: str,
    scale_change: bool,
    trans_change: bool,
    rotate_change: bool,
    skew_change: bool,
    rotate_angle: int = 0,
    skew_angle_change: int = 0,
    diameter: Optional[int] = None,
    shape_p: float = 1,
    bg_noise_p: float = 0.01,
    max_velocity: int = 1,
    max_scale_change: int = 1,
    max_shear_change: float = 1,
    max_angle_change: float = 3,
):
    """
    Draws a moving shape for `len` duration with maximum velocity of `max_velocity`.
    Arguments:
        shape_fn (Callable[[int, float, str], torch.Tensor]): The function generating the shape
        len (int): The number of subsequent frames to generate
        resolution (torch.Size): The WxH resolution of the total frame. Should not be smaller than r
        device (str): The device on which the shape will be generated
        scale_change (Boolean): True if scale varies
        trans_change (Boolean): True if translation is present
        rotate_change (Boolean): True if shape rotates
        rotate_angle (int): The amount of rotation in degrees per frame. Defaults to 0
        diameter (Optional[int]): The diameter of the shape. Defaults to a random number between 50 and 200
        shape_p (float): The probability of drawing an event from a Bernouilli distribution
                         (1 = full shape, 0 = empty)
        bg_noise_p (float): The probability of drawing an event for added background noise from a
                         Bernouilli distribution (0 = no noise, 1 = full noise)
        max_velocity: The maximum allowed velocity between frames (1 = maximum moves one pixel)
        max_angle_change: The maximum allowed velocity between frames (1 = maximum moves one pixel)
        max_scale_change: The max change in the diameter of the shape between frames
    Returns:
        A tensor of size (len, resolution)
    """
    assert resolution[0] >= 256 and resolution[1] >= 256, "Resolution must be >= 256"

    mask_r = 5
    images = torch.zeros(len, *resolution, dtype=torch.bool, device=device)
    labels = torch.zeros(len, 2)

    angle = 0
    angle_velocity = 0
    shear = 0
    shear_velocity = 0

    # initialise starting x, y, diameter
    min_resolution = torch.as_tensor(min(resolution[0], resolution[1]))
    min_size = (0.08 * min_resolution).int().to(device)
    max_size = (0.5 * min_resolution).int().to(device)
    if diameter is not None:
        diameter = diameter
    elif scale_change:
        diameter = torch.randint(low=min_size, high=max_size, size=(1,)).item()
    else:
        diameter = 100
    x = torch.randint(
        low=int(diameter / 2) + mask_r,
        high=resolution[0] - int(diameter / 2) - mask_r,
        size=(1,),
    ).item()
    y = torch.randint(
        low=int(diameter / 2) + mask_r,
        high=resolution[1] - int(diameter / 2) - mask_r,
        size=(1,),
    ).item()

    actual_diameter = diameter

    if scale_change:
        delta_scale = torch.distributions.Normal(loc=0, scale=0.2)
        scale = 2 * torch.rand((1,), device=device) - 1
    else:
        scale = torch.zeros((1,), device=device)

    if trans_change:
        deltas = torch.distributions.Normal(loc=0, scale=0.2)
        velocity = 2 * torch.rand((2,), device=device) - 1
    else:
        velocity = torch.zeros((2,), device=device)

    if rotate_change:
        angle = torch.randint(low=0, high=360, size=(1,)).item()
        angle_delta = torch.distributions.Normal(loc=0, scale=0.2)
        angle_velocity = (
            (angle_delta.sample((1,)) * max_angle_change * 2)
            .clip(-max_angle_change, max_angle_change)
            .to(device)
        )

    if skew_change:
        shear = torch.randint(low=0, high=20, size=(1,)).item()
        delta_shear = torch.distributions.Normal(loc=0, scale=0.6)
        shear_velocity = 2 * (torch.rand((1,), device=device) - 0.5)

    for i in range(images.shape[0]):
        x = x + velocity[0]
        y = y + velocity[1]

        noise = (
            torch.distributions.Bernoulli(probs=bg_noise_p)
            .sample(images[i].shape)
            .to(device)
        )
        images[i] += noise.bool()

        # Fill in shape
        img = shape_fn(int(diameter), p=shape_p, device=device)

        if rotate_change:
            rotated_img = rotate_tensor(img, angle)
        else:
            rotated_img = img

        if skew_change:
            skewed_img = skew_tensor(rotated_img, 0, shear)
        else:
            skewed_img = rotated_img

        rotated_img = skewed_img

        # if rotate_change:
        #     # angle = angle + rotate_angle
        #     angle = angle + (angle_delta.sample()).item()
        # else:
        #     angle = 0

        if skew_change:
            # to keep the shear movement
            if shear == 20 or shear == -20:
                shear_velocity = -1 * shear_velocity

            shear_velocity = (
                shear_velocity + delta_shear.sample((1,)).to(device)
            ).clip(-max_shear_change, max_shear_change)
            shear = (shear + shear_velocity[0]).clip(-20, 20)
        else:
            shear = 0

        # if int(rotated_img.size()[0]) > min(
        #     resolution[0] - mask_r - 1, resolution[1] - mask_r - 1
        # ):
        #     scale = -1 * abs(scale)
        # if int(rotated_img.size()[0]) < min(resolution[0], resolution[1]) / 10:
        #     scale = abs(scale)
        # if (
        #     x <= int(rotated_img.size()[0] / 2) + mask_r
        #     or x >= resolution[0] - int(rotated_img.size()[0] / 2) - mask_r
        # ):
        #     velocity[0] = -1 * velocity[0]
        #     scale = -1 * abs(scale)
        #     shear_velocity = -2*torch.sign(shear)*abs(shear_velocity)
        #     shear = (shear + shear_velocity).clip(-20,20)
        #     rotated_img = rotate_tensor(img, rotate_angle)
        #     skewed_img = skew_tensor(rotated_img, 0, shear)
        #     rotated_img = skewed_img

        # if (
        #     int(rotated_img.size()[0])
        #     >= min(resolution[0] - mask_r - 10, resolution[1] - mask_r - 10)
        #     or int(rotated_img.size()[1])
        #     >= min(resolution[0] - mask_r - 10, resolution[1] - mask_r - 10)
        #     or x <= int(rotated_img.size()[0] / 2) + mask_r
        #     or x >= resolution[0] - int(rotated_img.size()[0] / 2) - mask_r
        #     or y <= int(rotated_img.size()[1] / 2) + mask_r
        #     or y >= resolution[1] - int(rotated_img.size()[1] / 2) - mask_r
        # ):
        #     rotate_angle = -1*rotate_angle
        #     angle = angle + 2*rotate_angle
        #     shear_velocity = -2*torch.sign(shear)*abs(shear_velocity)
        #     shear = (shear + shear_velocity).clip(-20,20)
        #     rotated_img = rotate_tensor(img, rotate_angle)
        #     skewed_img = skew_tensor(rotated_img, 0, shear)
        #     rotated_img = skewed_img

        if rotate_change:
            angle = angle + angle_velocity.item()
            angle_velocity = (
                angle_velocity + angle_delta.sample((1,)).to(device)
            ).clip(-max_angle_change, max_angle_change)

        if scale_change:
            scale = (scale + delta_scale.sample((1,)).to(device)).clip(
                -max_scale_change, max_scale_change
            )

        if trans_change:
            velocity = (velocity + deltas.sample((2,)).to(device)).clip(
                -max_velocity, max_velocity
            )

        actual_diameter = (diameter + scale[0]).clip(min_size, max_size).item()
        diameter = int(round(actual_diameter))

        x = x.clip(
            int(rotated_img.size()[0] * np.sqrt(2) / 2) + mask_r,
            resolution[0] - int(rotated_img.size()[0] * np.sqrt(2) / 2) - mask_r - 1,
        )
        y = y.clip(
            int(rotated_img.size()[1] * np.sqrt(2) / 2) + mask_r,
            resolution[1] - int(rotated_img.size()[1] * np.sqrt(2) / 2) - mask_r - 1,
        )

        x_min = int(x.round() - rotated_img.size()[0] / 2)
        x_min_cropped = max(0, x_min)
        x_max = int(x_min + rotated_img.size()[0])
        x_max_cropped = min(resolution[0], x_max)
        y_min = int(y.round() - rotated_img.size()[1] / 2)
        y_min_cropped = max(0, y_min)
        y_max = int(y_min + rotated_img.size()[1])
        y_max_cropped = min(resolution[1], y_max)

        images[
            i, x_min_cropped:x_max_cropped, y_min_cropped:y_max_cropped
        ] += rotated_img.bool()[
            0 : x_min_cropped + x_max_cropped, 0 : y_min_cropped + y_max_cropped
        ]
        labels[i] = torch.tensor(
            [(x_min_cropped + x_max_cropped) // 2, (y_min_cropped + y_max_cropped) // 2]
        )

    return images.unsqueeze(1).float(), labels.unsqueeze(1)


if __name__ == "__main__":
    from shapes import *

    for fn in [circle_improved, square_improved, triangle_improved]:
        s, l = render_shape_improved(
            fn,
            len=128,
            resolution=(300, 300),
            shape_p=0.8,
            bg_noise_p=0.002,
            device="cuda",
            scale_change=True,
            trans_change=True,
            rotate_change=True,
            skew_change=True,
        )
