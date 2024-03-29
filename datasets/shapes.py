from typing import Callable
import torch


def gaussian_mask(r, min, max, dist, device):
    width = 2 * r + 1
    g = (r - torch.arange(0, width, 1, device=device)) ** 2
    grid = g + g.unsqueeze(0).T
    img = torch.zeros(width, width, device=device)
    img = torch.where(
        (grid < max) & (grid > min), dist.sample((width, width)).to(device), img
    )
    return img.bool()


def circle_improved(size, p, device):
    width = 1.75 * size
    r = size // 2
    g = (r - torch.arange(0, size, 1, device=device)) ** 2
    grid = g + g.unsqueeze(0).T
    img = torch.zeros(size, size, device=device)
    dist = torch.distributions.Bernoulli(probs=p).sample((size, size)).to(device)

    scale = 0.75 + size / 2000
    r_2 = scale * r
    r_1 = r
    outer_ring_2 = (r_1) ** 2 - width
    inner_ring_2 = (r_1) ** 2 - width * 3

    outer_ring_1 = (r_2) ** 2 - width
    inner_ring_1 = (r_2) ** 2 - width * 3

    img_2 = torch.where(
        (grid < outer_ring_2 + width) & (grid > outer_ring_2 - width), dist, img
    )

    img_1 = torch.where(
        (grid < outer_ring_1 + width) & (grid > outer_ring_1 - width), dist, img
    )

    img = img_1 + img_2

    return img


def triangle_improved(r, p, device, thickness=3):
    r = int(round(r / 2) * 2)

    if r < 100:
        thickness = 2

    mid_1 = r // 2
    outer_1 = torch.distributions.Bernoulli(probs=p).sample((r, r)).to(device).bool()
    outer_full_1 = (
        torch.distributions.Bernoulli(probs=1).sample((r, r)).to(device).bool()
    )

    # Outer_1
    outer_left_1 = outer_1[:mid_1, :mid_1].tril(0).flip(1).repeat_interleave(2, 1)

    outer_full_1[:mid_1] &= outer_left_1
    outer_full_1[mid_1:] &= outer_left_1.flip(0)

    # Inner
    inner_left = (
        torch.ones(mid_1 - thickness, mid_1 - thickness, device=device)
        .tril(2)
        .bool()
        .flip(0)
        .repeat_interleave(2, 1)
    )
    outer_full_1[thickness:mid_1, thickness : r - thickness] &= inner_left
    outer_full_1[mid_1:-thickness, thickness : r - thickness] &= inner_left.flip(0)

    scale = 0.75 + r / 2000
    r_2 = int(round(scale * r / 2) * 2)

    mid_2 = r_2 // 2
    outer_2 = (
        torch.distributions.Bernoulli(probs=p).sample((r_2, r_2)).to(device).bool()
    )
    outer_full_2 = (
        torch.distributions.Bernoulli(probs=1).sample((r_2, r_2)).to(device).bool()
    )

    # Outer_2
    outer_left_2 = outer_2[:mid_2, :mid_2].tril(0).flip(1).repeat_interleave(2, 1)

    outer_full_2[:mid_2] &= outer_left_2
    outer_full_2[mid_2:] &= outer_left_2.flip(0)

    # Inner
    inner_left = (
        torch.ones(mid_2 - thickness, mid_2 - thickness, device=device)
        .tril(2)
        .bool()
        .flip(0)
        .repeat_interleave(2, 1)
    )
    outer_full_2[thickness:mid_2, thickness : r_2 - thickness] &= inner_left
    outer_full_2[mid_2:-thickness, thickness : r_2 - thickness] &= inner_left.flip(0)

    r_1 = r
    size_padded = int(round(((r - r_2) / 2)))

    if (r_1 - r_2) % 2 == 0:
        size_padded_1 = int((r_1 - r_2) / 2)
        size_padded_2 = int((r_1 - r_2) / 2)
    else:
        size_padded_1 = int((r_1 - r_2) / 2)
        size_padded_2 = int((r_1 - r_2 + 1) / 2)

    outer_2_padded = torch.nn.functional.pad(
        outer_full_2,
        (
            int(round(1.5 * size_padded)),
            int(round(0.5 * size_padded)),
            size_padded_1,
            size_padded_2,
        ),
        "constant",
        0,
    )

    outer = outer_full_1 + outer_2_padded

    return outer


def square_improved(r, p, device, width=3):
    r_1 = r
    size_1 = r
    outer_1 = torch.distributions.Bernoulli(probs=p).sample((size_1, size_1)).to(device)
    inner_1 = torch.zeros(size_1 - width * 2, size_1 - width * 2, device=device)
    outer_1[width : r_1 - width, width : r_1 - width] = inner_1

    scale = 0.75 + r / 2000
    r_2 = int(round(scale * r / 2) * 2)

    size_2 = r_2
    outer_2 = torch.distributions.Bernoulli(probs=p).sample((size_2, size_2)).to(device)
    inner_2 = torch.zeros(size_2 - width * 2, size_2 - width * 2, device=device)
    outer_2[width : r_2 - width, width : r_2 - width] = inner_2

    if (r_1 - r_2) % 2 == 0:
        size_padded_1 = int((r_1 - r_2) / 2)
        size_padded_2 = int((r_1 - r_2) / 2)
    else:
        size_padded_1 = int((r_1 - r_2) / 2)
        size_padded_2 = int((r_1 - r_2 + 1) / 2)
    outer_2_padded = torch.nn.functional.pad(
        outer_2,
        (size_padded_1, size_padded_2, size_padded_1, size_padded_2),
        "constant",
        0,
    )

    return outer_2_padded + outer_1
