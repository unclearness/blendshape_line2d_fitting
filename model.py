import torch
import torch.nn as nn


def quaternion_to_matrix(quaternions: torch.Tensor) -> torch.Tensor:
    """
    Convert rotations given as quaternions to rotation matrices.

    Args:
        quaternions: quaternions with real part first,
            as tensor of shape (..., 4).

    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).
    """
    r, i, j, k = torch.unbind(quaternions, -1)
    two_s = 2.0 / (quaternions * quaternions).sum(-1)

    o = torch.stack(
        (
            1 - two_s * (j * j + k * k),
            two_s * (i * j - k * r),
            two_s * (i * k + j * r),
            two_s * (i * j + k * r),
            1 - two_s * (i * i + k * k),
            two_s * (j * k - i * r),
            two_s * (i * k - j * r),
            two_s * (j * k + i * r),
            1 - two_s * (i * i + j * j),
        ),
        -1,
    )
    return o.reshape(quaternions.shape[:-1] + (3, 3))


class BlendShape(nn.Module):
    def __init__(self, base, indices, identities,
                 expressions, is_offset=False) -> None:
        super().__init__()
        self.base = base  # (V, 3)
        self.base_unsqueezed = self.base.unsqueeze(0)
        self.indices = indices  # (VI, 3)
        self.identities = identities  # (I, V, 3)
        self.expressions = expressions  # (E, V, 3)
        if not is_offset:
            self.identities = identities - self.base_unsqueezed
            self.expressions = expressions - self.base_unsqueezed
            # print(self.identities[0], identities[0], self.base[0])
            # print(self.identities.shape, identities.shape, self.base.shape)
        self.identity_coeffs = None
        self.expression_coeffs = None
        self.morphed = None

    '''
    def to(self, device):
        # Manually move to members as they are not a subclass of nn.Module
        self.cameras = self.cameras.to(device)
        return self
    '''

    def forward(self, identity_coeffs, expression_coeffs) -> torch.Tensor:
        self.identity_coeffs = identity_coeffs.reshape(
            identity_coeffs.shape[0], 1, 1)  # (I, 1, 1)
        self.expression_coeffs = expression_coeffs.reshape(
            expression_coeffs.shape[0], 1, 1)  # (E, 1, 1)
        self.morphed = (
            self.base
            + torch.sum(self.identity_coeffs * self.identities, dim=0)
            + torch.sum(self.expression_coeffs * self.expressions, dim=0)
        )
        return self.morphed


class OrthoCamera(nn.Module):
    def __init__(self, device) -> None:
        super().__init__()
        self.scale = 1.0
        self.w2c_q = torch.tensor([1.0, .0, .0, .0], device=device)
        self.w2c_R = quaternion_to_matrix(self.w2c_q)
        self.w2c_t = torch.zeros((3,), device=device)
        self.points2d = None

    def forward(self, points) -> torch.Tensor:
        # Ensure unit quartanion
        self.w2c_q = self.w2c_q / torch.linalg.norm(self.w2c_q)
        self.w2c_R = quaternion_to_matrix(self.w2c_q)
        self.points2d = torch.t(self.w2c_R @
                                torch.t(points * self.scale)) + self.w2c_t
        self.points2d = points * self.scale + self.w2c_t
        return self.points2d
