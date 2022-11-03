import torch.nn.functional as F


def l2loss(reconstructed_x, original_x):
    """
    * 해당 배치의 L2 loss 구하기
    :param reconstructed_x: shape: (batch, SHAPE)
    :param original_x: shape: (batch, SHAPE)
    :return: 해당하는 배치의 L2 loss
    """

    # L2 loss 계산
    batch_l2loss = F.mse_loss(input=reconstructed_x,
                              target=original_x).cpu().detach().numpy()

    return batch_l2loss
