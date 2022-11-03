import torch.nn as nn

loss_fn = nn.MSELoss()


"""
nn.MSELoss()
def forward(self, input: Tensor, target: Tensor) -> Tensor:

사용 설명
batch size = 2 인 경우
input(예측한 값): model 의 output 값. ex) [[-0.3588, -0.0903,  0.0114], [-0.3502, -0.0834,  0.0395]]
target(실제 값): target 값. ex) [[-0.3588, -0.0903,  0.0114], [-0.3502, -0.0834,  0.0395]]

loss_fn = nn.MSELoss()
loss_fn(input, target)
"""
