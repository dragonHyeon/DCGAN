import torch.nn as nn


class AutoEncoder(nn.Module):
    def __init__(self):
        """
        * 모델 구조 정의
        """

        super(AutoEncoder, self).__init__()

        # 인코더
        self.encoder = nn.Sequential(
            # (N, 1, 32, 32) -> (N, 1024)
            nn.Flatten(),
            # (N, 1024) -> (N, 512)
            nn.Linear(in_features=1024, out_features=512),
            nn.ReLU(),
            # (N, 512) -> (N, 512)
            nn.Linear(in_features=512, out_features=512),
            nn.ReLU(),
            # (N, 512) -> (N, 256)
            nn.Linear(in_features=512, out_features=256),
            nn.ReLU(),
            # (N, 256) -> (N, 128)
            nn.Linear(in_features=256, out_features=128),
            nn.ReLU(),
            # (N, 128) -> (N, 64)
            nn.Linear(in_features=128, out_features=64),
            nn.ReLU(),
            # (N, 64) -> (N, 3)
            nn.Linear(in_features=64, out_features=3)
        )

        # 디코더
        self.decoder = nn.Sequential(
            # (N, 3) -> (N, 64)
            nn.Linear(in_features=3, out_features=64),
            nn.ReLU(),
            # (N, 64) -> (N, 128)
            nn.Linear(in_features=64, out_features=128),
            nn.ReLU(),
            # (N, 128) -> (N, 256)
            nn.Linear(in_features=128, out_features=256),
            nn.ReLU(),
            # (N, 256) -> (N, 512)
            nn.Linear(in_features=256, out_features=512),
            nn.ReLU(),
            # (N, 512) -> (N, 512)
            nn.Linear(in_features=512, out_features=512),
            nn.ReLU(),
            # (N, 512) -> (N, 1024)
            nn.Linear(in_features=512, out_features=1024)
        )

    def forward(self, x):
        """
        * 순전파
        :param x: 배치 개수 만큼의 입력. (N, 1, 32, 32)
        :return: 배치 개수 만큼의 인코딩 된 값, 디코딩 된 값, 디코딩 된 값의 이미지 형태
        """

        # 인코딩 된 값. (N, 1, 32, 32) -> (N, 3)
        encoded = self.encoder(x)
        # 디코딩 된 값. (N, 3) -> (N, 1024)
        decoded = self.decoder(encoded)
        # 디코딩 된 값의 이미지 형태. (N, 1024) -> (N, 1, 32, 32)
        reconstructed_x = decoded.view(x.shape)

        return encoded, decoded, reconstructed_x
