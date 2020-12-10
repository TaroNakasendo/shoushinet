import cv2
import numpy as np
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision


class ImageTransformer():
    """ 画像の前処理を行うクラス
    """

    def __init__(self, mean, std):
        self.data_transform = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),  # pytorchのテンソル型にする
            torchvision.transforms.Normalize(mean, std)  # 正規化
        ])

    def __call__(self, img):
        return self.data_transform(img)


mean = (0.5,)  # 平均
std = (0.5,)  # 標準偏差


class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=5)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(in_features=16 * 5 * 5, out_features=120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, len(classes))

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# test
test_data = torchvision.datasets.ImageFolder(
    "./images_for_test", transform=ImageTransformer(mean, std))
classes = test_data.classes

net = LeNet()
net.eval()  # 評価用に微分を行わないモードにする
PATH = './shoushi_net.pth'
net.load_state_dict(torch.load(PATH))  # 学習済みのモデルを読み込み

# リアルタイム判定
# 0番目のWebカメラからキャプチャーできるようにする。
cap = cv2.VideoCapture(0)
# cap = cv2.VideoCapture("test.mp4")  # キャプチャー済みのビデオを使う場合
if not cap.read()[0]:
    cap = cv2.VideoCapture(1)  # Mac対応

transformer = ImageTransformer(mean, std)
while True:
    _, frame = cap.read()  # 画像を取得
    # time.sleep(0.1)
    if frame is None:
        break
    short_side = min(frame.shape[0], frame.shape[1])  # 短辺
    offset_left = (frame.shape[1] - short_side) // 2  # 左端のオフセット
    frame = frame[:, offset_left:offset_left + short_side, :]  # 中央を正方形に切り抜き

    frame_32x32 = cv2.resize(frame, dsize=(32, 32))  # 32x32にリサイズ
    frame_32x32 = transformer(frame_32x32)
    frame_32x32 = frame_32x32[np.newaxis, :, :, :]

    with torch.no_grad():
        outputs = net(frame_32x32)
        max_val, predicted = torch.max(outputs, 1)

    label = '--' if max_val < 5 else classes[predicted[0]]
    max_v = f"{max_val.item(): >5.2f}"
    cv2.putText(frame, label, (20, 70),
                cv2.FONT_HERSHEY_PLAIN, 5, (0, 0, 255), 5, cv2.LINE_AA)
    cv2.putText(frame, max_v, (15, 120),
                cv2.FONT_HERSHEY_PLAIN, 2.5, (0, 0, 255), 2, cv2.LINE_AA)

    window_name = "Shoushi Predictor"
    cv2.imshow(window_name, frame)  # 画像をWindowに表示
    k = cv2.waitKey(100)  # 100ms 待つ

    # ESCボタン押下またはクローズボタンが押されたに終了
    if k == 27 or not cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE):
        cv2.destroyAllWindows()  # 終了
        break

cap.release()  # 後処理
