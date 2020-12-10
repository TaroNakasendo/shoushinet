import cv2
import numpy as np
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

# ハイパーパラメータ
batch_size = 4  # 1度に処理する枚数
epoch_num = 50  # 学習の回数


class ImageTransformer():
    """ 画像の前処理を行うクラス
    """

    def __init__(self, mean, std):
        self.data_transform = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),  # pytorchのテンソル型に変換する
            torchvision.transforms.Normalize(mean, std)  # 正規化
        ])

    def __call__(self, img):
        return self.data_transform(img)


mean = (0.5,)  # 平均
std = (0.5,)  # 標準偏差
dataset = torchvision.datasets.ImageFolder(
    "./images_for_train", transform=ImageTransformer(mean, std))

train_size = int(len(dataset) * 0.8)  # 8割を学習用
val_size = int(len(dataset) - train_size)  # 残りを検証用
train_data, val_data = torch.utils.data.random_split(
    dataset, [train_size, val_size])

trainloader = torch.utils.data.DataLoader(
    train_data, batch_size, shuffle=True)
valloader = torch.utils.data.DataLoader(val_data, batch_size)

classes = dataset.classes


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


net = LeNet()
# device = "cpu"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
net.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

start = time.time()

for epoch in range(epoch_num):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data[0].to(device), data[1].to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()

    vcorrect = 0
    vtotal = 0
    with torch.no_grad():
        for vdata in valloader:
            vimages, vlabels = vdata[0].to(device), vdata[1].to(device)
            voutputs = net(vimages)
            _, vpredicted = torch.max(voutputs.data, 1)
            vtotal += vlabels.size(0)
            vcorrect += (vpredicted == vlabels).sum().item()
    print(
        f'[epoch {epoch + 1: >3}/{epoch_num}] loss: {running_loss: >6.3f} acc: {vcorrect / vtotal: >5.2%}')

end = time.time()

print(f"time: {end - start:.2f}")
print('Finished Training')

PATH = './shoushi_net.pth'
torch.save(net.state_dict(), PATH)


# test
test_data = torchvision.datasets.ImageFolder(
    "./images_for_test", transform=ImageTransformer(mean, std))
testloader = torch.utils.data.DataLoader(test_data, batch_size)

net = LeNet()
net.eval()
PATH = './shoushi_net.pth'
net.load_state_dict(torch.load(PATH))

correct = 0
total = 0
with torch.no_grad():
    for data in testloader:
        images, labels = data
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(
    f'Accuracy of the network on the 10000 test images: {correct / total: >5.2%}')

class_correct = list(0. for i in range(len(classes)))
class_total = list(0. for i in range(len(classes)))
with torch.no_grad():
    for data in testloader:
        images, labels = data
        outputs = net(images)
        _, predicted = torch.max(outputs, 1)
        c = (predicted == labels).squeeze()
        for i in range(len(labels)):
            label = labels[i]
            class_correct[label] += c[i].item()
            class_total[label] += 1

for i in range(len(classes)):
    print(
        f'Accuracy of {classes[i]} : {class_correct[i] / (1 if class_total[i] == 0 else class_total[i]): >5.2%}')


# リアルタイム判定
# 0番目のWebカメラからキャプチャーできるようにする。
cap = cv2.VideoCapture(0)
if not cap.read()[0]:
    cap = cv2.VideoCapture(1)  # Mac対応

transformer = ImageTransformer(mean, std)
while True:
    _, frame = cap.read()  # 画像を取得
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
    k = cv2.waitKey(1)  # 1ms 待つ

    # ESCボタン押下またはクローズボタンが押されたに終了
    if k == 27 or not cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE):
        cv2.destroyAllWindows()  # 終了
        break

cap.release()  # 後処理
