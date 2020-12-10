import cv2
import datetime
import numpy as np
import os

view_size = 480  # 表示サイズ
folder = 'images_for_train'
# folder = 'images_for_test'

# 0番目のWebカメラからキャプチャーできるようにする。
cap = cv2.VideoCapture(0)
if not cap.read()[0]:
    cap = cv2.VideoCapture(1)  # Mac対応

tutrial_img = cv2.imread('shoushi.png')
t_h, t_w, _ = tutrial_img.shape

while True:
    _, frame = cap.read()  # 画像を取得
    short_side = min(frame.shape[0], frame.shape[1])  # 短辺
    offset_left = (frame.shape[1] - short_side) // 2  # 左端のオフセット
    frame2 = frame[:, offset_left:offset_left + short_side, :]  # 中央を正方形に切り抜き
    frame = cv2.resize(frame2, (view_size, view_size))
    canvas = np.ones((t_h, view_size + 50 + t_w, 3), np.uint8) * 255  # 真っ白
    canvas[50:50 + view_size, 50:50 + view_size] = frame  # キャプチャ画像
    canvas[:, 50 + view_size:] = tutrial_img  # チュートリアル画像

    # Windowに表示
    window_name = "Capture Shoushi Images"
    cv2.imshow(window_name, canvas)
    k = cv2.waitKey(1)  # 1ms 待つ

    if 48 <= k and k <= 57:  # 0～9が押された
        frame_32x32 = cv2.resize(frame, dsize=(32, 32))  # 32x32にリサイズ

        num = '10' if k == 48 else f'0{k - 48}'  # 押された数字
        folder_name = f"{folder}/{num}"  # フォルダ名

        if not os.path.isdir(folder_name):  # フォルダがない場合は作成
            os.makedirs(folder_name)

        yyyymmddhhmmss = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
        cv2.imwrite(f"{folder_name}/{yyyymmddhhmmss}.jpg",
                    frame_32x32)  # PNGで保存

    # ESCボタン押下または×ボタンが押された
    elif k == 27 or not cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE):
        cv2.destroyAllWindows()  # 終了
        break

cap.release()  # 後処理
