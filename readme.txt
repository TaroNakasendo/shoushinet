■ pythorch
conda install pytorch torchvision torchaudio cpuonly -c pytorch

■ opencv
conda install -c conda-forge opencv

をインストール後、

python3 train_and_test.py
で動かしてください。モデルが作成され、その後、リアルタイムで識別ができるようになります。

python3 realtime_predictor.py
では、訓練済みモデルを使って、リアルタイム識別のみを行います。