import sys
from pathlib import Path

print(f"python  {sys.argv[0]}  {sys.argv[1]}  {sys.argv[2]}  {sys.argv[3]}  {sys.argv[4]}")

exp_num = int(sys.argv[1])
method = int(sys.argv[2])
start = int(sys.argv[3])
end = int(sys.argv[4])

script_dir = Path(__file__).resolve().parent
repo_root = script_dir.parent
csv_dir = repo_root / "教師データ"
image_root = repo_root / "labeled_image"
output_root = repo_root / "mri-vit-classification" / "outputs" / "repro_cnn"
exp_dir = output_root / f"exp_{exp_num}"
data_csv_path = output_root / "data_v5.csv"
model_dir = exp_dir / "model"
img_accuracy_dir = exp_dir / "img" / "accuracy"
img_loss_dir = exp_dir / "img" / "loss"

##############################################################################
## 以下、wm_cnn_color_v5.ipynbから引用(コマンドライン引数のパラメータの変更あり) ##
##############################################################################

# %reset

# restsrt kernel
# Kernelの再起動
#from IPython.display import display_html
#def restartkernel() :
#    display_html("<script>Jupyter.notebook.kernel.restart()</script>",raw=True)
#restartkernel()    #Kernelの再起動

### モジュールのインポート

import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from tensorflow.keras.utils import to_categorical
import numpy as np
import matplotlib.pyplot as plt
import time
from PIL import Image
import os, glob
from sklearn import model_selection
import pandas as pd
import csv

df_FL = pd.read_csv(csv_dir / "MRI_list_FL_1146.csv", header=0)
df_T1 = pd.read_csv(csv_dir / "MRI_list_T1_1146.csv", header=0)
df_T2 = pd.read_csv(csv_dir / "MRI_list_T2_1146.csv", header=0)

df_FL = df_FL.drop(columns="Unnamed: 0", errors="ignore")
df_T1 = df_T1.drop(columns="Unnamed: 0", errors="ignore")
df_T2 = df_T2.drop(columns="Unnamed: 0", errors="ignore")

#直近の実験結果を表示
output_root.mkdir(parents=True, exist_ok=True)
if data_csv_path.exists():
    df_result = pd.read_csv(data_csv_path)
else:
    df_result = pd.DataFrame(
        columns=[
            "exp_num",
            "method",
            "start",
            "end",
            "axial",
            "axial_range",
            "batch_size",
            "epochs",
            "image_size",
            "lr",
            "decay",
            "num_of_images",
            "elapsed_time",
            "train_accuracy",
            "test_accuracy",
            "train_loss",
            "test_loss",
        ]
    )
    df_result.to_csv(data_csv_path, index=False)
df_result.tail(5)

if "exp_num" in df_result.columns and (df_result["exp_num"] == exp_num).any():
    print(f"exp_num={exp_num} already exists in {data_csv_path}")
    sys.exit(1)

if exp_dir.exists():
    print(f"Output directory already exists: {exp_dir}")
    sys.exit(1)

model_dir.mkdir(parents=True, exist_ok=True)
img_accuracy_dir.mkdir(parents=True, exist_ok=True)
img_loss_dir.mkdir(parents=True, exist_ok=True)

model_path = model_dir / f"wm_cnn_{exp_num}.h5"
acc_path = img_accuracy_dir / f"wm_accuracy_{exp_num}.png"
loss_path = img_loss_dir / f"wm_loss_{exp_num}.png"
existing_outputs = [path for path in [model_path, acc_path, loss_path] if path.exists()]
if existing_outputs:
    print(f"Output already exists for exp_num={exp_num}: {existing_outputs}")
    sys.exit(1)

### 画像表示テスト

#num = 500
#print(df_FL['name'][num])

# 特定のインデックスとカラムのデータを抽出する
#print(df_FL['name'][0])

#file = '../../labeled_image/FL/' + df_FL['name'][num]
#image = Image.open(file)
#data = np.array(image)
#plt.figure()
#plt.imshow(data)
#xlabel = df_FL['name'][num] + '  wm :' + str(df_FL['wm'][num])
#plt.xlabel(xlabel)
#plt.show()

### GPU動作確認

# TensorFlowがGPUを認識しているか確認
from tensorflow.python.client import device_lib
device_lib.list_local_devices()

### クラスの設定とハイパーパラメータ

#引数で受け取る
#コマンドライン引数によって与えられる #exp_num = 0 #実験番号
#コマンドライン引数によって与えられる #method = 5 #撮影方法
#method == 1 : T1
#method == 2 : T2
#method == 3 : FL
#method == 4 : T1+T2
#method == 5 : T2+FL
#method == 6 : FL+T1
#method == 7 : T1+T2+FL

#コマンドライン引数によって与えられる #start = 9
#コマンドライン引数によって与えられる #end = 15
axial = f"{start}_{end}" #識別に使用したaxial
axial_range = end - start + 1
batch_size = 32
epochs = 200
image_size =80 #画像の縦と横のピクセル数

lr = 0.0001
decay = 1e-6

classes = ['level_0', 'level_1', 'level_2', 'level_3', 'level_4'] # 識別するクラス
num_classes = len(classes) #クラス数
np.random.seed(1)

### データの作成

if method == 1:
    df_axial_all = df_T1
elif method == 2:
    df_axial_all = df_T2
elif method == 3:
    df_axial_all = df_FL
elif method == 4:
    df_axial_all = pd.concat([df_T1, df_T2], axis=0)
elif method == 5:
    df_axial_all = pd.concat([df_T2, df_FL], axis=0)
elif method == 6:
    df_axial_all = pd.concat([df_FL, df_T1], axis=0)
elif method == 7:
    df_axial_all = pd.concat([df_T1, df_T2, df_FL], axis=0)
else:
    print('error')

df_axial_all

#学習に用いるaxialの抽出
df_axial = df_axial_all[(df_axial_all['axial']>=start) & (df_axial_all['axial']<=end)]
#df_axial = df_FL[df_FL['axial']==start]
#df_axial = df_FL
num_of_images = df_axial.shape[0] # 全画像を使用
df_axial

#df_axial.info()
#print(df_axial.shape)
#print(df_axial.shape[0])

### 画像パスを使い、配列Xに画素のNumPy配列を格納する

import re

cnt=0
X = []
for i in df_axial['name']:   
    file = image_root / re.findall(r'^\w\w', i)[0] / i
    #print(file)
    image = Image.open(file)
    image = image.convert("RGB")
    image = image.resize((image_size, image_size,))
    data = np.array(image)
    #print(data.shape)
    X.append(data)
    cnt+=1
    if(cnt>=num_of_images):
        break;

#print(type(X))
#print(len(X))
#print(X)

cnt=0
Y = []
for i in df_axial['wm']:
    Y.append(i)
    cnt+=1
    if(cnt>=num_of_images):
        break;
        
#print(len(Y))
#print(Y[:10])

X = np.array(X) # 入力データの配列をnp.arrayに変換
Y = np.array(Y) # クラス名の配列をnp.arrayに変換
#print(X.shape)
#print(Y.shape)

# 学習用データとテスト用データに分割
x_train, x_test, y_train, y_test = model_selection.train_test_split(X, Y)

### 前処理

# 標準化
#x_train = x_train.astype("float")/255.
#x_test  = x_test.astype("float")/255.
x_train = x_train / 255.0
x_test  = x_test  / 255.0

# one_hot形式のnp.arrayに変換
y_train = to_categorical(y_train, num_classes)
y_test  = to_categorical(y_test, num_classes)

### モデルを組む

# 最後のDenseの出力はnum_classes、活性化関数はsoftmax
model = Sequential()
#model.add(Conv2D(filters=32, kernel_size=(3,3), padding='same', input_shape=(image_size, image_size,1), activation='relu'))
model.add(Conv2D(filters=32, kernel_size=(3,3), padding='same', input_shape=(image_size, image_size,3), activation='relu'))
model.add(Conv2D(filters=32, kernel_size=(3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))
model.add(Conv2D(filters=32, kernel_size=(3,3), padding='same', activation='relu'))
model.add(Conv2D(filters=32, kernel_size=(3,3), padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Flatten())
model.add(Dropout(0.25))
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))

#model.summary()

#from keras.utils.vis_utils import plot_model
#%matplotlib inline

#plot_model(model, "model.png", show_layer_names=True)
#plot_model(model, "model.png", show_layer_names=True)
#image=plt.imread("model.png")
#plt.figure(dpi = 150)
#plt.imshow(image)
#plt.show()

### 学習方法を定義

#from tensorflow.keras import optimizers

#opt = keras.optimizers.RMSprop(lr=0.0001, decay=1e-6) #cpu
lr_schedule = tf.keras.optimizers.schedules.InverseTimeDecay(
    initial_learning_rate=lr,
    decay_steps=1,
    decay_rate=decay,
)
opt = tf.keras.optimizers.RMSprop(learning_rate=lr_schedule) #gpu

model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

# 時間計測開始
startTime = time.time()

### 学習の実施

history = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, verbose=0, validation_data=(x_test, y_test))

### モデルの評価

# 学習時間の計測と表示
elapsed_time = time.time() - startTime
#print("elapsed_time = {:.1f} sec.".format(elapsed_time))

# モデルの保存
model.save(str(model_dir / f"wm_cnn_{exp_num}.h5"))

train_score = model.evaluate(x_train, y_train, verbose=0)
test_score = model.evaluate(x_test, y_test, verbose=0)

train_accuracy = train_score[1]
test_accuracy = test_score[1]
train_loss = train_score[0]
test_loss = test_score[0]

#print('Train accuracy = {:.4f}'.format(train_accuracy))
#print('Test accuracy = {:.4f}'.format(test_accuracy))
#print('Train loss = {:.4f}'.format(train_loss))
#print('Test loss = {:.4f}'.format(test_loss))

### 学習中の精度と誤差の変化をグラフ化

#print(history.history)

plt.figure()

#plt.plot(history.history['acc']) #cpu
#plt.plot(history.history['val_acc']) #cpu

plt.plot(history.history['accuracy']) #gpu
plt.plot(history.history['val_accuracy']) #gpu

plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.ylim(0.4, 1.0)
plt.legend(['train', 'test'], loc='upper left')
plt.grid()
plt.savefig(str(img_accuracy_dir / f"wm_accuracy_{exp_num}.png"))
#plt.show()

plt.figure()

#plt.plot(history.history['acc']) #cpu
#plt.plot(history.history['val_acc']) #cpu

plt.plot(history.history['loss']) #gpu
plt.plot(history.history['val_loss']) #gpu

plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.grid()
plt.savefig(str(img_loss_dir / f"wm_loss_{exp_num}.png"))
#plt.show()

### csvファイルに実行結果を保存

#df_result.to_csv("./data_v5.csv", index=False)

result = [exp_num, method, start, end, axial, axial_range, batch_size, epochs, image_size, lr, decay, num_of_images, elapsed_time, train_accuracy, test_accuracy, train_loss, test_loss]
print(result)

with open(data_csv_path, 'a', newline='') as f:
    writer = csv.writer(f, lineterminator='\n')
    writer.writerow(result)
    
#保存された結果を表示
df_result = pd.read_csv(data_csv_path)
df_result.tail(1)



########################################
## 以上、wm_cnn_color_v5.ipynbから引用 ##
########################################

# Kernelの再起動
try:
    from IPython.display import display_html
except ImportError:
    display_html = None

if display_html is not None:
    def restartkernel() :
        display_html("<script>Jupyter.notebook.kernel.restart()</script>",raw=True)
    restartkernel()    #Kernelの再起動