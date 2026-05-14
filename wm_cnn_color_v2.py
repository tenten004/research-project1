import sys

print(f"python  {sys.argv[0]}  {sys.argv[1]}  {sys.argv[2]}  {sys.argv[3]}")

exp_num = int(sys.argv[1])
start = int(sys.argv[2])
end = int(sys.argv[3])

##########################################################
## 以下、wm_cnn_color_v2.ipynbから引用(一部出力の変更あり) ##
##########################################################

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
from keras.utils  import np_utils
import numpy as np
import matplotlib.pyplot as plt
import time
from PIL import Image
import os, glob
from sklearn import model_selection
import pandas as pd
import csv

df_FL = pd.read_csv('../3_教師データ/labeled_image_list_FL_preprocess.csv', header=0)
df_T1 = pd.read_csv('../3_教師データ/labeled_image_list_T1_preprocess.csv', header=0)
df_T2 = pd.read_csv('../3_教師データ/labeled_image_list_T2_preprocess.csv', header=0)

df_FL = df_FL.drop(columns = 'Unnamed: 0')
df_T1 = df_T1.drop(columns = 'Unnamed: 0')
df_T2 = df_T2.drop(columns = 'Unnamed: 0')

#直近の実験結果を表示
df_result = pd.read_csv("./data.csv")
df_result.tail(30)

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

#コマンドライン引数によって与えられる #exp_num=120 #実験番号
#コマンドライン引数によって与えられる #start = 3
#コマンドライン引数によって与えられる #end = 7
method = "T2" #撮影方法
axial = f"{start}_{end}" #識別に使用したaxial
batch_size = 32
epochs = 200
image_size =80 #画像の縦と横のピクセル数

classes = ['level_0', 'level_1', 'level_2', 'level_3', 'level_4'] # 識別するクラス
num_classes = len(classes) #クラス数
np.random.seed(1)

### データの作成

#学習に用いるデータの抽出
df_axial = df_T2[(df_T2['axial']>=start) & (df_T2['axial']<=end)]
#df_axial = df_FL[df_FL['axial']==start]
#df_axial = df_FL
num_of_images = df_axial.shape[0] # 全画像を使用
df_axial

#df_axial.info()
#print(df_axial.shape)
#print(df_axial.shape[0])

### 画像パスを使い、配列Xに画素のNumPy配列を格納する

cnt=0
X = []
for i in df_axial['name']:
    file = str('../../labeled_image/T2/' + i)
    #print(i)
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
y_train = np_utils.to_categorical(y_train, num_classes)
y_test  = np_utils.to_categorical(y_test , num_classes)

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

### 学習方法を定義

#from tensorflow.keras import optimizers

#opt = keras.optimizers.RMSprop(lr=0.0001, decay=1e-6) #cpu
opt = tf.keras.optimizers.RMSprop(lr=0.0001, decay=1e-6) #gpu

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
model.save(f'./model/wm_cnn_{exp_num:}.h5')

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
plt.savefig(f'./img/accuracy/wm_accuracy_{exp_num:}.png')
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
plt.savefig(f'./img/loss/wm_loss_{exp_num:}.png')
#plt.show()

### csvファイルに実行結果を保存

#df_result.to_csv("./data.csv", index=False)

result = [exp_num, method, axial, batch_size, epochs, image_size, num_of_images, elapsed_time, train_accuracy, test_accuracy, train_loss, test_loss]
print(result)

with open('./data.csv', 'a', newline='') as f:
    writer = csv.writer(f, lineterminator='\n')
    writer.writerow(result)
    
#保存された結果を表示
df_result = pd.read_csv("./data.csv")
df_result.tail(1)

########################################
## 以上、wm_cnn_color_v2.ipynbから引用 ##
########################################

# Kernelの再起動
from IPython.display import display_html
def restartkernel() :
    display_html("<script>Jupyter.notebook.kernel.restart()</script>",raw=True)
restartkernel()    #Kernelの再起動