# research-project1
卒研のためのgithub

# 大脳白質病変CNN先行研究 再現実験 指示文

## 目的

先行研究「畳み込みニューラルネットワークを用いた大脳白質病変のグレード予測」における実験条件を可能な限り再現し，報告された性能（test accuracy ≒ 0.9209）に近い結果を得ることを目的とする．

---

# 再現対象

## 最良条件

* モデル：CNN
* 撮像法：FLAIR + T1
* axial：9-15
* range：7
* 画像サイズ：80×80
* クラス数：5（grade0-4）
* optimizer：RMSprop
* learning rate：0.0001
* decay：1e-6
* batch size：32
* epochs：200
* 評価指標：test accuracy

---

# データセット条件

## 使用データ

以下のcsvを使用する．

* MRI_list_FL_1146.csv
* MRI_list_T1_1146.csv
* MRI_list_T2_1146.csv

各csvには少なくとも以下の列が存在すること．

* name
* wm
* axial

---

# 使用画像条件

## 使用モダリティ

FLAIR画像 + T1強調画像

コード上では：

```python
method = 6
```

---

## 使用axial

以下の範囲のみ使用する．

```python
start = 9
end = 15
```

つまり：

* axial 9
* axial 10
* axial 11
* axial 12
* axial 13
* axial 14
* axial 15

の計7枚を使用する．

---

# 前処理条件

## 画像サイズ

すべての画像を以下へresizeする．

```python
image_size = 80
```

---

## カラーモード

RGB変換を使用する．

```python
image.convert("RGB")
```

---

## 正規化

0-255を0-1へ正規化する．

```python
x_train = x_train / 255.0
x_test = x_test / 255.0
```

---

# データ分割

## train/test split

以下を使用する．

```python
train_test_split(X, Y)
```

※ 注意：
このコードは患者単位分割ではなく，画像単位分割である可能性が高い．
先行研究再現フェーズでは，まずこの条件を忠実に再現する．

---

# CNN構造

```python
Conv2D(32, (3,3), padding='same', activation='relu')
Conv2D(32, (3,3), activation='relu')
MaxPooling2D((2,2))
Dropout(0.25)

Conv2D(32, (3,3), padding='same', activation='relu')
Conv2D(32, (3,3), padding='same', activation='relu')
MaxPooling2D((2,2))

Flatten()

Dropout(0.25)

Dense(512, activation='relu')

Dropout(0.5)

Dense(5, activation='softmax')
```

---

# 学習条件

## optimizer

```python
RMSprop(
    learning_rate=0.0001,
    decay=1e-6
)
```

---

## loss function

```python
categorical_crossentropy
```

---

## metrics

```python
accuracy
```

---

## batch size

```python
32
```

---

## epochs

```python
200
```

---

# 保存項目

以下を毎実験保存すること．

* train accuracy
* test accuracy
* train loss
* test loss
* 学習曲線
* loss曲線
* 実験条件

---

# 実験管理

## 実験番号

各実験にexp_numを割り当てる．

例：

```python
python train.py 1 6 9 15
```

意味：

* exp_num = 1
* method = 6（FLAIR+T1）
* start = 9
* end = 15

---

# 再現実験の到達目標

## 目標性能

```text
test accuracy ≒ 0.9209
```

完全一致ではなく，近い性能が得られるかを確認する．

---

# 再現後の次段階

再現完了後に以下を検討する．

1. 患者単位split導入
2. class imbalance対策
3. ViT導入
4. 全axial利用
5. augmentation導入

ただし，再現実験中は条件を変更しないこと．
