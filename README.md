# research-project1
卒研のためのgithub
① WeightedRandomSampler 導入（最優先）
目的

今の最大問題：

予測が 0/1 に崩壊している

これを止める。

なぜ必要か

現在の分布：

0: 1200前後
1: 1000前後
2: 200前後
3: 100前後
4: 20前後

極端。

普通にshuffleして学習すると、

1 batch中に class4 が存在しない

ことが多発する。

するとモデルは：

4を覚える意味がない

と判断する。

WeightedRandomSampler の役割

少数クラスを「何回も出す」。

つまり：

class4 を意図的に何度も学習

させる。

実装
1. ラベル一覧取得
labels = train_dataset.labels
2. クラス数カウント
import numpy as np

class_count = np.bincount(labels)
print(class_count)
3. 重み作成
weights = 1. / class_count

例：

0 -> 0.0008
4 -> 0.05

みたいになる。

つまり：

少数クラスほど大きい重み
4. 各サンプルに重み付与
sample_weights = [weights[t] for t in labels]
5. sampler作成
from torch.utils.data import WeightedRandomSampler

sampler = WeightedRandomSampler(
    sample_weights,
    num_samples=len(sample_weights),
    replacement=True
)
6. DataLoader変更
train_loader = DataLoader(
    train_dataset,
    batch_size=16,
    sampler=sampler,
    num_workers=4
)

※ shuffle=True は消す。

期待される変化
Before
pred:
0 0 1 0 1 1 0 0
After
pred:
0 1 2 1 3 0 4

みたいに、

「少数クラスを出す勇気」

が生まれる。

最重要ポイント

accuracy は下がる可能性ある。

でも：

macro-F1
recall

は改善する可能性が高い。

ここ超重要。

② class weight 付き CrossEntropy
目的

モデルへ：

「class4外したら重罪」

を教える。

なぜ必要？

今のCrossEntropyは：

class0 を外す
class4 を外す

をほぼ同じ重みで扱う。

でも実際は：

class4 は超貴重

だから重みを付ける。

実装
まずクラス数

例：

class_count = [1162,1111,232,137,24]
inverse frequency
weights = [1/x for x in class_count]

ただしそのままだと極端なので、

実用的にはこれくらい
weights = torch.tensor([
    1.0,
    1.0,
    3.0,
    5.0,
    10.0
]).to(device)
損失関数
criterion = nn.CrossEntropyLoss(weight=weights)

これだけ。

期待される変化

今：

4 を無視しても痛くない

↓

導入後：

4 を無視すると損失爆増
sampler と併用する理由
sampler

→ 「見る回数」を増やす

class weight

→ 「外した罰」を増やす

両方必要。

③ モデル保存基準を macro-F1 に変更
これ超重要

今たぶん：

best_acc

で保存してる。

これが危険。

なぜ？

accuracy最大化すると：

0/1だけ出すモデル

が有利。

今の状態

例えば：

0,1 が90%

なら、

全部0/1

でも accuracy 高くなる。

だから macro-F1

macro-F1 は：

各クラスを平等扱い

する。

なので：

class4 無視

すると点数が落ちる。

実装
sklearn
from sklearn.metrics import f1_score
validation後
macro_f1 = f1_score(
    y_true,
    y_pred,
    average="macro"
)
best更新
if macro_f1 > best_f1:
    best_f1 = macro_f1
    torch.save(model.state_dict(), save_path)
期待される変化

Before:

accuracy重視
↓
多数派特化

After:

全クラスを拾う方向
④ confusion matrix を毎epoch確認
目的

「本当に改善したか」を見る。

超重要

accuracyだけ見ると騙される。

今まさにそれ。

見るべきポイント
BEFORE
pred:
0/1 only
AFTER

最低でも：

2/3/4 が出始める

これが第一歩。

注意

最初は：

誤検出増加

する可能性高い。

でも正常。

なぜなら：

「少数クラスを出す練習」

が始まったから。

まず最初の目標
目標①
class2–4 が予測される
目標②
macro-F1改善
目標③
class別 recall改善

accuracyは後回し。
