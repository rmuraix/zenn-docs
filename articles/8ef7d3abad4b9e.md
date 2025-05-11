---
title: "Pruningについてまとめてみた"
emoji: "✂️"
type: "tech" # tech: 技術記事 / idea: アイデア
topics: ["ディープラーニング"]
published: false
---

## Pruningとは

Pruning（枝刈り）は、ニューラルネットワークにおいて重要度の低いパラメータ（重みやノード）を削除することで、**モデルをスリム化**する手法です。削除されるのは、学習済みのネットワークにおいて出力への影響が小さいと判断された構成要素であり、その判断には様々な基準や手法が存在します。
本記事では、Pruningの種類や設計の要素を整理します。

![](https://storage.googleapis.com/zenn-user-upload/73d451884012-20250511.png)
_[Learning both Weights and Connections for Efficient Neural Networks \[Han et al., 2015\]](https://arxiv.org/abs/1506.02626)_

## なぜPruningが必要か

近年のディープニューラルネットワークは、性能向上を追求するあまり、パラメータ数・モデルサイズともに急速に増大しています。例えば、[Llama 3.1](https://ai.meta.com/blog/meta-llama-3-1/) 405Bモデルでは、FP8形式であっても[推論時に約405GBのメモリを要する](https://huggingface.co/blog/llama31#inference-memory-requirements)とされています。一方、ハイエンドGPUである[NVIDIA A100](https://www.nvidia.com/ja-jp/data-center/a100/)で80GB、[H200](https://www.nvidia.com/ja-jp/data-center/h200/)でも141GBの容量しかなく、単一GPUでの推論は困難になりつつあります。
加えて、モバイル端末や組み込みデバイスのような計算資源が限られた環境では、大規模モデルをそのままデプロイすることは現実的とは言えません。
このような背景のもと、**モデルのサイズや計算負荷を削減しながら、可能な限り性能を維持する**ための手段として、Pruningが注目されています。

## Pruningの手順

Pruningは、一般的に以下の手順に従って行われます。

1. **モデルの事前学習**：まずは通常の方法でモデルを学習する。この段階では一切パラメータを削減せずに、モデル性能を最大化させることを目指す。
1. **Pruning対象の決定/実行**：重要度の低いパラメータを選別し値を0に置き換える。選別する粒度・基準は後述。
1. **Fine-tuningの実施**：Pruningによって性能低下を引き起こす場合があるので、Pruning後のネットワークを再学習する。一般的に、最初から学習するよりはFine-tuningを行ったほうが性能は良くなる。

## Pruningの粒度

ここからは、Pruning対象をどのように決定するかを見ていきます。
Pruningの**粒度**(granularity)とは、どの単位でパラメータを削減するかを示す概念です。粒度が細かいほど個々のパラメータ単位での削除、粗いほど構造単位での削除となります。以下に代表的な粒度を示します。

### Fine-grained

これは**非構造化Pruning**とも呼ばれています。この粒度では**個々の重み単位**でPruningを実行します。非常に柔軟性が高いためより多くのパラメータの削除が可能な一方、パターン化された削除方法ではないためハードウェアアクセラレーションを行うことは難しいとされています。

![](https://storage.googleapis.com/zenn-user-upload/10a220788a44-20250511.png)
_[MIT 6.5940](https://hanlab.mit.edu/courses/2024-fall-65940)が作成_

### 構造化Pruning

構造化されていないFine-grained Pruningは柔軟にパラメータを削減できる一方で、その不規則な削減パターンが原因で、ハードウェアアクセラレーションとの相性が悪いという課題がありました。限られた計算資源で効率的にモデルを構築するには、ハードウェアに最適化された手法を選ぶことが望ましいと言えます。
これに対して**構造化Pruning**では、パターン単位、ベクトル単位、カーネル単位、チャンネル単位など、あらかじめ決められた構造に従ってパラメータを削除することで、ハードウェア側での最適化を可能にします。本記事ではこの中から、特に「パターンに基づく方法」と「チャンネル単位の方法」の2つを紹介します。なお、ベクトル単位やカーネル単位の方法はこれらに類似したバリエーションのため、今回は説明を省略します。

![](https://storage.googleapis.com/zenn-user-upload/3a3b8c918fc9-20250511.png)
_[Mao et al., 2017](https://arxiv.org/abs/1705.08922)をもとに[MIT 6.5940](https://hanlab.mit.edu/courses/2024-fall-65940)が作成_

:::details この図の見方
![](https://storage.googleapis.com/zenn-user-upload/e0e18de9b0e0-20250511.png)
_[MIT 6.5940](https://hanlab.mit.edu/courses/2024-fall-65940)が作成_
:::

### パターンに基づく方法

この方法では特定のパターンに沿ってPruningを実行します。ここでは代表的な手法である「**M:N Sparsity**」を紹介します。

#### N:M Sparsity

これは連続するM個の要素のうち、N個をPruningする方法です。
下の図は2:4 Sparsityの例を示しています。連続する4つのパラメータのうち2つのパラメータを削除し、重み行列を圧縮しています。一番右に紫の行列が示されていますが、これは0になった重みがどこに位置していたのかを示す、2bitのインデックスです。

![](https://storage.googleapis.com/zenn-user-upload/c56ed41436c4-20250511.png)
_[Accelerating Inference with Sparsity Using the NVIDIA Ampere Architecture and NVIDIA TensorRT](https://developer.nvidia.com/blog/accelerating-inference-with-sparsity-using-ampere-and-tensorrt/)_

この手法は適切な再学習の実施により、Pruning前と同等の精度を維持できることが[示されています](https://developer.nvidia.com/blog/accelerating-inference-with-sparsity-using-ampere-and-tensorrt/)。
また、Nvidiaの[Ampere](https://www.nvidia.com/ja-jp/data-center/ampere-architecture/)以降のGPU（A100等）では2:4 Sparsityをハードウェアサポートしています。

### チャンネル単位の方法

チャンネル単位のPruning（Channel Pruning）は、構造化Pruningの中でも広く利用されている手法の1つであり、特に畳み込み層において効果的です。この手法では、各畳み込み層の出力チャンネル単位で不要なチャンネルを選別・削除します。
出力チャンネルを削除すると、それに対応するカーネル（フィルタ）全体も不要になるため、計算量削減が明確に現れます。たとえば、1つのチャンネルを削除することで、そのチャンネルに対するすべての重み行列とバイアス、さらにその後続の層の入力次元も削減できます。

![](https://storage.googleapis.com/zenn-user-upload/fcc5adf190e9-20250511.png)
_[MIT 6.5940](https://hanlab.mit.edu/courses/2024-fall-65940)が作成。層ごとに異なるPruning率を設定する方法は後述_

## Pruningの基準

ここまでは、Pruningをどのような粒度で実行するかに注目してきました。しかし、実際にどの重みやチャンネルを削除するかを決定するには、それを選別するための**基準**（Criterion）も重要です。Pruningの効果を最大限に引き出すためには、適切な基準に基づいて対象を選ぶ必要があります。
本章では以下の代表的な基準を紹介します。

- シナプスの選択方法
  - Magnitude-based
  - Scaling-based
  - Second-order-based
- ニューロンの選択方法
  - Percentage-of-zero-based
  - Regression-based

### Magnitude-based

この基準はヒューリスティックな方法で、「絶対値の大きな重みは重要な重みだろう」という洞察に基づいています。
各パラメータごとにこの基準を適用する場合、以下のようになります。

![](https://storage.googleapis.com/zenn-user-upload/28fd5ee9d754-20250511.png)
_[MIT 6.5940](https://hanlab.mit.edu/courses/2024-fall-65940)が作成_

また、行ごとに行う場合の例を以下に示しています。行ごとにL1-norm（またはL2-norm）を行うことによって、どの行をPruningするか決定していることが示されています。

![](https://storage.googleapis.com/zenn-user-upload/beb762343945-20250511.png)
_[MIT 6.5940](https://hanlab.mit.edu/courses/2024-fall-65940)が作成_

### Scaling-based

スケーリングに基づく方法は、ネットワークのスケーリング係数を指標として、各構成要素（チャンネルやノードなど）の重要度を評価し、削除するべき部分を決定するアプローチです。代表的な手法にはBatchNormのスケール係数を活用した方法があります。
BatchNorm層の出力は以下の式で表されます。

$$
y_i = \gamma \cdot \frac{x_i - \mu_B}{\sqrt{\sigma_B^2 + \epsilon}} + \beta
$$

ここで、$\gamma$ はスケール係数で、各出力チャンネルごとに学習されるパラメータです。
この $\gamma$ の大きさが小さいほど、そのチャンネルの出力が抑えられており、出力に対する寄与が少ないと考えられます。したがって、小さい $\gamma$ を持つチャンネルを削除対象としてPruningします。

### Second-order-based

この基準は、損失関数の2階微分（ヘッセ行列）を活用して、パラメータの削除による影響を推定するアプローチです。
これまで紹介したPruning基準は、重みの絶対値などを使って重要度を評価しますが、パラメータ削除による出力の変化を考慮していませんでした。一方、Second-order-basedの代表例である[Optimal Brain Damage [LeCun et al., 1989]](https://proceedings.neurips.cc/paper/1989/hash/6c9882bbac1c7093bd25041881277658-Abstract.html)では次のような式に基づいて削除による影響を推定します。

$$
\text{Saliency}(w_i) \approx \frac{1}{2} H_{ii} w_i^2
$$

- $H_{ii}$ は **誤差関数のヘッセ行列の対角要素**（2階微分）
- $w_i$ は対象の重み

:::details 式の導出方法

重み $w_i$ を0にした際、損失関数 $E(\mathbf{w})$がどれだけ増えるかを見積もります。

$$
\Delta E = E(\mathbf{w}_{\setminus i}) - E(\mathbf{w})
$$

つまり、削除による損失の変化をテイラー展開で近似します。
損失関数 $E(\mathbf{w})$ を現在の重みベクトル $\mathbf{w}$ のまわりで2次のテイラー展開すると以下のようになります。

<!-- textlint-disable -->

$$
E(\mathbf{w} + \Delta \mathbf{w}) \approx E(\mathbf{w}) + \nabla E(\mathbf{w})^T \Delta \mathbf{w} + \frac{1}{2} \Delta \mathbf{w}^T H \Delta \mathbf{w}
$$

<!-- textlint-enable -->

- $\nabla E(\mathbf{w})$：勾配ベクトル（1階微分）
- $H$：ヘッセ行列（2階微分の行列）
- $\Delta \mathbf{w}$：重みの変化（今回は $w_i \to 0$ の変化）

学習後のモデルでは、損失関数の勾配は0に近い（十分に収束している）と仮定します。

$$
\nabla E(\mathbf{w}) \approx 0
$$

これにより、テイラー展開の1次項は無視できて、以下が残ります。

$$
\Delta E \approx \frac{1}{2} \Delta \mathbf{w}^T H \Delta \mathbf{w}
$$

重み $w_i$ だけを削除（=0に設定）するので以下のようになります。

<!-- textlint-disable -->

$$
\Delta w_j =
\begin{cases}
-w_i & (j = i) \\
0 & (j \neq i)
\end{cases}
\Rightarrow \Delta \mathbf{w} = (0, ..., -w_i, ..., 0)
$$

<!-- textlint-enable -->

$$
\Delta E \approx \frac{1}{2} H_{ii} w_i^2
$$

ここで $H_{ii}$ はヘッセ行列の対角成分（つまり $\frac{\partial^2 E}{\partial w_i^2}$）です。

よって、重み $w_i$ を削除したときの誤差増加量の近似は以下のようになります。

$$
\text{Saliency}(w_i) \approx \frac{1}{2} H_{ii} w_i^2
$$

:::

この式から、損失への影響が小さくなるような重み（重みが小さい+損失の曲率が小さい）を選んで削除する、というのが基本方針です。

### Percentage-of-zero-based

https://arxiv.org/abs/1607.03250

出力における0の割合が高いチャンネルを削除する方法です。
0の割合（Average Percentage of Zeros）は以下の式で計算されます。

$$
APoZ^{(i)}_c = \frac{ \sum_{k=1}^N \sum_{j=1}^M f(O^{(i)}_{c,j}(k) = 0) }{N \times M}
$$

- $i$：レイヤー番号
- $c$：チャンネル番号
- $N$： バリデーションデータの総数
- $M$：各出力の空間次元
- $O^{(i)}_{c,j}(k)$：サンプル $k$ に対するレイヤー $i$ のチャンネル $c$ の出力の $j$ 番目の要素
- $f(\cdot)$：条件が真なら1、偽なら0を返す関数（indicator関数）

図として表すと以下のようになります。

![](https://storage.googleapis.com/zenn-user-upload/9e87befab849-20250512.png)
_[MIT 6.5940](https://hanlab.mit.edu/courses/2024-fall-65940)が作成_

「レイヤー $i$ にあるチャンネル $c$ の出力がゼロになる割合を、全バリデーションデータと全空間位置について平均したもの」がAPoZと言えるでしょう。

### Regression-based

https://arxiv.org/abs/1707.06168

回帰によって損失の変化を予測し、Pruning前の損失との差が小さくなるようなPruningを決定する方法です。

![](https://storage.googleapis.com/zenn-user-upload/0611e4feeaba-20250512.png)
_[MIT 6.5940](https://hanlab.mit.edu/courses/2024-fall-65940)が作成_

$$
Z = XW^T = \sum_{c=0}^{c_i-1}X_c {W_c}^T
$$

とおいたときに以下を解くことによって最適化を行います。

<!-- textlint-disable -->

$$
\arg\min_{\mathbf{W},\,\boldsymbol{\beta}} \left\| \mathbf{Z} - \hat{\mathbf{Z}} \right\|_F^2 = \left\| \mathbf{Z} - \sum_{c=0}^{c_i - 1} \beta_c \mathbf{X}_c \mathbf{W}_c^\top \right\|_F^2
$$

<!-- textlint-enable -->

- $\beta$：各チャンネルを使うか決める係数ベクトル。${\beta}_c=0$のときチャンネル$c$はPruningされる（長さ$c_i$）
- $\left\| \beta \right\|_0 \le N_c$

## 層ごとのPruning率の最適化

ここまで、Pruningの粒度（どの単位で削減するか）と基準（どのパラメータを削減するか）について見てきました。しかし、どの程度まで削減を行うか（Pruning率）も、モデル性能や推論効率に大きく影響する重要な要素です。

たとえば、チャンネル単位の構造化Pruningにおいて、すべての層に一律のPruning率を適用するのではなく、層ごとの特性に応じて異なる割合でPruningを行うことで、より高い精度を維持しつつ、推論時のレイテンシを抑えることが可能になります。

しかし、層ごとにどの程度Pruningを行うべきかを事前に知ることは容易ではありません。層によってモデル全体の性能に与える影響度は異なるため、一律の基準では最適なPruning率を見つけることができないのが実情です。

### 感度分析

このような状況に対して有効なアプローチのひとつが、**感度分析（Sensitivity Analysis）**です。感度分析では、各層に対して異なるPruning率を設定し、各率における性能の変化を評価します。そして、各層においてしきい値を下回らないようなPruning率をその層のPruning率とします。

これにより、性能への影響が小さい層を優先的に削減し、重要な層の情報は維持するといった戦略的なPruning設計が可能になります。

### AMC (AutoML for Model Compression)

感度分析は層ごとに異なるPruning率を設定することを可能にしましたが、各層を独立して考えており、層間の相互作用を考慮しているため、必ずしも最適とは限りません。また、最適なPruning率を自動的に探索できればより便利です。

AMCは、Pruningの問題を強化学習の問題として定式化することにより、Pruning率の自動探索を可能にしました。

AMCは人間が1週間かかった作業を数時間で実施することを可能にしました。

また、MobileNetを使った実験では性能を維持しながらレイテンシとメモリ使用量を改善しました。

### モデルのFine-tuning

Pruning後のモデルは性能が低下することがあるため、Fine-tuningによって精度を回復する必要があります。

Fine-tuning時の
