---
title: "効率的なGANs [TinyML]"
emoji: "🗜️"
type: "tech" # tech: 技術記事 / idea: アイデア
topics: ["ディープラーニング"]
published: true
---

## はじめに

Generative Adversarial Networks（GANs）は2010年代後半に登場し、画像生成・変換の分野で一躍注目を集めました。特にStyleGANシリーズなどはフォトリアリスティックな画像生成を可能にし、多くの研究や応用に繋がっています。

しかし近年はDiffusion Models（拡散モデル）の台頭により、GANは最先端ではなくなりつつあります。それでも、GANは生成モデル研究の重要な基盤であり、理解する価値は十分にあります。

そこで本記事では、MITの講義[TinyML and Efficient Deep Learning Computing](https://hanlab.mit.edu/courses/2024-fall-65940)の[Lecture 17](https://www.dropbox.com/scl/fi/6o45qs8xm20qhzkc192bv/Lec17-Efficient-GANs-Video-PointCloud.pdf?rlkey=71hrp50kjtl8zz8w7jntvbwn0&st=ywq378y5&dl=0)をもとに、効率的なGAN手法を3つ紹介します。

https://www.youtube.com/watch?v=o_60Yhb79W8

なお、正確性には十分気をつけていますが、筆者はプロフェッショナルではないので間違いを含んでいる可能性があることをご了承ください。

:::message
[TinyML and Efficient Deep Learning Computing](https://hanlab.mit.edu/courses/2024-fall-65940)は、主宰である[Han教授](https://hanlab.mit.edu/songhan)の研究室における研究成果を中心に扱っています。したがって本記事は、GAN圧縮研究を網羅的にカバーするものではありません。
:::

## GANの圧縮

https://arxiv.org/abs/2003.08936

https://hanlab.mit.edu/projects/gancompression

Liら(CVPR 2020)は、Conditional GANの圧縮方法を提案しています。以下は提案手法を表した図であり、大きく3段階に分かれています。

![](https://storage.googleapis.com/zenn-user-upload/fb8524fac8e9-20250721.png)
_Li et al., GAN Compression: Efficient Architectures for Interactive Conditional GANs. 2020　より_

<!-- textlint-disable -->

1. 知識蒸留： 学習済み教師モデルの出力 $G'(x)$ とチャンネル数を削減した生徒モデルの出力 $G(x)$ を近づけるように、Reconstruction Lossを用いて学習する。このとき、中間層の特徴マップも近づけるために、Disstillation Lossを用いる。さらに一般的なGANの損失関数も使用する(cGAN Loss)
2. より小さくて高性能なチャンネル幅の組み合わせを探索する
3. 探索により得られた最良の構成をFine-tuningする

Reconstruction Loss・Disstillation Loss・cGAN Lossはそれぞれ以下のように表されます。

$$
\mathcal{L}_{\text{recon}} =
\begin{cases}
\mathbb{E}_{x,y}\|G(x) - y\|_{1}, & \text{for paired cGANs}, \\
\mathbb{E}_{x}\|G(x) - G'(x)\|_{1}, & \text{for unpaired cGANs}.
\end{cases}
$$

$$
\mathcal{L}_{\text{distill}}
= \sum_{t=1}^{T} \| f_{t}(G_{t}(x)) - G'_{t}(x) \|_{2},
$$

$$
\mathcal{L}_{\text{cGAN}}
= \mathbb{E}_{x,y}[\log D(x,y)]
+ \mathbb{E}_{x}[\log(1 - D(x, G(x)))],
$$

<!-- textlint-enable -->

この手法による圧縮結果を示しています。代表的な条件付きGANである[CycleGAN](https://arxiv.org/abs/1703.10593)・[Pix2pix](https://arxiv.org/abs/1611.07004)・[GauGAN](https://arxiv.org/abs/1903.07291)について、推論速度とモデルサイズの大きな改善を達成しました。

![](https://storage.googleapis.com/zenn-user-upload/eca40cede74f-20250921.jpg)
*https://hanlab.mit.edu/projects/gancompression*

また手法による生成結果の例を以下に示しています。pix2pixとCycleGANのどちらにおいても多少のFIDの悪化が見られますが、引き換えに大きく推論速度を改善しています。その一方で、本手法を用いずに単にチャンネル数を0.25倍した場合（画像右下）ではFIDが大きく上昇し、シマウマへの変換が完全ではないことがわかります。

![](https://storage.googleapis.com/zenn-user-upload/0dc1fdaabfa9-20250921.png)
*https://hanlab.mit.edu/projects/gancompression*

:::details FIDとは

FIDは **Fréchet Inception Distance** の略で、生成画像と実画像の分布の距離を測る指標です。
画像認識モデルであるInception v3を使って画像を特徴ベクトルに変換し、それらを多次元ガウス分布に近似したうえでFréchet距離を計算します。

<!-- textlint-disable -->

$$
\text{FID} = \| \mu_r - \mu_g \|^2 + \mathrm{Tr}\left(\Sigma_r + \Sigma_g - 2(\Sigma_r \Sigma_g)^{1/2}\right)
$$

<!-- textlint-enable -->

- $\mu_r, \Sigma_r$：実画像の特徴ベクトルの平均と共分散
- $\mu_g, \Sigma_g$：生成画像の特徴ベクトルの平均と共分散

つまり、FIDが小さいほど「見た目の質と多様性が高い」生成モデルとみなされます。

:::

## AnyCost GAN

https://arxiv.org/abs/2103.03243

https://hanlab.mit.edu/projects/anycost-gan

GANは、ランダムベクトルや入力画像を潜在空間（latent space）にマッピングし、その潜在表現を変化させることで出力画像を制御できます。潜在空間の操作によって、顔画像であれば「表情」「髪型」などの属性を編集できることが知られています。しかしその生成速度が遅いため、潜在ベクトルを少しずつ変化させながら結果を確認するような、インタラクティブなアプリケーションには適さないという課題がありました。

このような課題に対し、Linら(CVPR 2021)はBlenderやMayaなどのレンダリングソフトウェアでのアプローチに目をつけました。レンダリングソフトウェアでは、光の計算（レイトレーシング）の際に計算する光の量を減らすことで高速なプレビューを実現しています。

![](https://storage.googleapis.com/zenn-user-upload/98807d41c324-20250921.png)
_[TinyML and Efficient Deep Learning Computing](https://hanlab.mit.edu/courses/2024-fall-65940)より_

このような手法をGANに適用し、画像編集の際は軽量なモデルを使い、満足したらフルサイズモデルで高品質な結果を得れば良いと考えたのです。

![](https://storage.googleapis.com/zenn-user-upload/9f8261bf44b2-20250921.png)
_[TinyML and Efficient Deep Learning Computing](https://hanlab.mit.edu/courses/2024-fall-65940)より_

これを実現するには異なる解像度・異なるチャンネル数で一貫した出力が得られるようにしなければなりません。以下にその方法をまとめます。

### マルチ解像度学習

通常のStyleGAN2では中間層の出力をそのまま低解像度画像にすると不自然になります。
Anycost GANでは、学習時に解像度をランダムサンプリングし、各解像度で自然な画像を出力するように訓練します。

![](https://storage.googleapis.com/zenn-user-upload/7ccbbc573602-20250921.png)
_[TinyML and Efficient Deep Learning Computing](https://hanlab.mit.edu/courses/2024-fall-65940)より_

### アダプティブチャンネル学習

計算コストをさらに下げるため、各層のチャンネル数をランダムに削減して訓練します。
重要度の高いチャンネルを優先的に保持し、削減後でも出力品質を維持できるよう工夫しています。
すべてのサブネットワークを同時に訓練し、低コスト版とフル版を重み共有しています。

![](https://storage.googleapis.com/zenn-user-upload/ba7f6723ef89-20250921.png)
_[TinyML and Efficient Deep Learning Computing](https://hanlab.mit.edu/courses/2024-fall-65940)より_

しかし、チャンネル削減だけではフル版と結果がズレしまいます。そのため、MSE損失と[LPIPS](https://richzhang.github.io/PerceptualSimilarity/)を組み合わせて、サブネット出力とフルモデル出力が近づくよう学習させます。

### ジェネレータ条件付き識別器

多様な解像度・チャンネル設定を同時に訓練すると、通常の識別器では対応しきれません。
Anycost GANでは、識別器に「ジェネレータの構造情報」を入力し、それに応じて特徴マップを変調します。例えば、「このサブネットは0.5×チャンネル構成」と識別器に知らせることで、すべてのサブネットに適切なフィードバックを与えられるようになります。

![](https://storage.googleapis.com/zenn-user-upload/137bd89f0977-20250921.gif)
*https://hanlab.mit.edu/projects/anycost-gan*

### 結果

以下のように、異なる解像度・モデルサイズにおいて、一貫した出力を得ることに成功しました。

![](https://storage.googleapis.com/zenn-user-upload/43ba8c6c86e6-20250921.gif)
*https://hanlab.mit.edu/projects/anycost-gan*

## 微分可能なデータ拡張

一般的な深層学習モデルと同様に、GANにおいてもデータの量が重要になります。少データでは識別器が過学習を起こし、生成画像が崩壊します。このようなデータの不足に対し、一般的にはデータ拡張が有効であると考えられていますが、GANの文脈ではどのように適用すれば良いでしょうか。いくつか考えてみます。

### 本物の画像のみを拡張する

識別器に入力されるデータのうち、「本物の画像」にのみカラージッターなどのデータ拡張を施した場合、どのような問題が起こるでしょうか。

結論から言うと、生成画像にデータ拡張の効果そのものが現れてしまい、学習がうまくいかなくなります。

理由は、識別器が「本物」と「偽物」を比較して識別しているためです。識別器は本物と偽物を区別する手がかりを探し、生成器は本物らしい画像を出力して識別器を欺こうとします。
このとき本物の画像だけにデータ拡張を施すと、生成器は「拡張の痕跡（CutOutやカラージッターなどの効果）」を模倣するほうが有利になります。その結果、生成器は拡張の効果を含んだ不自然な画像を生成し、識別器を欺く方向に学習が偏ってしまうのです。

![](https://storage.googleapis.com/zenn-user-upload/19ec97734c06-20250921.png)
_[TinyML and Efficient Deep Learning Computing](https://hanlab.mit.edu/courses/2024-fall-65940)より_

### 識別器の入力を拡張する

では、本物の画像だけでなく、生成画像にもデータ拡張を施してから識別器に入力したらどうなるでしょうか。

一見すると公平に見えますが、実はこれもうまくいきません。理由は、生成器がデータ拡張に一切関与していないため、生成器と識別器の間で学習のバランスが崩れるからです。

生成器の立場からすると、自分の出力した画像がそのまま識別器に渡されると思っています。しかし実際には、識別器に入力される直前でデータ拡張によって画像が加工されてしまいます。その結果、識別器が返すフィードバックは「生成器が出力した画像」ではなく、「拡張後の画像」に基づくものとなります。

つまり、生成器からすれば「自分の知らないところで勝手に書き換えられた画像」に対して評価を受けている状態です。この不一致が、生成器にとって正しい改善方向を学ぶ妨げとなり、最終的に訓練が不安定になります。

![](https://storage.googleapis.com/zenn-user-upload/0940c01b61e6-20250921.png)
_[TinyML and Efficient Deep Learning Computing](https://hanlab.mit.edu/courses/2024-fall-65940)より_

### Differentiable Augmentation

https://arxiv.org/abs/2006.10738

https://hanlab.mit.edu/projects/diffaug

Zhaoら（NeurIPS 2020）は、本物画像と生成画像の両方に、かつ生成器にも勾配が伝わる形で拡張を適用する方法を提案しました。

<!-- textlint-disable -->

具体的には識別器の更新時には $D(T(x)), D(T(G(z)))$ を使用します。そして、生成器の更新時には $D(T(G(z)))$ に基づいて勾配を逆伝播させます（拡張 $T$ は微分可能である必要あり）。このような設計により生成器は拡張を考慮した画像を生成できるようになりました。

<!-- textlint-enable -->

![](https://storage.googleapis.com/zenn-user-upload/cf13a14e6405-20250921.jpeg)
*https://hanlab.mit.edu/projects/diffaug*

このようなアプローチにより、データ効率が大幅に改善し、極小データセットでも高品質な生成が可能になりました。以下の右は、100-shotによる生成結果です。

![](https://storage.googleapis.com/zenn-user-upload/6a624a3fb66e-20250921.jpeg)
*https://hanlab.mit.edu/projects/diffaug*

![](https://storage.googleapis.com/zenn-user-upload/0d71d18d7f4b-20250921.png)
_Zhao et al., Differentiable Augmentation for Data-Efficient GAN Training. 2020 より_

## まとめ

本記事では、MITの講義 _TinyML and Efficient Deep Learning Computing_ をもとに、効率的なGANに関する3つの代表的な研究を紹介しました。

これらの研究は、GANをより軽量かつデータ効率よく利用するためのアプローチを示しています。拡散モデルが主流となりつつある現在でも、GANは生成モデル研究の重要な基盤であり、その効率化に向けた取り組みは引き続き価値のあるテーマです。
