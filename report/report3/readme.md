# requirements
```sh
$ python3 --version
# >> 3.5.2
$ pip3 install -r py3/reqirements.txt
$ make svr
```

# example
途中で得られる 11/20 などの分数はeps内にある点の割合を目安として表すものである。

## 使用方法を得る
```sh
$ ./svr
```

## ガウスカーネル, C=1000, eps=0.01, θ=2.236 で, sample40.dat に SVRを実行する。
```sh
$ ./svr sample_data/sample40.dat gauss --c 1000 --eps 0.01 --p 2.236
# >> 回帰式f(X)が出力される
$ ./svr sample_data/sample40.dat --p 2.236
# デフォルトで ガウスカーネル, C=1000, eps=0.01 なのでこれは上と同じ結果を返す。
```

## 分割数 5, 指標を平均絶対誤差 にして 交差検定を行う
```sh
$ ./svr sample_data/sample20.dat --cross 5 mean_abs
# >> c:2**7.54235 , p:2**1.5 | 0.0201272  (c:186.412 , p:2.82843)
# => c:186.412 , p:2.82843 が解として得られる。
```

## 0~1の範囲の100分割のグリッド上で作成したSVRを適用した結果を 20.dat に保存する。
```sh
$ ./svr sample_data/sample20.dat --c 186.412 --p 2.82843 --plot 20.dat
```

## 実際にプロットして精度を確かめてみる (2次元データなので --3dをつける)
```sh
$ ./py3/plotdata.py 20.dat sample_data/sample20.dat --3d
```

## 分割数 5, 指標を平均絶対誤差 にして 交差検定を行い、探索過程(及び精度)をプロットする
```sh
$ ./svr sample_data/sample20.dat --cross 5 mean_abs --plot > 20.dat
```

## 実際にプロットして探索過程を確かめてみる (2次元データなので --3dをつける)
```sh
$ ./py3/plotdata.py 20.dat --3d
```

