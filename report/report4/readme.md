# requirements
```sh
$ python3 --version
# >> 3.5.2
$ pip3 install -r py3/requirements.txt
$ make
# >> svr* will be made
```

# example
```sh
$python3 py3/multi_agent.py sample_data/id0001.csv @greedy
# どんな商品でも極力買おうとする「貪欲エージェント」によるオークションを行い、結果を見る
$python3 py3/multi_agent.py sample_data/id0001.csv @svr --show-process
# SVRの予測値を利用する「SVRエージェント」によるオークションを行い購入過程を見る。
$python3 py3/multi_agent.py sample_data/id0001.csv @greedy @svr
# 貪欲エージェントとSVRエージェントによるオークションを行い、結果を見る
```
