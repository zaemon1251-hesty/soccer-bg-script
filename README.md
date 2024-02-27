# sn-script

研究用のスクリプト

git filter-repo --subdirectory-filter sn-script で抽出したリポジトリ
（その際、Benchmarkのコードがローカルとgithub上から履歴もろとも吹き飛んだから、今後気をつける。なお、ラボのサーバにコードが残っていたため無事だった。）

## How to construct corpus

```bash
$ /path/to/python speech2text.py
...
$ /path/to/python transcribe_processor.py
```
