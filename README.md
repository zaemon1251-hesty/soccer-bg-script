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

### todo list

- タイミング分析
  - [ ] 上位チーム・下位チーム
    - すぐできるはず
  - [ ] チャンピオンズリーグ決勝など、大衆向け放送だとどうか
    - すぐできるはず
  - [ ] 話者特定
    - [ ] 実況者・実況者スタイル分類分析
  - [ ] ドメインスポーツ広げる
  - [ ] 計算社会学 認知言語学に広げる（言語話者間の違いをより分析すると）
