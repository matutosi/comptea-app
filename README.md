# comptea — 組成表を構造化データにする

**comptea** (Composition Table to Data Easy) は，植生学 (植物社会学) の**組成表**を
スキャンした画像から，構造化データ (縦持ちの表) を取り出す道具です．

古い資料の組成表は，そのまま OCR にかけてもうまく読めません．

- **場所によって文字種が違う** — 学名 (ラテン文字)・和名 (カタカナ)・階層 (英数記号)・
  被度 (`5 4 3 2 1 + r ・`)
- **解像度が低い** — 1980 年代の印刷を写したものが多い
- **表の形が資料ごとに違う** — 1 地点を 2 段に折り返す，A0 の折り込みに表が 5 つ並ぶ，
  90 度回して組む

そこで，**物体検出で領域を捉え，行と列はアルゴリズムで決め，セルごとに読む**という
順序をとります．

## できること

- 大きな折り込みを，表ごとの画像に切り分ける
- 表頭・種名の列・組成部を検出し，行と列の格子を作る
- セルを読み，種名の辞書と被度の規則で補正する
- 縦持ちの表 (1 行 = 1 地点 × 1 種) に組み，機械でできる検査にかける

実際の資料 148 表で通したときの成績は，**縦持ち 47,756 行・95.6% が検証を通る**
状態です (残りは目視に回します)．

## 2 つの使い方

### CUI (まとめて処理する)

```bash
pip install -e .[all]        # 中核だけなら pip install -e .

python cli/run_pipeline.py <画像> --workdir work/<名前>   # 格子を作る
python cli/run_ocr.py work/<名前>                          # セルを読む
python cli/build_table.py work/<名前>                      # 縦持ちに組む
python cli/export_data.py work --out out --tag <資料名>    # まとめて書き出す

# 続きのページがあるとき(ファイル名に枝番を付けておく)
python cli/run_pipeline.py <画像>-2.jpg --workdir work/<名前>-2  # 塊を切り出す
python cli/link_pages.py work --out out                          # 表につなぐ
```

**入れなくても動きます** (`cli/` は，入っていなければリポジトリの置き場を
自分で `sys.path` に足します)．依存は工程ごとに分けてあり，
`.[detect]` は検出 (torch・ultralytics)，`.[read]` は読み取り (easyocr)，
`.[sheet]` は PDF (PyMuPDF)，`.[web]` は Streamlit です．

### ライブラリとして

```python
from comptea import correct_text
correct_text.correct_comp('5・5')       # {'corrected': '5;5', 'status': 'OK'}

from comptea import pipeline           # 段を，同じプロセスで動かす
code, log = pipeline.run('grid', ['表.png', '--workdir', 'work/x'])
```

辞書と重みはパッケージに同梱してあるので，**どこから呼んでも開けます**．
折り込みを切り分けるだけなら `python -m comptea.split_sheet <PDF> --outdir <置き場>`．

**途中に 3 つの段階**を経ます．格子・読み取り・組み上がりを目で確かめてから
先へ進む作りです．`export_data.py` は，`Need Check` のセルを画像に切り出して
CSV から辿れるようにします．

組成表の下の流し込み (出現1回の種・調査地・調査年月日・出典) は，紙面に
収まらないと**次のページへあふれます**．あふれた先は本文や写真だけのページに
見えるので，そのままでは格子の段で「組成表が無い」として止まります
(検出クラス `once_species` は，表の下にある形でしか学習していないため，
単独で置かれた塊は出ません)．**ファイル名に枝番を付けておけば止まりません**
(下記)．枝番を付けない一枚だけを扱うときは `read_once_page.py` を使います．

### 続きのページは，ファイル名の枝番で示す

どのページが続きかはノンブル (紙面のページ番号) の順で分かるはずですが，
**紙面の配置の都合で崩れることがあります**．そこで**ファイル名で明示**します．

```
組成表だけ        xxx.jpg
組成表と続き      xxx-1.jpg (組成表)   xxx-2.jpg (続き)
```

枝番は 1 から，**読む順**に振ります．組成表が 2 ページ以上に渡っても構いません
(そのときは本体の行がページごとに出ます．`link_pages.py` が知らせます)．

枝番が付いていれば，`run_pipeline.py` は**表が無くても止まりません**．
続きのページとして塊と注記を切り出し，`link_pages.py` が表につなぎます．

```bash
python cli/link_pages.py work --out out
```

つなぐのは**文字列としてつないでから一度だけ解析する**ためです．そうすると
地点の引き継ぎ・ページで割れた種名・前ページに残った注記の見出しが，
いずれも自動で解けます (1 ページずつ解析すると，どれも解けません)．

### GUI (Streamlit — 1 枚ずつ試す)

工程ごとにアプリを分けてあります．重いもの (検出・読み取り) を分けることで，
無料枠でも動かせるようにしています．

| アプリ | やること | 入力 → 出力 |
|:---|:---|:---|
| `apps/1_split` | 折り込みを表ごとに切る (**PDF も可**) | 画像・PDF → 画像 (zip) |
| `apps/2_grid` | 検出して格子を作る (**PDF も可**) | 画像・PDF → `grid.zip` |
| `apps/3_read` | セルを読む・**表頭を表にする**・読みを直す | 画像 + `grid.zip` → `read.zip` |
| `apps/4_table` | 縦持ちに組んで検査 | `read.zip` → `table.zip` |

```bash
streamlit run apps/2_grid/streamlit_app.py
```

**工程のあいだは zip で受け渡します**．次の工程に要るものが一式入っているので，
入れ忘れが起きません．工程の分け方は，CUI の 3 つの段階とそのまま対応しています．

**2・3・4 には見本が入っている**ので，前の工程を通さずにその場で試せます
(`examples/sample.jpg`・`sample_grid.zip`・`sample_read.zip`)．

**3 では，読んだ字の隣に実物のセルが出ます**．`corrected` はその場で直せ，
直した値は同じ規則で検証したうえで zip に反映されます (`note` に `hand`)．
長い表は出す行数を選べます (`Need Check` だけに絞ることもできます)．

各ページの上には**全体像**を，横には**4 工程の一覧**を出しています．
share.streamlit.io に登録したあと，Secrets に次のように書いておくと，
一覧が**ほかのページへのリンク**になります (書かなければただの一覧のままです)．

```toml
[urls]
1_split = "https://....streamlit.app"
2_grid  = "https://....streamlit.app"
3_read  = "https://....streamlit.app"
4_table = "https://....streamlit.app"
```

**扱える大きさには上限があります** (無料枠のメモリに合わせて測って決めました)．

| アプリ | 上限 | 測った山 |
|:---|:---|---:|
| 1 切り分け | **150 Mpx** (A0 の折り込み 9344 x 12873 = 120 Mpx は通る) | 520 MB |
| 2 格子・3 読み取り | 学習時と同じ縮尺で検出できること (長辺 6600 px ほど) | 432-781 MB |

**折り込みは 1 で表ごとに切ってから 2 へ渡してください**．
1 の画面には，切り出した表ごとに「2 に渡せるか」が出ます
(手元の折り込み 23 枚では，切り出した 64 表のうち 34 表が渡せました．
残りは手元で `python cli/run_pipeline.py <画像>` を使います)．

**1 枚の紙面に表が 2 つ以上あるときは，2 が表ごとに分けて出します**．
別々の表なので，1 つずつ 3 へ渡してください．

## テスト

```bash
pip install -r requirements-dev.txt
pytest                # 実データの要らないもの(数秒)
pytest --runslow      # 検出と読み取りも実際に走らせる(30 秒ほど)
```

**実データが無くても走ります**．辞書と見本 (`examples/`) だけで完結し，
画像の要るものは印字を模した小さな配列を組み立てて確かめます．

| ファイル | 見るもの |
|:---|:---|
| `tests/test_correct_text.py` | 被度・常在度・階層・表頭の値の補正 |
| `tests/test_names.py` | 種名の辞書との突き合わせ |
| `tests/test_comp_table.py` | 括弧付きのセルの読み分け・割れた値の繕い |
| `tests/test_header.py` | 項目名の寄せ・文章形式の表頭・1 回出現種 |
| `tests/test_ink.py` | 黒画素から測る道具 (画数・罫線・破線) |
| `tests/test_geometry.py` | 等間隔の格子・境が字を割る回数 |
| `tests/test_checks.py` | 格子の検査 4 つ (鳴るべきときに鳴るか) |
| `tests/test_wiring.py` | モジュールをまたぐ参照が実在するか (移し忘れを捕まえる) |
| `tests/test_grid_parts.py` | 短冊・種名の列・階層の列 (組み立てた紙面で) |
| `tests/test_row_heights.py` | 行の高さをそろえる後処理 (細い行・位相・半分の刻み・上下端) |
| `tests/test_row_skew.py` | 紙面の傾きをセルの座標だけで直す後処理 (傾いた紙面・水平な紙面・列の少ない表) |
| `tests/test_row_track.py` | 行を単位の連なりとして追う後処理 (列ごとの y のずれ・拍による行数の検算．`slow` は実データ) |
| `tests/test_header_lines.py` | 表頭の項目行を OCR の検出器の箱から作る (束ね方・帯・表題の除外．`slow` は描いた表頭で検出器を走らせる) |
| `tests/test_apps.py` | Streamlit の 4 アプリを画面まで走らせる |
| `tests/test_shared.py` | 工程のあいだの受け渡し (zip・見本・検出の返り) |
| `tests/test_pipeline.py` | 見本 1 枚の通し (`slow` は 3 段を続けて回す) |

歯止めにしているのは，**実物を見て決めた判断**です
(`III(+-4)` を `III(1-4)` にしない，1 文字の読みは完全一致だけ採る，など)．
どれも「黙って別の値になる」型で，通してみても気づけません．
経緯は [docs/lessons.md](docs/lessons.md) にあります．

## 出力の形

`comp_table_long.csv` は縦持ちで，1 行が 1 地点 × 1 種です．

| 列 | 中身 |
|:---|:---|
| `plot` | 地点番号 (左から 1, 2, …) |
| `row_no` | 表の中の行番号 |
| `j_name` / `s_name` | 和名 / 学名 |
| `layer` | 階層 (`B1` `B2` `S` `K` など) |
| `cover` / `sociability` | 被度 / 群度 |
| `constancy` | 常在度 (群落の要約列のとき) |
| `status` | `OK` / `Need Check` / `suggested` / `absent` |
| `note` | セルごとの疑わしさ (`interpolated` `snapped` `retry` `roman` …) |
| `source` | `body` (表の本体) / `once` (1 回出現種の流し込み) |

## 仕組み

- [docs/pipeline.md](docs/pipeline.md) — 工程の流れとアルゴリズム
- [docs/lessons.md](docs/lessons.md) — 規則づくりの知見と，測って取り下げた案
- [docs/vegetation_science.md](docs/vegetation_science.md) — 分野の背景 (被度階級・階層・常在度)
- [docs/architecture.md](docs/architecture.md) — コードの構成
- [.claude/skills/comptea/](.claude/skills/comptea/) — Claude Code から通しで回すスキル．
  段階ごとの見どころ (`references/checkpoints.md`)，崩れ方と直し方
  (`references/failure-modes.md`)，画像を読むときの約束
  (`references/reading-guide.md`)，読み手に渡す文面
  (`references/read-cells-prompt.md`)．**このリポジトリが正**です
- [eval/README.md](eval/README.md) — 物差し (**非公開の資料が要るので，ここからは動きません**)

作業ディレクトリを省くと，**いまいる場所**の `work/<画像名>/` に書きます．

## 重み

`comptea/weights/comptea.pt` は YOLO11n を組成表で学習したものです．検出クラスは 12 で，
`comp` (セル) は検出せず，`col` × `row` から計算で作ります．

## 見本の画像

`examples/sample.jpg` は，動かして確かめるための 1 枚です．

> **出典**: 宮脇昭 (編) 1984.『日本植生誌 近畿』至文堂.
>
> 組成表の 1 ページを，動作を示すための引用として掲載しています．

学習に使った資料そのものは，著作権のため公開していません．

## ライセンス

MIT (`LICENSE` を見てください)．ただし `examples/sample.jpg` は引用であり，
この条件の対象外です．
