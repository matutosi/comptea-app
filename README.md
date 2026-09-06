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
pip install -r requirements.txt

python cli/run_pipeline.py <画像> --workdir work/<名前>   # 格子を作る
python cli/run_ocr.py work/<名前>                          # セルを読む
python cli/build_table.py work/<名前>                      # 縦持ちに組む
python cli/export_data.py work --out out --tag <資料名>    # まとめて書き出す
```

**途中に 3 つの段階**を経ます．格子・読み取り・組み上がりを目で確かめてから
先へ進む作りです．`export_data.py` は，`Need Check` のセルを画像に切り出して
CSV から辿れるようにします．

### GUI (Streamlit — 1 枚ずつ試す)

工程ごとにアプリを分けてあります．重いもの (検出・読み取り) を分けることで，
無料枠でも動かせるようにしています．

| アプリ | やること | 入力 → 出力 |
|:---|:---|:---|
| `apps/1_split` | 折り込みを表ごとに切る | 画像 → 画像 (zip) |
| `apps/2_grid` | 検出して格子を作る | 画像 → `located.csv` |
| `apps/3_read` | セルを読む | 画像 + `located.csv` → `ocred.csv` |
| `apps/4_table` | 縦持ちに組んで検査 | `ocred.csv` → `comp_table_long.csv` |

```bash
streamlit run apps/2_grid/streamlit_app.py
```

工程のあいだは CSV で受け渡します．段階の分け方とそのまま対応しています．

**大きな画像は扱えません** (長辺 4000 px まで)．A0 の折り込みは CUI を使ってください．

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

作業ディレクトリを省くと `comptea/work/<画像名>/` に書きます．

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
