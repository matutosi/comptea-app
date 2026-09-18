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

実際の資料の全 147 表で通すと，**縦持ちは 47,097 行** (2026-09-16 時点)，
**要確認 (`Need Check`) は 924 件**でした (2026-09-17 時点．要確認は目視に回します)．

## 入れ方

**Python 3.10 以上**が要ります．

```bash
pip install -e .[all]        # 中核だけなら pip install -e .
```

依存は工程ごとに分けてあり，必要なものだけを足せます．

| extra | 中身 | 使う工程 |
|:---|:---|:---|
| `detect` | torch・torchvision・ultralytics・OpenCV | 段階 1 (検出) |
| `read` | easyocr・OpenCV | 段階 2 (読み取り) |
| `sheet` | PyMuPDF | PDF の読み込み・折り込みの切り分け |
| `web` | Streamlit | GUI |
| `dev` | pytest・coverage・streamlit・OpenCV | テスト (requirements-test.txt と同じ) |
| `all` | `detect`・`read`・`sheet`・`web` | — |

`run_pipeline.py`・`run_ocr.py`・`build_table.py`・`link_pages.py`・`read_once_page.py` は，
入れていなくてもリポジトリの置き場を自分で `sys.path` に足します．
**`crop_cells.py`・`apply_text.py`・`export_data.py` は足さない**ので，
`pip install -e .` してから使ってください．

### 外の読み手 (任意)

EasyOCR に加えて，外の読み手を重ねられます (`run_ocr.py --reader multi`)．
**読み手ごとに落とすセルが違う**ので，重ねると読めるセルが増えます
(和名 277 → 428，学名 462 → 548 のセルが辞書に当たるようになりました)．
**どちらも，入っていなければ黙って飛ばします**．入れていない環境でも
`--reader multi` は落ちず，EasyOCR だけで読みます．

yomitoku は段階 3 でも使います．表の下の注記から調査地・調査年月日・出典を読み，
表頭で空いている地点の情報を埋めます (`site_notes`)．
入れていなければ，この差し込みは起きません．

| 読み手 | 入れ方 |
|:---|:---|
| yomitoku | 主環境を汚さないよう別の環境に入れます．`python -m venv --system-site-packages <置き場>/venv_yomi` して `pip install yomitoku==0.14.0`．**`--system-site-packages` は主環境の torch を使い回すため**です (入れ直すと数 GB)．重みは初回の呼び出しで取りに行きます |
| NDLOCR-Lite | [ndl-lab/ndlocr-lite](https://github.com/ndl-lab/ndlocr-lite) を置いて `pip install ordered-set` を足すだけです．**ONNX で GPU が要りません** |

### 環境変数

置き場は**環境変数だけ**で決めます (公開リポジトリなので，コードに既定のパスは持ちません．
`tests/test_no_local_paths.py` が見張ります)．毎回指さなくて済むよう，
環境に登録しておくと楽です (Windows は `setx`，POSIX は `~/.profile` などに `export`)．

| 変数 | 指すもの |
|:---|:---|
| `COMPTEA_YOMI_PY` | yomitoku を入れた環境の python |
| `COMPTEA_NDLOCR` | NDLOCR-Lite を置いた場所 |
| `COMPTEA_DEVICE` | 読み手を動かす装置 (`cpu` / `cuda`)．`--device` が優先し，どちらも無ければ自動 |
| `COMPTEA_CORE` | 中核 (`comptea` パッケージ) の場所を上書きする (古い名前 `COMPTEA_YOLO` も読みます) |
| `COMPTEA_DATA` | `eval/` だけが使う，正解データの置き場 ([eval/README.md](eval/README.md)) |

## CUI (まとめて処理する)

**途中に 3 つの段階**を経ます．格子・読み取り・組み上がりを目で確かめてから
先へ進む作りです．作業ディレクトリ (`--workdir`) を省くと，
**いまいる場所**の `work/<画像名>/` に書きます．

```bash
# 段階 1: 格子を作る (located.csv と，格子を描いた画像)
python cli/run_pipeline.py <画像> --workdir work/<名前>

# 段階 2: セルを読む (ocred.csv と，目視に回すセルの一覧 review.tsv)
python cli/run_ocr.py work/<名前>                    # --reader multi・--device cpu も可
python cli/crop_cells.py work/<名前> --what review    # 目視に回すセルを work/<名前>/crops/ に切り出す
python cli/apply_text.py work/<名前> --tsv fixes.tsv # 目で読んだ字 (cell_id と text) を戻す

# 段階 3: 縦持ちに組む (comp_table_long.csv ほか．下の「出力の形」)
python cli/build_table.py work/<名前>                # 非出現のセルも残すなら --keep-absent，注記を読まないなら --no-notes

# まとめて書き出す (Need Check のセルを画像に切り出し，CSV から辿れるようにする)
python cli/export_data.py work --out out --tag <資料名>
```

`apply_text.py` で戻した読みは，EasyOCR の読みと**同じ補正**を通します．

折り込みを切り分けるだけなら `python -m comptea.split_sheet <PDF> --outdir <置き場>`．

### 続きのページ

組成表の下の流し込み (出現1回の種・調査地・調査年月日・出典) は，紙面に
収まらないと**次のページへあふれます**．どのページが続きかは
**ファイル名の枝番で示します** (`xxx-1.jpg` が組成表，`xxx-2.jpg` が続き．枝番は読む順)．

```bash
python cli/run_pipeline.py <画像>-2.jpg --workdir work/<名前>-2  # 表が無くても止まらず，塊を切り出す
python cli/link_pages.py work --out out                          # つないでから一度だけ解析する
```

枝番を付けずに 1 枚だけ扱うときは `read_once_page.py` を使います
(塊を切り出す → 読んだ文字列から行を作る，の 2 回に分けて呼びます)．

```bash
python cli/read_once_page.py <画像> --out <置き場>
python cli/read_once_page.py <画像> --out <置き場> --text <置き場>/once.txt --table <続き元の表> --start-plot <地点番号>
```

なぜ枝番なのか (ノンブルの順では決められない理由) と，つなぐと何が解けるかは
[docs/pipeline.md の 11 節](docs/pipeline.md#11-続きのページをつなぐ) にあります．

### ライブラリとして

```python
from comptea import correct_text
correct_text.correct_comp('5・5')       # {'corrected': '5;5', 'status': 'OK'}

from comptea import pipeline           # 段を，同じプロセスで動かす
code, log = pipeline.run('grid', ['表.png', '--workdir', 'work/x'])
```

辞書と重みはパッケージに同梱してあるので，**どこから呼んでも開けます**．

## GUI (Streamlit — 1 枚ずつ試す)

工程ごとにアプリを分けてあります．重いもの (検出・読み取り) を分けることで，
無料枠でも動かせるようにしています．**アプリごとに `apps/*/requirements.txt`** があります．

| アプリ | やること | 入力 → 出力 |
|:---|:---|:---|
| `apps/1_split` | 折り込みを表ごとに切る (**PDF も可**) | 画像・PDF → 画像 (zip) |
| `apps/2_grid` | 検出して格子を作る (**PDF も可**) | 画像・PDF → `grid.zip` |
| `apps/3_read` | セルを読む・**表頭を表にする**・読みを直す | 画像 + `grid.zip` → `read.zip` |
| `apps/4_table` | 縦持ちに組んで検査 | `read.zip` → `table.zip` |

```bash
streamlit run apps/1_split/streamlit_app.py
streamlit run apps/2_grid/streamlit_app.py
streamlit run apps/3_read/streamlit_app.py
streamlit run apps/4_table/streamlit_app.py
```

**工程のあいだは zip で受け渡します**．次の工程に要るものが一式入っているので，
入れ忘れが起きません．工程の分け方は，CUI の 3 つの段階とそのまま対応しています
(CUI との違いは [docs/pipeline.md の「CUI と GUI の対応」](docs/pipeline.md#cui-と-gui-の対応))．

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

**扱える大きさには上限があります** (無料枠のメモリに合わせて，CPU 版の torch で測って決めました)．

| アプリ | 上限 | 測った山 |
|:---|:---|---:|
| 1 切り分け | **150 Mpx** (A0 の折り込み 9344 x 12873 = 120 Mpx は通る) | 520 MB |
| 2 格子 | 検出の縮尺 `imgsz` 2560 まで (紙面の長辺 6600 px ほど) | 432 MB (imgsz 1280) - 781 MB (2560) |
| 3 読み取り | 同上 | 507 MB (見本) |

**折り込みは 1 で表ごとに切ってから 2 へ渡してください**．
1 の画面には，切り出した表ごとに「2 に渡せるか」が出ます
(手元の折り込み 23 枚では，切り出した 64 表のうち 34 表が渡せました．
残りは手元で `python cli/run_pipeline.py <画像>` を使います)．

**1 枚の紙面に表が 2 つ以上あるときは，2 が表ごとに分けて出します**．
別々の表なので，1 つずつ 3 へ渡してください．

## 出力の形

段階 3 (`build_table.py`) は，作業ディレクトリに次のファイルを書きます．

| ファイル | 中身 |
|:---|:---|
| `comp_table_long.csv` | 縦持ちの表 (1 行 = 1 地点 × 1 種)．**これが正** |
| `comp_table_wide.csv` | 目で確かめる用の横持ち (組成表の見た目) |
| `plot_table.csv` | 表頭の属性 (1 行 = 1 地点)．表頭が無ければ書かない |
| `checks.txt` | 機械でできる検査の結果 (標準出力と同じ) |
| `run_info.json` | 段ごとの版と設定 (段階 1・2・3 がそれぞれ書き足す) |

`comp_table_long.csv` の列は次のとおりです．

| 列 | 中身 |
|:---|:---|
| `source_image` | 格子を作った画像 |
| `plot` | 地点番号 (左から 1, 2, …) |
| `row_no` | 格子の行番号 (1 回出現種の行では空) |
| `j_name` / `s_name` | 和名 / 学名 |
| `layer` | 階層 (`B1` `B2` `S` `K` など) |
| `layer_raw` | 階層として読めなかった読み (あるときだけ列ができる) |
| `cover` / `sociability` | 被度 / 群度 |
| `constancy` | 常在度 (群落の要約列のとき) |
| `comp_raw` | 組成のセルの読み (分ける前) |
| `status` | 検証の結果 (下の表) |
| `note` | セルごとの疑わしさ (下の表．複数は `;` でつなぐ) |
| `source` | `body` (表の本体) / `once` (1 回出現種の流し込み) |
| `table` | 1 ページに表が 2 つ以上あるときだけ付く，ページの中の表の番号 (`plot_table.csv` にも付く) |

`export_data.py` は全部の表を縦に積んで `comp_table_long_<tag>.csv` に書き，
次の列を足します (`plot_table_<tag>.csv` にも `corpus`・`table`・`table_in_sheet` が付きます)．

| 列 | 中身 |
|:---|:---|
| `corpus` | `--tag` で渡した資料名 |
| `table` | 作業ディレクトリの名前 (表ごとに 1 つ) |
| `table_in_sheet` | 上の表の `table` (ページの中の表の番号) を名前を変えて残したもの |
| `image` | `Need Check` の行だけ，切り出したセルの画像のフルパス (`--no-crop` のときは空) |

### `status`

| 値 | 意味 |
|:---|:---|
| `OK` | 規則と辞書で検証が通った |
| `Need Check` | 値として読めない形．辞書に無い種名や，短すぎて候補を採らなかった読みは印字のまま残す |
| `suggested` | 種名を辞書の近い名前に寄せた (候補が複数なら `;` でつなぐ．別種に化けうるので目視に回す) |
| `multi` | 階層が複数 (`S;K` など)．誤りとは限らないが目視で確かめる |
| `absent` | 非出現のセル．`build_table.py --keep-absent` のときだけ残る |

### `note`

| 値 | 付ける所 | 意味 |
|:---|:---|:---|
| `interpolated` | `axes.py`・`blocks.align_block_bottoms` | 検出の無い所を内挿した境 (格子の画像で金色) |
| `snapped` | `axes.py` | 字を避けて谷へずらした境 (橙) |
| `on_text` | `axes.py` | ずらしても字に重なったままの境 (赤) |
| `row_fixed` | `row_heights.py` | 行の高さをそろえて直した行 |
| `y_shifted:±N` / `y_followed` / `y_fitted` | `row_track.py` | 列ごとの行のずれを追って動かしたセル |
| `heading` / `name_only` / `legend` | `row_kinds.py` | 見出し・学名だけの行・凡例 (行は落とさない) |
| `flow` | `comp_table.mark_flow` | 表に入り込んだ流し込みの文章 |
| `retry` | `ocr.py` | 読み直して差し替えたセル |
| `roman` | `ocr.py` | ローマ数字として読み直したセル |
| `ndl` | `pipeline/read.py` | NDLOCR-Lite で読めたセル (読みは `text_ndl`) |
| `moved` | `comp_table.py` | 隣のセルにまたがった値を分け直した |
| `sname_from_jname` | `comp_table.py` | 空の学名を和名から補った |
| `hand` | `apps/3_read` | 画面で人が直した |

## テスト

```bash
pip install -r requirements-test.txt   # CI と同じ (重い依存を入れない)
pytest                # 実データの要らないもの (30 秒ほど)
pip install -r requirements-dev.txt    # 検出と読み取りも走らせるとき
pytest --runslow      # 検出と読み取りも実際に走らせる (`slow` の印)
python -m coverage run -m pytest && python -m coverage report   # 網羅の度合い
```

**CI は Python 3.10 と 3.12 で回します**．手元も 3.12 で回すのが確実です
(Windows なら `py -3.12 -m pytest`．既定の `python` の版に rapidfuzz などが
入っていないと，テストを集める段で落ちます)．

**実データが無くても走ります**．辞書と見本 (`examples/`) だけで完結し，
画像の要るものは印字を模した小さな配列を組み立てて確かめます．
実データの格子を使うもの (`test_known_defects.py`・`test_table_find.py` の一部など) は，
環境変数 `COMPTEA_GRIDS`・`COMPTEA_PARTS`・`COMPTEA_SCAN` (`test_row_track.py` は `COMPTEA_WORK`，
`test_table_find.py` の取り置いた目印は `COMPTEA_TF_CACHE`)
で置き場を指したときだけ走り，無ければ飛ばします．

`tests/test_<モジュール名>.py` が，おおむね `comptea/<モジュール名>.py` に対応します．
このほかに，横断で見るものがあります．

- `test_wiring.py` — モジュールをまたぐ参照が実在するか (移し忘れを捕まえる)
- `test_no_local_paths.py` — 文書とコードに手元のパスが無いか
- `test_apps.py`・`test_shared.py`・`test_pipeline.py` — 4 アプリ・工程の受け渡し・見本 1 枚の通し

歯止めにしているのは，**実物を見て決めた判断**です
(`III(+-4)` を `III(1-4)` にしない，1 文字の読みは完全一致だけ採る，など)．
どれも「黙って別の値になる」型で，通してみても気づけません．
経緯は [docs/lessons.md の「採否の記録」](docs/lessons.md#採否の記録) にあります．

## 文書

- [docs/pipeline.md](docs/pipeline.md) — 工程の流れとアルゴリズム
- [docs/architecture.md](docs/architecture.md) — コードの構成と約束事
- [docs/lessons.md](docs/lessons.md) — 規則づくりの原則と，測って決めた採否の表
- [docs/lessons_history.md](docs/lessons_history.md) — その詳しい経緯 (日付順)
- [docs/vegetation_science.md](docs/vegetation_science.md) — 分野の背景 (被度階級・階層・常在度)
- [eval/README.md](eval/README.md) — 物差し (**非公開の資料が要るので，ここからは動きません**)
- [.claude/skills/comptea/](.claude/skills/comptea/) — Claude Code から通しで回すスキル．
  段階ごとの見どころ (`references/checkpoints.md`)，崩れ方と直し方
  (`references/failure-modes.md`)，画像を読むときの約束
  (`references/reading-guide.md`)，読み手に渡す文面
  (`references/read-cells-prompt.md`)．**このリポジトリが正**です

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
