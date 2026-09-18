# コードの構成

comptea のコードがどこに何を置き，どうつながっているかの地図である．
ここにはコードの構成だけを書き，ほかは次の文書に分けてある．

| 知りたいこと | 文書 |
|:---|:---|
| 入れ方・環境変数・CUI と GUI の使い方・出力の列と値 | [README](../README.md) |
| 段ごとのアルゴリズムと規則 (順序・閾値と定数名) | [pipeline.md](pipeline.md) |
| 規則づくりの原則と，測って決めた採否 | [lessons.md](lessons.md) |
| 実験の数字と経緯 (日付順) | [lessons_history.md](lessons_history.md) |
| 物差し (`eval/`) の使い方 | [eval/README.md](../eval/README.md) |
| 分野の用語 (組成表・被度・階層) | [vegetation_science.md](vegetation_science.md) |

## 全体の形

スキャンした組成表の画像から，縦持ちの表 (1 行 = 1 地点 × 1 種) を作る．
工程は **格子 → 読み取り → 組み上げ** の 3 段で，段ごとに人が目で確かめてから先へ進む
(段の中身は [pipeline.md の全体像](pipeline.md#全体像))．

**物体検出 (YOLO11) は領域 (表頭・種名の列・組成部) までにとどめ，行と列の境は
黒画素から決める**．このため，格子を組むモジュールの多くは検出の箱と画像の両方を受け取る．

## ディレクトリ

| 置き場 | 中身 |
|:---|:---|
| `comptea/` | 中核のパッケージ．モジュールどうしは相対 import (`from . import ink`) |
| `comptea/pipeline/` | 段の入口 (`grid.py`・`read.py`・`table.py`) と共通の準備 (`common.py`) |
| `comptea/weights/` | 検出の重み `comptea.pt` (`comptea.WEIGHTS`) |
| `comptea/*.txt` | 種名の辞書 (`j_name.txt`・`s_name.txt`・`js_name.txt`) |
| `cli/` | 段ごとの薄い CUI |
| `apps/` | 段ごとの Streamlit アプリ (`1_split`〜`4_table`) と共通の `_shared.py`．アプリごとに `requirements.txt` を持つ |
| `eval/` | 物差し．**正解ラベルと正解表は公開していない**ので，この repo だけでは回せない |
| `tests/` | 辞書と `examples/` だけで回る試験 |
| `examples/` | 出典を添えて引用した見本の 1 ページ (`sample.jpg`) と，段ごとの結果の zip |
| `.claude/skills/comptea/` | Claude Code から工程を回すスキル (段階の手引き・読み取りの手引き・失敗の型) |
| `.claude/skills/vegtable-to-csv-by-ai/` | コードを使わず AI が画像を読んで CSV にするスキル (手引き・転記表から CSV への変換・検算) |

## 段と入口

同じ段を CUI と GUI から呼ぶ．**段の本体は `comptea/pipeline/` の 1 か所だけ**にあり，
`cli/` と `apps/` はそれを呼ぶだけである (CUI と GUI の違いは
[pipeline.md の「CUI と GUI の対応」](pipeline.md#cui-と-gui-の対応))．

| 段 | 本体 | CUI | GUI |
|:---|:---|:---|:---|
| 切り分け (段の前) | `split_sheet.main` | `python -m comptea.split_sheet` | `apps/1_split` |
| 段階 1 格子 | `pipeline/grid.py` | `cli/run_pipeline.py` | `apps/2_grid` |
| 段階 2 読み取り | `pipeline/read.py` | `cli/run_ocr.py` (目視は `crop_cells.py` → `apply_text.py`) | `apps/3_read` |
| 段階 3 組み上げ | `pipeline/table.py` | `cli/build_table.py` | `apps/4_table` |
| 続きのページ | `grid.save_continuation` (`page_group.py`・`once_page.py`) | `cli/link_pages.py`・`cli/read_once_page.py` | `apps/2_grid` (切り出しまで) |
| まとめて書き出す | — | `cli/export_data.py` | — |

- `pipeline.run(stage, argv)` は段を**呼んだプロセスの中で**動かし，`(終了コード, ログ)` を返す．
  アプリはこれを使う (段ごとに Python を起こすと，そのたびに torch の読み込みを払う)．
- `run_pipeline.py` は名前に反して**段階 1 だけ**を回す．
- 続きのページ (枝番 `xxx-1`・`xxx-2`) の扱いは [pipeline.md の 11 節](pipeline.md#11-続きのページをつなぐ) が正．
  コードの側では，`grid.save_continuation` が表の無い枝番付きのページを止めずに塊を切り出し，
  `link_pages.py` がつないでから解析する．

### 段階 1 の中の順序

`grid.main` の中は次の順である．

1. `split_sheet.check_rotation` (横倒しなら `rotate_page` で 2 通りに回し，`row` が多く出た向きを採る)
2. `detect` (YOLO の検出)
3. `deskew_page` (傾いていれば回して検出し直す)
4. `split_page` (表ごとに分ける)．中は `filters.drop_unsupported_headers` → `blocks.split_tables` →
   `resplit_parts` (切れ目があれば部分画像に切り出してやり直す) → `table_split.split_side_by_side`
5. `widen_tables` (地点の多い表を `strips.detect_wide` で短冊に分けて検出し直す)
6. `build_tables` → 表ごとに `build_one`．`row` が 1 本も無くて失敗した表は，行の閾値を
   `ROW_RETRY_CONF` (10%) に下げて検出し直し，1 回だけやり直す

`build_one` の中は次の順で，**順序に意味がある** (理由は pipeline.md の各節)．

1. `filters.looks_like_fragment` (切れ端なら止める)
2. `_clean_detections`: `table_split.drop_stray_plot_rows`・`drop_stray_name_cols` → `name_col.name_columns_from_ink` → `one_plot.columns_from_ink`
3. `locate.locate_items` → `filters.looks_like_no_table` → `locate.number_cells` → `layer_col.fix_columns`
4. `body_rows.rows_from_body` (行が大きく落ちていたら格子を組み直す)
5. `row_heights.fix_row_heights` → `blocks.align_block_bottoms`
6. `_polish_grid`: `col_edges.fix_column_edges` → `fix_edges_by_crossings` → `align_header_columns` →
   `checks.*` → `row_skew.fix_skew` → `row_track.check_beats`・`fix_offsets` →
   `col_edges.fix_name_layer_edge` → `layer_col.refit_layer_width` → `join_layer_comp_edge` →
   `row_kinds.mark_rows` → `header_lines.shear_header_values`
7. `located.csv` と重ね描きの画像を書く

## モジュールの地図

1 行に 1 つ．「主な関数」は外から呼ばれるもの．

### 入力と切り分け

| モジュール | 受け持ち | 主な関数 |
|:---|:---|:---|
| `split_sheet.py` | 紙面 (画像・PDF) を読み，折り込みを表ごとの画像と注記の画像に切る．検出の `imgsz` を決める | `load_page`・`find_tables`・`blob_boxes`・`split_by_blobs`・`note_boxes`・`check_rotation`・`cut_table`・`auto_imgsz` |
| `deskew.py` | 組成部の左右の黒画素から紙面の傾きを測る | `estimate`・`deskew_to` |

### 検出と領域

| モジュール | 受け持ち | 主な関数 |
|:---|:---|:---|
| `detect.py` | YOLO の結果を表にし，クラスごとのしきい値で絞る | `filter_by_conf`・`yolo_results2csv` |
| `strips.py` | 地点の多い表を短冊に分けて検出し，元の座標へ戻す | `needs_strips`・`detect_wide` |
| `table_split.py` | 1 枚に載る別々の表を検出の手掛かりで見分ける．表頭の外の項目行を捨てる | `split_side_by_side`・`name_band_cuts`・`drop_stray_plot_rows`・`drop_stray_name_cols` |
| `filters.py` | 組んではいけないもの (切れ端・表の無いページ・重複・流し込みの中の行) を外す | `looks_like_fragment`・`looks_like_no_table`・`remove_dup_ranges` |
| `blocks.py` | 紙面を**表** (決してまとめない) と**段** (折り返し) に分ける | `split_tables`・`split_blocks`・`drop_stray_anchors`・`align_block_bottoms` |
| `name_col.py` | 検出されなかった種名の列を黒画素の山から作る | `name_columns_from_ink` |
| `one_plot.py` | 1 調査区を 2 段に組んだ紙面で，階層と被度の列を黒画素から作る | `columns_from_ink` |

### 格子 (行・列・組み立て)

| モジュール | 受け持ち | 主な関数 |
|:---|:---|:---|
| `locate.py` | 上を呼んで**格子を組む本体**．表頭の帯・階層の列・上下端の補いを含む | `locate_items`・`number_cells` |
| `axes.py` | 1 つの軸の境を検出から組み立て，字を避けてずらす | `locate_edges`・`snap_edges`・`ink_profile` |
| `body_rows.py` | 組成部の縦の範囲 (流し込みの手前まで) と，行の刻みと境 | `body_extent`・`body_bottom`・`running_text_top`・`find_gutter`・`body_extent_ink`・`expected_row_edges`・`lattice_rows`・`rows_from_body` |
| `row_heights.py` | 行の高さをそろえ，半分の刻みを直し，上下端を埋める | `fix_row_heights`・`refine_pitch`・`is_half_pitch` |
| `row_skew.py` | 残った傾きをセルの座標だけで直す | `fix_skew` |
| `row_track.py` | 行を単位の連なりとして追い，列ごとの y のずれを直す．拍で行数を検算する | `fix_offsets`・`check_beats`・`clean_rules` |
| `row_kinds.py` | 行の種類 (見出し・学名だけの行・凡例・流し込み) を `note` に付ける | `mark_rows` |
| `col_edges.py` | 列の境 (印字の隙間の格子・字を割らない位置・表頭の列の合わせ・和名と階層の境) | `plot_gaps`・`fix_column_edges`・`fix_edges_by_crossings`・`crossing_counts`・`align_header_columns`・`fix_name_layer_edge` |
| `col_reach.py` | 検出の外 (右端・左端) にある地点の列を足す | `reach_right`・`reach_left` |
| `layer_col.py` | 階層の列を見つけ，組成部の先頭と右端の地点でない列を整理する | `fix_columns`・`find_layer_column`・`refit_layer_width`・`join_layer_comp_edge` |
| `header_lines.py` | 表頭の帯 (項目名の行と値の行) を作り，境を字の隙間へ寄せる | `header_bands`・`bands_from_value_lines`・`bands_from_pairs`・`slant_profile`・`extend_top`・`shear_header_values` |
| `header_cols.py` | 表頭の独文と和文を分ける縦の境 | `split_x` |
| `checks.py` | できた格子を独立した尺度で検査する | `check_grid_rows`・`check_grid_columns`・`check_row_heights`・`check_header_rows`・`check_source_image` |
| `count_plots.py` | 表頭の帯から地点数を数える物差し (検出にも格子にも依らない) | `plots_from_header` |

### 読み取り

| モジュール | 受け持ち | 主な関数 |
|:---|:---|:---|
| `ocr.py` | EasyOCR でセルを読む．字種を絞った読み直し・常在度の頭の数え直しを含む．読み手は初めて使うときに作る | `get_reader`・`ocr_images_df`・`retry_empty_comp`・`fix_roman_heads` |
| `left_rule.py` | 組成部の左端の列の読む箱を，左の縦罫線より右へ押し出す (格子は変えない) | `fit_rule`・`push_first_column` |
| `cell_rule.py` | セルごとに読む箱の端の縦罫線を外す (格子は変えない) | `trim_rules` |
| `read_region.py` | 領域を 1 回読み，位置でセルに割り当てて読み手を重ねる | `read_cells`・`assign`・`pick` |
| `tiles.py` | 大きな紙面を分けて読み，元の座標へ戻す | `read_tiled` |
| `ndl.py`・`yomi.py` | 外の読み手 (NDLOCR-Lite・yomitoku)．置き場は環境変数で指し，無ければ飛ばす | `NdlReader`・`YomiReader` |
| `device.py` | 読み手を動かす装置を 1 か所で決める (`--device` → `COMPTEA_DEVICE` → 自動) | `pick`・`forget` |
| `ink.py` | 二値化と黒画素の測り方 (各段で共有) | `binarize`・`ratio`・`erase_box_lines`・`dashed_rows`・`head_strokes`・`glyph_gate` |

段階 2 の本体 `pipeline/read.py` は，`move_off_rules` (読む箱を罫線から外す)・
`retry_targets` → `retry_cells` → `retry_until_stable` (NDLOCR-Lite での読み直し)・
`recorrect_cells` (読み直さずに補正だけ当て直す) を持つ．

### 補正と組み上げ

| モジュール | 受け持ち | 主な関数 |
|:---|:---|:---|
| `correct_text.py` | 辞書と規則で補正し，その場で検証して `status` を付ける | `correct_cell`・`correct_comp`・`correct_constancy`・`correct_layer`・`correct_name`・`correct_header_value` |
| `download_species_names.py` | 維管束植物和名チェックリストから辞書を作る | `download_species_names` |
| `comp_table.py` | 縦持ちの表に組む．括弧付きのセルの意味を列の多数決で決める | `comp_table`・`column_head_kinds`・`split_comp`・`mark_flow`・`to_wide` (目で見る用) |
| `parse_text.py` | 流し込み (表頭・1 回出現の種・注記) を読み順に並べて解析する | `reading_order`・`parse_header_text`・`parse_once_species`・`parse_site_notes` |
| `plot_table.py` | 表頭から地点ごとの属性の表を組む | `plot_table` |
| `site_notes.py` | 表の下の注記から調査地・調査年月日・出典を差し込む | `apply_notes`・`read_site_info`・`usable_value` |
| `page_group.py` | ファイル名の枝番からページの組を作る | `split_name`・`group_parts` |
| `once_page.py` | 表の無いページから 1 回出現の種の塊を探す | `find_block` |

### 道具と約束の番人

| モジュール | 受け持ち | 主な関数 |
|:---|:---|:---|
| `source.py` | 格子を作ったのと同じ画像を開く (`source_image` 列) | `open_image`・`check` |
| `compare.py` | 2 つの通しを比べる物差し (`eval/compare_runs.py` が使う) | `diff_long`・`boundary_ink`・`boundary_ink_halves` |
| `pipeline/common.py` | 段の準備と，段ごとの版と設定の記録 (`run_info.json`) | `setup`・`workdir`・`write_run_info`・`run_info_line` |
| `draw_rect.py` | 箱を `note` で色分けして描く | `draw_rects_df` |
| `note.py` | セルの `note` に印を足す (`;` 区切り．空・NaN を `nan` にしない) | `add`・`text` |
| `util_file.py` | ファイル操作 (時刻付きの名前・zip) | `now`・`zip_now` |

### 控え (工程につないでいない別案)

| モジュール | 中身 | 使われ方 |
|:---|:---|:---|
| `table_find.py` | 検出器を使わず，タイル分割 OCR の目印で表全体の箱を作る (`find_by_marks`・`check_boxes`) | 工程では使わない．試験だけ |
| `noyolo.py` | 検出器を使わず，粗い区切りと OCR の文字列から部分を推定する (`guess_parts`・`guess_parts_v2`・`guess_layout`) | 工程では使わない．`table_find` が `guess_pitch` だけ借りる |

控えにした理由は [lessons.md の採否の記録](lessons.md#採否の記録) (切り分け・格子) にある．

## 段ごとの版 (正と控え)

同じ仕事に複数の実装がある段の一覧．**「正」だけが工程から呼ばれる**．

| 仕事 | 正 | 控え・分岐 |
|:---|:---|:---|
| 折り込みの切り分け | `split_sheet.find_tables` (空白の帯 + 縮めた塊) | `table_find` (目印)．yomitoku・DocLayout-YOLO のレイアウト解析は候補から外した (コードは無い) |
| 1 枚に載る別々の表 | `blocks.split_tables` (縦) → `grid.resplit_parts` → `table_split.split_side_by_side` (横) | — |
| 格子 | `detect.py` → `locate.py` | `noyolo.py` |
| 格子の分岐 | `one_plot.py` (1 調査区の 2 段組)・`strips.py` (地点の多い表)・`name_col.py` (種名の列の補い) | 条件に当たる表だけで働く |
| 表頭の縦の境 | `header_cols.py` (黒画素) | 検出の `header_col` は領域の左端だけに使う |
| 表頭の行 | `header_lines.bands_from_value_lines` (値の行ごと) | 値の行が足りなければ項目名の箱から (`bands_from_pairs`)，それも足りなければ投影 (`locate._header_bands_from_names`) |
| 組成部の行 | `axes.locate_edges` (`locate_items` の中) → `body_rows.rows_from_body` (`lattice_rows`) → `row_heights` → `row_skew` → `row_track` | `--no-snap`・`--no-track` で後ろを切れる |
| 読み取り | `--reader easyocr` (既定) | `multi`・`ai`・`both` ([pipeline.md の「読み方 (`--reader`)」](pipeline.md#読み方---reader)) |
| 段階 1 の読み手 | EasyOCR だけ (`header_lines` の `reader.detect` など) | — |
| 後処理 (段階 3) | `correct_text` → `comp_table` (`row_kinds` の印を使う) → `plot_table` → `site_notes` | `--no-notes`・`--keep-absent` |

## 検出のクラス

検出器が学習したのは次の 12 クラス (正解ラベルは公開していない)．

| クラス | 中身 |
|:---|:---|
| `row` | 組成部の行 (種) |
| `plot_row` | 表頭の項目行 |
| `col` | 組成部の列 (地点) |
| `sname` | 学名の列 |
| `species_col` | 和名の列 |
| `header` | 表頭 (表の形か流し込み) |
| `table` | 組成表の全体 |
| `layer` | 階層の列 |
| `header_col` | 表頭の項目名の列 |
| `once_species` | 表の下の 1 回出現の種 |
| `plot_no`・`date` | `plot_row` にまとめた (番号は残し，他のクラスの番号をずらさない) |

組成部のセル `comp` は検出クラスではなく，`locate.py` が `col` × `row` から計算する擬似クラスである．
表頭の**列**はラベルを付けていない (`col` が表頭まで伸びる)．表頭の**行**は `plot_row` として付ける
(付けないと，`row` と同じ見かけのものを前景と背景の両方として学ぶ)．

## 物差し (`eval/`) との関係

- `eval/eval_grid.py` は**素の格子** (`locate_items` まで) を正解ラベルと比べる．
  `rows_from_body`・`fix_columns`・上下端の処理は通らないので，そこを変えたら全表を通して
  `eval/compare_runs.py` で置き場ごと比べる．
- `eval/eval_read.py` は読み取りと組み上がった表を正解表と比べる．`scan_blocks.py` は段の丸ごとの
  落ちを数え，`label_gaps.py` は付け漏れた `row` のラベルを補う．
- `eval/make_labels.py`・`build_dataset.py` は，検査を通った格子から教師データ (labelme・YOLO) を作る．
  既存の train / val の分け方は保つ．
- データの置き場は `eval/_data.py` が環境変数で決める ([eval/README.md](../eval/README.md))．

## 設計上の約束

- **手元のパスを持たない**．辞書と重みはパッケージの中を基準に開き (`comptea.data_path`)，
  外の読み手とデータの置き場は環境変数だけで指す (`yomi.DEFAULT_PYS`・`ndl.DEFAULT_DIRS` は空)．
  `tests/test_no_local_paths.py` が字面で見張る．環境変数の一覧は [README](../README.md#環境変数) が正．
- **作業ディレクトリに依らない**．入口は入れたパッケージか repo の中核を import する
  (`sys.path` を足す入口の別は README の「入れ方」)．`common.setup` も `package_dir` から同じことをする．
- **重いものは import 時に読まない**．`comptea/__init__.py` は torch・ultralytics・easyocr を読まず，
  使う関数の中で読む (`ocr.get_reader` など)．依存は extras (`detect`・`read`・`sheet`・`web`) に分けてある (README の「入れ方」)．
- **外の読み手は任意**．入っていなければ黙って飛ばし，工程は EasyOCR だけで進む．
- **座標は画像に紐づく**．格子の座標は `located.csv` の `source_image` が指す画像 (傾きを直した写し・
  部分画像) のもので，突き合わせは必ず `source.open_image` を通す (元画像で代用しない)．
- **失敗は黙らない**．段は `df.attrs['warnings']` で警告を返し，セルごとの疑いは `note` 列 (`;` 区切り) で
  下流へ運ぶ．値の一覧は [README の「出力の形」](../README.md#出力の形) が正．
- **出力は縦持ち** (`comp_table.py`)．`to_wide` は目で見るためだけにある．
- **同じ処理を写さない**．表の切り出しは `split_sheet.cut_table`，装置の選び方は `device.py` に 1 つだけ置く．
  関数の移動や分割は `tests/test_wiring.py` が字面で見張る．
- **結果には版と設定を残す** (`run_info.json`)．
- 資料 (スキャン・ラベル・正解表) は著作権のため公開しない．`examples/sample.jpg` だけを出典付きで置く．
