# 出力する CSV の仕様

出力は **2 つの縦持ちの CSV** (UTF-8 BOM 付き．Excel で文字化けしない)．
表ごとに項目が違っても列が崩れないよう，どちらも 1 行 = 1 つの値にする．

| ファイル | 1 行 | 中身 |
|:---|:---|:---|
| `sites.csv` | 表 × 地点 × 項目 | 地点の情報 (表頭・注記) と表の情報 |
| `composition.csv` | 表 × 地点 × 種 × 階層 | 組成 (本体と 1 回出現の種) |

**種の情報 (科名など) は書かない**．学名・和名は**印字のまま**
(旧名・誤植も直さない．直したくなったら `note` に書く)．

## 共通の列

| 列 | 中身 |
|:---|:---|
| `file` | 入力のファイル名 (拡張子なし)．続きのページをつないだときは表のあるページ |
| `table` | 表の番号．印字の `Tab. 34` を `Tab.34` の形で (空白なし) |
| `plot` | 地点の番号．表頭の**通し番号** (`Lfd. Nr.`) をそのまま．無ければ左から 1, 2, 3 … 常在度表では列の番号 (群落の番号) |
| `value_raw` | 印字のまま (直す前) の文字列 |
| `status` | `ok`・`check` (読みに自信が無い・規則に合わない・決められない)・`multi` (composition だけ．値は確かだが階層が `S;K` のまま 1 つに決まらない) |
| `note` | `check` の理由，括弧付き，下線，印字の誤植，続きのページから取った (`続き: <ファイル名>`) など．短く |

## sites.csv

| 列 | 中身 | 例 |
|:---|:---|:---|
| `item_ja` | 項目名 (和文) | `海抜高` |
| `item_de` | 項目名 (独文．単位の括弧は除く) | `Höhe ü. Meer` |
| `value` | 値 (単位を除き，形をそろえる) | `640`・`SS-67`・`1983-08-03`・`SE` |
| `unit` | 単位 | `m`・`m2`・`%`・`°`・`cm` |
| `source` | `header` (表頭) か `note` (表の下の注記) か `title` (表題) | |

- **表の情報** (表題・群落名) は `plot` を空にして書く．
  `item_ja` = `表題`，`item_de` = `Titel`，`value` = 和文の表題，`value_raw` = 独文も含めた全体．
- **群落区分** (`Spalte`) は地点ごとに `item_ja` = `群落区分` として書く (値は `1`・`2`・`b1` など)．
- 注記は地点ごとに分ける: `調査地` (`Lage d. Aufn.`)，`調査年月日` (`Datum d. Aufn.`)，
  `出典` (`Nachweis`)．全地点に共通の注記も，地点ごとの行に展開してある．
- 表頭の通し番号は `plot` 列そのものなので，項目としては書かない．
- 出現種数は item_ja を `出現種数`，item_de を `Artenzahl` にそろえる．
- 日付は `YYYY-MM-DD`．日が無ければ `YYYY-MM`．
- 項目名の和文・独文が片方しか無いときは，無い方を空にする．

例:

```csv
file,table,plot,item_ja,item_de,value,unit,value_raw,source,status,note
kinki_010,Tab.34,,表題,Titel,ハマゴウ―アキグミ群落(1)，ハマゴウ―ハイネズ群集(2)およびハマエンドウ―テリハノイバラ群落(3),,Tab. 34. ハマゴウ―アキグミ群落(1)… Vitex rotundifolia-…,title,ok,
kinki_010,Tab.34,1,群落区分,Spalte,1,,1,header,ok,
kinki_010,Tab.34,1,調査番号,Feld-Nr.,SS-67,,SS 67,header,ok,
kinki_010,Tab.34,1,調査面積,Größe d. Probefläche,2,m2,2,header,ok,
kinki_010,Tab.34,9,調査面積,Größe d. Probefläche,0.5,m2,0. 5,header,ok,
kinki_010,Tab.34,1,出現種数,Artenzahl,7,,7,header,ok,
kinki_010,Tab.34,1,調査地,Lage d. Aufn.,熊野郡久美浜町湊宮,,Lfd. Nr. 1，3：Kumihama-cho，Kumano-gun 熊野郡久美浜町湊宮 (6. Juli 1983),note,ok,
kinki_010,Tab.34,1,調査年月日,Datum d. Aufn.,1983-07-06,,6. Juli 1983,note,ok,
```

## composition.csv

| 列 | 中身 | 例 |
|:---|:---|:---|
| `group_ja` | 見出しの行 (和文)．1 回出現の種は空 | `群落区分種` |
| `group_de` | 見出しの行 (独文) | `Trennart d. Gesellschaft` |
| `s_name` | 学名 (印字のまま．`var.` `f.` も) | `Lotus corniculatus var. japonicus` |
| `j_name` | 和名 (カタカナ．印字のまま) | `ミヤコグサ` |
| `layer` | 階層 (印字のまま．複数は `;`)．列が無い表は空 | `B1`・`S;K` |
| `cover` | 被度 `5 4 3 2 1 + r`．常在度表では被度の範囲 (`+-3`) | `+` |
| `sociability` | 群度 `1`〜`5`．**省略された群度 1 は補って `1`** | `1` |
| `constancy` | 常在度 `I`〜`V` (常在度表だけ) | `IV` |
| `source` | `body` (表の本体) か `once` (1 回出現の種) | |

- **非出現 (`・`・空欄) の行は書かない**．
- 同じ種が複数の階層に出るときは，階層ごとに行を分ける (値のある行だけ)．
- 例:

```csv
file,table,plot,group_ja,group_de,s_name,j_name,layer,cover,sociability,constancy,value_raw,source,status,note
kinki_010,Tab.34,1,群落区分種,Trennart d. Gesellschaft,Elaeagnus umbellata,アキグミ,,5,4,,5・4,body,ok,
kinki_010,Tab.34,5,群落区分種,Trennart d. Gesellschaft,Elaeagnus umbellata,アキグミ,,+,2,,+・2,body,ok,
kinki_010,Tab.34,9,群落区分種,Trennart d. Gesellschaft,Rosa wichuraiana,テリハノイバラ,,5,5,,5・5,body,ok,
kinki_010,Tab.34,8,群落区分種,Trennart d. Gesellschaft,Cocculus orbiculatus,アオツヅラフジ,,+,2,,+・2,body,ok,
kinki_010,Tab.34,9,群落区分種,Trennart d. Gesellschaft,Cocculus orbiculatus,アオツヅラフジ,,+,1,,+,body,ok,
kinki_010,Tab.34,1,,,Artemisia princeps,ヨモギ,,1,2,,1・2,once,ok,
kinki_010,Tab.34,6,,,Hemerocallis fulva var. littorea,ハマカンゾウ,,1,1,,1・1,once,ok,
kinki_010,Tab.34,4,随伴種,Begleiter,Ischaenum anthephoroides,ケカモノハシ,,+,2,,+・2,body,ok,印字は Ischaenum (Ischaemum の誤植)
```

## 置き場

1 回の依頼 (入力の束) につき 1 つの出力ディレクトリにまとめる．
既定は `<入力のあるディレクトリ>/vegtable_csv/`．

```
vegtable_csv/
  sites.csv
  composition.csv
  work/            ← ページ画像・見渡し・切り出し (消してよい)
  reading_log.md   ← 紙面の型・何を読んだか・迷った点・check の理由
```

表が複数あっても CSV は 1 組にまとめ，`table` 列で分ける．
