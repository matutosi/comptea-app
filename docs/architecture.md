# comptea architecture

This file describes the codebase — architecture, commands, conventions. The
narrative of how the rules were arrived at (and which ideas were measured and
dropped) is in [lessons.md](lessons.md); the stage-by-stage walkthrough is in
[pipeline.md](pipeline.md).

## Project Overview

**comptea** (Composite Table to Data Easy) extracts structured data from scanned images of vegetation composition tables found in books and reports. The pipeline: image preprocessing → object detection (YOLO/Detectron2) → region location → OCR → text correction → tabular output.

The project is bilingual (Japanese/English) and targets ecological vegetation survey data.

## Layout

| Directory | What is in it |
|:---|:---|
| `comptea/` | The core modules. What used to be one 1,673-line `split_wide.py` is five files by concern: `strips.py` (wide tables), `col_edges.py` (column boundaries), `table_split.py` (telling two tables apart on one sheet), `body_rows.py` (the body's vertical extent and row boundaries) and `checks.py` (the independent checks). It is a package: modules import each other relatively (`from . import ink`), the dictionaries and weights are resolved against the package directory, and **nothing needs a particular working directory** any more (until 2026-09-07 everything had to run inside `comptea/`) |
| `comptea/web/` | The older all-in-one Streamlit pages, kept as they were |
| `comptea/pipeline/` | The three stages themselves — `grid.py`, `read.py`, `table.py` — with `common.py` shared between them. `pipeline.run(stage, argv)` drives one **in the calling process**: until 2026-09-07 the apps spawned a new Python per stage and paid the torch import (5 s) every time. Measured with the CPU wheel Streamlit Cloud installs, detection peaks at 432 MB and reading at 507 MB, which fits the ~1 GB free tier (the local CUDA build reaches 1,375 MB and is what made this look impossible) |
| `cli/` | Thin command-line entry points over those stages (`run_pipeline` → `run_ocr` → `build_table`, plus `crop_cells`, `apply_text`, `export_data`) |
| `apps/` | One Streamlit app per stage, each with its own `requirements.txt` |
| `eval/` | The yardsticks. **They need the labelled scans and truth tables, which are not published**, so they cannot be run from this repository |
| `tests/` | Runs on the dictionaries and `examples/` alone — no source material needed |

An earlier R implementation and a Detectron2 prototype exist in the project's private
history; neither is included here.

## Pipeline stages

4. **Detection** (`detect.py`): YOLO object detection. Confidence thresholds are per class (see `detect.filter_by_conf()`). The command-line path and the yardsticks all pass `--conf 30 --conf-col 20`: at 30 the `col` of a whole block is lost on some pages (kinki_047's left block scores 0.24), which drops every row in that block. `imgsz` defaults to `auto`, which scales it with
   the longest side so the page meets the detector at the scale it was trained on
   (long side 3300 px ↔ `imgsz` 1280): a book page still resolves to 1280, an
   oversized fold-out to 1856–2560. Feeding a 4809-px-tall table at 1280 costs two
   thirds of its rows
5. **Location** (`locate.py`): Turns the detected `row` / `col` boxes into cell coordinates. A page can hold more than one block — a single-plot table is often set in two columns — so `split_blocks()` divides the page at the species-name columns and builds a grid per block; a page can also hold more than one **table**, stacked vertically, which is a different thing entirely (separate header, separate plots, never merged), and `split_tables()` cuts those apart at the top of each header, the caller keeping one workdir per table; `number_cells()` then numbers rows continuously across blocks and columns within each. Boundaries come from the detections themselves; only the gaps where a row or column went undetected are interpolated (`locate_edges()`). A boundary that lands on text is nudged to the middle of the nearest gap in the ink profile (`snap_edges()`). Removes duplicate overlapping detections and reports what it could not resolve via `df.attrs['warnings']`
6. **OCR** (`ocr.py`): EasyOCR (ja+en) reads text from each located cell region. Images are binarized and trimmed before OCR.
7. **Text correction** (`correct_text.py`): Post-OCR corrections for species names (edit distance against `j_name.txt` / `s_name.txt`), layer codes, and composition values. Each correction validates its own result and returns a `status` (`OK` / `Need Check` / `suggested` / `multi`). A composition cell may also be a **constancy** value — `IV(+-3)`, meaning "constancy IV, cover range +–3" — which appears when the sheet is a synoptic table whose columns are communities rather than plots (2 of the 68 fold-out tables). `correct_constancy()` normalises it (`+`/`r` are legitimate heads for occurrences below constancy I; a missing hyphen inside the brackets is restored; a Roman numeral read inside the brackets is a `1`, since only cover ranges live there; a trailing stray character is the closing bracket misread). What it does **not** do is decide what the cell means: a bracketed cell is `IV(+-3)` (constancy IV, cover range) in a community column and `2(3-4)` (cover 2, sociability range) in a single-plot column, **and one table holds both** — 9 of 22_p2's 25 columns are single plots, 7 of 11_p1's 14. Reading a printed `2` as `II` cost those columns their cover values. `comp_table.column_head_kinds()` votes per column on whether the heads are Roman or Arabic, and `split_comp(kind=…)` fills either `constancy` + cover range or `cover` + `sociability` accordingly; the vote also settles the `1`/`I` confusion the glyphs make unavoidable cell by cell. A table with both kinds is flagged in the warnings
8. **Running-text parsing** (`parse_text.py`, `plot_table.py`): The header of a
   single-plot table and the once-only species list below the table are set as running
   text, so they cannot be cut into rows and columns. The whole region is OCR'd, the
   fragments are re-joined in reading order (`reading_order`), and only then parsed —
   an item or a species name can straddle a line. `plot_table.py` turns the parsed
   header into a separate table, one row per plot, columns being the header items
   (plot number, date, altitude, …). Japanese item names win over the German ones:
   they OCR more reliably
9. **Table assembly** (`comp_table.py`): Builds the long-format table — one row per plot × species

### Object classes detected

The 12 classes the detector was trained on (the labelled dataset itself is not
published; the counts are from it):

| Class | Description | Labels |
|-------|-------------|-------:|
| `row` | Data rows (species) | 675 |
| `plot_row` | Header item rows | 209 |
| `col` | Data columns (plots) | 138 |
| `sname` | Scientific name column | 44 |
| `species_col` | Japanese name column | 43 |
| `header` | Header block (tabular or running text) | 33 |
| `table` | Entire composition table | 25 |
| `layer` | Vegetation layer column | 22 |
| `header_col` | Header item-name column | 22 |
| `once_species` | Listing of species occurring once, below the table | 15 |
| `plot_no`, `date` | Folded into `plot_row` | 0 |

`comp` (a single cell of the composition body) is **not** a detected class: `locate.py`
computes it as a grid from `col` × `row`. The header **columns** are not labelled —
`col` spans the header on the tabular pages — but the header **rows** are, as
`plot_row`: they look like `row` and leaving them unlabelled taught the detector to
read the same thing as both foreground and background. `plot_no` and `date` were the
same object under two more names and are folded in; their ids stay so the rest do not
shift.

### Shared utilities

- `split_sheet.py`: Cuts an oversized fold-out sheet into one image per table, and
  works out the `imgsz` that puts the page back at the scale the detector was trained
  at. A sheet like `s01115` (A0, 9344 × 12873 at 300 dpi) carries two or three separate
  tables and is 16× the area of a book page; fed in whole, the detector finds no rows
  at all. The cut is made at blank bands — **vertically first, then horizontally inside
  each vertical part**, never the other way round, or the gap between the species-name
  column and the composition body would split a single table. Loads PDFs through
  PyMuPDF, lifting the embedded scan rather than re-rendering it
- `strips.py`: Detects a table with far more plots than the detector was trained on
  by cutting the composition body into groups of plots, the species-name columns kept
  at the head of each strip, then mapping the boxes back to the original coordinates so
  `locate.py` never learns the table was wide. Not a scale problem: a 37-plot table
  yields **zero** `row` detections at any `imgsz`, because a row box that wide is
  outside the training distribution — the same table at 8 plots per strip detects 102.
  Strips overlap by one plot so a plot never lands on a strip edge, and when the page
  holds more than one table stacked vertically the strip is cut to that table's height
  (`y_range`), or the strip re-detects the other table and undoes the split. It also splits
  **two tables set side by side** on one sheet (their gutter can be 23 px, narrower
  than the gaps between plots, so `split_sheet.py` cannot see it): two species-name
  columns each with their own header means two tables, whereas one table folded into
  two blocks has a single header. A name column the detector missed altogether
  (sheet 23, the two tables 50 px apart) still shows in the ink: a band of body
  columns three times as dark as the rest is a species-name column, and
  `name_band_cuts()` cuts there. The strips' own `header` / `plot_row` boxes are
  trusted only inside the header band of the whole-table detection
  (`drop_stray_plot_rows(ref=…)`): strips invent headers inside the body, and one
  such header cost a table the bottom quarter of its rows. Whenever a left/right cut
  is found, `run_pipeline.py` does **not** divide the detections: it crops each side
  out of the image (`resplit_parts()`), runs `split_sheet.find_tables()` on the crop
  to catch tables stacked below that a full-height neighbour had hidden from the
  whitespace pass, and re-runs itself on each piece at that piece's own scale
  (workdirs `_s1`, `_s2`, `_s1p2`). `check_grid_columns()` then compares the finished
  grid's column edges against the ink gaps — an independent measure — and warns when
  they disagree, while `fix_column_edges()` rebuilds those edges from the gaps when
  that measurably reduces the ink the boundaries land on (one column too many makes
  the pitch a few px short and every boundary drifts into the values further right:
  22_p2 had 26 columns of 118 px where the print has 25 of 125). It runs **after**
  the rows are settled and keeps the outer edges: swapping columns while the grid is
  being built moved the body's left edge and halved the row count on another table.
  The header can also sit a few tens of px to the side of the body, so
  `locate._shift_edges_to_band()` slides the header's copy of the boundaries (never
  changing their number, or the header items would stop lining up with the plots). The vertical extent of the body (`body_extent()`) is taken from the
  `col` detections at **both** ends, then cut below the header that straddles the top
  and above the once-only species; taking the top from the grid itself left the first
  species group under the header unread in 28 of 68 fold-out tables (~371 rows), and
  since `check_grid_rows()` counted its valleys inside the same extent, the row check
  could not see it. The bottom is where the once-only species' running text
  begins (`body_bottom()`): a band is running text when the column boundaries
  are filled **and** the gutter between the name columns and the body (found from
  the ink of the upper half, `find_gutter()`) is filled — a dense row of values
  fills the boundaries but not the gutter, a group heading fills the gutter but
  not the boundaries, and horizontal rules (long runs of ink, skewed or not) are
  stripped first. That test needs the boundaries to be **countably** filled, which
  a wide table defeats: across 17_p1's 116 plots even running text leaves 62% of
  the boundaries in the gaps between letters, against 80% for the body — a real
  difference, but rescaling it per table cuts at the first run of dense rows
  instead (128 rows lost). What separates them is the **spacing of the letters**
  (`running_text_top()`): dilate each band horizontally by a third of a cell
  width and running text barely grows (1.5–1.7×, its letters already nearly
  touch) while a row of values grows 2.4–5.7× — even 05_p2's `4.3 4.3 …` rows
  grow 2.4×. The list is always at the foot of the table, so the search runs
  **upwards** from the bottom and stops when the run stops being mostly
  text-like; walking down from the top, or allowing a couple of stray bands
  without that density floor, chains through the body (12_p1 lost 152 rows).
  Letter spacing alone is not enough either: a synoptic table's own values
  (`III(+-2)`, eight characters) are as tightly set as prose, and 11_p1 lost
  35 rows of body to it, so a band must **also** fill the gutter between the
  name columns and the body — prose crosses it, the body never does (11_p1's
  body measures 0.000, 17_p1's once-only list 0.54–1.00).
  `fix_edges_by_crossings()` then nudges each interior column boundary to
  where it splits the fewest printed characters. Values are set around a
  centre dot (`2・3`), so a boundary a few px off drops the last glyph into
  the next cell and neither cell reads as a value — 52% of the remaining
  doubtful cells had ink touching a boundary. The objective **counts blobs
  rather than weighing ink**: a body that is nine tenths `・` puts every
  boundary in a relative valley (17_p1 scored 0.56 of the median) while still
  cutting the rare values. The room to move is a twelfth of a column: over
  49 tables, ±0.08 of the pitch un-splits 2,337 values at the cost of 135
  moving to a neighbouring plot, where ±0.22 un-splits 3,462 but moves 962
  (21_p3's `III(+-4)` shifted from plot 13 to 14). Below 12 columns the rule
  is off entirely — kinki_047 has two interior boundaries and 90-px columns,
  and free movement cost it its perfect truth-table score (1.000 → 0.625) Rows, when the detector's `row` boxes fall short, are rebuilt
  from the fact that **row height is constant within a table**
  (`expected_row_edges()`): candidate pitches are the autocorrelation peaks of
  the ink profile, measured on the body (dashed box borders masked out, or
  they add a half period) and on the name columns separately — either one
  alone can double: the body when it is all `・`, the names when a species
  spans two layer rows and the name is printed only on the first. A peak
  counts only if it stands out from the flat baseline (prominence ≥ 0.15 —
  a `・`-only body has no period at all and its "peaks" are noise), the
  shortest pitch within 0.1 of the most prominent wins, and a table shorter
  than 12 rows keeps the detector's row height (too little signal). The grid
  then walks down at that pitch, snapping each edge to the nearest ink
  minimum of the **body** (values sit a few px off the names in typewritten
  tables; the names are used only on rows where the body is blank). Scoring
  candidates by the ink under their edges was tried and dropped — a longer
  pitch has more freedom to find a gap and always looks better.
  Ink valleys alone
  over-split (a dirty gap yields three valleys) and under-split (a run of
  `・`-only rows is one valley). `drop_stray_plot_rows(heads=False)` runs on every table, not only
  the strips: header item rows are detected all down the body, and left in they cut
  the body's rows a second time as header values and widen the header band until the
  layer column (blank header) can no longer be told apart. `check_row_heights()`
  (coefficient of variation of row heights) and `check_header_rows()` (header rows
  inside the body) are the two checks that catch the failure modes the row and column
  counts both pass
- `deskew.py`: Fold-out sheets are scanned up to 1° off (9 of 68 tables shift
  by 0.3–1.0 row heights between the left and right end of the body), and a
  horizontal row boundary then cuts values at one end whatever the pitch. The
  angle is the cross-correlation lag between the row-ink profiles of the left
  and right third of the **body** (measured on the page as a whole, or on the
  `col` boxes with the header inside, the estimate can even flip sign), and the
  page is rotated and re-detected only when the shift exceeds 0.3 rows — an
  angle threshold fired on 17 book pages and cost one of them two columns.
  Coordinates downstream refer to the rotated copy `<work>_deskew.png`
- `ocr.py`: a composition cell whose reading does not validate as a value is
  re-read with the character set narrowed to `0-9 + r ・` (typewritten
  `2.2` reads as `ムっム` otherwise; 08_p2 went from 35 to 7 unreadable
  values), the replacement flagged `retry` in `note` so it reaches the
  reviewer — narrowing can also turn `+` into `4`. A cell that still fails
  is read once more with brackets and Roman numerals added
  (`CONSTANCY_ALLOW`): without them a bracketed `2(3-4)` came back as
  `36+23`, and not one of 22_p2's 451 doubtful cells could be recovered.
  The narrow set is tried first, so a table without brackets reads exactly
  as before; 22_p2 went from 451 doubtful cells to 124. Box borders crossing a cell
  are erased first, **horizontal lines only**: erasing thin vertical lines took
  the typewritten `1` with them
- `name_col.py`: When `sname` or `species_col` is missing — either both (3 tables) or
  just one (28 of the 68 fold-out tables, which is why one sheet had no Japanese names
  at all) — builds the missing box from the ink left of the body: the scientific-name and Japanese-name columns
  are two hills in the column-wise ink profile with a valley between them that never
  reaches zero (long names and group headings straddle the gutter, so a blank
  threshold cannot split them). Left is always the scientific name — true of all 33
  labelled pages — and the valley matched the detector's own boundary within 60 px on
  every table that had one
- `eval_grid.py` measures the raw grid (`locate_items` only) against the labelled
  pages; it does not run `rows_from_body`, `fix_columns` or the extent logic, so a
  change there must be measured by running `run_pipeline.py` over the 88 pages and
  comparing the work directories
- `count_plots.py`: The column yardstick for fold-outs — counts the plots from the
  header band, taking the most evenly spaced row of ink blobs (the running plot-number
  row), and compares that against the grid's column count. Independent of both the
  detections and the ink gaps the grid is built from
- `locate.looks_like_fragment()`: A cut-out with neither a header nor a species-name
  column (a sheet's corner label, a title and legend band) is stopped before it grows a
  grid from a few stray `row` / `col` boxes and is counted as a table
- `layer_col.py`: Finds the layer column when it sits inside the composition body
  rather than in the gap `locate._guess_layer_column()` looks in. The tell is the
  header: plot number, date and altitude fill every plot column and leave the layer
  column blank — 0.07–0.16 of the other columns' ink against 0.85 and up for a real
  plot. Retyping those cells as `layer` is enough; `comp_table.py` re-ranks the
  remaining columns, so plot numbering fixes itself. `fix_columns()` walks the
  leading body columns in order and also throws out what is not a plot column at
  all: a column sitting on a species-name box (or three times wider than the
  rest), and the blank gutter between the names and the body (blank header *and*
  blank body); a blank-header column that does carry ink is the layer column,
  even when a gutter column precedes it
- `make_labels.py` / `build_dataset.py`: Turn a finished grid back into labelme and
  YOLO annotations, cut into page-sized tiles, so a new source can be trained on
  without labelling it by hand. Only tables whose checks pass are used, and the
  existing train/val split is preserved so before/after comparisons stay honest
- `util_file.py`: File operations (timestamped names, zip, directory management)
- `preprocess_image.py` / `preprocess_image_web.py`: Image preprocessing (deskew, grayscale, binarization, noise removal)
- `crop_image.py`: Cuts the located regions out of the page image
- `draw_rect.py`: Draws the boxes, colouring each by its `note` (`on_text` red /
  `snapped` orange / `interpolated` gold — darkest first, most worth looking at first)
- `progress.py`: Redirects stdout into a Streamlit widget so long runs show progress
- `download_species_names.py`: Fetches the Japanese/scientific name list from the
  Vascular Plant Japanese Name Checklist (ver. 1.10), the dictionary `correct_text.py`
  matches against

### Command line (`cli/`)

The same pipeline driven from the command line rather than Streamlit, for reading a
table end to end in one go: `run_pipeline.py` (the whole run), `crop_cells.py`,
`run_ocr.py`, `apply_text.py`, `build_table.py`, with `_common.py` shared between them.
Three stages — the grid, the reading, the assembled table — are rendered as images to
be eyeballed before the run continues; `docs/references/` holds the stage guide, the
reading guide, and the known failure modes. OCR is EasyOCR-led.

## Running it

```bash
python cli/run_pipeline.py <image> --workdir work/<name>   # the grid
python cli/run_ocr.py work/<name>                          # read the cells
python cli/build_table.py work/<name>                      # assemble

streamlit run apps/2_grid/streamlit_app.py                 # or one stage in the browser
```

The entry points import the package — installed (`pip install -e .`) or, failing
that, from this repository — and can be called from anywhere without changing the
working directory (`COMPTEA_CORE` overrides where the core is looked for).

## Key dependencies

ultralytics (YOLO11), streamlit, easyocr, python-Levenshtein, rapidfuzz, PIL/Pillow,
pandas, numpy, torch, opencv, PyMuPDF — pinned in `requirements.txt` (whole pipeline)
and, per stage, in `apps/*/requirements.txt`.

## Notes

- Output is **long format**: one row per plot × species (`comp_table.py`). `to_wide()` exists only for eyeballing.
- Stages report trouble through `df.attrs['warnings']` rather than failing silently — `locate.py` and `comp_table.py` both do this, and the Streamlit pages display it.
- Per-cell doubts travel in a `note` column set by `locate.py` (`interpolated` / `snapped` / `on_text`), drawn in colour on the grid overlay and carried through OCR into the long table.
- Domain background (composition tables, cover-abundance classes, layer codes) is in `docs/vegetation_science.md`. Read it before looking up vegetation-science terms elsewhere.
- YOLO model weights ship with the package at `comptea/weights/comptea.pt` (`comptea.WEIGHTS`)
- `comptea` is an ordinary package; `pip install -e .` makes it importable, and `cli/_common.setup()` falls back to the repository path when it is not installed
- The source material (scans, labels, truth tables) is not published, for copyright; `examples/sample.jpg` is a single page quoted with its source
- GPU (CUDA) is optional for inference but recommended for training
