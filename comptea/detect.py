from pathlib import Path
import pandas as pd
import util_file

# 検出が0件のときに返る列
EMPTY_COLS = ['obj_class', 'obj_name', 'confidence', 'x1', 'y1', 'x2', 'y2']

def to_pandas(df):
    """`results[0].to_df()` の返りを pandas にそろえる

    **ultralytics の版で中身が変わる**(2026-09-06 に Streamlit Cloud で発覚)．
    8.3.94 は pandas を返すが，新しい版は **polars** を返すので，
    そのままだと `'Series' object has no attribute 'apply'` で落ちる．
    `to_dicts()` を通して pandas に直す
    (`to_pandas()` は pyarrow を要るので使わない)．
    """
    if isinstance(df, pd.DataFrame):
        return df
    if hasattr(df, 'to_dicts'):
        return pd.DataFrame(df.to_dicts())
    return pd.DataFrame(df)


# separate box column into 4 columns
def split_box_column(df):
    df = to_pandas(df)
    # 検出が0件だと to_df() は列の無い空のDataFrameを返す．
    # そのまま進むと 'box' が無くて KeyError になり，
    # 「1件も検出できなかった」ことが読み取れない例外で落ちる．
    # 手元の88枚のうち6枚がこれだった(2026-08-24)
    if len(df) == 0 or 'box' not in df.columns:
        return pd.DataFrame(columns=EMPTY_COLS)
    df['box'] = df['box'].apply(str) # convert from dict to str
    df[['x1', 'y1', 'x2', 'y2']] = df['box'].str.split(',', expand=True)
    df['x1'] = df['x1'].str.replace("{'x1': ", '')
    df['y1'] = df['y1'].str.replace("'y1': " , '')
    df['x2'] = df['x2'].str.replace("'x2': " , '')
    df['y2'] = df['y2'].str.replace("'y2': " , '')
    df['y2'] = df['y2'].str.replace("}", '')
    df = df.drop(columns=['box'])
    # round and convert x1, y1, x2, y2 to float
    df[['x1', 'y1', 'x2', 'y2']] = df[['x1', 'y1', 'x2', 'y2']].apply(lambda x: round(x.astype(float)).astype(int))
    df = df.rename(columns={'name'        : 'obj_name'})
    df = df.rename(columns={'class'       : 'obj_class'})
    df = df.rename(columns={'source_image': 'file_name'})
    return df

def filter_by_conf(detection_results, conf_default, conf_by_class=None):
    """クラスごとの信頼度でYOLOの検出結果を絞り込む

    クラスによって検出の信頼度の出方が違うため，クラス別に閾値を変えられるようにする．
    予測時にはこの中で一番低い閾値を使い，ここでクラスごとに切り直す．

    Args:
        detection_results: YOLOの検出結果(Resultsのリスト)
        conf_default     : 既定の閾値(0.0-1.0)
        conf_by_class    : クラス名をキー，閾値を値にした辞書．例 {'col': 0.3}
    Returns:
        絞り込んだ検出結果(Resultsのリスト)
    """
    if not conf_by_class:
        return detection_results
    filtered = []
    for res in detection_results:
        names = res.names
        keep = [
            i for i, (cls, conf) in enumerate(zip(res.boxes.cls.tolist(), res.boxes.conf.tolist()))
            if conf >= conf_by_class.get(names[int(cls)], conf_default)
        ]
        filtered.append(res[keep])
    return filtered

def yolo_results2csv(detection_results, source_image, model_path, output_dir):
    """YOLOの物体検出結果をCSVファイルに保存
    Args:
        detection_results: YOLOの検出結果オブジェクト
        source_image     : 元画像のパスまたはオブジェクト
        model_path       : YOLOモデルのパス
        output_dir       : CSVの保存ディレクトリ
    """
    if not isinstance(source_image, str):
        source_image = source_image.name
    df = detection_results[0].to_df(decimals=2)
    df = split_box_column(df)
    df['source_image'] = source_image.replace('\\', '/')
    df['model'] = Path(model_path).name
    yolo_result_csv = f'yolo_results_{util_file.now()}.csv'
    df.to_csv(Path(output_dir, yolo_result_csv), index=False)
    return yolo_result_csv
