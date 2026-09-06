import pandas as pd
from PIL import Image, ImageOps
import os
import re


def crop_images_df_path(path):
    """
    crop_images_dfのラッパー
    
    dataframeのパスをもとに画像を抽出
    """
    df = pd.read_csv(path)
    crop_images_df(df)


def crop_images_df(df):
    """
    crop_imageのラッパー
    """
    for index, row in df.iterrows():
        crop_image("images/example.jpg", row["x1"], row["y1"], row["x2"], row["y2"], row["obj_name"])

def crop_image(input_path, x1, y1, x2, y2, obj="", save_image = False):
    """
    指定された座標に基づいて画像を切り取り、新しいファイルとして保存します。

    Args:
        input_path (str): 元の画像のパス。
        x1 (int): 切り取り領域の左上のx座標。
        y1 (int): 切り取り領域の左上のy座標。
        x2 (int): 切り取り領域の右下のx座標。
        y2 (int): 切り取り領域の右下のy座標。
        obj (str): オブジェクトの名称

    Returns:
        str or None: 切り取った画像を保存したパス。エラーが発生した場合は None を返します。
    
    Examples:
    image_path = "images/example.jpg"
    x1, y1, x2, y2 = 300, 300, 800, 800
    output_path = crop_image(image_path, x1, y1, x2, y2)
    """
    try:
        img = Image.open(input_path)
        box = (x1, y1, x2, y2) # crop range: left, top, right, bottom
        cropped_img = img.crop(box)
        # output path
        directory, filename = os.path.split(input_path)
        name, ext = os.path.splitext(filename)
        output_filename = f"croped_{name}_{obj}_{x1}_{y1}_{x2}_{y2}{ext}"
        output_path = os.path.join(directory, output_filename)
        cropped_img.save(output_path)
        return output_path
    except FileNotFoundError:
        print(f"エラー: 指定されたパスにファイルが見つかりません。- {input_path}")
        return None
    except Exception as e:
        print(f"エラー: 画像の処理中に問題が発生しました。- {e}")
        return None

