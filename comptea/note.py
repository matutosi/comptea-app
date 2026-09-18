"""セルの `note` (気になった点を `;` でつないだ文字列) を扱う

`str(v or '')` の形は，CSV から読んだ空の note (NaN) を真とみなして
`'nan'` という文字にしてしまう (`nan;ndl` のような note が 398 件あった．2026-09-18)．
note を読んだり足したりするときは，ここを通す．
"""
import math


def text(v):
    """セルの値を文字列にする．None・NaN は ''"""
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return ''
    return str(v)


def add(old, tag):
    """note に印を 1 つ足す．区切りは `;`．すでにあれば足さない"""
    parts = [p for p in text(old).split(';') if p]
    if tag in parts:
        return ';'.join(parts)
    return ';'.join(parts + [tag])
