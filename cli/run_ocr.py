"""段階2: セルを読む

    python cli/run_ocr.py ...

中身は `comptea.pipeline.read`．CLI とアプリで同じものを呼ぶ．
"""
import os
import sys

try:
    from comptea.pipeline import read
except ImportError:                                  # 入れていないとき
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from comptea.pipeline import read

if __name__ == '__main__':
    sys.exit(read.main())
