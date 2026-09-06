"""段階3: 縦持ちに組んで検査する

    python cli/build_table.py ...

中身は `comptea.pipeline.table`．CLI とアプリで同じものを呼ぶ．
"""
import os
import sys

try:
    from comptea.pipeline import table
except ImportError:                                  # 入れていないとき
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from comptea.pipeline import table

if __name__ == '__main__':
    sys.exit(table.main())
