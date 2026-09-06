"""段階1: 検出と格子を作る

    python cli/run_pipeline.py ...

中身は `comptea.pipeline.grid`．CLI とアプリで同じものを呼ぶ．
"""
import os
import sys

try:
    from comptea.pipeline import grid
except ImportError:                                  # 入れていないとき
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from comptea.pipeline import grid

if __name__ == '__main__':
    sys.exit(grid.main())
