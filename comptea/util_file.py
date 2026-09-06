import glob
import os
from datetime import datetime
from pathlib import Path
from shutil import make_archive, rmtree

def renew_dir(dir):
    if Path(dir).exists():
        rmtree(dir)
    Path.mkdir(dir)
    return dir

def now():
    return datetime.now().strftime('%Y-%m-%d-%H-%M-%S')

def zip_now(to_be_zipped_dir, zip_name = "zipped_file"):
    zip_base = f'{zip_name}_{now()}'
    to_be_zipped_dir = Path(to_be_zipped_dir)
    make_archive(zip_base, format='zip', root_dir=to_be_zipped_dir)
    zip_file = f'{zip_base}.zip'
    return zip_file

def tree(path, layer=0, is_last=False, indent_current='　'):
    if not Path(path).is_absolute():
        path = str(Path(path).resolve())
    current = path.split('/')[::-1][0]
    if layer == 0:
        print('<'+current+'>')
    else:
        branch = '└' if is_last else '├'
        print('{indent}{branch}<{dirname}>'.format(indent=indent_current, branch=branch, dirname=current))
    paths = [p for p in glob.glob(path+'/*') if os.path.isdir(p) or os.path.isfile(p)]
    def is_last_path(i):
        return i == len(paths)-1
    for i, p in enumerate(paths):    # 再帰的に表示
        indent_lower = indent_current
        if layer != 0:
            indent_lower += '　　' if is_last else '│　'
        if os.path.isfile(p):
            branch = '└' if is_last_path(i) else '├'
            print('{indent}{branch}{filename}'.format(indent=indent_lower, branch=branch, filename=p.split('/')[::-1][0]))
        if os.path.isdir(p):
            tree(p, layer=layer+1, is_last=is_last_path(i), indent_current=indent_lower)


if __name__ == "__main__":

    wd = Path(__file__).parent
    tmp_dir = Path(wd, "tmp")

    zipped = zip_now(tmp_dir, zip_name = "zipped_file")
    print(zipped)

