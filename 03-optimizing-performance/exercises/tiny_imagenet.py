# wget http://cs231n.stanford.edu/tiny-imagenet-200.zip

import os
from os import listdir
from os.path import join, isfile, isdir, basename
from shutil import copyfile

input_dir = 'C:/DATA/Projects/DataSets/tiny-imagenet-200'
output_dir = 'C:/DATA/Projects/DataSets/tiny-imagenet-200/output'


def dir_content(path_dir, item_type='all'):
    if item_type == 'all':
        return [join(path_dir, f) for f in listdir(path_dir)]
    elif item_type == 'file':
        return [join(path_dir, f) for f in listdir(path_dir) if isfile(join(path_dir, f))]
    elif item_type == 'dir':
        return [join(path_dir, f) for f in listdir(path_dir) if isdir(join(path_dir, f))]


os.mkdir(output_dir)

with open(output_dir + '/annotation.txt', 'w') as annotate:
    class_dirs = dir_content(input_dir + '/train', item_type='dir')
    for i, d in enumerate(class_dirs):
        f = dir_content(d + '/images', item_type='file')[0]
        annotate.write(basename(f) + ' ' + str(i) + '\n')
        dest = output_dir + '/' + basename(f)
        copyfile(f, dest)
        print(f'{basename(d)}: {dest}')
