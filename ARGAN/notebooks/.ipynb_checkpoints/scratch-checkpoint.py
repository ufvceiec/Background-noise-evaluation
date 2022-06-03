import glob
import random
path_one = '../dataset/temples/temple_0/*.png'
path_two = '../dataset/temples_ruins/temple_0_ruins_0/*.png'

if __name__ == '__main__':
    path_col_one = glob.glob(path_one)[:10]
    path_col_two = glob.glob(path_two)[:10]

    zipped = list(zip(path_col_one, path_col_two))

    zipped.extend(zipped)
    random.shuffle(zipped)
    print('eof')
