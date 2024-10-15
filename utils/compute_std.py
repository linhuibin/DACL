import numpy as np
import os


def round_up(value):
    # 替换内置round函数,实现保留2位小数的精确四舍五入
    return round(value * 100) / 100.0


def compute_std(path):
    with open(os.path.join(path, 'done.txt'), mode="a+") as f:  # a+ 追加读，和追加写，a只追加写
        f.seek(0)  # 指针默认在末尾，归为到文件头
        b = []
        lines = f.readlines()
        for line in lines:
            if line.isspace():
                continue
            else:
                if line.startswith('target acc'):
                    line = line.strip()  # 去掉换行符
                    a = float(line.split(":")[-1]) * 100
                    b.append(a)
        # 求均值
        arr_mean = round_up(np.mean(b))
        # 求标准差
        arr_std = round_up(np.std(b, ddof=1))
        f.write("mean_std: %.2f" % arr_mean)
        f.write('+')
        f.write("%.2f\n" % arr_std)
    f.close()


if __name__ == '__main__':
        compute_std('/mlspace/DeepDG/scripts/domainnet_resnet50/env5/SAGM_DG/')
