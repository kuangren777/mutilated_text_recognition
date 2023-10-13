# -*- coding: utf-8 -*-
# @Time    : 2023/10/13 13:11
# @Author  : KuangRen777
# @File    : csv_writer.py
# @Tags    :
import csv


def csv_writer(
        csv_file_path: str,
        new_row: list,
):
    # 打开 CSV 文件以追加数据
    with open(csv_file_path, mode='a', newline='', encoding='gbk') as file:
        writer = csv.writer(file)

        # 将新行写入 CSV 文件
        writer.writerow(new_row)

    print(f"[INFO] {new_row} has successfully add into {csv_file_path}.")
