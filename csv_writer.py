# -*- coding: utf-8 -*-
# @Time    : 2023/10/13 13:11
# @Author  : KuangRen777
# @File    : csv_writer.py
# @Tags    :
import csv
from datetime import datetime


def csv_writer(
        csv_file_path: str,
        new_row: list,
        current_time: bool) -> None:
    if current_time:
        # 获取当前时间并将其格式化为字符串
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        new_row.append(current_time)

    # 打开 CSV 文件以追加数据
    with open(csv_file_path, mode='a', newline='', encoding='gbk') as file:
        writer = csv.writer(file)

        # 将新行写入 CSV 文件
        writer.writerow(new_row)

    print(f"[INFO] {new_row} has successfully add into {csv_file_path}.")
