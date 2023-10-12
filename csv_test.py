# -*- coding: utf-8 -*-
# @Time    : 2023/10/12 14:37
# @Author  : KuangRen777
# @File    : csv_test.py
# @Tags    :
import csv

# 定义要写入的数据
data_to_write = [
    ["姓名", "年龄", "城市"],
    ["小明", 25, "北京"],
    ["小红", 30, "上海"],
    ["小李", 28, "广州"]
]

# 指定CSV文件的名称
csv_file_name = "test.csv"

# 逐行写入数据，每次增加一行都打开并关闭文件
for row in data_to_write:
    with open(csv_file_name, mode="a", newline="", encoding="gbk") as file:
        writer = csv.writer(file)
        writer.writerow(row)

print(f"{len(data_to_write)} 行数据已成功写入 {csv_file_name} 文件。")

