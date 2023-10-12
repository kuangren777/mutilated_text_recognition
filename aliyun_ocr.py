# -*- coding: utf-8 -*-
# @Time    : 2023/10/12 16:39
# @Author  : KuangRen777
# @File    : aliyun_ocr.py
# @Tags    :
# -*- coding: utf-8 -*-
# This file is auto-generated, don't edit it. Thanks.
import os
import sys
import read_ini
import json
import csv

from typing import List

from alibabacloud_tea_openapi.client import Client as OpenApiClient
from alibabacloud_tea_openapi import models as open_api_models
from alibabacloud_darabonba_stream.client import Client as StreamClient
from alibabacloud_tea_util import models as util_models

ini = read_ini.get_config()

# 指定CSV文件的名称
csv_file_name = "predict_accuracy.csv"

# 从JSON文件中加载字典
with open('my_dict.json', 'r') as file:
    loaded_dict = json.load(file)


def count_chinese_characters(text, character):
    count = 0
    for char in text:
        if char == character:
            count += 1
    return count


class Sample:
    def __init__(self):
        pass

    @staticmethod
    def create_client(
            access_key_id: str,
            access_key_secret: str,
    ) -> OpenApiClient:
        """
        使用AK&SK初始化账号Client
        @param access_key_id:
        @param access_key_secret:
        @return: Client
        @throws Exception
        """
        config = open_api_models.Config(
            # 必填，您的 AccessKey ID,
            access_key_id=access_key_id,
            # 必填，您的 AccessKey Secret,
            access_key_secret=access_key_secret
        )
        # Endpoint 请参考 https://api.aliyun.com/product/ocr-api
        config.endpoint = f'ocr-api.cn-hangzhou.aliyuncs.com'
        return OpenApiClient(config)

    @staticmethod
    def create_api_info() -> open_api_models.Params:
        """
        API 相关
        @param path: params
        @return: OpenApi.Params
        """
        params = open_api_models.Params(
            # 接口名称,
            action='RecognizeGeneral',
            # 接口版本,
            version='2021-07-07',
            # 接口协议,
            protocol='HTTPS',
            # 接口 HTTP 方法,
            method='POST',
            auth_type='AK',
            style='V3',
            # 接口 PATH,
            pathname=f'/',
            # 接口请求体内容格式,
            req_body_type='json',
            # 接口响应体内容格式,
            body_type='json'
        )
        return params

    @staticmethod
    def main(index: int) -> (int, int, str):
        # 请确保代码运行环境设置了环境变量 ALIBABA_CLOUD_ACCESS_KEY_ID 和 ALIBABA_CLOUD_ACCESS_KEY_SECRET。
        # 工程代码泄露可能会导致 AccessKey 泄露，并威胁账号下所有资源的安全性。以下代码示例使用环境变量获取 AccessKey 的方式进行调用，仅供参考，建议使用更安全的 STS 方式，更多鉴权访问方式请参见：https://help.aliyun.com/document_detail/378659.html
        client = Sample.create_client(ini['access_id'], ini['secret'])
        params = Sample.create_api_info()
        # 需要安装额外的依赖库，直接点击下载完整工程即可看到所有依赖。
        body = StreamClient.read_from_file_path(f'./data/big_picture/{index}.png')
        # runtime options
        runtime = util_models.RuntimeOptions()
        request = open_api_models.OpenApiRequest(
            stream=body
        )
        # 复制代码运行请自行打印 API 的返回值
        # 返回值为 Map 类型，可从 Map 中获得三类数据：响应体 body、响应头 headers、HTTP 返回的状态码 statusCode。
        response = client.call_api(params, request, runtime)

        text = json.loads(response['body']['Data'])['content']
        count = count_chinese_characters(text, loaded_dict[f'{index}'])

        print(f'index:{index}')
        print(f'count:{count}')
        return index, count, text

    @staticmethod
    async def main_async(index: int) -> (int, int, str):
        # 请确保代码运行环境设置了环境变量 ALIBABA_CLOUD_ACCESS_KEY_ID 和 ALIBABA_CLOUD_ACCESS_KEY_SECRET。
        # 工程代码泄露可能会导致 AccessKey 泄露，并威胁账号下所有资源的安全性。以下代码示例使用环境变量获取 AccessKey 的方式进行调用，仅供参考，建议使用更安全的 STS 方式，更多鉴权访问方式请参见：https://help.aliyun.com/document_detail/378659.html
        client = Sample.create_client(ini['access_id'], ini['secret'])
        params = Sample.create_api_info()
        # 需要安装额外的依赖库，直接点击下载完整工程即可看到所有依赖。
        body = StreamClient.read_from_file_path(f'./data/big_picture/{index}.png')
        # runtime options
        runtime = util_models.RuntimeOptions()
        request = open_api_models.OpenApiRequest(
            stream=body
        )
        # 复制代码运行请自行打印 API 的返回值
        # 返回值为 Map 类型，可从 Map 中获得三类数据：响应体 body、响应头 headers、HTTP 返回的状态码 statusCode。
        response = await client.call_api_async(params, request, runtime)
        text = json.loads(response['body']['Data'])['content']
        count = count_chinese_characters(text, loaded_dict[f'{index}'])

        print(f'index:{index}')
        print(f'count:{count}')
        return index, count, text


# if __name__ == '__main__':
#     Sample.main(sys.argv[1:])

# if __name__ == '__main__':
#     import asyncio
#
#     results = []
#
#     async def run_all():
#         result = await asyncio.gather(*(Sample.main_async(i) for i in range(621)))
#         results.append(result)
#         print(result)
#
#     asyncio.run(run_all())
#
#     print(results)


if __name__ == '__main__':
    import concurrent.futures

    results = []

    # 使用线程池并发处理，max_workers 控制并发数
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        # map 方法会为函数参数中的每一个元素创建一个新的线程
        for result in executor.map(Sample.main, range(621)):
            results.append(result)
            print(f'\naaaaaaaaaaaa{result}aaaaaaaaaaaaaaaaa\n')

    print(results)

    # 逐行写入数据，每次增加一行都打开并关闭文件
    for row in results:
        with open(csv_file_name, mode="a", newline="", encoding="gbk") as file:
            writer = csv.writer(file)
            writer.writerow(row)

    print(f"{len(results)} 行数据已成功写入 {csv_file_name} 文件。")
