# -*- coding: utf-8 -*-
# @Time    : 2023/10/12 16:45
# @Author  : KuangRen777
# @File    : read_ini.py
# @Tags    :
import configparser


def get_config():
    # 创建ConfigParser对象
    config = configparser.ConfigParser()

    # 读取INI文件
    config.read('config.ini')

    config_dict = {}

    # 检查节和键是否存在
    if config.has_section('ocr'):
        if config.has_option('ocr', 'login_password'):
            # 获取配置值
            config_dict['login_password'] = config.get('ocr', 'login_password')
        else:
            print("The login_password section does not exist in the INI file.")
        if config.has_option('ocr', 'access_id'):
            config_dict['access_id'] = config.get('ocr', 'access_id')
        else:
            print("The access_id section does not exist in the INI file.")
        if config.has_option('ocr', 'secret'):
            config_dict['secret'] = config.get('ocr', 'secret')
        else:
            print("The secret section does not exist in the INI file.")
    else:
        print("The 'ocr' section does not exist in the INI file.")

    return config_dict

# # 修改配置值
# config.set('database', 'host', 'new_host')

# # 保存修改后的INI文件
# with open('config.ini', 'w') as config_file:
#     config.write(config_file)


if __name__ == '__main__':
    a = get_config()
    print(a)
