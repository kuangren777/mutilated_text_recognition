# coding: utf-8
from PIL import Image, ImageDraw, ImageFont, ImageOps, ImageEnhance
import random
import numpy as np
import cv2
import os


def random_brightness(img, min_factor=0.5, max_factor=1.5):
    """
    对图像进行随机亮度变换

    Args:
        img: 原图像
        min_factor: 亮度调整因子下限，大于1表示增强，小于1表示减弱
        max_factor: 亮度调整因子上限，大于1表示增强，小于1表示减弱

    Returns:
        变换后的图像
    """
    enhancer = ImageEnhance.Brightness(img)
    factor = random.uniform(min_factor, max_factor)
    img = enhancer.enhance(factor)
    return img


# 定义一个函数，用于创建文件夹
def create_folder_if_not_exist(folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)


def add_gaussian_noise(img, mean=0, std=1):
    """
    对图像增加随机高斯噪声

    Args:
        img: 原图像
        mean: 高斯分布均值，默认为0
        std: 高斯分布标准差，默认为1

    Returns:
        添加噪声后的图像
    """
    row, col, ch = img.shape
    gauss = np.random.normal(mean, std, (row, col, ch))
    gauss = gauss.reshape(row, col, ch)
    noisy = img + gauss
    return noisy.astype(np.uint8)


def random_scale(img):
    # 随机生成缩放比例
    scale_factor = random.uniform(0.5, 1.5)
    # 计算缩放后的大小
    new_size = tuple(int(dim * scale_factor) for dim in img.size)
    # 缩放图片
    img = ImageOps.scale(img, new_size)
    return img


def delete_some(img):
    # 在图像上绘制随机的线段，以删除部分字符
    draw = ImageDraw.Draw(img)
    for i in range(10):
        x1 = random.randint(0, img.width)
        y1 = random.randint(0, img.height)
        x2 = random.randint(0, img.width)
        y2 = random.randint(0, img.height)
        draw.line((x1, y1, x2, y2), fill='white', width=9)
    return img


def random_shift(img):
    # 随机生成平移偏移量
    shift_x = random.randint(-30, 30)
    shift_y = random.randint(-30, 30)

    # 创建仿射变换矩阵
    matrix = (1, 0, shift_x, 0, 1, shift_y)

    # 对图像进行仿射变换
    img = img.transform(img.size, Image.AFFINE, matrix)
    return img


def gus_blur(img):
    # 对彩色图像进行高斯模糊处理
    img = cv2.GaussianBlur(img, (5, 5), 0)

    img = add_gaussian_noise(img)
    return img


a = ['的', '一', '是', '在', '不', '了', '有', '和', '人', '这', '中', '大', '为', '上', '个', '国', '我', '以', '要',
     '他', '时', '来', '用', '们', '生', '到', '作', '地', '于', '出', '就', '分', '对', '成', '会', '可', '主', '发',
     '年', '动', '同', '工', '也', '能', '下', '过', '子', '说', '产', '种', '面', '而', '方', '后', '多', '定', '行',
     '学', '法', '所', '民', '得', '经', '十', '三', '之', '进', '着', '等', '部', '度', '家', '电', '力', '里', '如',
     '水', '化', '高', '自', '二', '理', '起', '小', '物', '现', '实', '加', '量', '都', '两', '体', '制', '机', '当',
     '使', '点', '从', '业', '本', '去', '把', '性', '好', '应', '开', '它', '合', '还', '因', '由', '其', '些', '然',
     '前', '外', '天', '政', '四', '日', '那', '社', '义', '事', '平', '形', '相', '全', '表', '间', '样', '与', '关',
     '各', '重', '新', '线', '内', '数', '正', '心', '反', '你', '明', '看', '原', '又', '么', '利', '比', '或', '但',
     '质', '气', '第', '向', '道', '命', '此', '变', '条', '只', '没', '结', '解', '问', '意', '建', '月', '公', '无',
     '系', '军', '很', '情', '者', '最', '立', '代', '想', '已', '通', '并', '提', '直', '题', '党', '程', '展', '五',
     '果', '料', '象', '员', '革', '位', '入', '常', '文', '总', '次', '品', '式', '活', '设', '及', '管', '特', '件',
     '长', '求', '老', '头', '基', '资', '边', '流', '路', '级', '少', '图', '山', '统', '接', '知', '较', '将', '组',
     '见', '计', '别', '她', '手', '角', '期', '根', '论', '运', '农', '指', '几', '九', '区', '强', '放', '决', '西',
     '被', '干', '做', '必', '战', '先', '回', '则', '任', '取', '据', '处', '队', '南', '给', '色', '光', '门', '即',
     '保', '治', '北', '造', '百', '规', '热', '领', '七', '海', '口', '东', '导', '器', '压', '志', '世', '金', '增',
     '争', '济', '阶', '油', '思', '术', '极', '交', '受', '联', '什', '认', '六', '共', '权', '收', '证', '改', '清',
     '美', '再', '采', '转', '更', '单', '风', '切', '打', '白', '教', '速', '花', '带', '安', '场', '身', '车', '例',
     '真', '务', '具', '万', '每', '目', '至', '达', '走', '积', '示', '议', '声', '报', '斗', '完', '类', '八', '离',
     '华', '名', '确', '才', '科', '张', '信', '马', '节', '话', '米', '整', '空', '元', '况', '今', '集', '温', '传',
     '土', '许', '步', '群', '广', '石', '记', '需', '段', '研', '界', '拉', '林', '律', '叫', '且', '究', '观', '越',
     '织', '装', '影', '算', '低', '持', '音', '众', '书', '布', '复', '容', '儿', '须', '际', '商', '非', '验', '连',
     '断', '深', '难', '近', '短', '判', '突', '素', '育', '调', '房', '屋', '易', '精', '负', '责', '言', '退', '摆',
     '官', '吵', '痛', '注', '排', '供', '河', '态', '封', '古', '往', '迈', '般', '候', '逐', '浓', '芳', '慢', '随',
     '燥', '赞', '露', '森', '句', '柴', '翻', '墙', '足', '颜', '值', '号', '喝', '敢', '跳', '妇', '找', '飞', '哪',
     '玉', '脸', '懂', '孩', '帮', '紧', '状', '脱', '买', '草', '独', '针', '绝', '细', '详', '睡', '破', '纪', '震',
     '穿', '疼', '藏', '降', '盛', '怒', '照', '遇', '底', '抓', '警', '沉', '括', '暗', '害', '咳', '赚', '泪', '剧',
     '补', '牛', '骗', '凭', '劳', '落', '盖', '销', '韦', '码', '梦', '伙', '财', '圆', '险', '惨', '魔', '测', '香',
     '踏', '卷', '块', '拒', '忠', '郎', '抱', '骂', '辞', '园', '训', '败', '姐', '糖', '川', '旅', '饭', '盘', '坦',
     '牌', '秀', '卧', '醒', '羽', '毁', '懒', '伯', '喊', '肉', '幸', '卫', '析', '甲', '旁', '练', '念', '仙', '腰',
     '席', '窗', '霜', '掉', '弹', '剂', '丝', '喂', '洞', '掌', '姓', '敬', '哭', '悟', '渐', '豆', '烧', '磨', '著',
     '姑', '枝', '浅', '紫', '锁', '唯', '咬', '忙', '船', '措', '贵', '避', '仁', '废', '贯', '盐', '朵', '馆', '渔',
     '惯', '坐', '翼', '恢', '减', '康', '裹', '枯', '折', '胞', '饰', '薄', '蒸', '怜', '扇', '摸', '屈', '郁', '跨',
     '撒', '狠', '鼠', '挣', '泉', '偿', '愁', '占', '炸', '哩', '盆', '麻', '辣']
a_min = ['的', '一', '是', '在', '不', '了', '有', '和', '人', '这', '中', '大', '为', '上', '个', '国', '我', '以', '要']
b = ["now", '1']
RENDER = False
ALL = 1000
TRAIN = 0.7
VAL = 0.2
TEST = 0.1


font = ImageFont.truetype('./data_pre_set/HYSongYunLangHeiW-1.ttf', size=80)

idx = 0
for char in a:
    print(idx, char)
    idx += 1
    for time in range(ALL):
        img = Image.new('RGB', (128, 128), color='white')
        draw = ImageDraw.Draw(img)
        draw.text((12, 12), f'{char}', font=font, fill='black')

        # 在图像上绘制随机的线段，以删除部分字符
        img = delete_some(img)

        # 随机偏移
        img = random_shift(img)

        # 随机放大缩小
        # img = random_scale(img)

        # 随机对比度
        img = random_brightness(img)

        img.save(f'./data_pre_set/temp/temp.png')

        # 加载图像
        img = cv2.imread(rf'./data_pre_set/temp/temp.png')

        if img is None:
            print("Failed to read image")
        else:
            # 高斯模糊
            img = gus_blur(img)

            # 随机噪音
            img = add_gaussian_noise(img)

            if RENDER:
                # 窗口显示模组
                cv2.imshow('Rotated Image', img)
                cv2.waitKey(0)
                cv2.destroyAllWindows()

            # 保存为文件
            # 1
            img = Image.fromarray(img)

            if time < ALL * TRAIN:
                create_folder_if_not_exist(f'./train/{char}/')
                img.save(f'./train/{char}/{time}.png')
            elif ALL * TRAIN < time < ALL * (TRAIN + VAL):
                create_folder_if_not_exist(f'./val/{char}/')
                img.save(f'./val/{char}/{time}.png')
            else:
                create_folder_if_not_exist(f'./test/{char}/')
                img.save(f'./test/{char}/{time}.png')
            # create_folder_if_not_exist(f'./enhance/{char}/')
            # img.save(f'./enhance/{char}/{time}.png')

            # 2
            # cv2.imwrite(f'./enhance/{char}.png', blur_img)

            # 3
            # filename = f'./enhance/{char}.png'
            # filename_unicode = filename.encode('unicode_escape').decode()
            # cv2.imwrite(filename_unicode, rotated)

            # 4
            # import locale
            #
            # filename = f'./enhance/{char}.png'
            #
            # # 获取系统默认的编码方式
            # sys_encoding = locale.getpreferredencoding()
            #
            # # 转换文件名为系统默认编码方式
            # filename_encoded = filename.encode(sys_encoding)
            #
            # # 保存旋转后的图像
            # cv2.imwrite(filename_encoded, rotated)

            # 5
            # 编码文件名
            # import locale
            # filename = f'{char}.png'
            # sys_encoding = locale.getpreferredencoding()
            # filename_encoded = filename.encode(sys_encoding)
            # cv2.imwrite(filename_encoded, rotated)