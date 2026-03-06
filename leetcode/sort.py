#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys

def sort_numbers_in_file():
    # 检查命令行参数
    if len(sys.argv) != 2:
        print("用法：python 脚本名.py 目标文件名")
        print("示例：python sort.py numbers.txt")
        sys.exit(1)

    file_name = sys.argv[1]

    try:
        # 读取文件：去除空白、过滤空行
        with open(file_name, 'r', encoding='utf-8') as f:
            raw_lines = [line.strip() for line in f.readlines() if line.strip()]

        # 转换为【整数】并升序排序
        numbers = [int(num) for num in raw_lines]
        sorted_numbers = sorted(numbers)

        # 转换为字符串行
        output_lines = [str(num) for num in sorted_numbers]

        # 覆盖写入原文件
        with open(file_name, 'w', encoding='utf-8') as f:
            f.write('\n'.join(output_lines) + '\n')

        print(f"✅ 整数排序完成！文件 {file_name} 已更新")

    except FileNotFoundError:
        print(f"❌ 错误：未找到文件 {file_name}")
        sys.exit(1)
    except ValueError:
        print(f"❌ 错误：文件中包含非整数内容，无法排序")
        sys.exit(1)
    except PermissionError:
        print(f"❌ 错误：没有权限读写文件 {file_name}")
        sys.exit(1)
    except Exception as e:
        print(f"❌ 未知错误：{str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    sort_numbers_in_file()
