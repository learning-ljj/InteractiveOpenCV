# 检查Python版本是否在3.11-3.13范围内
import sys  # 导入系统模块用于版本检查


# 检查当前Python版本是否符合要求
if sys.version_info < (3, 11) or sys.version_info > (3, 13):
    # 如果版本不在3.11-3.13范围内，打印警告信息
    print(
        "警告: 不支持的Python版本 {ver}，请使用3.11-3.13版本".format(
            ver=".".join(map(str, sys.version_info))  # 格式化当前版本号为字符串
        )
    )
