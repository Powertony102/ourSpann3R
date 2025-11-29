import re
import sys
from collections import defaultdict
import argparse

# 脚本功能：分析日志文件中的FPS数据，按scene分组计算平均FPS

def parse_log_file(file_path, fps_data):
    """
    解析单个日志文件，提取scene名称和对应的FPS数值（优先从'Complete metrics'行提取'fps'值），并按scene分组。
    :param file_path: 日志文件路径
    :param fps_data: 字典，用于存储scene到FPS列表的映射
    """
    current_scene = None
    try:
        with open(file_path, 'r') as f:
            for line_num, line in enumerate(f, start=1):
                line_lower = line.lower()
                
                # 提取scene名称（格式：🚩Processing sceneXXXX_XX ...）
                scene_match = re.search(r'processing\s+scene(\w+_\w+)', line_lower)
                if scene_match:
                    current_scene = scene_match.group(1)
                    continue  # 继续读取下一行
                
                # 提取Complete metrics行中的'fps'值（字典格式）
                if "complete metrics" in line_lower:
                    fps_match = re.search(r"'fps':\s*(\d+\.\d+)", line_lower)
                    if fps_match and current_scene:
                        try:
                            fps_value = float(fps_match.group(1))
                            fps_data[current_scene].append(fps_value)
                        except ValueError:
                            print(f"警告: 文件 {file_path} 第 {line_num} 行 FPS 值格式错误，已跳过: {line.strip()}")
                    else:
                        print(f"警告: 文件 {file_path} 第 {line_num} 行 匹配失败（缺失fps或当前scene）: {line.strip()}")
                
                # 可选：提取Inference FPS行（如果需要补充数据，格式：Inference FPS (frames/s): xx.xx）
                # inference_match = re.search(r'inference fps \(frames/s\):\s*(\d+\.\d+)', line_lower)
                # if inference_match and current_scene:
                #     fps_value = float(inference_match.group(1))
                #     fps_data[current_scene].append(fps_value)
    
    except FileNotFoundError:
        print(f"错误: 文件 {file_path} 不存在，已跳过。")
    except Exception as e:
        print(f"错误: 处理文件 {file_path} 时发生异常: {str(e)}")

def calculate_average_fps(fps_data):
    """
    计算每个scene的平均FPS值，保留三位小数。
    :param fps_data: 字典，scene到FPS列表的映射
    :return: 字典，scene到平均FPS的映射
    """
    averages = {}
    for scene, fps_list in fps_data.items():
        if fps_list:
            avg = sum(fps_list) / len(fps_list)
            averages[scene] = round(avg, 3)  # 保留三位小数
        else:
            print(f"警告: scene {scene} 没有有效的FPS数据。")
    return averages

def main():
    # 命令行参数解析，支持多个文件输入
    parser = argparse.ArgumentParser(description="分析日志文件中的FPS数据，按scene分组计算平均值。")
    parser.add_argument('files', nargs='+', help="一个或多个日志文件路径，例如: log1.txt log2.txt")
    args = parser.parse_args()

    # 存储FPS数据的字典：scene -> [fps1, fps2, ...]
    fps_data = defaultdict(list)

    # 处理每个输入文件
    for file_path in args.files:
        parse_log_file(file_path, fps_data)

    # 计算平均值
    averages = calculate_average_fps(fps_data)

    # 输出结果，按scene名称排序
    for scene in sorted(averages.keys()):
        print(f"{scene}: {averages[scene]}")

if __name__ == "__main__":
    main()