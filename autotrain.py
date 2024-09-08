import subprocess
import time

# 运行次数的限制（如果需要无限循环，可以去掉这个限制）
max_iterations = 1000  # 可以根据需要调整或删除

# 交替运行的循环
for i in range(max_iterations):
    print(f"Iteration {i+1}")
    
    # 运行第一个 Python 文件
    subprocess.run(["python", "selfplay.py"], check=True)
    
    # 运行第二个 Python 文件
    subprocess.run(["python", "train.py"], check=True)
    
    # 如果需要在每次运行后暂停，可以使用 time.sleep(seconds)
    # time.sleep(1)  # 暂停1秒

print("Finished running scripts.")