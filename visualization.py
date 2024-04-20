import matplotlib.pyplot as plt
import numpy as np
import os

HOME_DIR = "D:\\codes\\毕设项目"
result_path = os.path.join(HOME_DIR, "results")

models = ['bert', 'xlnet']
data_types = ['random', 'slim']
tasks = ['cola', 'mrpc', 'qnli', 'qqp', 'rte', 'sst-2', 'sts-b', 'wnli']
reduction_rate = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]


def draw_plt(y_random, y_slim, x, model, task):
    plt.plot(x, y_random, 'r', marker='.', markersize=10)
    plt.plot(x, y_slim, 'b', marker='.', markersize=10)
    plt.title('Reduction Accuracy - {} - {}'.format(model, task))
    plt.xlabel('Reduction Rate')
    plt.ylabel('Accuracy')
    # for a, b in zip(x, y1):
    #     plt.text(a, b, b, ha='center', va='bottom', fontsize=10)
    # for a, b in zip(x, y2):
    #     plt.text(a, b, b, ha='center', va='bottom', fontsize=10)
    plt.legend(['Random', 'Slim'])
    plt.show()


x = reduction_rate
y_random = []
y_slim = []
for model in models:
    for task in tasks:
        for data_type in data_types:
            for rate in reduction_rate:
                file_path = os.path.join(result_path, "{}\\{}\\{}\\eval_results.txt".format(model, task, 'original')) if rate == 0.0 else os.path.join(result_path, "{}\\{}\\{}\\{}\\eval_results.txt".format(model, task, data_type, rate))
                print("---{}---".format(file_path))
                with open(file_path) as f:
                    lines = f.readlines()
                    result = 0
                    if len(lines) == 1:
                        result = eval(lines[0][lines[0].find('=') + 1:])
                        print(result)
                    if len(lines) == 3:
                        result = eval(lines[1][lines[1].find('=') + 1:])
                        print(result)
                    if data_type == "random":
                        y_random.append(result)
                    if data_type == "slim":
                        y_slim.append(result)
        draw_plt(y_random, y_slim, x, model, task)
        y_random = []
        y_slim = []

