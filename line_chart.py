import pandas as pd # 导入pandas库用来处理csv文件
import matplotlib.pyplot as plt # 导入matplotlib.pyplot并用plt简称
unrate = pd.read_csv('0634.csv') # 读csv文件
data = unrate[0:] # 取前12行数据

plt.plot(data['epoch'], data['label0'], color='deeppink', label='label 0')
plt.plot(data['epoch'], data['label1'], color='orange', label='label 1')
plt.plot(data['epoch'], data['label2'], color='limegreen', label='label 2')
plt.plot(data['epoch'], data['label3'], color='dodgerblue', label='label 3')
plt.legend()

plt.xlabel('epoch') # 给x轴数据加上名称
plt.ylabel('avg_dice_loss') # 给y轴数据加上名称
plt.title('Dice Loss Line Graph in Training Period') # 给整个图表加上标题

plt.show() # 将刚画的图显示出来
