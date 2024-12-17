import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

filepath = r"C:\Users\viswa teja\Desktop\ML workshop\ml-robotics-workshop\level-1\tasks\datasets\simple_dataset.csv";
dataset = pd.read_csv(filepath)

id = dataset['ID'].to_numpy()
weight = dataset['Weight (kg)'].to_numpy()
height = dataset['Height (cm)'].to_numpy()


print("The mean value of Weight is ", np.mean(weight))
print("The mean value of Height is ", np.mean(height))

#plotting a bar graph for Weights 

plt.bar(id, weight)
plt.title('Weight Distribution')
plt.xlabel('ID')
plt.ylabel('Weight (kg)')
plt.show()

#plotting a bar graph for Heights 

plt.bar(id, height)
plt.title('Height Distribution')
plt.xlabel('ID')
plt.ylabel('Height (cm)')
plt.show()

