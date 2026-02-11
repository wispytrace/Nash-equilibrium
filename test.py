
import time
import numpy as np
import json
array1 = np.zeros((3,4,5))
json_dict = {"x": array1.tolist()}
a = json.dumps(json_dict)
print(a, type(a))
b = json.loads(a)
print(b, type(b))


gama = 2.1*(2*np.sqrt(10)*2)**(0.65) + 2*2.1*(np.sqrt(10))*(13)**(0.35) + 2.1*2*np.sqrt(10)*2
print(gama*np.sqrt(5)*0.59)
# class Agent:
#     def __init__(self,id,value) -> None:
#         self.id = id
#         self.value = value

# def fibonacci(n):
#     if n <= 0:
#         return 0
#     elif n == 1:
#         return 1
#     else:
#         return fibonacci(n - 1) + fibonacci(n - 2)

# agent0 = Agent(0, 1)
# agent1 = Agent(1, 2)
# agent2 = Agent(2, 3)

# agents = [agent0, agent1 ,agent2]

# epochs = 10000

# start = time.time()
# for i in range(epochs):
#     for id, agent in enumerate(agents):
#         fibonacci(17)
        
# end = time.time()

# print(f"cost: {end - start}")