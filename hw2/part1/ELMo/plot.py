import matplotlib.pyplot as plt
import pickle
import numpy as np
import random

with open('train_loss','rb') as f:
    loss = pickle.load(f)

print(len(loss))
# x = range(0,50000,1000)

# y = [random()/100 for _ in range(50000)]
# y = np.random.rand(len(x))
# y2 = np.random.rand(len(x))
x = range(len(loss))
# y = np.log2(loss)
# y = np.power(np.full(len(loss),2),np.array(loss))
y = np.exp(loss)
print(len(x))

plt.plot(x,y)
# plt.plot(x,y2)
plt.xlabel('steps')
plt.ylabel('perplexity')
plt.show()
# plt.savefig('perplexity.png')
