import pandas as pd
import numpy as np
import copy
import matplotlib.pyplot as plt

method_names = ['DQN']
pendulum_data_dir = '../../result/single_catchpigs_result/single_catchpigs_result.csv'

pendulum_df = pd.read_csv(pendulum_data_dir, dtype=np.float64)

for method in method_names:
    df = copy.deepcopy(pendulum_df.loc[:, pendulum_df.columns.str.contains(method)])
    mean_reward = df.apply(lambda x: x.mean(), axis=1)
    plt.plot(range(len(mean_reward)), mean_reward)

plt.legend(method_names)
plt.title('Reward Curve')
plt.xlabel('Epoch')
plt.ylabel('Reward')

plt.savefig('reward_curve.png')
plt.show()

