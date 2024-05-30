import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Read the CSV file into a pandas DataFrame
df = pd.read_csv("/work/valex1377/LLMSpeculativeSampling/scripts/self_speculative_accepted_number_2to30_gamma5tokenAtten.csv", header=None)

# Determine the maximum length of the sequences
max_length = df.apply(lambda row: len(row.str.split(',')), axis=1).max()

# Pad the sequences with -1 to match the maximum length
df = df.apply(lambda row: row.str.split(',').apply(lambda seq: [int(x) if x.strip() else -1 for x in seq] + [-1] * (max_length - len(seq))), axis=1)

# Convert the DataFrame to a NumPy array
data = np.array(df.values.tolist())

# Plot the 2D lines
for i in range(data.shape[0]):
    plt.plot(data[i], label=f'Sequence {i+1}')

plt.xlabel('Index')
plt.ylabel('Value')
plt.legend()
plt.title('2D Line Plot of Sequences')
plt.show()




# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D

# # Read the CSV file into a pandas DataFrame
# df = pd.read_csv("/work/valex1377/LLMSpeculativeSampling/scripts/self_speculative_accepted_number_2to30_gamma5tokenAtten.csv", header=None)

# # Determine the maximum length of the sequences
# max_length = df.apply(lambda row: len(row.str.split(',')), axis=1).max()

# # Pad the sequences with -1 to match the maximum length
# df = df.apply(lambda row: row.str.split(',').apply(lambda seq: [int(x) if x.strip() else -1 for x in seq] + [-1] * (max_length - len(seq))), axis=1)

# # Convert the DataFrame to a NumPy array
# data = np.array(df.values.tolist())

# # Plot the 3D graph
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')

# for i in range(data.shape[0]):
#     x = np.arange(data.shape[1])
#     y = np.full_like(x, i)
#     z = data[i]
#     ax.plot(x, y, z)

# ax.set_xlabel('X Label')
# ax.set_ylabel('Y Label')
# ax.set_zlabel('Z Label')

# plt.show()
