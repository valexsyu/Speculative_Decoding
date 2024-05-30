import numpy as np

# Load the data from the CSV file
data = np.genfromtxt('/work/valex1377/LLMSpeculativeSampling/scripts/speculative_la_accepted_rate_68m_g4_200to200_1000GenWithEos_3_.csv', delimiter=',')

# Reshape the data to the desired shape
new_data = data.reshape(5, 20)

# Existing data
existing_data = np.loadtxt('/work/valex1377/LLMSpeculativeSampling/scripts/speculative_la_accepted_rate_68m_g4_200to200_1000GenWithEos_.csv', delimiter=',')


# Concatenate existing data with new data
combined_data = np.concatenate((existing_data, new_data), axis=0)

# Save the combined data back to the CSV file
np.savetxt('/work/valex1377/LLMSpeculativeSampling/scripts/speculative_la_accepted_rate_68m_g4_200to200_1000GenWithEos_.csv', combined_data, delimiter=',')
