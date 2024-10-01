import numpy as np
import pickle

# Load the heart attack risk prediction model
with open('saved_steps.pkl', 'rb') as file:
    data = pickle.load(file)

model_loaded = data["model"]

# Define the State Space: Assuming the state is the patient's profile
state_space = ['Age', 'Cholesterol', 'BP_systolic', 'BP_diastolic', 'Heart Rate',
               'Diabetes', 'Family History', 'Smoking', 'Obesity',
               'Alcohol Consumption', 'Exercise Hours Per Week',
               'Previous Heart Problems', 'Medication Use',
               'BMI', 'Triglycerides', 'Sleep Hours Per Day', 'Sex', 'Diet']

# Define the Action Space: Assuming the actions are the suggested lifestyle changes
action_space = ['quit smoking', 'lose weight', 'exercise more', 'eat a healthy diet']

# Define hyperparameters
alpha = 0.1  # learning rate
gamma = 0.9  # discount factor
epsilon = 0.1  # exploration rate
num_episodes = 2
# Define the reward function
def reward(predict_type):
    return 1 - predict_type  # Higher probability means higher risk, so we invert it

# Hash function to convert state to a string
def hash_state(state):
    return ','.join([str(s) for s in state])

# Initialize the Q-values dictionary
Q = {}

# Q-learning algorithm
for episode in range(num_episodes):
    # Define the initial state (e.g., a default or random patient's profile)
    initial_state = [40, 200, 120, 80, 70, 0, 0, 0, 0, 0, 2, 0, 1, 25, 150, 7, 1, 0]
    state = initial_state

    done = False
    while not done:
        # Choose an action
        if np.random.rand() < epsilon:
            action = np.random.choice(len(action_space))
        else:
            state_str = hash_state(state)
            if state_str not in Q:
                Q[state_str] = np.zeros(len(action_space))
            action = np.argmax(Q[state_str])

        # Take the action and observe the next state and reward
        new_person = {state_space[i]: list(state)[i] for i in range(len(state_space))}
        new_person['Diet'] = action
        x = dicti_vals(new_person)
        predict_type = model_loaded.predict_proba(x)[:, 1]
        r = reward(predict_type)

        # Update the Q-value
        new_state_str = hash_state(new_person)
        if new_state_str not in Q:
            Q[new_state_str] = np.zeros(len(action_space))
        Q[state_str][action] = (1 - alpha) * Q[state_str][action] + alpha * (r + gamma * np.max(Q[new_state_str]))

        # Update the state
        state = new_person.values()