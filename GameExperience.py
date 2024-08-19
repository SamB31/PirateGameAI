import numpy as np

class GameExperience(object):
    def __init__(self, model, target_model, max_memory=100, discount=0.95):
        self.model = model
        self.target_model = target_model
        self.max_memory = max_memory
        self.discount = discount
        self.memory = list()
        self.num_actions = model.output_shape[-1]

    def remember(self, episode):
        self.memory.append(episode)
        if len(self.memory) > self.max_memory:
            del self.memory[0]

    def predict(self, envstate):
        return self.model.predict(envstate)[0]

    def get_target_q_values(self, next_states):
        # Get Q values for next_states using the main network
        next_q_values = self.model.predict(next_states)
        
        # Get the actions that would be chosen by the main network
        best_actions = np.argmax(next_q_values, axis=1)
        
        # Get Q values from the target network
        target_q_values = self.target_model.predict(next_states)
        
        # Select Q values for the best actions
        selected_q_values = target_q_values[np.arange(target_q_values.shape[0]), best_actions]
        
        return selected_q_values

    def get_data(self, data_size=10):
        env_size = self.memory[0][0].shape[1]
        mem_size = len(self.memory)
        data_size = min(mem_size, data_size)
        inputs = np.zeros((data_size, env_size))
        targets = np.zeros((data_size, self.num_actions))

        for i, j in enumerate(np.random.choice(range(mem_size), data_size, replace=False)):
            envstate, action, reward, envstate_next, game_over = self.memory[j]
            inputs[i] = envstate
            targets[i] = self.predict(envstate)
            Q_sa = self.get_target_q_values(envstate_next.reshape(1, -1))[0]
            if game_over:
                targets[i, action] = reward
            else:
                targets[i, action] = reward + self.discount * Q_sa

        return inputs, targets