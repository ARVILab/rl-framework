import torch.optim as optim


class DQNOptimizer(object):
    def __init__(self,
                 policy,
                 target_policy,
                 mini_batch_size,
                 discount,
                 lr,
                 update_epochs):

        self.policy = policy
        self.target_policy = target_policy
        self.mini_batch_size = mini_batch_size
        self.discount = discount
        self.update_epochs = update_epochs

        self.epsilon = 1e-8
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr, eps=self.epsilon)

    def update(self, storage):
        loss_avg = 0
        n_updates = 0

        for e in range(self.update_epochs):
            data_generator = storage.sample(self.mini_batch_size)
            for sample in data_generator:

                states, actions, rewards, next_states, masks = sample

                q_values = self.policy.get_value(states)
                next_q_values = self.target_policy.get_value(next_states)

                q_value = q_values.gather(1, actions)

                next_q_value = next_q_values.max(1)[0].view(-1, 1)

                expected_q_value = rewards + self.discount * next_q_value * masks

                loss = (q_value - expected_q_value.data).pow(2).mean()

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                loss_avg += loss.item()
                n_updates += 1

        return loss_avg / n_updates
