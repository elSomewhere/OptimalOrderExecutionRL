import random
from typing import Tuple, List, Optional
import torch
from torch.distributions import Categorical, Beta
from torch.nn import init
from tqdm import tqdm

from RLAnalytics import execute_and_plot_schedule, run_schedule, plot_lob_data_with_volume
from RLEnvironment import RLEnvironment
from RLNetwork import RLAgent, TSM_LSTM
import ScheduleExecutor
from DataProvision import DataFeed


class ActorCriticNetwork_continuous_beta_simple(RLAgent):
    def __init__(self, num_hidden, lob_levels, aux_features):
        super(ActorCriticNetwork_continuous_beta_simple, self).__init__(lob_levels=lob_levels, aux_features=aux_features, time_depth=1, dtype=torch.float)
        # Common layers
        self.fc1 = torch.nn.Linear(self.number_of_features, num_hidden)
        self.fc2 = torch.nn.Linear(num_hidden, num_hidden)

        # Actor head - outputs alpha and beta parameters for the Beta distribution
        self.actor_alpha_head = torch.nn.Linear(num_hidden, 1)
        self.actor_beta_head = torch.nn.Linear(num_hidden, 1)

        # Critic head
        self.critic_head = torch.nn.Linear(num_hidden, 1)  # Output a single value

    def forward(self, x):
        x = x.squeeze(2)
        x = torch.nn.functional.relu(self.fc1(x))
        x = torch.nn.functional.relu(self.fc2(x))

        # Actor output - compute alpha and beta parameters
        alpha = torch.nn.functional.softplus(self.actor_alpha_head(x)) + 1  # Ensure alpha > 1
        beta = torch.nn.functional.softplus(self.actor_beta_head(x)) + 1  # Ensure beta > 1
        distribution = Beta(alpha, beta)

        # Critic output
        state_values = self.critic_head(x)

        return distribution, state_values




class ActorCriticNetwork_continuous_beta_complex(RLAgent):
    def __init__(self,
                 levels,
                 aux_features,
                 time_depth,
                 conv_channels_a,
                 conv_channels_b,
                 conv_channels_c,
                 translation_layer,
                 hidden_features_per_layer,
                 dropout=0.2,
                 dtype=torch.float
                 ):
        super(ActorCriticNetwork_continuous_beta_complex, self).__init__(levels, aux_features, time_depth, dtype)
        self.volumeWeightedPrice = torch.nn.Conv2d(in_channels=1, out_channels=conv_channels_a, kernel_size=(1, 2), stride=(1, 2))
        init.xavier_uniform_(self.volumeWeightedPrice.weight)
        self.volumeWeightedPrice_act = torch.nn.ReLU()
        self.microPrice = torch.nn.Conv2d(in_channels=conv_channels_a, out_channels=conv_channels_b, kernel_size=(1, 2), stride=(1, 2))
        init.xavier_uniform_(self.microPrice.weight)
        self.microPrice_act = torch.nn.ReLU()
        self.weightedMidPrice = torch.nn.Conv2d(in_channels=conv_channels_b, out_channels=conv_channels_c, kernel_size=(1, levels), stride=(1, levels))
        init.xavier_uniform_(self.weightedMidPrice.weight)
        self.weightedMidPrice_act = torch.nn.ReLU()
        self.totalstate_to_rnn = torch.nn.Linear(conv_channels_c + self.aux_features, translation_layer).type(self.dtype)
        init.xavier_uniform_(self.totalstate_to_rnn.weight)
        self.totalstate_to_rnn_act = torch.nn.ReLU()
        self.encoder = TSM_LSTM(
            number_of_features=translation_layer,
            hidden_features_per_layer=hidden_features_per_layer,
            dropout=dropout,
            dtype=dtype
        )
        # Actor head
        self.actor_alpha_head = torch.nn.Linear(hidden_features_per_layer[-1], 1)
        self.actor_beta_head = torch.nn.Linear(hidden_features_per_layer[-1], 1)
        init.xavier_uniform_(self.actor_alpha_head.weight)
        init.xavier_uniform_(self.actor_beta_head.weight)
        # Critic head
        self.critic_head = torch.nn.Linear(hidden_features_per_layer[-1], 1)  # Output a single value
        init.xavier_uniform_(self.critic_head.weight)

    # input needs an extra dimension because conv2 always asumes a layer channel
    def forward(self, x):
        '''
        Expects shape: [batchsize, statesize, timesteps]
        '''
        # add a convolution dimension
        x = x.unsqueeze(1)
        # split lob and agent state, they will be processed seaparately in the feature layers
        lob = x[:, :, 0:(self.levels * 4), :]
        aux = x[:, :, -self.aux_features::, :]
        # create volume weighted midprice
        x_vwp = self.volumeWeightedPrice_act(self.volumeWeightedPrice(lob.transpose(2, 3)))
        # create microprice
        x_mp = self.microPrice_act(self.microPrice(x_vwp))
        # create weighted midprice
        x_weightedMidPrice = self.weightedMidPrice_act(self.weightedMidPrice(x_mp)).squeeze(3)
        x_concatenated = torch.cat([x_weightedMidPrice, aux.squeeze(1)], 1)
        # combine midprice and agent state
        x_final_to_rnn = self.totalstate_to_rnn_act(self.totalstate_to_rnn(x_concatenated.transpose(1, 2)).transpose(1, 2))
        # flatten time
        output, all_hidden_states = self.encoder(x_final_to_rnn)
        # Actor output - compute alpha and beta parameters
        alpha = torch.nn.functional.softplus(self.actor_alpha_head(output[:, :, -1])) + 1  # Ensure alpha > 1
        beta = torch.nn.functional.softplus(self.actor_beta_head(output[:, :, -1])) + 1  # Ensure beta > 1
        distribution = Beta(alpha, beta)
        # Critic output
        state_values = self.critic_head(output[:, :, -1])
        return distribution, state_values




class RLEnvironment_continuous_beta(RLEnvironment):
    def __init__(self, policy_network, buffer_size, resubmit_each_tick, price_depth_penalty):
        super(RLEnvironment_continuous_beta, self).__init__(policy_network, buffer_size, resubmit_each_tick, price_depth_penalty)
        self.epsilon = 1.0
        self.epsilon_decay = 0.999
        self.epsilon_min = 0.01
        self.gamma = 0.99

    def _select_agent_action(self, state: torch.Tensor) -> Tuple[float, Optional[torch.Tensor], Optional[torch.Tensor]]:
        # Normalize the input state
        state_norm = (state - state.mean()) / (state.std() + 1e-5)

        # Pass state through the network to get action logits and state value
        distribution, state_value = self.policy_network(state_norm)
        action = distribution.sample()
        log_prob = distribution.log_prob(action)
        action_out = action.item() * 2

        # Return the selected action, its log probability, and the state value
        return action_out, log_prob, state_value

    def _select_agent_action_inference(self, state: torch.Tensor) -> Tuple[float, Optional[torch.Tensor], Optional[torch.Tensor]]:
        # Normalize the input state
        state_norm = (state - state.mean()) / (state.std() + 1e-5)

        # Pass state through the network to get action logits and state value
        distribution, state_value = self.policy_network(state_norm)
        action = distribution.mean

        log_prob = distribution.log_prob(action)
        action_out = action.item() * 2

        # Return the selected action, its log probability, and the state value
        return action_out, log_prob, state_value

    def _update_policy_multistep(self, all_values_per_step: List[torch.Tensor], all_states_per_step: List[torch.Tensor], all_action_per_step: List[float], all_log_prob_per_step: List[Optional[torch.Tensor]], all_rewards_per_step: List[float], cumulative_reward: float, done: bool) -> torch.Tensor:
        returns = []
        R = cumulative_reward  # Starting with the final reward

        # Backpropagate the discounted cumulative reward to each step
        for step in reversed(range(len(all_states_per_step))):
            returns.insert(0, R)
            R = R * self.gamma  # Discount the reward for the next earlier step

        # Convert returns to a PyTorch tensor
        returns = torch.tensor(returns, dtype=torch.float32)

        actor_losses = []
        critic_losses = []
        for log_prob, value, R in zip(all_log_prob_per_step, all_values_per_step, returns):
            advantage = R - value.squeeze()

            # Actor loss: -log_prob * advantage
            actor_loss = -log_prob * advantage
            actor_losses.append(actor_loss)

            # Critic loss: F.mse_loss or another suitable loss function
            critic_loss = torch.nn.functional.mse_loss(value.squeeze(0), torch.tensor([R]))
            critic_losses.append(critic_loss)

        # Take the mean of the accumulated losses
        actor_loss = torch.stack(actor_losses).mean()
        critic_loss = torch.stack(critic_losses).mean()

        # Update both actor and critic
        self.optimizer.zero_grad()
        total_loss = actor_loss + critic_loss
        total_loss.backward()
        self.optimizer.step()
        return total_loss


if __name__ == '__main__':
    import pandas as pd
    import numpy as np
    np.random.seed(123)
    torch.manual_seed(123)

    # get the data
    data = pd.read_csv("LOBL2.csv")
    data = data.set_index(data.iloc[:, 0]).iloc[:, 1::]
    # data = data / data.mean()
    # shorten the data for reasonable train times
    average_timestep = np.nanmean(data.index.diff())
    int(3 * 1e6) / average_timestep * 3

    # data = data.iloc[500:1000, :]

    # data metrics to get some good training params, 1§e6 = 1s
    num_ticks = data.shape[0]
    average_tick_duration = np.round(np.nanmean(data.index.diff()))
    average_tick_duration_s = average_tick_duration / 1e6
    max_time = int(3 * 1e6)
    max_volume = int(3 * 1e6)
    side = "bid"
    resubmit_each_tick = False
    price_level_offset = 0.5

    # other params
    num_schedule_elements = 3
    schedule_twap = {}
    for i in range(num_schedule_elements):
        schedule_twap.update({i: ScheduleExecutor.LimitEvent(side, max_time, max_volume)})
    schedule_twap.update({len(schedule_twap): ScheduleExecutor.MarketEvent(side, max_time)})

    # network
    # policy_network = ActorCriticNetwork_continuous_simple(256, 10, 3)
    policy_network = ActorCriticNetwork_continuous_beta_complex(10, 3, 10, 16, 16, 16, 128, [64], 0.0, torch.float)

    # training
    epochs = 3
    env = RLEnvironment_continuous_beta(policy_network, policy_network.time_depth, False, 0.5)
    all_average_policy_losses = {}
    all_actions = {}
    all_average_rewards = {}
    for epoch in range(epochs):
        # collect metrics
        epoch_rewards = {}
        epoch_policy_losses = {}
        epoch_actions = {}
        # randomly sample
        max_time_of_schedule = np.sum([i.max_time for j, i in schedule_twap.items()])
        last_slice_starts_at_index = np.argmax(data.index > (data.index[-1] - max_time_of_schedule))
        starting_indexes = np.arange(last_slice_starts_at_index)
        # shuffle the slices should make training a bit more varied and help generalize
        np.random.shuffle(starting_indexes)
        episode = 0
        # iterate the episodes
        # for start_index in tqdm(starting_indexes, desc='Episode Progress', unit='start_index', leave=True):
        for start_index in starting_indexes:
            # Randomly select a start index for the data slice
            data_slice = data.iloc[start_index::]
            data_feed = DataFeed(data_slice, levels=10)
            final_policy_loss, all_values_per_step, all_states_per_step, all_action_per_step, all_log_prob_per_step, all_rewards_per_step, cumulative_reward, done = env.run_episode_multistep(data_feed, schedule_twap, side)
            print(f"Epoch {epoch} of {epochs}, Episode {episode} of {len(starting_indexes)} - idx: {start_index} Log_p: {([round(i.item(), 2) for i in all_log_prob_per_step])}  Act:  {([round(i,2) for i in all_action_per_step])} rewards: {([round(i, 2) for i in all_rewards_per_step])}, Cum reward: {(round(cumulative_reward, 2))} policy loss: {(round(final_policy_loss.item(), 2))}")
            # Update the progress bar description with the latest metrics
            epoch_rewards.update({data.index[start_index]: cumulative_reward})
            epoch_actions.update({data.index[start_index]: torch.tensor(all_action_per_step).numpy()})
            episode += 1
        avg_reward = np.mean(list(epoch_rewards.values()))
        # tqdm.write(f"AVERAGE REWARD PER EPOCH: {avg_reward:.2f}")
        print(f"AVERAGE REWARD PER EPOCH: {avg_reward:.2f}")
        all_average_rewards.update({epoch: np.mean(list(epoch_rewards.values()))})
        all_average_policy_losses.update({epoch: torch.mean(torch.tensor(list(epoch_policy_losses.values()))).item()})
        all_actions.update({epoch: epoch_actions})
    # get metrics
    print("DONE TRAINING")
    all_average_rewards = pd.Series(all_average_rewards)
    all_average_policy_losses = pd.Series(all_average_policy_losses)
    final_epoch_actions = all_actions[list(all_actions.keys())[-1]]
    final_epoch_actions_df = pd.DataFrame.from_dict(final_epoch_actions, orient='index')

    # Initialize environment with given agent and schedule, run and generate the RL agent schedule
    data_feed = DataFeed(data, levels=10)
    time_start = random.choice(data.index)
    print(time_start)
    env.reset(schedule_twap, data_feed, side, start_time_at=time_start)
    all_values_per_step, all_states_per_step, all_action_per_step, all_log_prob_per_step, all_rewards_per_step, cumulative_reward, done = env.multistep_infer()
    schedule_rl = {}
    for i in range(len(all_action_per_step)):
        schedule_rl.update({i: ScheduleExecutor.LimitEvent(side, max_time, max_volume * all_action_per_step[i])})
    schedule_rl.update({len(schedule_rl): ScheduleExecutor.MarketEvent(side, max_time)})

    # plot the benchmark twap schedule
    data_gen_twap = data_feed.raw_data_yield_fun(time_start)
    trades_collect_all_twap, tick_data_bids_all_twap, tick_data_asks_all_twap, carryover_vol_twap, all_filled_twap, placed_limit_orders_collect_all_twap, placed_market_orders_collect_all_twap = run_schedule(schedule=schedule_twap, data_gen=data_gen_twap, levels=10, resubmit_each_tick=resubmit_each_tick, price_level_offset=price_level_offset)
    plot_lob_data_with_volume(schedule_twap, data_gen_twap, resubmit_each_tick=resubmit_each_tick, price_level_offset=price_level_offset)

    # plot the RL agents schedule
    data_gen_rl = data_feed.raw_data_yield_fun(time_start)
    trades_collect_all_rl, tick_data_bids_all_rl, tick_data_asks_all_rl, carryover_vol_rl, all_filled_rl, placed_limit_orders_collect_all_rl, placed_market_orders_collect_all_rl = run_schedule(schedule=schedule_rl, data_gen=data_gen_rl, levels=10, resubmit_each_tick=resubmit_each_tick, price_level_offset=price_level_offset)
    plot_lob_data_with_volume(schedule_rl, data_gen_rl, resubmit_each_tick=resubmit_each_tick, price_level_offset=price_level_offset)

    # Initialize cumulative reward and execution values
    cumulative_reward = 0
    exec_price_rl_cumulative = 0
    exec_volume_rl_cumulative = 0
    exec_price_twap_cumulative = 0
    exec_volume_twap_cumulative = 0
    cumulative_rewards = {}
    keys_rl = set(trades_collect_all_rl.keys())
    keys_twap = set(trades_collect_all_twap.keys())
    # Find the intersection of both sets to get common keys
    common_keys = keys_rl.intersection(keys_twap)
    common_keys = sorted(common_keys)
    for timestep in common_keys:
        trades_rl = trades_collect_all_rl[timestep]
        trades_twap = trades_collect_all_twap[timestep]

        # Update RL execution price and volume for this timestep
        for trade in trades_rl:
            for p, v in trade.price_levels.items():
                exec_price_rl_cumulative += p * v
                exec_volume_rl_cumulative += v

        # Update TWAP execution price and volume for this timestep
        for trade in trades_twap:
            for p, v in trade.price_levels.items():
                exec_price_twap_cumulative += p * v
                exec_volume_twap_cumulative += v

        # Calculate VWAPs for RL and TWAP up to current timestep
        rl_vwap = (exec_price_rl_cumulative / exec_volume_rl_cumulative) if exec_volume_rl_cumulative > 0 else 0
        twap_vwap = (exec_price_twap_cumulative / exec_volume_twap_cumulative) if exec_volume_twap_cumulative > 0 else 0

        # Update the cumulative reward
        cumulative_reward += twap_vwap - rl_vwap

        # Optionally store the cumulative reward at each timestep if needed for analysis
        cumulative_rewards[timestep] = cumulative_reward
    cumulative_rewards_df = pd.Series(cumulative_rewards).sort_index()
    cumulative_rewards_df.plot()
