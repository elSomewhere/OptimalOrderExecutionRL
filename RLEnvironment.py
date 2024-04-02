from collections import deque
from typing import Tuple, List, Optional
import torch
from torch import optim
from ScheduleExecutor import ScheduleExecutor


class RLEnvironment:
    '''
    - The RL environment benchmarks a TWAP order execution schedule vs the RL modified TWAP order execution setup. RL does not modify timing, only volume of a given order in the schedule
    - step() simulates the execution of a single order in the schedule. Multiple calls to step() collects multiple rewards, when the schedule is exhausted the policy is updated with this array of multiple rewards. These multiple rewards are noisy, but less delayed.
    - multistep() simulates the entire schedule and provides only a single cumulative reward at the end of the schedule. This reward is more accurate, but also delayed.
    - A given step executes a given order. The limit order is backtested thru time until either it is exhausted or the next order in the schedule arrives.
    - the actual backtest is a bit involved given we only have L2 data. While a resting partially filled limit order is simulated, the actual mechanics in the sim procedure re-submits the limit order each tick to have a more conservative PnL estimation.
    - why more conservative: because this requires each tick to make a sufficient downwards movement to fill a bid limit that is always re-conditioned to rest slightly below the newest top bid, instead of its original nesting level farther above
    - (that one would always fill if resting in a downwards market)
    - a given episode executes a given schedule. Hence, step() provides an array of multiple rewards per schedule, multistep() provides a single reward per schedule (step provides rewards during the schedule, multistep calculated at the end of the schedule)
    '''

    def __init__(self, policy_network, buffer_size, resubmit_each_tick=True, price_depth_penalty=-0.1):
        self.executor: ScheduleExecutor = ScheduleExecutor(resubmit_each_tick, price_depth_penalty)
        self.data_feed = None
        self.levels = 10
        self.policy_network = policy_network
        self.optimizer = optim.Adam(self.policy_network.parameters(), lr=0.01)
        self.state_queue = deque(maxlen=buffer_size)
        self.resubmit_each_tick = resubmit_each_tick
        self.price_depth_penalty = price_depth_penalty

    def reset(self, schedule, data_feed, side, start_time_at):
        '''
        Resets data feeds and backtester and states fields
        '''
        self.data_feed = data_feed
        self.executor.reset(schedule, data_feed, side, start_time_at)

    def step(self):
        '''
        reward is caluclated at each individual step
        '''
        current_state = self.get_global_state()
        self.state_queue.append(current_state)
        statebuffer = self.get_buffered_global_state()
        action, log_probs, value = self.select_agent_action(statebuffer)
        done, trades_rl, trades_twap = self.executor.step(action)
        step_reward = self.executor.get_reward(trades_rl, trades_twap)
        return action, log_probs, value, step_reward, statebuffer, done

    def step_infer(self):
        '''
        reward is caluclated at each individual step
        '''
        current_state = self.get_global_state()
        self.state_queue.append(current_state)
        statebuffer = self.get_buffered_global_state()
        action, log_probs, value = self.select_agent_action_inference(statebuffer)
        done, trades_rl, trades_twap = self.executor.step(action)
        step_reward = self.executor.get_reward(trades_rl, trades_twap)
        return action, log_probs, value, step_reward, statebuffer, done

    def multistep(self):
        '''
        If this is called, rewards are calculated in a cumulative fashion at the end of a given schedule
        '''
        # this executes a complete bucket
        done = False
        all_action_per_step: List[float] = []
        all_log_prob_per_step = []
        all_states_per_step = []
        all_rewards_per_step = []
        all_values_per_step = []
        while not done:
            action, log_probs, value, step_reward, statebuffer, done = self.step()
            all_states_per_step += [statebuffer]
            all_action_per_step += [action]
            all_log_prob_per_step += [log_probs]
            all_rewards_per_step += [step_reward]
            all_values_per_step += [value]
        cumulative_reward = self.executor.get_reward(self.executor.trades_collect_rl, self.executor.trades_collect_twap)
        return all_values_per_step, all_states_per_step, all_action_per_step, all_log_prob_per_step, all_rewards_per_step, cumulative_reward, done

    def multistep_infer(self):
        '''
        If this is called, rewards are calculated in a cumulative fashion at the end of a given schedule
        '''
        # this executes a complete bucket
        done = False
        all_action_per_step: List[float] = []
        all_log_prob_per_step = []
        all_states_per_step = []
        all_rewards_per_step = []
        all_values_per_step = []
        while not done:
            action, log_probs, value, step_reward, statebuffer, done = self.step_infer()
            all_states_per_step += [statebuffer]
            all_action_per_step += [action]
            all_log_prob_per_step += [log_probs]
            all_rewards_per_step += [step_reward]
            all_values_per_step += [value]
        cumulative_reward = self.executor.get_reward(self.executor.trades_collect_rl, self.executor.trades_collect_twap)
        return all_values_per_step, all_states_per_step, all_action_per_step, all_log_prob_per_step, all_rewards_per_step, cumulative_reward, done

    def run_episode_multistep(self, data_feed, schedule, side, start_time_at=0):
        self.reset(schedule, data_feed, side, start_time_at)
        all_values_per_step, all_states_per_step, all_action_per_step, all_log_prob_per_step, all_rewards_per_step, cumulative_reward, done = self.multistep()
        if not done:
            raise Exception("Multistep is supposed to be done")
        final_policy_loss = self.update_policy_multistep(all_values_per_step, all_states_per_step, all_action_per_step, all_log_prob_per_step, all_rewards_per_step, cumulative_reward, done)
        return final_policy_loss, all_values_per_step, all_states_per_step, all_action_per_step, all_log_prob_per_step, all_rewards_per_step, cumulative_reward, done

    def run_episode_multistep_infer(self, data_feed, schedule, side, start_time_at=0):
        self.reset(schedule, data_feed, side, start_time_at)
        all_values_per_step, all_states_per_step, all_action_per_step, all_log_prob_per_step, all_rewards_per_step, cumulative_reward, done = self.multistep_infer()
        if not done:
            raise Exception("Multistep is supposed to be done")
        return all_values_per_step, all_states_per_step, all_action_per_step, all_log_prob_per_step, all_rewards_per_step, cumulative_reward, done

    def get_agent_state(self) -> torch.tensor:
        '''
        get current state of the agent
        '''
        pending_vol_rl = self.executor.total_volume_rl - self.executor.total_remaining_vol_rl
        pending_ratio = pending_vol_rl / self.executor.total_volume_rl
        exec_ratio = pending_vol_rl / self.executor.total_volume_rl
        time_ratio = self.executor.index / self.executor.length
        return torch.tensor([pending_ratio, exec_ratio, time_ratio])

    def get_lob_state(self) -> torch.tensor:
        '''
        get current (private) state of the environment / market
        '''
        # Initialize an empty tensor for the specified number of levels
        res = torch.empty(self.data_feed.levels * 4)
        # Sort and get the top N levels of bids and asks
        sorted_bids = sorted(self.executor.market_executor_rl.backtester.lob.bids.items(), key=lambda x: -x[0])[:self.data_feed.levels]
        sorted_asks = sorted(self.executor.market_executor_rl.backtester.lob.asks.items(), key=lambda x: x[0])[:self.data_feed.levels]
        # Fill the tensor with bid and ask data
        for i in range(self.data_feed.levels):
            # For bids
            if i < len(sorted_bids):
                bid_price, bid_volume = sorted_bids[i]
            else:
                bid_price, bid_volume = 0, 0  # Default values if level is missing
            # For asks
            if i < len(sorted_asks):
                ask_price, ask_volume = sorted_asks[i]
            else:
                ask_price, ask_volume = 0, 0  # Default values if level is missing
            # Fill the tensor
            res[i * 4] = bid_price
            res[i * 4 + 1] = bid_volume
            res[i * 4 + 2] = ask_price
            res[i * 4 + 3] = ask_volume
        return res

    def get_global_state(self):
        '''
        Get combined agent and environment state
        '''
        # Combine market state and agent state
        market_state = self.get_lob_state()  # Assuming this function is implemented
        agent_state = self.get_agent_state()
        observation = torch.cat([market_state, agent_state])
        return observation

    def get_buffered_global_state(self):
        '''
        Get combined agent and environment state, buffered thru time
        '''
        statebuffer = list(self.state_queue)
        reshaped_input = torch.stack(statebuffer, 0).unsqueeze(0).transpose(1, 2)
        return reshaped_input

    def select_agent_action(self, statebuffer: torch.Tensor) -> Tuple[float, Optional[torch.Tensor], Optional[torch.Tensor]]:
        return self._select_agent_action(statebuffer)

    def select_agent_action_inference(self, statebuffer: torch.Tensor) -> Tuple[float, Optional[torch.Tensor], Optional[torch.Tensor]]:
        return self._select_agent_action_inference(statebuffer)

    def update_policy_multistep(self, all_values_per_step: List[torch.Tensor], all_states_per_step: List[torch.Tensor], all_action_per_step: List[float], all_log_prob_per_step: List[Optional[torch.Tensor]], all_rewards_per_step: List[float], cumulative_reward: float, done: bool) -> torch.Tensor:
        return self._update_policy_multistep(all_values_per_step, all_states_per_step, all_action_per_step, all_log_prob_per_step, all_rewards_per_step, cumulative_reward, done)

    def _update_policy_multistep(self, all_values_per_step: List[torch.Tensor], all_states_per_step: List[torch.Tensor], all_action_per_step: List[float], all_log_prob_per_step: List[Optional[torch.Tensor]], all_rewards_per_step: List[float], cumulative_reward: float, done: bool) -> torch.Tensor:
        pass

    def _select_agent_action(self, state: torch.Tensor) -> Tuple[float, Optional[torch.Tensor], Optional[torch.Tensor]]:
        pass

    def _select_agent_action_inference(self, state: torch.Tensor) -> Tuple[float, Optional[torch.Tensor], Optional[torch.Tensor]]:
        pass