The following code implements a "toy" RL example for optimal order execution. Due to the nature of data used (less granular L2 / MBP as opposed to full granular L3 / MBO), the RL environment relies on a set of assumptions that are a departure from a fully realistic simulated market. Despite the simplified model, the general approach and employed concepts are transferrable to a more complete RL application, which I shortly discuss in more detail at the end of the notebook.

The dataset used in this example is way too small to realistically train an RL agent that is able to generalize over multiple conditions. Instead, we knowingly overfit the RL framework on the given small dataset, avoiding the train / test / validate split otherwise used in a proper setting. While the result is not usable in practice, it is sufficient to prove that the framework has an ability to learn on a provided dataset.

On a further note - The proposed setup also puts a bit more focus on the environment design than on the actual RL learning methodology. This is deliberate - we'd rather have a properly designed RL environment and iterate on the RL learning method rather than have the newest RL learning method but a crappy environment to operate in (garbage in, garbage out...). Plus, there's a lot of available RL libraries to play around with, but the simulation environment for financial markets is a much more closed space, especially if we want full simulation granularity.

We will initially discuss the environment design. Towards the end of the notebook, the environement designed and discussed is employed in an example triaining loop. This particular implementation uses a actor-critic approach, cumulative multistep rewards, achieved via benchmarking against a vanilla TWAP order execution strategy with an LSTM and some tricks for smartly parsing the LOB data in the policy network.

This is purely didactic code for presentation purposes. 
For fully featured performant exchange simulation environments with fully modelled stochastic agents, messaging, event & latency simuations, contact Author:

estebanlanter86@gmail.com
