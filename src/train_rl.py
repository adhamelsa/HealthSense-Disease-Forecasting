import gym
from stable_baselines3 import DQN

# بيئة تجريبية (CartPole مثال)
env = gym.make('CartPole-v1')
model = DQN('MlpPolicy', env, verbose=1)
model.learn(total_timesteps=10000)
model.save('models/dqn_cartpole')
