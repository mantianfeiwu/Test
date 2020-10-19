#   Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

#-*- coding: utf-8 -*-

import gym
import numpy as np
import parl
from parl.utils import logger

from agent import Agent
from model import Model
from algorithm import DDPG  # from parl.algorithms import DDPG
from parl.utils import action_mapping
from parl.utils import ReplayMemory
from rlschool import make_env


ACTOR_LR = 1e-3  # Actor网络的 learning rate
CRITIC_LR = 1e-3  # Critic网络的 learning rate
GAMMA = 0.99  # reward 的衰减因子
TAU = 0.001  # 软更新的系数
MEMORY_SIZE = int(1e6)  # 经验池大小
MEMORY_WARMUP_SIZE = 500  # 预存一部分经验之后再开始训练
BATCH_SIZE = 256
REWARD_SCALE = 0.01  # reward 缩放系数
NOISE = 0.05  # 动作噪声方差
TRAIN_EPISODE = 1e6  # 训练的总episode数


# 训练一个episode
def run_episode(agent, env, rpm):
    obs = env.reset()
    total_reward = 0
    steps = 0
    while True:
        steps += 1
        batch_obs = np.expand_dims(obs, axis=0)
        action = agent.predict(batch_obs.astype('float32'))
        action = np.squeeze(action)
        # 增加探索扰动, 输出限制在 [-1.0, 1.0] 范围内
        action = np.clip(np.random.normal(action, NOISE), -1.0, 1.0)
        # 动作映射到对应的实际动作取值范围内，
        action = action_mapping(action, env.action_space.low[0], env.action_space.high[0])

        next_obs, reward, done, info = env.step(action)


        rpm.append(obs, action, REWARD_SCALE * reward, next_obs, done)

        if rpm.size() > MEMORY_WARMUP_SIZE:
            (batch_obs, batch_action, batch_reward, batch_next_obs,
            batch_done) = rpm.sample_batch(BATCH_SIZE)
            critic_cost = agent.learn(batch_obs, batch_action, batch_reward, batch_next_obs,
                                batch_done)

        obs = next_obs
        total_reward += reward

        if done:
            break
    return total_reward, steps


# 评估 agent, 跑 5 个episode，总reward求平均
def evaluate(env, agent, render=False):
    eval_reward = []
    for i in range(5):
        obs = env.reset()
        total_reward = 0
        steps = 0
        while True:
            batch_obs = np.expand_dims(obs, axis=0)
            action = agent.predict(batch_obs.astype('float32'))
            action = np.clip(action, -1.0, 1.0)
            action = np.squeeze(action)
            action = action_mapping(action, env.action_space.low[0], env.action_space.high[0])

            steps += 1
            next_obs, reward, done, info = env.step(action)

            obs = next_obs
            total_reward += reward

            if render:
                env.render()
            if done:
                break
        eval_reward.append(total_reward)
    return np.mean(eval_reward)


def main():
    env = make_env("Quadrotor", task="hovering_control")

    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    # 使用PARL框架创建agent
    model = Model(act_dim)
    algorithm = DDPG(
        model, gamma=GAMMA, tau=TAU, actor_lr=ACTOR_LR, critic_lr=CRITIC_LR)
    agent = Agent(algorithm, obs_dim, act_dim)

    # 创建经验池
    rpm = ReplayMemory(int(MEMORY_SIZE), obs_dim, act_dim)


    episode = 0
    test_flag = 0
    while episode < TRAIN_EPISODE:
        total_reward = run_episode(agent, env, rpm)
        episode += 1
        if(episode // 10 >= test_flag):
            test_flag += 1
            eval_reward = evaluate(env, agent, render=False)
            logger.info('episode:{}    Test reward:{}'.format(
                episode, eval_reward))

            ckpt = 'model_dir/steps_{}.ckpt'.format(episode)
            agent.save(ckpt)
if __name__ == '__main__':
    main()
