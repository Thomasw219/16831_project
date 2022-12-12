import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from collections import deque
from tqdm import tqdm

from .utils import FreezeParameters
from .memory import Memory
from .networks import RSSM, FusionDecoder, FusionEncoder, MLPDistribution

class DreamerV2:
    def __init__(self, cfg, obs_shape, action_dim, name):
        self.cfg = cfg
        self.obs_shape = obs_shape
        self.action_dim = action_dim
        self.mem = Memory(cfg.memory)
        self.wm = WorldModel(cfg.world_model, obs_shape, action_dim)
        self.feat_dim = self.wm.rssm.get_output_dim()
        self.wm.to(self.cfg.device)
        self.wm_opt = optim.Adam(self.wm.parameters(), lr=self.cfg.world_model.lr, eps=self.cfg.world_model.eps, weight_decay=self.cfg.world_model.wd)
        self.behavior = ActorCritic(cfg.behavior, self.feat_dim, self.action_dim)
        self.behavior.to(self.cfg.device)

        self.state = None

        self.global_step = 0
        self.summary_writer = SummaryWriter(name) if name is not None else None

    def step(self, obs):
        with torch.no_grad():
            state, feat = self.wm.encode_obs(self.wm.preprocess_obs(obs, device=self.cfg.device), self.state, self.cfg.device)
            action = self.behavior.get_action(feat)
            self.state = self.wm.encode_action(action, state)
        return action.cpu().squeeze().numpy()

    def random_action(self):
        return np.random.uniform(low=-1, high=1, size=(2,))

    def reset(self):
        self.mem.reset_episode()
        self.state = None

    def log_episode(self):
        metrics = {"episodes/" + k : v for k, v in self.mem.last_episode_metrics().items()}
        self.log(metrics)

    def save_memory(self):
        # TODO: DO
        pass

    def load_memory(self):
        raise NotImplementedError("No load memory function yet :(")

    def load_networks(self, wm_path, behavior_path):
        self.wm.load_state_dict(torch.load(wm_path))
        self.behavior.load_state_dict(torch.load(behavior_path))

    def save_networks(self, desc):
        torch.save(self.wm.state_dict(), f"./wm_{desc}.pth")
        torch.save(self.behavior.state_dict(), f"./behavior_{desc}.pth")

    def record(self, **transition):
        self.mem.step(**transition)

    def train_model_step(self, data):
        data['is_firsts'][0, :, 0] = 1
        with torch.autocast('cuda' if 'cuda' in self.cfg.device else 'cpu'):
            loss, _, _, metrics = self.wm.loss(data)
        self.wm_opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.wm.parameters(), self.cfg.world_model.clip)
        self.wm_opt.step()
        metrics = {"model/train/" + k : v for k, v in metrics.items()}
        return metrics

    def test_model_step(self, data):
        data['is_firsts'][0, :, 0] = 1
        with torch.autocast('cuda' if 'cuda' in self.cfg.device else 'cpu'):
            with torch.no_grad():
                _, _, _, metrics = self.wm.loss(data)
        metrics = {"model/test/" + k : v for k, v in metrics.items()}
        return metrics

    def train_behavior_step(self, data):
        data['is_firsts'][0, :, 0] = 1
        with torch.autocast('cuda' if 'cuda' in self.cfg.device else 'cpu'):
            with torch.no_grad():
                _, _, posts, _ = self.wm.loss(data)
            post = {k : torch.concat([post[k] for post in posts], dim=0).detach() for k in posts[0].keys()}
            with FreezeParameters(self.wm):
                trajectories = self.generate_trajectories(post, data['is_lasts'].reshape(-1, 1))
                if self.global_step % 25 == 0:
                    self.summary_writer.add_histogram("action_means_0", trajectories['action_means'][:, :, 0].flatten(), global_step=self.global_step)
                    self.summary_writer.add_histogram("action_means_1", trajectories['action_means'][:, :, 1].flatten(), global_step=self.global_step)
                actor_loss, critic_loss, metrics = self.behavior.loss(trajectories)

        self.behavior.actor_opt.zero_grad()
        actor_loss.backward()
        actor_grad_norm = nn.utils.clip_grad_norm_(self.behavior.actor.parameters(), self.cfg.behavior.actor_clip)
        metrics['actor_grad_norm'] = actor_grad_norm
        self.behavior.actor_opt.step()

        self.behavior.critic_opt.zero_grad()
        critic_loss.backward()
        critic_grad_norm = nn.utils.clip_grad_norm_(self.behavior.critic.parameters(), self.cfg.behavior.critic_clip)
        metrics['critic_grad_norm'] = critic_grad_norm
        self.behavior.critic_opt.step()

        self.update_critic()

        metrics = {"behavior/train/" + k : v for k, v in metrics.items()}
        return metrics

    def test_behavior_step(self, data):
        data['is_firsts'][0, :, 0] = 1
        with torch.autocast('cuda' if 'cuda' in self.cfg.device else 'cpu'):
            with torch.no_grad():
                _, _, posts, _ = self.wm.loss(data)
                post = {k : torch.concat([post[k] for post in posts], dim=0).detach() for k in posts[0].keys()}
                trajectories = self.generate_trajectories(post, data['is_lasts'].reshape(-1, 1))
                if self.global_step % 25 == 0:
                    self.summary_writer.add_histogram("action_means_0", trajectories['action_means'][:, :, 0].flatten(), global_step=self.global_step)
                    self.summary_writer.add_histogram("action_means_1", trajectories['action_means'][:, :, 1].flatten(), global_step=self.global_step)
                _, _, metrics = self.behavior.loss(trajectories)
        metrics = {"behavior/test/" + k : v for k, v in metrics.items()}
        return metrics

    def train_offline_behavior_step(self, data):
        with torch.autocast('cuda' if 'cuda' in self.cfg.device else 'cpu'):
            _, _, posts, _ = self.wm.loss(data)
            feats = self.wm.rssm.get_feat_t_b(posts)
            is_lasts = data['is_lasts']
            data['feats'] = feats
            discounts = torch.where(is_lasts > self.cfg.termination_threshold, torch.zeros_like(is_lasts), self.cfg.behavior.gamma * torch.ones_like(is_lasts))
            weights = torch.cumprod(torch.cat([torch.ones_like(discounts[0:1]) * self.cfg.behavior.gamma, discounts[:-1]], dim=0), dim=0)
            data['discounts'] = discounts
            data['weights'] = weights
            actor_loss, critic_loss, metrics = self.behavior.offline_loss(data)

        self.behavior.actor_opt.zero_grad()
        actor_loss.backward()
        actor_grad_norm = nn.utils.clip_grad_norm_(self.behavior.actor.parameters(), self.cfg.behavior.actor_clip)
        metrics['actor_grad_norm'] = actor_grad_norm
        self.behavior.actor_opt.step()

        self.behavior.critic_opt.zero_grad()
        critic_loss.backward()
        critic_grad_norm = nn.utils.clip_grad_norm_(self.behavior.critic.parameters(), self.cfg.behavior.critic_clip)
        metrics['critic_grad_norm'] = critic_grad_norm
        self.behavior.critic_opt.step()

        self.update_critic()

        metrics = {"behavior/train/" + k : v for k, v in metrics.items()}
        return metrics

    def update_critic(self):
        self.behavior.critic_update_counter += 1
        if self.cfg.behavior.update_type == 'hard':
            self.behavior.critic_update_counter %= self.behavior.cfg.hard.critic_update_every
            if self.behavior.critic_update_counter == 0:
                self.behavior.hard_update_target_critic()
        elif self.cfg.behavior.update_type == 'polyak':
            self.behavior.polyak_update_target_critic()
        else:
            raise NotImplementedError("Critic update type not implemented")

    def test_offline_behavior_step(self, data):
        with torch.autocast('cuda' if 'cuda' in self.cfg.device else 'cpu'):
            with torch.no_grad():
                _, _, posts, _ = self.wm.loss(data)
                feats = self.wm.rssm.get_feat_t_b(posts)
                is_lasts = data['is_lasts']
                data['feats'] = feats
                discounts = torch.where(is_lasts > self.cfg.termination_threshold, torch.zeros_like(is_lasts), self.cfg.behavior.gamma * torch.ones_like(is_lasts))
                weights = torch.cumprod(torch.cat([torch.ones_like(discounts[0:1]) * self.cfg.behavior.gamma, discounts[:-1]], dim=0), dim=0)
                data['discounts'] = discounts
                data['weights'] = weights
                _, _, metrics = self.behavior.offline_loss(data)

        metrics = {"behavior/test/" + k : v for k, v in metrics.items()}
        return metrics

    def train_test_model(self, steps):
        for _ in tqdm(range(steps), desc="Model Training"):
            data = self.mem.train_sample(self.cfg.device)
            train_metrics = self.train_model_step(data)
            if self.global_step % self.cfg.test_every == 0:
                test_data = self.mem.test_sample(self.cfg.device)
                test_metrics = self.test_model_step(test_data)
                metrics = {**train_metrics, **test_metrics}
            else:
                metrics = train_metrics
            self.log(metrics)

    def train_test_behavior(self, steps):
        for _ in tqdm(range(steps), desc="Imagination Behavior Training"):
            data = self.mem.train_sample(self.cfg.device)
            train_metrics = self.train_behavior_step(data)
            if self.global_step % self.cfg.test_every == 0:
                test_data = self.mem.test_sample(self.cfg.device)
                test_metrics = self.test_behavior_step(test_data)
                metrics = {**train_metrics, **test_metrics}
            else:
                metrics = train_metrics
            self.log(metrics)

    def train_test_offline_behavior(self, steps):
        for _ in tqdm(range(steps), desc="Offline Behavior Training"):
            data = self.mem.train_sample(self.cfg.device)
            train_metrics = self.train_offline_behavior_step(data)
            if self.global_step % self.cfg.test_every == 0:
                test_data = self.mem.test_sample(self.cfg.device)
                test_metrics = self.test_offline_behavior_step(test_data)
                metrics = {**train_metrics, **test_metrics}
            else:
                metrics = train_metrics
            self.log(metrics)

    def train_test_joint(self, steps):
        for _ in tqdm(range(steps), desc="Joint Model/Behavior Training"):
            data = self.mem.train_sample(self.cfg.device)
            model_train_metrics = self.train_model_step(data)
            behavior_train_metrics = self.train_behavior_step(data)
            if self.global_step % self.cfg.test_every == 0:
                test_data = self.mem.test_sample(self.cfg.device)
                model_test_metrics = self.test_model_step(test_data)
                behavior_test_metrics = self.test_behavior_step(test_data)
                metrics = {**model_train_metrics, **behavior_train_metrics, **model_test_metrics, **behavior_test_metrics}
            else:
                metrics = {**model_train_metrics, **behavior_train_metrics}
            self.log(metrics)

    def train_test_offline_joint(self, steps):
        for _ in tqdm(range(steps), desc="Joint Model/Behavior Training"):
            data = self.mem.train_sample(self.cfg.device)
            model_train_metrics = self.train_model_step(data)
            behavior_train_metrics = self.train_offline_behavior_step(data)
            if self.global_step % self.cfg.test_every == 0:
                test_data = self.mem.test_sample(self.cfg.device)
                model_test_metrics = self.test_model_step(test_data)
                behavior_test_metrics = self.test_offline_behavior_step(test_data)
                metrics = {**model_train_metrics, **behavior_train_metrics, **model_test_metrics, **behavior_test_metrics}
            else:
                metrics = {**model_train_metrics, **behavior_train_metrics}
            self.log(metrics)

    def generate_trajectories(self, start_state, is_last):
        beginning_is_last = is_last
        prev_state = start_state
        prev_feat = self.wm.rssm.get_feat_b(start_state)
        feats = [prev_feat]
        actions = [torch.zeros_like(self.behavior.actor.forward_reparameterize(prev_feat)[0])]
        action_means = []
        action_entropies = []
        for _ in range(self.cfg.behavior.horizon):
            action, action_mean, _, action_entropy = self.behavior.actor.forward_reparameterize(prev_feat)
            state = self.wm.rssm.img_step(prev_state, action, sample=True)
            feat = self.wm.rssm.get_feat_b(state)
            feats.append(feat)
            actions.append(action)
            action_means.append(action_mean)
            action_entropies.append(action_entropy)
            prev_state = state
            prev_feat = feat

        feats = torch.stack(feats)
        actions = torch.stack(actions)
        action_means = torch.stack(action_means)
        action_entropies = torch.stack(action_entropies)
        is_lasts = self.wm.discount_decoder(feats).mean
        is_lasts[0] = beginning_is_last
        discounts = torch.where(is_lasts > self.cfg.termination_threshold, torch.zeros_like(is_lasts), self.cfg.behavior.gamma * torch.ones_like(is_lasts))
        rewards = self.wm.reward_decoder(feats).mean
        weights = torch.cumprod(torch.cat([torch.ones_like(discounts[0:1]) * self.cfg.behavior.gamma, discounts[:-1]], dim=0), dim=0)
        return dict(
            feats=feats,
            discounts=discounts,
            rewards=rewards,
            weights=weights,
            actions=actions,
            action_means=action_means,
            action_entropies=action_entropies,
        )

    def log(self, metrics):
        if self.summary_writer is None:
            return
        memory_metrics = {"memory/" + k : v for k, v in self.mem.get_metrics().items()}
        metrics = {**metrics, **memory_metrics}
        for k, v in metrics.items():
            self.summary_writer.add_scalar(k, v, self.global_step)
        if self.global_step % self.cfg.video_every == 0:
            if len(self.mem.test_episodes) > 0:
                real_images, reconstructed_images = self.wm.create_video(self.mem.test_sample(self.cfg.device))
                self.summary_writer.add_images("test_goal_real", real_images[:, 0, 0].unsqueeze(1).repeat(1, 3, 1, 1), self.global_step, dataformats='NCHW')
                self.summary_writer.add_images("test_goal_reconstructed", reconstructed_images[:, 0, 0].unsqueeze(1).repeat(1, 3, 1, 1), self.global_step, dataformats='NCHW')
            if len(self.mem.train_episodes) > 0:
                real_images, reconstructed_images = self.wm.create_video(self.mem.train_sample(self.cfg.device))
                self.summary_writer.add_images("train_goal_real", real_images[:, 0, 0].unsqueeze(1).repeat(1, 3, 1, 1), self.global_step, dataformats='NCHW')
                self.summary_writer.add_images("train_goal_reconstructed", reconstructed_images[:, 0, 0].unsqueeze(1).repeat(1, 3, 1, 1), self.global_step, dataformats='NCHW')

        if self.global_step % self.cfg.save_every == 0 and self.global_step > 0:
            self.save_networks(self.global_step)

        self.global_step += 1

class WorldModel(nn.Module):
    def __init__(self, cfg, obs_shape, action_dim):
        super().__init__()
        self.cfg = cfg
        self.goal_shape = obs_shape['goal']
        self.vector_shape = obs_shape['vector']
        self.action_dim = action_dim

        self.encoder = FusionEncoder(self.goal_shape[0], self.vector_shape[0], cfg.goal_encoder, cfg.vector_encoder, cfg.fusion_encoder)
        self.rssm = RSSM(action_dim, self.encoder.get_output_dim(), **cfg.rssm)
        feature_dim = self.rssm.get_output_dim()
        self.obs_decoder = FusionDecoder(feature_dim, self.goal_shape[0], self.vector_shape[0], cfg.goal_decoder, cfg.vector_decoder)
        self.reward_decoder = MLPDistribution(feature_dim, **cfg.reward_decoder)
        self.discount_decoder = MLPDistribution(feature_dim, **cfg.discount_decoder)

    def loss(self, data, state=None):
        data = self.preprocess(data)
        observations = data['observations']
        batch_shape = observations['goals'].shape[:-len(self.img_shape)]
        batch_obs = dict(
            images=observations['goals'].reshape((-1,) + self.img_shape),
            vectors=observations['vectors'].reshape((-1,) + self.vector_shape),
        )
        embeddings = self.encoder(batch_obs)
        embeddings = embeddings.reshape(batch_shape + (embeddings.shape[-1],))

        posts, priors = self.rssm.observe(embeddings, data['actions'], is_firsts=data['is_firsts'], post=state)
        kl_loss, kl_val, post_entropy, prior_entropy = self.rssm.kl_loss(posts, priors)

        feats = self.rssm.get_feat_t_b(posts)
        feats = feats.reshape((-1, feats.shape[-1]))
        image_loss, vector_loss, image_mse, vector_mse = self.obs_decoder.get_nll_mse(feats, batch_obs)
        reward_loss, reward_mse = self.reward_decoder.get_nll_mse(feats, data['rewards'].reshape((-1, 1)))
        discount_loss = -torch.mean(self.discount_decoder(feats).log_prob(data['is_lasts'].reshape((-1, 1))))
        model_loss = self.cfg.image_scale * image_loss + self.cfg.vector_scale * vector_loss + self.cfg.reward_scale * reward_loss + self.cfg.discount_scale * discount_loss + self.cfg.beta * kl_loss

        metrics = dict(
            kl_loss=kl_loss,
            kl_val=kl_val,
            image_loss=image_loss,
            vector_loss=vector_loss,
            reward_loss=reward_loss,
            discount_loss=discount_loss,
            model_loss=model_loss,
            prior_entropy=prior_entropy,
            post_entropy=post_entropy,
            image_mse=image_mse,
            vector_mse=vector_mse,
            reward_mse=reward_mse,
        )
        last_state = posts[-1]
        return model_loss, last_state, posts, metrics

    def encode_obs(self, obs, prior=None, device='cpu'):
        if prior is None:
            prev_state = self.rssm.initial(1, torch.float32, device)
            prev_action = torch.zeros(1, self.action_dim, dtype=torch.float32, device=device)
            prior = self.rssm.img_step(prev_state, prev_action, sample=True)

        obs_embed = self.encoder(obs)
        logits = self.rssm.posterior_net(torch.cat([prior['deter'], obs_embed], dim=-1)).reshape(1, self.rssm.stoch, self.rssm.discrete)
        stoch = self.rssm.get_stoch(logits, sample=True)
        post = dict(
            logits=logits,
            stoch=stoch,
            deter=prior['deter'],
        )
        return post, self.rssm.get_feat_b(post)

    def encode_action(self, action, post):
        prior = self.rssm.img_step(post, action)
        return prior

    def decode(self, states):
        feats = self.rssm.get_feat_t_b(states)
        image_dist, vector_dist = self.obs_decoder(feats)
        reward_dist = self.reward_decoder(feats)
        discount_dist = self.discount_decoder(feats)
        return dict(
            observations=dict(
                images=image_dist.mean,
                vectors=vector_dist.mean,
            ),
            rewards = reward_dist.mean,
            is_lasts = discount_dist.mean,
        )

    def preprocess_obs(self, obs, dtype=torch.float32, device='cpu'):
        goal = torch.tensor(obs['desired_goal'], dtype=dtype, device=device).unsqueeze(0)
        vector = torch.tensor(obs['observation'], dtype=dtype, device=device).unsqueeze(0)
        return dict(
            goal=goal,
            vector=vector,
        )

    def preprocess(self, data):
        return data

class ActorCritic(nn.Module):
    def __init__(self, cfg, feature_dim, action_dim):
        super().__init__()
        self.cfg = cfg
        self.feature_dim = feature_dim

        self.actor = MLPDistribution(feature_dim, output_dim=action_dim, **cfg.actor)
        self.actor_opt = optim.Adam(self.actor.parameters(), **cfg.actor_opt)
        self.critic = MLPDistribution(feature_dim, **cfg.critic)
        self.critic_opt = optim.Adam(self.actor.parameters(), **cfg.critic_opt)
        self.target_critic = MLPDistribution(feature_dim, **cfg.critic)
        self.hard_update_target_critic()

        self.critic_update_counter = 0

    def get_action(self, feat):
        return self.actor(feat).sample()

    def loss(self, trajectories):
        with FreezeParameters(self.target_critic):
            lambda_return = self.compute_lambda_return(trajectories)
        actor_loss, actor_metrics = self.get_actor_loss(trajectories, lambda_return)
        critic_loss, critic_metrics = self.get_critic_loss(trajectories, lambda_return)
        return actor_loss, critic_loss, {**actor_metrics, **critic_metrics}

    def offline_loss(self, trajectories):
        with FreezeParameters(self.target_critic):
            lambda_return = self.compute_lambda_return(trajectories)
        actor_bc_loss, actor_bc_metrics = self.get_actor_bc_loss(trajectories, lambda_return)
        critic_loss, critic_metrics = self.get_critic_loss(trajectories, lambda_return)
        return actor_bc_loss, critic_loss, {**actor_bc_metrics, **critic_metrics}

    def compute_lambda_return(self, trajectories):
        feats = trajectories['feats']
        rewards = trajectories['rewards']
        discounts = trajectories['discounts']
        target_values = self.target_critic(feats).mean
        trajectories['target_values'] = target_values
        indices = reversed(range(rewards.shape[0]))
        v = deque()
        for i in indices:
            if len(v) == 0:
                v_t = target_values[i]
            else:
                v_tp1 = (1 - self.cfg.lamb) * target_values[i + 1] + self.cfg.lamb * v[0]
                v_t = rewards[i] + discounts[i] * v_tp1
            v.appendleft(v_t)
        return torch.stack(list(v))

    def get_actor_loss(self, trajectories, lambda_return):
        weights = trajectories['weights']
        weighted_return = torch.mean(lambda_return[1:-2] * weights[1:-2])
        entropy = torch.mean(trajectories['action_entropies'])
        mean_abs = torch.abs(trajectories['action_means'])
        pre_tanh_bound_penalty = torch.mean(torch.where(mean_abs > self.cfg.pre_tanh_bound, mean_abs - self.cfg.pre_tanh_bound, torch.zeros_like(mean_abs)))
        if self.cfg.policy_grad == 'dynamics':
            # Backprop through dynamics to get policy gradient
            loss = -weighted_return - self.cfg.eta * entropy #+ self.cfg.pre_tanh_bound_penalty_scale * pre_tanh_bound_penalty
        elif self.cfg.policy_grad == 'reinforce':
            # REINFORCE policy gradient
            feats = trajectories['feats']
            action_log_prob = self.actor.forward(feats[:-2]).log_prob(trajectories['actions'][1:-1].detach())
            loss = torch.mean(-action_log_prob * (lambda_return[1:-1] - self.target_critic(feats[:-2]).mean).detach().squeeze(-1))
        metrics = dict(
            img_weighted_return=weighted_return,
            actor_loss=loss,
            entropy=entropy,
            pre_tanh_bound_penalty=pre_tanh_bound_penalty,
        )
        return loss, metrics

    def get_actor_bc_loss(self, trajectories, lambda_return):
        feats = trajectories['feats']
        actions = trajectories['actions']
        mean_action_log_prob = torch.mean(self.actor.forward(feats[:-1]).log_prob(actions[1:]))
        loss = -mean_action_log_prob

        weights = trajectories['weights']
        weighted_return = torch.mean(lambda_return[1:-2] * weights[1:-2])
        metrics = dict(
            action_log_prob=mean_action_log_prob,
            img_weighted_return=weighted_return,
        )
        return loss, metrics

    def get_critic_loss(self, trajectories, lambda_return):
        lambda_return = lambda_return.detach()
        feats = trajectories['feats'].detach()
        weights = trajectories['weights'].detach()
        v_dist = self.critic(feats[:-1])
        weighted_nll = -torch.mean(v_dist.log_prob(lambda_return[:-1]) * weights[:-1].squeeze(-1))
        metrics=dict(
            img_weighted_value_nll=weighted_nll,
            critic_loss=weighted_nll,
        )
        return weighted_nll, metrics

    def hard_update_target_critic(self):
        for target_param, param in zip(self.target_critic.parameters(), self.critic.parameters()):
            target_param.data.copy_(param.data)

    def polyak_update_target_critic(self):
        for target_param, param in zip(self.target_critic.parameters(), self.critic.parameters()):
            target_param.data.copy_((1 - self.cfg.polyak.alpha) * target_param.data + self.cfg.polyak.alpha * param)

    def to(self, device):
        self.actor.to(device)
        self.actor_opt = optim.Adam(self.actor.parameters(), **self.cfg.actor_opt)
        self.critic.to(device)
        self.critic_opt = optim.Adam(self.critic.parameters(), **self.cfg.critic_opt)
        self.target_critic.to(device)
        self.hard_update_target_critic()
