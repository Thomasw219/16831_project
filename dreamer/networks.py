import torch
import torch.nn as nn
import torch.distributions as dists

def get_activation(activation):
    if activation == 'elu':
        return nn.ELU
    elif activation == 'relu':
        return nn.ReLU
    else:
        return NotImplementedError("Activation not implemented yet")

class StandardMLP(nn.Module):
    def __init__(self, input_dim, layer_sizes=[400, 400, 400, 400], output_dim=1, activate_last=False, activation='elu'):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.layer_sizes = layer_sizes
        self.activation = get_activation(activation)

        if len(layer_sizes) == 0:
            if not activate_last:
                self.network = nn.Linear(input_dim, output_dim)
            else:
                self.network = nn.Sequential(nn.Linear(input_dim, output_dim), self.activation())
            return

        layer_list = [nn.Linear(self.input_dim, self.layer_sizes[0]), self.activation()]
        for i in range(len(self.layer_sizes) - 1):
            layer_list.append(nn.Linear(self.layer_sizes[i], self.layer_sizes[i + 1]))
            layer_list.append(self.activation())
        layer_list.append(nn.Linear(self.layer_sizes[-1], output_dim))
        if activate_last:
            layer_list.append(self.activation())

        self.network = nn.Sequential(*layer_list)

    def get_output_dim(self):
        return self.output_dim

    def forward(self, x):
        return self.network(x)

class ConvEncoder(nn.Module):
    def __init__(self, input_shape, cnn_depth=48, cnn_kernels=[4, 4, 4, 4], activation='elu'):
        super().__init__()
        self.input_shape = input_shape
        self.activation = get_activation(activation)

        layer_list = []
        last_depth = input_shape[0]
        for i, size in enumerate(cnn_kernels):
            depth = 2 ** i * cnn_depth
            layer_list.append(nn.Conv2d(last_depth, depth, size, 2))
            layer_list.append(self.activation())
            last_depth = depth

        self.network = nn.Sequential(*layer_list)

    def forward(self, x):
        batch_shape = x.shape[:-3]
        return self.network(x).reshape(batch_shape + (-1,))

    def get_output_dim(self):
        test = torch.zeros(1, *self.input_shape, dtype=self.network[0].weight.dtype)
        return self.forward(test).shape[-1]

class FusionEncoder(nn.Module):
    def __init__(self, image_input_shape, vector_input_dim, conv_encoder_params, vector_encoder_params, fusion_encoder_params):
        super().__init__()
        self.conv_encoder = ConvEncoder(image_input_shape, **conv_encoder_params)
        self.vector_encoder = StandardMLP(vector_input_dim, **vector_encoder_params)

        self.fusion_encoder = StandardMLP(self.conv_encoder.get_output_dim() + self.vector_encoder.get_output_dim(), **fusion_encoder_params)

    def get_output_dim(self):
        return self.fusion_encoder.get_output_dim()

    def forward(self, obs):
        image_features = self.conv_encoder.forward(obs['images'] if 'images' in obs.keys() else obs['image'])
        vector_features = self.vector_encoder.forward(obs['vectors'] if 'vectors' in obs.keys() else obs['vector'])
        return self.fusion_encoder.forward(torch.cat([image_features, vector_features], dim=-1))

class ConvDecoder(nn.Module):
    def __init__(self, input_dim, output_shape, cnn_depth=48, cnn_kernels=[5, 5, 6, 6], activation='elu'):
        super().__init__()
        self.input_shape = input_dim
        self.activation = get_activation(activation)

        self.initial_channels = 32 * cnn_depth
        self.linear = nn.Linear(input_dim, self.initial_channels)
        layer_list = []
        in_channels = self.initial_channels
        for i, size in enumerate(cnn_kernels):
            if i < len(cnn_kernels) - 1:
                depth = 2 ** (len(cnn_kernels) - i - 2) * cnn_depth
            else:
                depth = output_shape[0]
            layer_list.append(nn.ConvTranspose2d(in_channels, depth, size, 2))
            if i < len(cnn_kernels) - 1:
                layer_list.append(self.activation())
            in_channels = depth

        self.conv_layers = nn.Sequential(*layer_list)

    def forward(self, x):
        x = self.linear(x)
        x = x.reshape(-1, self.initial_channels, 1, 1)
        means = self.conv_layers(x)
        return dists.Independent(dists.Normal(means, torch.ones_like(means)), 3)

class MLPDistribution(nn.Module):
    def __init__(self, input_dim, layer_sizes=[400, 400, 400, 400], output_dim=1, activation='elu', dist='mse', min_std=0.1):
        super().__init__()
        self.dist = dist
        self.min_std = torch.tensor(min_std, requires_grad=False)
        if self.dist in ['mse', 'bernoulli']:
            self.output_dim = output_dim
        elif self.dist in ['normal', 'tanh_normal']:
            self.output_dim = output_dim * 2
        else:
            return NotImplementedError("Distribution type not implemented")
        self.mlp = StandardMLP(input_dim, layer_sizes=layer_sizes, output_dim=self.output_dim, activate_last=False, activation=activation)

    def forward(self, x):
        output = self.mlp(x)
        if self.dist == 'mse':
            return dists.Independent(dists.Normal(output, 1), 1)
        elif self.dist == 'bernoulli':
            return dists.Independent(dists.Bernoulli(torch.sigmoid(output)), 1)
        else:
            mean, std = output[..., :self.output_dim // 2], torch.max(nn.functional.softplus(output[..., self.output_dim // 2:]), self.min_std)
            if self.dist == 'normal':
                return dists.Independent(dists.Normal(mean, std), 1)
            if self.dist == 'tanh_normal':
                dist = dists.Independent(dists.Normal(mean, std), 1)
                return dists.TransformedDistribution(dist, [dists.TanhTransform()])

    def forward_reparameterize(self, x):
        output = self.mlp(x)
        if self.dist == 'tanh_normal':
            mean, std = output[..., :self.output_dim // 2], torch.max(nn.functional.softplus(output[..., self.output_dim // 2:]), self.min_std)
            normal_dist = dists.Independent(dists.Normal(mean, std), 1)
            return torch.tanh(mean + std * torch.normal(torch.zeros_like(mean), torch.ones_like(std))), mean, std, normal_dist.entropy()
        else:
            raise NotImplementedError("Function not implemented yet for this distribution type")

    def get_nll_mse(self, feat, target):
        dist = self.forward(feat)
        nll = -torch.mean(dist.log_prob(target))
        mse = torch.mean(torch.square(dist.mean - target))
        return nll, mse

class FusionDecoder(nn.Module):
    def __init__(self, input_dim, image_output_shape, vector_output_dim, conv_decoder_params, vector_decoder_params):
        super().__init__()
        self.conv_decoder = ConvDecoder(input_dim, image_output_shape, **conv_decoder_params)
        self.vector_decoder = MLPDistribution(input_dim, output_dim=vector_output_dim, **vector_decoder_params)

    def forward(self, x):
        return self.conv_decoder(x), self.vector_decoder(x)

    def get_nll_mse(self, feats, obs):
        image_dist = self.conv_decoder(feats)
        vector_dist = self.vector_decoder(feats)
        image_nll = -torch.mean(image_dist.log_prob(obs['images']))
        vector_nll = -torch.mean(vector_dist.log_prob(obs['vectors']))
        image_mse = torch.mean(torch.square(image_dist.mean - obs['images']))
        vector_mse = torch.mean(torch.square(vector_dist.mean - obs['vectors']))
        return image_nll, vector_nll, image_mse, vector_mse

class RSSM(nn.Module):
    def __init__(self, action_dim, obs_embed_dim, stoch=30, discrete=32, deter=200, hidden=200, activation='elu', kl_balance=0.8, kl_free=0.0, kl_free_avg=True):
        super().__init__()
        self.stoch = stoch
        self.discrete = discrete
        self.deter = deter
        self.hidden = hidden
        self.activation = get_activation(activation)
        self.kl_balance = kl_balance
        self.kl_free = torch.tensor(kl_free, requires_grad=False)
        self.kl_free_avg = kl_free_avg

        self.process_gru_input = nn.Sequential(nn.Linear(stoch * discrete + action_dim, deter), self.activation())
        self.gru_cell = nn.GRUCell(input_size=deter, hidden_size=deter)
        self.prior_net = nn.Sequential(nn.Linear(deter, hidden), self.activation(), nn.Linear(hidden, stoch * discrete))
        self.posterior_net = nn.Sequential(nn.Linear(deter + obs_embed_dim, hidden), self.activation(), nn.Linear(hidden, stoch * discrete))

    def initial(self, batch_size, dtype, device):
        return dict(
            logits=torch.zeros([batch_size, self.stoch, self.discrete], dtype=dtype, device=device),
            stoch=torch.zeros([batch_size, self.stoch * self.discrete], dtype=dtype, device=device),
            deter=torch.zeros([batch_size, self.deter], dtype=dtype, device=device),
            post=False,
        )

    def observe(self, obs_embeds, actions, is_firsts, post=None):
        # Expects dimensions (time_steps, batch_size, *feature_dims)
        if post is None:
            post = self.initial(obs_embeds.shape[1], obs_embeds.dtype, obs_embeds.device)
        posts = []
        priors = []
        for action, obs_embed, is_first in zip(actions, obs_embeds, is_firsts):
            prior, post = self.obs_step(post, action, obs_embed, is_first)
            posts.append(post)
            priors.append(prior)
        return posts, priors

    def imagine(self, actions, prior=None):
        # Expects dimensions (time_steps, batch_size, *feature_dims)
        if prior is None:
            prior = self.initial(actions.shape[1], actions.dtype, actions.device)
        priors = []
        for action in actions:
            prior = self.img_step(prior, action)
            priors.append(prior)
        return priors

    def obs_step(self, prev_state, prev_action, obs_embed, is_first, sample=True):
        filter = torch.logical_not(is_first.to(prev_action.dtype).squeeze(0))
        prev_state['logits'] = prev_state['logits'] * filter.reshape(filter.shape[0], *[1 for i in range(len(prev_state['logits'].shape[1:]))])
        prev_state['deter'] = prev_state['deter'] * filter.reshape(filter.shape[0], *[1 for i in range(len(prev_state['deter'].shape[1:]))])
        prev_state['stoch'] = prev_state['stoch'] * filter.reshape(filter.shape[0], *[1 for i in range(len(prev_state['stoch'].shape[1:]))])
        prev_action = prev_action * filter.reshape(filter.shape[0], *[1 for i in range(len(prev_action.shape[1:]))])
        prior = self.img_step(prev_state, prev_action, sample)
        logits = self.posterior_net(torch.cat([prior['deter'], obs_embed], dim=-1)).reshape(prev_action.shape[0], self.stoch, self.discrete)
        stoch = self.get_stoch(logits, sample)
        post = dict(
            logits=logits,
            stoch=stoch,
            deter=prior['deter'],
        )
        return prior, post

    def img_step(self, prev_state, prev_action, sample=True):
        prev_stoch = prev_state['stoch']
        batch_size = prev_stoch.shape[0]
        prev_stoch = prev_stoch.reshape((batch_size, self.stoch * self.discrete,))
        deter = self.gru_cell(self.process_gru_input(torch.cat([prev_stoch, prev_action], dim=-1)), prev_state['deter'])
        logits = self.prior_net(deter).reshape(prev_action.shape[0], self.stoch, self.discrete)
        stoch = self.get_stoch(logits, sample)
        prior = dict(
            logits=logits,
            stoch=stoch,
            deter=deter,
        )
        return prior

    def get_dist(self, logits):
        return dists.Independent(dists.OneHotCategorical(logits=logits), 1)

    def get_stoch(self, logits, sample):
        if sample:
            dist = self.get_dist(logits)
            probs = nn.functional.softmax(logits, dim=-1)
            stoch = dist.sample() + probs - probs.detach()
        else:
            modes = torch.argmax(logits, dim=-1)
            stoch = nn.functional.one_hot(modes, num_classes=self.discrete)
        return stoch

    def get_feat_b(self, state):
        stoch = state['stoch']
        shape = (stoch.shape[0], self.stoch * self.discrete,)
        stoch = stoch.reshape(shape)
        return torch.cat([stoch, state['deter']], dim=-1)

    def get_feat_t_b(self, states):
        l = []
        for state in states:
            l.append(self.get_feat_b(state))
        return torch.stack(l)

    def get_output_dim(self):
        return self.stoch * self.discrete + self.deter

    def kl_loss(self, posts, priors):
        post_logits = []
        prior_logits = []
        for post, prior in zip(posts, priors):
            post_logits.append(post['logits'])
            prior_logits.append(prior['logits'])
        post_logits = torch.stack(post_logits)
        prior_logits = torch.stack(prior_logits)

        post_grad_val = kl_val = dists.kl_divergence(self.get_dist(post_logits), self.get_dist(prior_logits.detach()))
        prior_grad_val = dists.kl_divergence(self.get_dist(post_logits.detach()), self.get_dist(prior_logits))

        post_grad_loss = torch.mean(post_grad_val)
        prior_grad_loss = torch.mean(prior_grad_val)

        kl_loss = self.kl_balance * prior_grad_loss + (1 - self.kl_balance) * post_grad_loss
        return kl_loss, torch.mean(kl_val), torch.mean(self.get_dist(post_logits).entropy()), torch.mean(self.get_dist(prior_logits).entropy())
