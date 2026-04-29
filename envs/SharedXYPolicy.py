import numpy as np
import torch as th
from torch import nn

from gymnasium import spaces

from stable_baselines3.common.distributions import DiagGaussianDistribution
from stable_baselines3.common.policies import ActorCriticPolicy

from config import StokesCylinderConfig


class SharedXYActorCriticPolicy(ActorCriticPolicy):
    """
    One-stage shared-layout policy:
    - xy head: does NOT see target-conditioned obs tail (last 2 dims are masked to zero)
    - ctrl head (r/omega/inflow): sees full observation
    This enforces structure-level decoupling: target cannot directly condition x/y outputs.
    """

    def __init__(self, *args, **kwargs):
        self.hidden_dim = int(kwargs.pop("shared_xy_hidden_dim", 256))
        super().__init__(*args, **kwargs)

    def _build(self, lr_schedule) -> None:
        if not isinstance(self.action_space, spaces.Box):
            raise ValueError("SharedXYActorCriticPolicy requires continuous Box action space.")
        if not isinstance(self.action_dist, DiagGaussianDistribution):
            raise ValueError("SharedXYActorCriticPolicy currently supports only DiagGaussianDistribution.")

        action_dim = int(np.prod(self.action_space.shape))
        self.n_cyl = int(StokesCylinderConfig.NUM_CYLINDERS)
        self.layout_dim = 3 * self.n_cyl
        if action_dim < self.layout_dim:
            raise ValueError(
                f"Action dim {action_dim} too small for shared-xy policy (need >= {self.layout_dim}). "
                "Ensure logic_box free-layout action shape is used."
            )
        self.xy_dim = 2 * self.n_cyl
        self.tail_dim = action_dim - self.layout_dim
        self.ctrl_dim = self.n_cyl + self.tail_dim

        feat_dim = int(self.features_dim)
        h = int(self.hidden_dim)

        self.xy_net = nn.Sequential(
            nn.Linear(feat_dim, h),
            nn.Tanh(),
            nn.Linear(h, h),
            nn.Tanh(),
            nn.Linear(h, self.xy_dim),
        )
        self.ctrl_net = nn.Sequential(
            nn.Linear(feat_dim, h),
            nn.Tanh(),
            nn.Linear(h, h),
            nn.Tanh(),
            nn.Linear(h, self.ctrl_dim),
        )
        self.value_net = nn.Sequential(
            nn.Linear(feat_dim, h),
            nn.Tanh(),
            nn.Linear(h, h),
            nn.Tanh(),
            nn.Linear(h, 1),
        )
        self.log_std = nn.Parameter(th.ones(action_dim) * float(self.log_std_init))

        if self.ortho_init:
            for module, gain in (
                (self.features_extractor, np.sqrt(2.0)),
                (self.xy_net, 0.01),
                (self.ctrl_net, 0.01),
                (self.value_net, 1.0),
            ):
                module.apply(lambda m: self.init_weights(m, gain=gain))

        self.optimizer = self.optimizer_class(self.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs)

    def _mask_target_tail(self, obs: th.Tensor) -> th.Tensor:
        # Observation shape: [..., obs_dim], where last 2 dims are dx,dy to current target.
        out = obs.clone()
        if out.shape[-1] >= 2:
            out[..., -2:] = 0.0
        return out

    def _extract_actor_features(self, obs):
        full_feat = super().extract_features(obs, self.pi_features_extractor)
        if not isinstance(obs, th.Tensor):
            raise ValueError("SharedXYActorCriticPolicy expects tensor observations.")
        xy_obs = self._mask_target_tail(obs)
        xy_feat = super().extract_features(xy_obs, self.pi_features_extractor)
        return full_feat, xy_feat

    def _distribution_from_obs(self, obs):
        full_feat, xy_feat = self._extract_actor_features(obs)
        mean_xy = self.xy_net(xy_feat)  # [B, 2N] -> x_all, y_all
        mean_ctrl = self.ctrl_net(full_feat)  # [B, N + tail]
        mean_r = mean_ctrl[:, : self.n_cyl]
        x_all = mean_xy[:, : self.n_cyl]
        y_all = mean_xy[:, self.n_cyl :]

        # Environment decode expects layout order:
        # [x0, y0, r0, x1, y1, r1, ...]
        layout = th.stack([x_all, y_all, mean_r], dim=2).reshape(mean_xy.shape[0], self.layout_dim)
        if self.tail_dim > 0:
            mean_tail = mean_ctrl[:, self.n_cyl :]
            mean_actions = th.cat([layout, mean_tail], dim=1)
        else:
            mean_actions = layout
        return self.action_dist.proba_distribution(mean_actions=mean_actions, log_std=self.log_std)

    def forward(self, obs: th.Tensor, deterministic: bool = False):
        dist = self._distribution_from_obs(obs)
        actions = dist.get_actions(deterministic=deterministic)
        log_prob = dist.log_prob(actions)
        vf_feat = super().extract_features(obs, self.vf_features_extractor)
        values = self.value_net(vf_feat)
        actions = actions.reshape((-1, *self.action_space.shape))
        return actions, values, log_prob

    def get_distribution(self, obs):
        return self._distribution_from_obs(obs)

    def _predict(self, observation, deterministic: bool = False):
        return self.get_distribution(observation).get_actions(deterministic=deterministic)

    def evaluate_actions(self, obs, actions):
        dist = self._distribution_from_obs(obs)
        log_prob = dist.log_prob(actions)
        entropy = dist.entropy()
        vf_feat = super().extract_features(obs, self.vf_features_extractor)
        values = self.value_net(vf_feat)
        return values, log_prob, entropy

    def predict_values(self, obs):
        vf_feat = super().extract_features(obs, self.vf_features_extractor)
        return self.value_net(vf_feat)


class SharedGeometryActorCriticPolicy(ActorCriticPolicy):
    """
    One-stage shared-geometry policy:
    - layout head (x/y/r) does NOT see target-conditioned obs tail
    - tail head (omega/inflow) sees full observation
    This enforces a single learned geometry with target-specific control.
    """

    def __init__(self, *args, **kwargs):
        self.hidden_dim = int(kwargs.pop("shared_geometry_hidden_dim", 256))
        self.layout_log_std_init = float(kwargs.pop("layout_log_std_init", -1.5))
        self.tail_log_std_init = float(kwargs.pop("tail_log_std_init", 0.0))
        super().__init__(*args, **kwargs)

    def _build(self, lr_schedule) -> None:
        if not isinstance(self.action_space, spaces.Box):
            raise ValueError("SharedGeometryActorCriticPolicy requires continuous Box action space.")
        if not isinstance(self.action_dist, DiagGaussianDistribution):
            raise ValueError("SharedGeometryActorCriticPolicy supports only DiagGaussianDistribution.")

        action_dim = int(np.prod(self.action_space.shape))
        self.n_cyl = int(StokesCylinderConfig.NUM_CYLINDERS)
        self.layout_dim = 3 * self.n_cyl
        if action_dim < self.layout_dim:
            raise ValueError(
                f"Action dim {action_dim} too small for shared-geometry policy "
                f"(need at least {self.layout_dim}). Use logic_box free-layout action shape."
            )
        self.tail_dim = action_dim - self.layout_dim

        feat_dim = int(self.features_dim)
        h = int(self.hidden_dim)

        self.layout_net = nn.Sequential(
            nn.Linear(feat_dim, h),
            nn.Tanh(),
            nn.Linear(h, h),
            nn.Tanh(),
            nn.Linear(h, self.layout_dim),
        )
        self.tail_net = (
            nn.Sequential(
                nn.Linear(feat_dim, h),
                nn.Tanh(),
                nn.Linear(h, h),
                nn.Tanh(),
                nn.Linear(h, self.tail_dim),
            )
            if self.tail_dim > 0
            else None
        )
        self.value_net = nn.Sequential(
            nn.Linear(feat_dim, h),
            nn.Tanh(),
            nn.Linear(h, h),
            nn.Tanh(),
            nn.Linear(h, 1),
        )
        log_std_init = th.ones(action_dim) * float(self.log_std_init)
        log_std_init[: self.layout_dim] = float(self.layout_log_std_init)
        if self.tail_dim > 0:
            log_std_init[self.layout_dim :] = float(self.tail_log_std_init)
        self.log_std = nn.Parameter(log_std_init)

        if self.ortho_init:
            modules = [
                (self.features_extractor, np.sqrt(2.0)),
                (self.layout_net, 0.01),
                (self.value_net, 1.0),
            ]
            if self.tail_net is not None:
                modules.append((self.tail_net, 0.01))
            for module, gain in modules:
                module.apply(lambda m: self.init_weights(m, gain=gain))

        self.optimizer = self.optimizer_class(self.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs)

    def _mask_target_tail(self, obs: th.Tensor) -> th.Tensor:
        out = obs.clone()
        if out.shape[-1] >= 2:
            out[..., -2:] = 0.0
        return out

    def _extract_actor_features(self, obs):
        if not isinstance(obs, th.Tensor):
            raise ValueError("SharedGeometryActorCriticPolicy expects tensor observations.")
        full_feat = super().extract_features(obs, self.pi_features_extractor)
        layout_obs = self._mask_target_tail(obs)
        layout_feat = super().extract_features(layout_obs, self.pi_features_extractor)
        return full_feat, layout_feat

    def _distribution_from_obs(self, obs):
        full_feat, layout_feat = self._extract_actor_features(obs)
        mean_layout = self.layout_net(layout_feat)
        if self.tail_dim > 0:
            mean_tail = self.tail_net(full_feat)
            mean_actions = th.cat([mean_layout, mean_tail], dim=1)
        else:
            mean_actions = mean_layout
        return self.action_dist.proba_distribution(mean_actions=mean_actions, log_std=self.log_std)

    def forward(self, obs: th.Tensor, deterministic: bool = False):
        dist = self._distribution_from_obs(obs)
        actions = dist.get_actions(deterministic=deterministic)
        log_prob = dist.log_prob(actions)
        vf_feat = super().extract_features(obs, self.vf_features_extractor)
        values = self.value_net(vf_feat)
        actions = actions.reshape((-1, *self.action_space.shape))
        return actions, values, log_prob

    def get_distribution(self, obs):
        return self._distribution_from_obs(obs)

    def _predict(self, observation, deterministic: bool = False):
        return self.get_distribution(observation).get_actions(deterministic=deterministic)

    def evaluate_actions(self, obs, actions):
        dist = self._distribution_from_obs(obs)
        log_prob = dist.log_prob(actions)
        entropy = dist.entropy()
        vf_feat = super().extract_features(obs, self.vf_features_extractor)
        values = self.value_net(vf_feat)
        return values, log_prob, entropy

    def predict_values(self, obs):
        vf_feat = super().extract_features(obs, self.vf_features_extractor)
        return self.value_net(vf_feat)
