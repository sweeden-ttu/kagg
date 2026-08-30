import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Any, Optional

from kaggriculture_adapter import (
    COMPETITION_TURNS_PER_DAY,
    CROPS,
    EPISODE_STEPS,
    FARMER_ACTIONS,
    MARKET_ACTIONS,
    NUM_FARMER_ACTIONS,
    NUM_HANDS,
    NUM_MARKET_ACTIONS,
    encode_path_b_observation,
    get_action_masks,
)

# ==============================================================================
# SECTION 1: SYSTEM OBSERVATION PARSER & FEATURE EXTRACTOR
# ==============================================================================

class KaggricultureJSONParser:
    """Parse official Kaggriculture observations for Path B feature tensors."""

    CROP_TYPES = CROPS
    SHOP_TYPES = ["BAKERY", "PIZZA", "GROCERY", "BREWERY", "MILL", "JUICE_BAR", "SALAD_BAR"]

    def __init__(self, grid_size: Tuple[int, int] = (10, 10)):
        self.grid_size = grid_size
        self.crop_to_idx = {crop: i for i, crop in enumerate(self.CROP_TYPES)}
        self.shop_to_idx = {shop: i for i, shop in enumerate(self.SHOP_TYPES)}

    def parse_observation(self, obs: Dict[str, Any]) -> Dict[str, Any]:
        return encode_path_b_observation(obs, player_id=obs.get("player", 0))


class KaggricultureFeatureExtractor(nn.Module):
    """
    Dual-branch feature extractor mapping spatial observations (CNN) and global
    normalized numerical context (MLP) into a high-capacity joint latent space.
    """
    def __init__(self, spatial_channels: int = 9, numeric_dim: int = 55, latent_dim: int = 512):
        super().__init__()
        
        # CNN Branch
        self.cnn_branch = nn.Sequential(
            nn.Conv2d(spatial_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
        )
        with torch.no_grad():
            sample = torch.zeros(1, spatial_channels, 10, 10)
            self._cnn_out_dim = self.cnn_branch(sample).shape[1]
        
        # MLP Branch
        self.mlp_branch = nn.Sequential(
            nn.Linear(numeric_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # Unified Fusion Projection Layer
        self.fusion = nn.Sequential(
            nn.Linear(self._cnn_out_dim + 256, latent_dim),
            nn.LayerNorm(latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, latent_dim),
            nn.LayerNorm(latent_dim),
            nn.ReLU()
        )

    def forward(self, tiles: torch.Tensor, numeric: torch.Tensor) -> torch.Tensor:
        cnn_out = self.cnn_branch(tiles)
        mlp_out = self.mlp_branch(numeric)
        fused_input = torch.cat([cnn_out, mlp_out], dim=-1)
        return self.fusion(fused_input)


# ==============================================================================
# SECTION 2: HIERARCHICAL ACTION DECODER NETWORK
# ==============================================================================

class HierarchicalDQNBranching(nn.Module):
    """
    An advanced Dueling Double DQN that replaces flat output heads with:
      - A primary Farmer Action Verb branch (15 outputs)
      - A state-conditioned Crop Parameter branch (5 outputs), evaluated conditionally
      - Multiple independent Hand branch action heads (6 heads * 15 actions each)
      - An autoregressive RNN Market Order decoder capable of outputting a sequence
        of up to 10 market transactions per step.
    """
    def __init__(self, 
                 extractor: KaggricultureFeatureExtractor, 
                 latent_dim: int = 512, 
                 shared_dim: int = 256,
                 num_farmer_verbs: int = NUM_FARMER_ACTIONS,
                 num_crops: int = len(CROPS),
                 num_hand_actions: int = NUM_FARMER_ACTIONS,
                 num_hands: int = NUM_HANDS,
                 num_market_actions: int = NUM_MARKET_ACTIONS,
                 max_market_orders: int = 10):
        super().__init__()
        
        self.extractor = extractor
        self.shared_dim = shared_dim
        self.num_farmer_verbs = num_farmer_verbs
        self.num_crops = num_crops
        self.num_hand_actions = num_hand_actions
        self.num_hands = num_hands
        self.num_market_actions = num_market_actions
        self.max_market_orders = max_market_orders
        
        # Shared representations mapping unified latent state to a narrower value vector
        self.shared_dense = nn.Sequential(
            nn.Linear(latent_dim, shared_dim),
            nn.BatchNorm1d(shared_dim),
            nn.ReLU(),
            nn.Linear(shared_dim, shared_dim),
            nn.BatchNorm1d(shared_dim),
            nn.ReLU()
        )
        
        # 1. State Value Stream V(s)
        self.value_stream = nn.Sequential(
            nn.Linear(shared_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
        
        # 2. Farmer Advantage Streams
        self.farmer_verb_adv = nn.Sequential(
            nn.Linear(shared_dim, 128),
            nn.ReLU(),
            nn.Linear(128, num_farmer_verbs)
        )
        self.crop_parameter_adv = nn.Sequential(
            nn.Linear(shared_dim + num_farmer_verbs, 128),
            nn.ReLU(),
            nn.Linear(128, num_crops)
        )
        
        # 3. Hands Advantage Streams
        self.hand_advs = nn.ModuleList([
            nn.Sequential(
                nn.Linear(shared_dim, 128),
                nn.ReLU(),
                nn.Linear(128, num_hand_actions)
            ) for _ in range(num_hands)
        ])
        
        # 4. Autoregressive Market Advantage Decoder
        self.market_proj = nn.Linear(shared_dim, 128)
        self.market_gru_cell = nn.GRUCell(input_size=128 + num_market_actions, hidden_size=shared_dim)
        self.market_action_embedding = nn.Embedding(num_market_actions, num_market_actions)
        self.market_action_embedding.weight.data.copy_(torch.eye(num_market_actions))
        self.market_action_embedding.weight.requires_grad = False
        
        self.market_order_adv = nn.Sequential(
            nn.Linear(shared_dim, 128),
            nn.ReLU(),
            nn.Linear(128, num_market_actions)
        )

    def forward(self,
                tiles: torch.Tensor,
                numeric: torch.Tensor,
                market_history: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Calculates joint state action Q-values across the hierarchical heads.
        """
        batch_size = tiles.size(0)
        
        # Extract and fuse state representations
        latent_state = self.extractor(tiles, numeric)
        shared_state = self.shared_dense(latent_state)
        
        # Compute uniform value baseline V(s)
        V = self.value_stream(shared_state) # Shape: (B, 1)
        
        # --- FARMER BRANCH Q-VALUES ---
        A_farmer_verb = self.farmer_verb_adv(shared_state) # (B, num_verbs)
        Q_farmer_verb = V + (A_farmer_verb - A_farmer_verb.mean(dim=-1, keepdim=True))
        
        crop_input = torch.cat([shared_state, A_farmer_verb], dim=-1)
        A_crop = self.crop_parameter_adv(crop_input) # (B, num_crops)
        Q_crop = V + (A_crop - A_crop.mean(dim=-1, keepdim=True))
        
        # --- HAND BRANCHES Q-VALUES ---
        Q_hands = []
        for hand_head in self.hand_advs:
            A_hand = hand_head(shared_state) # (B, num_hand_actions)
            Q_h = V + (A_hand - A_hand.mean(dim=-1, keepdim=True))
            Q_hands.append(Q_h)
            
        # --- AUTOREGRESSIVE MARKET ORDER DECODER ---
        Q_market_sequence = []
        hx = shared_state # Initial cell state
        proj_state = self.market_proj(shared_state) # Context vector: (B, 128)
        prev_action = torch.zeros(batch_size, dtype=torch.long, device=tiles.device) # PASS token
        
        for t in range(self.max_market_orders):
            prev_emb = self.market_action_embedding(prev_action)
            gru_input = torch.cat([proj_state, prev_emb], dim=-1)

            hx = self.market_gru_cell(gru_input, hx)

            A_market_t = self.market_order_adv(hx)
            Q_market_t = V + (A_market_t - A_market_t.mean(dim=-1, keepdim=True))
            Q_market_sequence.append(Q_market_t)

            if market_history is not None:
                prev_action = market_history[:, t].long()
            else:
                prev_action = Q_market_t.argmax(dim=-1)
            
        Q_market = torch.stack(Q_market_sequence, dim=1) # (B, max_market_orders, num_market_actions)

        return {
            "value": V,
            "farmer_verb": Q_farmer_verb,
            "crop_parameter": Q_crop,
            "hands": Q_hands,
            "market": Q_market
        }


# ==============================================================================
# SECTION 3: MULTI-LEVEL DYNAMIC ACTION MASKING
# ==============================================================================

class HierarchicalActionMasker:
    """Delegates to shared adapter masks (canonical verb indices)."""

    @staticmethod
    def get_dynamic_masks(obs: Dict[str, Any]) -> Dict[str, np.ndarray]:
        return get_action_masks(obs)

def apply_hierarchical_masks(q_values: Dict[str, torch.Tensor], 
                             masks: Dict[str, np.ndarray], 
                             device: torch.device) -> Dict[str, torch.Tensor]:
    masked_q = {}
    
    fv_mask_tensor = torch.as_tensor(masks["farmer_verb"], dtype=torch.bool, device=device)
    masked_q["farmer_verb"] = torch.where(fv_mask_tensor, q_values["farmer_verb"], torch.tensor(-1e9, device=device))
    
    cc_mask_tensor = torch.as_tensor(masks["crop_parameter"], dtype=torch.bool, device=device)
    masked_q["crop_parameter"] = torch.where(cc_mask_tensor, q_values["crop_parameter"], torch.tensor(-1e9, device=device))
    
    masked_q["hands"] = q_values["hands"]
    
    m_mask_tensor = torch.as_tensor(masks["market"], dtype=torch.bool, device=device).unsqueeze(0).unsqueeze(1)
    masked_q["market"] = torch.where(m_mask_tensor, q_values["market"], torch.tensor(-1e9, device=device))
    
    masked_q["value"] = q_values["value"]
    return masked_q


def break_pass_spawn_deadlock(
    farmer_verb_q: torch.Tensor,
    farmer_verb_mask: np.ndarray,
    *,
    pass_penalty: float = 12.0,
) -> torch.Tensor:
    """Soft-penalize PASS when locomotion is legal so the agent leaves spawn.

    Early-game masks often allow only PASS + a couple of moves. A PASS-biased
    policy then never moves, never unlocks DIG/PLANT, and ties Fallow Finn at
    the 3000-coin floor for a full 720-step season.
    """
    mask = np.asarray(farmer_verb_mask, dtype=bool)
    if mask.shape[-1] < 9 or not bool(mask[..., 0]):
        return farmer_verb_q
    move_legal = any(bool(mask[..., i]) for i in (5, 6, 7, 8))  # N,S,W,E
    if not move_legal:
        return farmer_verb_q
    out = farmer_verb_q.clone()
    out[..., 0] = out[..., 0] - pass_penalty
    return out


# ==============================================================================
# SECTION 4: TRAINING, TD TARGETS, & DOUBLE DQN OPTIMIZATION
# ==============================================================================

class HierarchicalDoubleDQNLearner:
    """
    Engine coordinating experiences, optimizing the Hierarchical Dueling DDQN model.
    """
    def __init__(self,
                 online_net: HierarchicalDQNBranching,
                 target_net: HierarchicalDQNBranching,
                 optimizer: torch.optim.Optimizer,
                 gamma: float = 0.995,
                 tau: float = 0.001,
                 huber_delta: float = 1.0):
        self.online = online_net
        self.target = target_net
        self.optimizer = optimizer
        self.gamma = gamma
        self.tau = tau
        self.huber_delta = huber_delta
        self.target.load_state_dict(self.online.state_dict())

    def compute_loss(
        self, batch: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        device = batch["reward"].device
        arange = torch.arange(batch["reward"].size(0), device=device)

        market_hist = batch.get("action_market")
        q_current = self.online(batch["tiles"], batch["numeric"], market_history=market_hist)
        V = q_current["value"].squeeze(-1)

        def _adv_component(q_head: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
            return q_head[arange, actions] - V

        adv_sum = _adv_component(q_current["farmer_verb"], batch["action_verb"])
        adv_sum = adv_sum + _adv_component(q_current["crop_parameter"], batch["action_crop"])
        for i in range(self.online.num_hands):
            adv_sum = adv_sum + _adv_component(
                q_current["hands"][i], batch["action_hands"][:, i]
            )
        for step in range(self.online.max_market_orders):
            adv_sum = adv_sum + _adv_component(
                q_current["market"][:, step, :], batch["action_market"][:, step]
            )
        total_q_online = V + adv_sum

        with torch.no_grad():
            q_next_online = self.online(batch["next_tiles"], batch["next_numeric"])
            V_next = q_next_online["value"].squeeze(-1)

            best_verb = q_next_online["farmer_verb"].argmax(dim=-1)
            best_crop = q_next_online["crop_parameter"].argmax(dim=-1)
            best_hands = [q_next_online["hands"][i].argmax(dim=-1) for i in range(self.online.num_hands)]
            best_market = q_next_online["market"].argmax(dim=-1)

            q_next_target = self.target(batch["next_tiles"], batch["next_numeric"])
            V_tgt = q_next_target["value"].squeeze(-1)

            def _tgt_adv(q_head: torch.Tensor, best: torch.Tensor) -> torch.Tensor:
                return q_head[arange, best] - V_tgt

            tgt_adv = _tgt_adv(q_next_target["farmer_verb"], best_verb)
            tgt_adv = tgt_adv + _tgt_adv(q_next_target["crop_parameter"], best_crop)
            for i in range(self.online.num_hands):
                tgt_adv = tgt_adv + _tgt_adv(q_next_target["hands"][i], best_hands[i])
            for step in range(self.online.max_market_orders):
                tgt_adv = tgt_adv + _tgt_adv(
                    q_next_target["market"][:, step, :], best_market[:, step]
                )
            total_target_next_q = V_tgt + tgt_adv
            td_target = batch["reward"] + self.gamma * (1.0 - batch["done"]) * total_target_next_q

        per_sample_loss = F.smooth_l1_loss(
            total_q_online, td_target, beta=self.huber_delta, reduction="none"
        )
        if "weights" in batch:
            loss = (per_sample_loss * batch["weights"]).mean()
        else:
            loss = per_sample_loss.mean()
        return loss, per_sample_loss.detach()

    def compute_bc_loss(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Behavioral cloning loss: cross-entropy on each hierarchical action head.

        PASS (index 0) dominates expert traces; down-weight it on farmer/market
        heads so cloning does not collapse to an all-PASS policy that ties Finn.
        """
        market_hist = batch.get("action_market")
        q_out = self.online(batch["tiles"], batch["numeric"], market_history=market_hist)

        verb = batch["action_verb"]
        verb_w = torch.where(
            verb == 0,
            torch.full_like(verb, 0.2, dtype=torch.float32),
            torch.full_like(verb, 2.0, dtype=torch.float32),
        )
        components = [
            (F.cross_entropy(q_out["farmer_verb"], verb, reduction="none") * verb_w).mean(),
            F.cross_entropy(q_out["crop_parameter"], batch["action_crop"]),
        ]
        for i in range(self.online.num_hands):
            components.append(
                F.cross_entropy(q_out["hands"][i], batch["action_hands"][:, i])
            )
        for step in range(self.online.max_market_orders):
            m_act = batch["action_market"][:, step]
            m_w = torch.where(
                m_act == 0,
                torch.full_like(m_act, 0.2, dtype=torch.float32),
                torch.full_like(m_act, 1.5, dtype=torch.float32),
            )
            components.append(
                (
                    F.cross_entropy(q_out["market"][:, step, :], m_act, reduction="none")
                    * m_w
                ).mean()
            )
        return torch.stack(components).mean()

    def update_target_network(self):
        for target_p, online_p in zip(self.target.parameters(), self.online.parameters()):
            target_p.data.copy_(self.tau * online_p.data + (1.0 - self.tau) * target_p.data)


# ==============================================================================
# SECTION 5: COMPETITIVE RELATIVE REWARD SHAPING
# ==============================================================================

class CompetitiveRewardShaper:
    """
    Shapes step rewards with relative bank equity plus a **kinematic**
    invest-vs-liquidate trajectory balancer.

    Episode metadata shows ``avg_score`` / ``min_score`` rising from ~700→~3100 over
    a season while ``sum_score ≈ 2 × avg_score`` (balanced opponents).  Step rewards
    use money-delta / ``bank_scale``; the competitive margin term scales with total
    stake so late-meta games (~6k combined bank) weight relative lead appropriately.

    Policy values are dynamic: we accumulate invest vs liquidate action mass over the
    episode and reward actions that steer the running mix toward a season schedule:

    - **Up front** (progress≈0): target invest/liquidate = **25/75**
    - **End** (progress≈1): target invest/liquidate = **75/25**

    Progress is kinematic on the competition calendar (30 days × 24 hours), not a
    fixed day gate list. Call ``reset_episode()`` at each self-play episode start
    (bootstrap resets automatically per episode parse).
    """

    SEASON_DAYS: int = 30
    HOURS_PER_DAY: int = COMPETITION_TURNS_PER_DAY  # 24

    def __init__(
        self,
        parser: KaggricultureJSONParser,
        *,
        bank_scale: float = 100.0,
        margin_scale: float = 500.0,
        stake_reference: float = 6000.0,
        clip: float = 20.0,
        bankruptcy_penalty: float = -10.0,
        invest_share_start: float = 0.25,
        invest_share_end: float = 0.75,
        mix_bonus_scale: float = 0.8,
        schedule_affinity: float = 0.25,
    ):
        self.parser = parser
        self.bank_scale = bank_scale
        self.margin_scale = margin_scale
        self.stake_reference = stake_reference
        self.clip = clip
        self.bankruptcy_penalty = bankruptcy_penalty
        self.invest_share_start = float(invest_share_start)
        self.invest_share_end = float(invest_share_end)
        self.mix_bonus_scale = float(mix_bonus_scale)
        self.schedule_affinity = float(schedule_affinity)

        # Invest: plant / dig / buy / hire / land / build / animals.
        self._invest_farmer = {
            FARMER_ACTIONS["PLANT"],
            FARMER_ACTIONS["DIG"],
            FARMER_ACTIONS["BUILD_COOP"],
            FARMER_ACTIONS["BUILD_PASTURE"],
            FARMER_ACTIONS["BUY_ANIMAL"],
        }
        self._invest_market = {
            MARKET_ACTIONS["BUY_SEED"],
            MARKET_ACTIONS["BUY_PRODUCT"],
            MARKET_ACTIONS["BUY_ANIMAL"],
            MARKET_ACTIONS["HIRE"],
            MARKET_ACTIONS["BUY_LAND"],
        }
        # Liquidate: harvest + sell.
        self._liquidate_farmer = {FARMER_ACTIONS["HARVEST"]}
        self._liquidate_market = {MARKET_ACTIONS["SELL"]}

        self._invest_mass: Dict[int, float] = {0: 0.0, 1: 0.0}
        self._liquidate_mass: Dict[int, float] = {0: 0.0, 1: 0.0}

    def reset_episode(self) -> None:
        """Clear invest/liquidate trajectories (call at each episode start)."""
        self._invest_mass = {0: 0.0, 1: 0.0}
        self._liquidate_mass = {0: 0.0, 1: 0.0}

    @staticmethod
    def money_delta_reward(prev_money: float, cur_money: float, bank_scale: float = 100.0) -> float:
        """Per-step reward from bank change (matches self-play env and bootstrap parser)."""
        return (cur_money - prev_money) / bank_scale

    def season_progress(self, obs: Dict[str, Any]) -> float:
        """Kinematic season progress in [0, 1] from day/hour (or step if present)."""
        if obs.get("step") is not None:
            try:
                step = int(obs["step"])
                return float(np.clip(step / float(max(EPISODE_STEPS - 1, 1)), 0.0, 1.0))
            except (TypeError, ValueError):
                pass
        day = max(1, int(obs.get("day", 1) or 1))
        hour = max(0, int(obs.get("hour", 0) or 0))
        ticks = (day - 1) * self.HOURS_PER_DAY + hour
        total = self.SEASON_DAYS * self.HOURS_PER_DAY
        return float(np.clip(ticks / float(max(total - 1, 1)), 0.0, 1.0))

    def target_invest_share(self, progress: float) -> float:
        """Lerp invest share: start 25% → end 75% (liquidate is the complement)."""
        t = float(np.clip(progress, 0.0, 1.0))
        return self.invest_share_start + (self.invest_share_end - self.invest_share_start) * t

    def phase_gates(self) -> Dict[str, Any]:
        """Return kinematic mix schedule (for config / logging)."""
        return {
            "mode": "kinematic_invest_liquidate_mix",
            "season_days": self.SEASON_DAYS,
            "hours_per_day": self.HOURS_PER_DAY,
            "invest_liquidate_start": [
                self.invest_share_start,
                1.0 - self.invest_share_start,
            ],
            "invest_liquidate_end": [
                self.invest_share_end,
                1.0 - self.invest_share_end,
            ],
            "mix_bonus_scale": self.mix_bonus_scale,
            "schedule_affinity": self.schedule_affinity,
            "turns_per_day_ref": COMPETITION_TURNS_PER_DAY,
        }

    def trajectory_snapshot(self, player_id: int = 0) -> Dict[str, float]:
        """Running invest/liquidate mass and empirical invest share for one seat."""
        inv = float(self._invest_mass.get(int(player_id), 0.0))
        liq = float(self._liquidate_mass.get(int(player_id), 0.0))
        total = inv + liq
        return {
            "invest_mass": inv,
            "liquidate_mass": liq,
            "invest_share": (inv / total) if total > 0 else 0.5,
        }

    def _action_sets(
        self,
        action_verb: Optional[int],
        action_market: Optional[Any],
        action_hands: Optional[Any],
    ) -> Tuple[set, set]:
        farmer_idxs: set = set()
        market_idxs: set = set()
        if action_verb is not None:
            farmer_idxs.add(int(action_verb))
        if action_hands is not None:
            for h in np.asarray(action_hands).reshape(-1).tolist():
                farmer_idxs.add(int(h))
        if action_market is not None:
            for m in np.asarray(action_market).reshape(-1).tolist():
                market_idxs.add(int(m))
        return farmer_idxs, market_idxs

    def _classify_action(
        self,
        action_verb: Optional[int],
        action_market: Optional[Any],
        action_hands: Optional[Any],
    ) -> Tuple[float, float]:
        """Return (invest_weight, liquidate_weight) for this step's actions."""
        farmer_idxs, market_idxs = self._action_sets(action_verb, action_market, action_hands)
        invest = 0.0
        liquidate = 0.0
        if farmer_idxs & self._invest_farmer or market_idxs & self._invest_market:
            invest = 1.0
        if farmer_idxs & self._liquidate_farmer or market_idxs & self._liquidate_market:
            liquidate = 1.0
        # Same step can both plant and sell — count both trajectories.
        return invest, liquidate

    def trajectory_mix_bonus(
        self,
        obs: Dict[str, Any],
        *,
        action_verb: Optional[int] = None,
        action_market: Optional[Any] = None,
        action_hands: Optional[Any] = None,
    ) -> float:
        """Bonus that steers running invest/liquidate mix toward the kinematic target."""
        player_id = int(obs.get("player", 0) or 0)
        invest_w, liquidate_w = self._classify_action(action_verb, action_market, action_hands)
        if invest_w <= 0.0 and liquidate_w <= 0.0:
            return 0.0

        progress = self.season_progress(obs)
        target = self.target_invest_share(progress)
        # Instantaneous affinity: prefer the side the schedule wants *now*.
        affinity = self.schedule_affinity * (
            invest_w * (2.0 * target - 1.0) + liquidate_w * (1.0 - 2.0 * target)
        )

        # Update trajectories, then reward reducing mix error.
        self._invest_mass[player_id] = self._invest_mass.get(player_id, 0.0) + invest_w
        self._liquidate_mass[player_id] = self._liquidate_mass.get(player_id, 0.0) + liquidate_w
        inv = self._invest_mass[player_id]
        liq = self._liquidate_mass[player_id]
        actual = inv / max(inv + liq, 1e-6)
        error = actual - target  # >0 means over-invested vs schedule
        # Investing when under-target is good; liquidating when over-invested is good.
        correction = self.mix_bonus_scale * (liquidate_w * error - invest_w * error)
        return float(affinity + correction)

    def shape_reward(
        self,
        obs: Dict[str, Any],
        raw_reward: float,
        *,
        action_verb: Optional[int] = None,
        action_market: Optional[Any] = None,
        action_hands: Optional[Any] = None,
    ) -> float:
        """Competitive shaping: equity term + kinematic invest/liquidate mix."""
        player_idx = obs.get("player", 0)
        opp_idx = 1 - player_idx
        farms = obs.get("farms", [])

        if len(farms) < 2:
            return float(np.clip(raw_reward, -self.clip, self.clip))

        my_money = float(farms[player_idx].get("money", 0.0))
        opp_money = float(farms[opp_idx].get("money", 0.0))

        if my_money <= 0.0 and opp_money > 0.0:
            return float(np.clip(self.bankruptcy_penalty, -self.clip, self.clip))

        stake = max(1.0, (my_money + opp_money) / self.stake_reference)
        competitive_delta = stake * (my_money - opp_money) / self.margin_scale
        mix_bonus = self.trajectory_mix_bonus(
            obs,
            action_verb=action_verb,
            action_market=action_market,
            action_hands=action_hands,
        )
        return float(
            np.clip(raw_reward + competitive_delta + mix_bonus, -self.clip, self.clip)
        )