import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Any, Optional

from kaggriculture_adapter import (
    COMPETITION_TURNS_PER_DAY,
    CROPS,
    DAILY_HIRE_TARGET,
    EPISODE_STEPS,
    FARMER_ACTIONS,
    MARKET_ACTIONS,
    NUM_FARMER_ACTIONS,
    NUM_HANDS,
    NUM_MARKET_ACTIONS,
    TARGET_WHEAT_PLANTS,
    daily_hire_orders_wanted,
    encode_path_b_observation,
    empty_neighbor_move_indices,
    farm_plant_census,
    farm_tour_move_index,
    get_action_masks,
    land_buy_wanted,
    target_plant_count,
)

# Path B uses channel-encoded tiles (B, 9, 10, 10) + numeric (B, 55).
# Do NOT import KaggricultureFeatureExtractor from kaggriculture_rl.dqn here —
# that class expects a dict of tensors and one-hots tiles internally.


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


class PathBFeatureExtractor(nn.Module):
    """Path B dual-branch extractor: channel tiles (B,9,10,10) + numeric (B,55).

    Incompatible with ``kaggriculture_rl.dqn.KaggricultureFeatureExtractor``,
    which expects a dict of tensors and one-hots ``tiles`` internally.
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


# Backward-compatible alias. Prefer PathBFeatureExtractor in new code.
# Distinct from kaggriculture_rl.dqn.KaggricultureFeatureExtractor (dict/one-hot).
KaggricultureFeatureExtractor = PathBFeatureExtractor


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
                 extractor: PathBFeatureExtractor, 
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


# Farmer verbs that convert land into money (vs locomotion / PASS).
_FARM_VERBS = (
    FARMER_ACTIONS["DIG"],
    FARMER_ACTIONS["WATER"],
    FARMER_ACTIONS["PLANT"],
    FARMER_ACTIONS["HARVEST"],
)
_GROW_VERBS = (
    FARMER_ACTIONS["DIG"],
    FARMER_ACTIONS["WATER"],
    FARMER_ACTIONS["PLANT"],
)
_MOVE_VERBS = (
    FARMER_ACTIONS["NORTH"],
    FARMER_ACTIONS["SOUTH"],
    FARMER_ACTIONS["WEST"],
    FARMER_ACTIONS["EAST"],
)


def prefer_farm_invest_actions(
    farmer_verb_q: torch.Tensor,
    farmer_verb_mask: np.ndarray,
    market_q: Optional[torch.Tensor] = None,
    market_mask: Optional[np.ndarray] = None,
    *,
    observation: Optional[Dict[str, Any]] = None,
    grow_bonus: float = 10.0,
    harvest_bonus: float = 12.0,
    harvest_early_penalty: float = 10.0,
    move_explore_bonus: float = 4.0,
    move_expand_bonus: float = 9.0,
    move_tour_bonus: float = 11.0,
    buy_seed_bonus: float = 10.0,
    buy_seed_surplus_penalty: float = 15.0,
    seed_surplus_threshold: int = 1,
    target_plants: Optional[int] = None,
    expensive_market_penalty: float = 20.0,
    sell_bonus: float = 8.0,
    hire_bonus: float = 24.0,
    buy_land_bonus: float = 28.0,
    daily_hire_target: Optional[int] = None,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Soft-boost farm actions; morning HIRE; optional NE land; restock seed.

    Never skip an engine-legal mature HARVEST (that regressed Finn). While a
    one-shot crop is still in its watering window, prefer WATER / expansion
    MOVE over early HARVEST. Restock BUY_SEED while below the land-scaled
    plant cap. Early-hour HIRE ramps 4→8 (fib, cash reserve); one BUY_LAND
    is boosted when affordable. Animals stay penalized.
    """
    f_mask = np.asarray(farmer_verb_mask, dtype=bool)
    farmer_out = farmer_verb_q.clone()
    census = (
        farm_plant_census(observation)
        if observation is not None
        else {
            "plants": 0,
            "seed_count": 0,
            "shed_count": 0,
            "standing_plant": 0,
            "standing_watered": 0,
            "standing_harvestable": 0,
            "standing_mature": 0,
        }
    )
    seed_count = int(census["seed_count"])
    shed_count = int(census["shed_count"])
    plant_count = int(census["plants"])
    if target_plants is None:
        plant_cap = (
            target_plant_count(observation)
            if observation is not None
            else int(TARGET_WHEAT_PLANTS)
        )
    else:
        plant_cap = int(target_plants)
    want_more_plants = plant_count < plant_cap

    grow_legal = any(
        idx < f_mask.shape[-1] and bool(f_mask[..., idx]) for idx in _GROW_VERBS
    )
    harvest = FARMER_ACTIONS["HARVEST"]
    harvest_legal = harvest < f_mask.shape[-1] and bool(f_mask[..., harvest])
    standing_mature = bool(census.get("standing_mature"))
    standing_watered = bool(census.get("standing_watered"))
    standing_plant = bool(census.get("standing_plant"))

    for idx in _GROW_VERBS:
        if idx < f_mask.shape[-1] and bool(f_mask[..., idx]):
            farmer_out[..., idx] = farmer_out[..., idx] + grow_bonus
    if harvest_legal:
        if standing_mature or not standing_plant:
            farmer_out[..., harvest] = farmer_out[..., harvest] + harvest_bonus
        else:
            # Engine-legal but still in the bonus window — keep watering /
            # expanding rather than Walter-style day-2 1-unit cuts.
            farmer_out[..., harvest] = farmer_out[..., harvest] - harvest_early_penalty

    # Leave a watered immature plant when we have a spare seed to plant.
    expand_now = (
        want_more_plants
        and seed_count > 0
        and standing_plant
        and standing_watered
        and not standing_mature
    )
    expand_dirs = (
        empty_neighbor_move_indices(observation) if observation is not None else ()
    )
    tour_idx = farm_tour_move_index(observation) if observation is not None else None
    if expand_now and expand_dirs:
        for idx in expand_dirs:
            if idx < f_mask.shape[-1] and bool(f_mask[..., idx]):
                farmer_out[..., idx] = farmer_out[..., idx] + move_expand_bonus
    elif (
        tour_idx is not None
        and standing_watered
        and plant_count >= 2
        and tour_idx < f_mask.shape[-1]
        and bool(f_mask[..., tour_idx])
    ):
        farmer_out[..., tour_idx] = farmer_out[..., tour_idx] + move_tour_bonus
    elif not expand_now and not grow_legal and not (harvest_legal and standing_mature):
        for idx in _MOVE_VERBS:
            if idx < f_mask.shape[-1] and bool(f_mask[..., idx]):
                farmer_out[..., idx] = farmer_out[..., idx] + move_explore_bonus

    market_out: Optional[torch.Tensor] = None
    if market_q is not None:
        market_out = market_q.clone()
        buy_seed = MARKET_ACTIONS["BUY_SEED"]
        land_idx = MARKET_ACTIONS["BUY_LAND"]
        n_orders = int(market_out.shape[-2])
        buy_legal = True
        land_legal = True
        if market_mask is not None:
            m_mask = np.asarray(market_mask, dtype=bool)
            if m_mask.ndim == 1 and buy_seed < m_mask.shape[0]:
                buy_legal = bool(m_mask[buy_seed])
            elif m_mask.ndim >= 2 and buy_seed < m_mask.shape[-1]:
                buy_legal = bool(m_mask[..., 0, buy_seed])
            if m_mask.ndim == 1 and land_idx < m_mask.shape[0]:
                land_legal = bool(m_mask[land_idx])
            elif m_mask.ndim >= 2 and land_idx < m_mask.shape[-1]:
                land_legal = bool(m_mask[..., 0, land_idx])

        hire_idx = MARKET_ACTIONS["HIRE"]
        hire_legal = True
        if market_mask is not None:
            m_mask = np.asarray(market_mask, dtype=bool)
            if m_mask.ndim == 1 and hire_idx < m_mask.shape[0]:
                hire_legal = bool(m_mask[hire_idx])
            elif m_mask.ndim >= 2 and hire_idx < m_mask.shape[-1]:
                hire_legal = bool(m_mask[..., 0, hire_idx])
        hire_todo = 0
        if hire_legal and observation is not None:
            hire_kwargs = {}
            if daily_hire_target is not None:
                hire_kwargs["target"] = int(daily_hire_target)
            hire_todo = daily_hire_orders_wanted(observation, **hire_kwargs)
        want_land = bool(
            land_legal and observation is not None and land_buy_wanted(observation)
        )
        # Liquidate shed before expanding: Hana-style chunked sells (up to 3).
        sell_slots = 0
        if shed_count > 0:
            sell_slots = 1
            if shed_count >= 30:
                sell_slots = 2
            if shed_count >= 60:
                sell_slots = 3
        cursor = sell_slots
        land_slot = cursor if want_land else None
        if want_land:
            cursor += 1
        # Keep room for a seed restock after hire burst (≤10 market orders).
        hire_budget = max(0, n_orders - cursor - (1 if want_more_plants else 0))
        hire_todo = min(hire_todo, hire_budget)
        hire_start = cursor
        seed_slot = hire_start + hire_todo
        want_seeds = want_more_plants and seed_count < max(
            1, plant_cap - plant_count, int(seed_surplus_threshold)
        )
        sell = MARKET_ACTIONS["SELL"]
        for t in range(n_orders):
            is_hire_slot = hire_todo > 0 and hire_start <= t < hire_start + hire_todo
            is_land_slot = land_slot is not None and t == land_slot
            is_sell_slot = t < sell_slots
            if buy_legal and want_seeds and t == seed_slot:
                market_out[..., t, buy_seed] = (
                    market_out[..., t, buy_seed] + buy_seed_bonus
                )
            else:
                market_out[..., t, buy_seed] = (
                    market_out[..., t, buy_seed] - buy_seed_surplus_penalty
                )
            market_out[..., t, MARKET_ACTIONS["BUY_ANIMAL"]] = (
                market_out[..., t, MARKET_ACTIONS["BUY_ANIMAL"]]
                - expensive_market_penalty
            )
            if is_land_slot:
                market_out[..., t, land_idx] = (
                    market_out[..., t, land_idx] + buy_land_bonus
                )
                pass_idx = MARKET_ACTIONS["PASS"]
                market_out[..., t, pass_idx] = (
                    market_out[..., t, pass_idx] - buy_land_bonus * 0.5
                )
            else:
                market_out[..., t, land_idx] = (
                    market_out[..., t, land_idx] - expensive_market_penalty
                )
            if is_hire_slot:
                market_out[..., t, hire_idx] = (
                    market_out[..., t, hire_idx] + hire_bonus
                )
                pass_idx = MARKET_ACTIONS["PASS"]
                market_out[..., t, pass_idx] = (
                    market_out[..., t, pass_idx] - hire_bonus * 0.5
                )
            else:
                market_out[..., t, hire_idx] = (
                    market_out[..., t, hire_idx] - expensive_market_penalty
                )
            if is_sell_slot:
                market_out[..., t, sell] = market_out[..., t, sell] + sell_bonus
            else:
                market_out[..., t, sell] = (
                    market_out[..., t, sell] - expensive_market_penalty * 0.25
                )
    return farmer_out, market_out


def _bc_farmer_verb_weights(verb: torch.Tensor) -> torch.Tensor:
    """Per-sample CE weights: mild PASS↓, farm↑ — avoid over-suppressing waits."""
    w = torch.full_like(verb, 1.25, dtype=torch.float32)
    w = torch.where(verb == FARMER_ACTIONS["PASS"], torch.full_like(w, 0.5), w)
    for idx in _FARM_VERBS:
        w = torch.where(verb == idx, torch.full_like(w, 2.5), w)
    for idx in _MOVE_VERBS:
        w = torch.where(verb == idx, torch.full_like(w, 1.0), w)
    return w


def _bc_market_action_weights(m_act: torch.Tensor) -> torch.Tensor:
    """Per-sample CE weights: mild PASS↓, BUY_SEED↑, animal/hire soft↓."""
    w = torch.full_like(m_act, 1.25, dtype=torch.float32)
    w = torch.where(m_act == MARKET_ACTIONS["PASS"], torch.full_like(w, 0.5), w)
    w = torch.where(m_act == MARKET_ACTIONS["BUY_SEED"], torch.full_like(w, 2.5), w)
    for key in ("BUY_PRODUCT", "BUY_LAND"):
        w = torch.where(m_act == MARKET_ACTIONS[key], torch.full_like(w, 1.0), w)
    # Animal/hire loops bankrupt seed-only policies that should clear Finn first.
    for key in ("BUY_ANIMAL", "HIRE"):
        w = torch.where(m_act == MARKET_ACTIONS[key], torch.full_like(w, 0.6), w)
    w = torch.where(m_act == MARKET_ACTIONS["SELL"], torch.full_like(w, 2.0), w)
    return w


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

        Expert traces are dominated by PASS and locomotion. Down-weight PASS,
        keep moves moderate, and strongly up-weight DIG/WATER/PLANT/HARVEST plus
        market BUY_SEED so cloning learns the farming loop needed to beat Finn.
        """
        market_hist = batch.get("action_market")
        q_out = self.online(batch["tiles"], batch["numeric"], market_history=market_hist)

        verb = batch["action_verb"]
        verb_w = _bc_farmer_verb_weights(verb)
        components = [
            (F.cross_entropy(q_out["farmer_verb"], verb, reduction="none") * verb_w).mean(),
            F.cross_entropy(q_out["crop_parameter"], batch["action_crop"]),
        ]
        for i in range(self.online.num_hands):
            hand_act = batch["action_hands"][:, i]
            hand_w = _bc_farmer_verb_weights(hand_act)
            components.append(
                (
                    F.cross_entropy(q_out["hands"][i], hand_act, reduction="none") * hand_w
                ).mean()
            )
        for step in range(self.online.max_market_orders):
            m_act = batch["action_market"][:, step]
            m_w = _bc_market_action_weights(m_act)
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

        margin = (my_money - opp_money) / self.margin_scale
        stake = max(1.0, (my_money + opp_money) / self.stake_reference)
        # Amplify leads with stake; do not amplify deficits (avoids desperation).
        if margin >= 0.0:
            competitive_delta = stake * margin
        else:
            competitive_delta = margin
        mix_bonus = self.trajectory_mix_bonus(
            obs,
            action_verb=action_verb,
            action_market=action_market,
            action_hands=action_hands,
        )
        return float(
            np.clip(raw_reward + competitive_delta + mix_bonus, -self.clip, self.clip)
        )