"""Kaggle Kaggriculture Path B agent export."""
import os
import sys
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from kaggriculture_adapter import decode_path_b_action, parse_observation, select_hand_farm_verbs
from kaggriculture_path_b_rebuild import (
    KaggricultureJSONParser,
    PathBFeatureExtractor,
    HierarchicalDQNBranching,
    HierarchicalActionMasker,
    apply_hierarchical_masks,
    break_pass_spawn_deadlock,
    prefer_farm_invest_actions,
)


def _resolve_model_path() -> str:
    here = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        os.path.join(here, "models", "model.pth"),
        os.path.join(here, "model.pth"),
        "/kaggle/working/models/model.pth",
        "/kaggle/working/run/models/model.pth",
    ]
    for path in candidates:
        if os.path.isfile(path):
            return path
    raise FileNotFoundError(
        "model.pth not found. Tried: " + ", ".join(candidates)
    )


class Agent:
    def __init__(self):
        self.device = torch.device("cpu")
        self.parser = KaggricultureJSONParser()
        extractor = PathBFeatureExtractor(latent_dim=512)
        self.net = HierarchicalDQNBranching(extractor, latent_dim=512, shared_dim=256)
        model_path = _resolve_model_path()
        try:
            state = torch.load(model_path, map_location=self.device, weights_only=True)
        except TypeError:
            state = torch.load(model_path, map_location=self.device)
        self.net.load_state_dict(state)
        self.net.eval()

    def act(self, obs, action_space=None):
        agent_obs = parse_observation(obs)
        parsed = self.parser.parse_observation(agent_obs)
        tiles_t = torch.as_tensor(parsed["tiles"], dtype=torch.float32, device=self.device).unsqueeze(0)
        numeric_t = torch.as_tensor(parsed["numeric"], dtype=torch.float32, device=self.device).unsqueeze(0)
        masks = HierarchicalActionMasker.get_dynamic_masks(agent_obs)
        with torch.no_grad():
            q_out = self.net(tiles_t, numeric_t)
            masked_q = apply_hierarchical_masks(q_out, masks, self.device)
            masked_q["farmer_verb"] = break_pass_spawn_deadlock(
                masked_q["farmer_verb"], masks["farmer_verb"]
            )
            farm_verb, farm_market = prefer_farm_invest_actions(
                masked_q["farmer_verb"],
                masks["farmer_verb"],
                masked_q["market"],
                masks.get("market"),
                observation=agent_obs,
            )
            masked_q["farmer_verb"] = farm_verb
            if farm_market is not None:
                masked_q["market"] = farm_market
            verb_idx = int(masked_q["farmer_verb"].argmax(dim=-1).item())
            crop_idx = int(masked_q["crop_parameter"].argmax(dim=-1).item())
            hands = select_hand_farm_verbs(agent_obs)
            market_seq = masked_q["market"].argmax(dim=-1).squeeze(0)
            market = [int(market_seq[t].item()) for t in range(self.net.max_market_orders)]
        return decode_path_b_action(verb_idx, crop_idx, hands, market, agent_obs)


_AGENT = Agent()


def agent(obs, cfg=None):
    return _AGENT.act(obs)
