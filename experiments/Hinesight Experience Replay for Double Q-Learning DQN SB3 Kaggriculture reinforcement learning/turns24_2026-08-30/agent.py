"""Kaggle Kaggriculture Path B agent export."""
import os
import sys
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from kaggriculture_adapter import decode_path_b_action, parse_observation
from kaggriculture_path_b_rebuild import (
    KaggricultureJSONParser,
    KaggricultureFeatureExtractor,
    HierarchicalDQNBranching,
    HierarchicalActionMasker,
    apply_hierarchical_masks,
)


class Agent:
    def __init__(self):
        self.device = torch.device("cpu")
        self.parser = KaggricultureJSONParser()
        extractor = KaggricultureFeatureExtractor(latent_dim=512)
        self.net = HierarchicalDQNBranching(extractor, latent_dim=512, shared_dim=256)
        model_path = os.path.join(os.path.dirname(__file__), "models", "model.pth")
        self.net.load_state_dict(torch.load(model_path, map_location=self.device))
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
            verb_idx = int(masked_q["farmer_verb"].argmax(dim=-1).item())
            crop_idx = int(masked_q["crop_parameter"].argmax(dim=-1).item())
            hands = [int(masked_q["hands"][i].argmax(dim=-1).item()) for i in range(self.net.num_hands)]
            market_seq = masked_q["market"].argmax(dim=-1).squeeze(0)
            market = [int(market_seq[t].item()) for t in range(self.net.max_market_orders)]
        return decode_path_b_action(verb_idx, crop_idx, hands, market, agent_obs)
