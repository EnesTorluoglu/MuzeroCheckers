# Deneyim tekrarı (Replay Buffer) implementasyonu.

import random
import numpy as np
import torch
from collections import deque

from .config import CONFIG

class GameTrace:
    """Bir oyunun tüm verilerini saklamak için bir yapı."""
    def __init__(self):
        self.observations = []  # Gözlemler (ağa verilecek formatta tensorler)
        self.actions = []       # Seçilen aksiyon ID'leri (int)
        self.rewards = []       # Anlık ödüller (float)
        self.policies = []      # MCTS politikaları (numpy array veya tensor)
        self.values = []        # MCTS/oyun sonu değerleri (float)
        self.game_over_reason = None # Oyun nasıl bitti (P1_WINS, P2_WINS, DRAW)
        self.total_steps = 0

    def add_step(self, observation, action, reward, policy, value):
        self.observations.append(observation)
        self.actions.append(action)
        self.rewards.append(reward)
        self.policies.append(policy) # MCTS'ten gelen tam politika (96,)
        self.values.append(value) # Hedef değer (n-step return veya oyun sonu)
        self.total_steps += 1

    def __len__(self):
        return self.total_steps

class ReplayBuffer:
    def __init__(self, config):
        self.window_size = config.get('replay_buffer_size', 2000)
        self.batch_size = config['batch_size']
        self.num_unroll_steps = config['num_unroll_steps']
        self.td_steps = config['td_steps']
        self.buffer = deque(maxlen=self.window_size)
        self.device = config.get('device', torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        # MuZero paper: absorbing states için gerekli
        self.action_space_size = config['action_space_size']
        self.discount_factor = config.get('discount_factor', 0.997)
        self.min_game_steps = config.get('min_game_steps_for_buffer', 8)

    def save_game(self, game_trace: GameTrace):
        """Tamamlanmış bir oyunu buffer'a ekler."""
        if len(game_trace) < self.min_game_steps:
            print(f"Uyarı: Oyun çok kısa ({len(game_trace)} adım), buffer'a eklenmiyor. Minimum: {self.min_game_steps}")
            return
        self.buffer.append(game_trace)
    
    def _compute_target_value(self, game, index):
        """
        Compute n-step target value for a given position.
        MuZero paper: V(s_t) = sum_{i=0}^{n-1} gamma^i * r_{t+1+i} + gamma^n * V(s_{t+n})
        """
        # Absorbing state: value = 0
        if index >= game.total_steps:
            return 0.0
        
        bootstrap_index = index + self.td_steps
        value = 0.0
        
        # Bootstrap value from future state
        if bootstrap_index < game.total_steps:
            value = game.values[bootstrap_index] * (self.discount_factor ** self.td_steps)
        # else: game ended before bootstrap, so bootstrap value = 0
        
        # Sum discounted rewards from index+1 to bootstrap_index
        for i in range(min(self.td_steps, game.total_steps - index - 1)):
            reward_idx = index + 1 + i
            if reward_idx < game.total_steps:
                value += (self.discount_factor ** i) * game.rewards[reward_idx]
        
        return value

    def sample_batch(self):
        """
        Buffer'dan eğitim için bir batch örnekler.
        MuZero Paper: Tüm pozisyonlardan uniform örnekleme + absorbing states.
        """
        if len(self.buffer) < self.batch_size / 10 and len(self.buffer) < 10:
            print(f"Buffer'da yeterli oyun yok ({len(self.buffer)}), örnekleme yapılamıyor.")
            return None

        observations_batch = []
        actions_hist_batch = []
        target_rewards_batch = []
        target_policies_batch = []
        target_values_batch = []

        games_indices = [random.randint(0, len(self.buffer) - 1) for _ in range(self.batch_size)]
        
        # Uniform policy for absorbing states (computed once)
        uniform_policy = np.ones(self.action_space_size, dtype=np.float32) / self.action_space_size

        for game_idx in games_indices:
            game = self.buffer[game_idx]
            
            # MuZero Paper: Sample uniformly from ALL positions in the game
            # Absorbing states will be used for steps that extend beyond game end
            start_idx = random.randint(0, game.total_steps - 1)

            # Gözlemler: s_start_idx (representation network'e verilecek)
            observations_batch.append(game.observations[start_idx].to(self.device))

            # Actions: K adet aksiyon (absorbing states için random)
            actions_segment = []
            for k in range(self.num_unroll_steps):
                action_idx = start_idx + k
                if action_idx < game.total_steps:
                    actions_segment.append(game.actions[action_idx])
                else:
                    # Absorbing state: random action
                    actions_segment.append(random.randint(0, self.action_space_size - 1))
            actions_hist_batch.append(torch.tensor(actions_segment, dtype=torch.long, device=self.device))

            # Hedefler: K+1 policy/value, K reward
            segment_target_rewards = []
            segment_target_policies = []
            segment_target_values_n_step = []

            for k in range(self.num_unroll_steps + 1):
                current_step_idx = start_idx + k
                is_absorbing = current_step_idx >= game.total_steps
                
                if is_absorbing:
                    # MuZero Paper: Absorbing state targets
                    segment_target_policies.append(torch.tensor(uniform_policy, dtype=torch.float32, device=self.device))
                    segment_target_values_n_step.append(0.0)
                    if k < self.num_unroll_steps:
                        segment_target_rewards.append(0.0)
                else:
                    # Normal state: use actual game data
                    segment_target_policies.append(torch.tensor(game.policies[current_step_idx], dtype=torch.float32, device=self.device))
                    segment_target_values_n_step.append(self._compute_target_value(game, current_step_idx))
                    if k < self.num_unroll_steps:
                        segment_target_rewards.append(game.rewards[current_step_idx])
            
            target_rewards_batch.append(torch.tensor(segment_target_rewards, dtype=torch.float32, device=self.device))
            target_policies_batch.append(torch.stack(segment_target_policies))
            target_values_batch.append(torch.tensor(segment_target_values_n_step, dtype=torch.float32, device=self.device))

        if not observations_batch:
            return None

        return (
            torch.stack(observations_batch),   # (B, C, H, W)
            torch.stack(actions_hist_batch),   # (B, K)
            torch.stack(target_rewards_batch), # (B, K)
            torch.stack(target_policies_batch),# (B, K+1, action_space)
            torch.stack(target_values_batch)   # (B, K+1)
        )

    def __len__(self):
        return len(self.buffer)

    def get_total_samples(self):
        """Buffer'daki tüm oyunlardaki toplam adım sayısını döndürür."""
        return sum(len(game_trace) for game_trace in self.buffer)

    def get_state(self):
        """Buffer'ın mevcut durumunu (oyun listesi) döndürür."""
        return list(self.buffer)

    def set_state(self, state_list):
        """Buffer'ı verilen durumla (oyun listesi) ayarlar."""
        self.buffer = deque(state_list, maxlen=self.window_size)

    def is_ready(self):
        """Buffer'da eğitim için yeterli veri olup olmadığını kontrol eder."""
        min_games_for_training = CONFIG.get("min_games_for_training", self.batch_size // 2 if self.batch_size > 1 else 1)
        if min_games_for_training == 0: min_games_for_training = 1
        return len(self.buffer) >= min_games_for_training

# Test için
if __name__ == '__main__':
    # Dummy config
    config_test = {
        'replay_buffer_size': 100,
        'batch_size': 4,
        'num_unroll_steps': 3, # K
        'td_steps': 2,         # n
        'observation_channels': 3,
        'board_rows': 8,
        'board_cols': 3,
        'action_space_size': 96,
        'discount_factor': 0.9,
        'device': torch.device('cpu')
    }
    CONFIG.update(config_test)

    replay_buffer = ReplayBuffer(CONFIG)

    # Örnek birkaç oyun ekleyelim
    for i in range(10):
        game = GameTrace()
        num_steps_in_game = random.randint(config_test['num_unroll_steps'] + config_test['td_steps'] + 1, 20)
        for step in range(num_steps_in_game):
            obs = torch.randn(config_test['observation_channels'], config_test['board_rows'], config_test['board_cols'])
            act = random.randint(0, config_test['action_space_size'] - 1)
            rew = random.random()
            pol = np.random.rand(config_test['action_space_size']).astype(np.float32)
            pol /= np.sum(pol)
            val = random.random()
            game.add_step(obs, act, rew, pol, val)
        replay_buffer.save_game(game)
        print(f"Oyun {i} eklendi, {len(game)} adım. Buffer boyutu: {len(replay_buffer)}")

    if replay_buffer.is_ready():
        print("\nBuffer eğitim için hazır.")
        batch_data = replay_buffer.sample_batch()
        if batch_data:
            obs_b, act_b, rew_b, pol_b, val_b = batch_data
            print(f"Örneklenmiş batch şekilleri:")
            print(f"  Observations: {obs_b.shape}") # (B, C, H, W)
            print(f"  Actions:      {act_b.shape}") # (B, K)
            print(f"  Rewards:      {rew_b.shape}") # (B, K)
            print(f"  Policies:     {pol_b.shape}") # (B, K+1, ActionSpace)
            print(f"  Values:       {val_b.shape}") # (B, K+1)
        else:
            print("Batch örneklenemedi.")
    else:
        print("\nBuffer henüz eğitim için hazır değil.")