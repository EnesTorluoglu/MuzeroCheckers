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
        self.window_size = config.get('replay_buffer_size', 10000) # Kaç oyun saklanacak
        self.batch_size = config['batch_size']
        self.num_unroll_steps = config['num_unroll_steps'] # Dinamik ağın kaç adım açılacağı
        self.td_steps = config['td_steps'] # n-adımlı TD öğrenmesi için
        self.buffer = deque(maxlen=self.window_size) # Tamamlanmış oyunları saklar
        self.device = config.get('device', torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    def save_game(self, game_trace: GameTrace):
        """Tamamlanmış bir oyunu buffer'a ekler."""
        # if len(game_trace) < self.num_unroll_steps + self.td_steps :
        if len(game_trace) <  8:  #self.num_unroll_steps + self.td_steps: # Yeterli adım yoksa ekleme
            print(f"Uyarı: Oyun çok kısa ({len(game_trace)} adım), buffer'a eklenmiyor. Minimum: 8")
            return
        self.buffer.append(game_trace)

    def sample_batch(self):
        """
        Buffer'dan eğitim için bir batch örnekler.
        Her örnek, bir oyundan alınan ardışık (num_unroll_steps + 1) adımdan oluşur.
        Dönüş: observations_batch, actions_batch, rewards_batch, policies_batch, values_batch, target_values_batch
        target_values_batch, n-step return'ler veya oyun sonu değerleridir.
        """
        if len(self.buffer) < self.batch_size / 10 and len(self.buffer) < 10: # Yeterli oyun yoksa (batch_size'ın %10'u veya min 10 oyun)
             # Bu eşik ayarlanabilir.
            print(f"Buffer'da yeterli oyun yok ({len(self.buffer)}), örnekleme yapılamıyor.")
            return None

        # Rastgele oyunlar seç (batch_size kadar)
        # Aynı oyundan birden fazla segment seçilebilir, bu yüzden replace=True ile oyunları seçebiliriz.
        # Veya her batch elemanı farklı bir oyundan gelsin diye replace=False.
        # Şimdilik, her batch elemanı farklı bir oyundan (veya aynı oyundan farklı başlangıç noktası).
        
        observations_batch = []
        actions_hist_batch = [] # Her segment için (num_unroll_steps) aksiyon
        target_rewards_batch = [] # Her segment için (num_unroll_steps) ödül hedefi
        target_policies_batch = [] # Her segment için (num_unroll_steps+1) politika hedefi (s0...sK için)
        target_values_batch = []   # Her segment için (num_unroll_steps+1) değer hedefi (s0...sK için)

        games_indices = [random.randint(0, len(self.buffer) - 1) for _ in range(self.batch_size)]

        for game_idx in games_indices:
            game = self.buffer[game_idx]
            # Oyundan rastgele bir başlangıç noktası seç (segmentin sığacağı şekilde)
            # s_t, a_t, r_{t+1}, p_t, v_t
            # Bir segment (s_i, a_i, ..., a_{i+K-1}) için, hedefimiz s_i'deki politika ve değer,
            # ve r_{i+1}, ..., r_{i+K} ve s_{i+K}'daki değer.
            # Gözlem s_i'yi alacağız.
            # Aksiyonlar a_i, ..., a_{i+K-1} (K adet)
            # Ödüller r_{i+1}, ..., r_{i+K} (K adet) -> bunlar dynamics network için hedefler
            # Politikalar p_i, ..., p_{i+K} (K+1 adet) -> prediction network için hedefler
            # Değerler v_i, ..., v_{i+K} (K+1 adet) -> prediction network için hedefler (n-step)
            
            # Maksimum başlangıç indeksi, segmentin ve n-step return'ün sığması için:
            # game.total_steps - (num_unroll_steps + 1) - td_steps + 1
            # Veya daha basit: game.total_steps - num_unroll_steps - td_steps
            # Eğer td_steps=0 ise (sadece son adımın değeri), game.total_steps - num_unroll_steps
            # Bir segmentin son elemanı s_k olur, bunun için n-step target hesaplarız: R_k + gamma^n * V(s_{k+n})
            # Bu yüzden s_{k+n}'e kadar veri olmalı.
            # Başlangıç adımı `idx` için, `num_unroll_steps` boyunca açılım yapılır.
            # Yani, `s_idx` ... `s_{idx + num_unroll_steps}` durumları kullanılır.
            # `target_value` için `s_{idx + num_unroll_steps}` kullanılır ve buradan `td_steps` ileri bakılır.
            
            # Segmentin ilk durumu s_k olsun. Bu durum için gözlem, politika ve değer hedefini alırız.
            # Sonra K adım boyunca (a_k, r_{k+1}, s_{k+1}), ..., (a_{k+K-1}, r_{k+K}, s_{k+K})
            # Politikalar p_k, ..., p_{k+K} ve değerler v_k, ..., v_{k+K} (n-step ile hesaplanır)

            # Bir segmentin başlangıç adımı (index) için s_idx kullanılır.
            # Bu s_idx için gözlem, p_idx ve v_idx (n-step) hedeftir.
            # Sonraki K adım için (a_idx, ..., a_{idx+K-1}) ve (r_{idx+1}, ..., r_{idx+K}) hedeftir.
            start_limit = game.total_steps - (self.num_unroll_steps + self.td_steps) 
            if start_limit < 0: start_limit = 0 # Eğer oyun çok kısaysa (bu durum save_game'de engellenmeliydi ama yine de kontrol) 

            if game.total_steps <= self.num_unroll_steps: # Çok kısa oyun (bu da engellenmeliydi)
                # Bu oyunu atla, başka bir tane dene (veya None döndür)
                # print(f"Uyarı: Örnekleme sırasında çok kısa bir oyunla karşılaşıldı ({len(game)} adım). Atlanıyor.")
                # TODO: Bu durumu daha iyi ele al. Belki batch'i doldurmak için tekrar dene.
                # Şimdilik bu oyunu atlayıp, batch boyutunu küçültebiliriz (bu da sorunlu).
                # Veya None döndürmek daha iyi.
                continue # Bu iterasyonu atla, batch eksik kalabilir. En iyisi baştan kontrol.

            start_idx = random.randint(0, start_limit if start_limit >=0 else 0) 

            # Gözlemler: s_start_idx
            # Bu, representation network'e verilecek.
            observations_batch.append(game.observations[start_idx])

            # Aksiyonlar: a_{start_idx} ... a_{start_idx + num_unroll_steps - 1} (K adet)
            # Bunlar dynamics network'e verilecek.
            actions_segment = game.actions[start_idx : start_idx + self.num_unroll_steps]
            actions_hist_batch.append(torch.tensor(actions_segment, dtype=torch.long, device=self.device))

            # Hedefler:
            # target_rewards: r_{start_idx+1} ... r_{start_idx + num_unroll_steps} (K adet)
            # target_policies: p_{start_idx} ... p_{start_idx + num_unroll_steps} (K+1 adet)
            # target_values: v^{n-step}_{start_idx} ... v^{n-step}_{start_idx + num_unroll_steps} (K+1 adet)
            
            segment_target_rewards = []
            segment_target_policies = []
            segment_target_values_n_step = []

            for k in range(self.num_unroll_steps + 1): # s_0 ... s_K için (K+1 durum)
                current_step_idx = start_idx + k
                
                # Politika hedefi: MCTS'ten gelen politika
                segment_target_policies.append(torch.tensor(game.policies[current_step_idx], dtype=torch.float32, device=self.device))
                
                # Anlık ödül (eğer k > 0 ise, yani s_1...s_K için r_k)
                # s_0 için ödül hedefi yok (ya da bir önceki adımdan gelen)
                if k < self.num_unroll_steps: # Sadece K adım için ödül var (r_1 ... r_K)
                    segment_target_rewards.append(game.rewards[current_step_idx + 1]) # r_{t+1}
                
                # Değer hedefi (n-step return)
                # V(s_t) = r_{t+1} + g*r_{t+2} + ... + g^{n-1}*r_{t+n} + g^n * V_{target}(s_{t+n})
                # V_{target}(s_{t+n}) oyun bitmişse 0, değilse ağdan gelen değer.
                # MuZero makalesinde, target value, MCTS'in arama sonucundaki value'su olabilir veya oyun sonu değeri.
                # Şimdilik, game.values içinde saklananları (oyun sırasında hesaplanan n-step veya oyun sonu) kullanalım.
                # Bu game.values, agent tarafından MCTS sonuçları veya gerçek oyun sonu değerleri ile doldurulmalı.
                # Eğer game.values, sadece terminal adımdaki oyun sonu değerini içeriyorsa,
                # o zaman n-step return'ü burada hesaplamamız gerekir.
                # Varsayım: game.values[t] = s_t için n-step hedeftir.
                
                n_step_value_target_idx = current_step_idx + self.td_steps
                if n_step_value_target_idx < game.total_steps:
                    # Oyun bitmediyse, td_steps sonraki adımdaki değeri al (eğer varsa, yoksa ağdan alınır)
                    # Şimdilik game.values içinde o anki adıma ait MCTS değeri olduğunu varsayalım.
                    # n-step return'ü burada hesaplayalım.
                    discounted_reward_sum = 0
                    for i in range(self.td_steps):
                        reward_idx = current_step_idx + i + 1
                        if reward_idx < game.total_steps:
                            discounted_reward_sum += (CONFIG.get('discount_factor', 0.997)**i) * game.rewards[reward_idx]
                        else: # Oyun bittiyse, daha fazla ödül yok
                            break
                    
                    last_step_value = 0
                    if n_step_value_target_idx < game.total_steps: # Hala oyun içindeysek
                        # Bu V(s_{t+n}) olmalı. game.values[t] MCTS değeri veya oyun sonu değeri.
                        # Eğer game.values[t] zaten n-step target ise direkt kullanırız.
                        # Eğer game.values[t] sadece o anki V(s_t) ise, V(s_{t+n}) olarak game.values[n_step_value_target_idx] kullanırız.
                        # Şimdilik, game.values[t]'nin o anki adımdaki (MCTS veya oyun sonu) value olduğunu varsayalım.
                        last_step_value = game.values[n_step_value_target_idx] 
                    # Eğer n_step_value_target_idx >= game.total_steps ise, oyun bitti, last_step_value = 0 (çünkü ödül yok)
                        
                    n_step_return = discounted_reward_sum + (CONFIG.get('discount_factor', 0.997)**self.td_steps) * last_step_value
                    segment_target_values_n_step.append(n_step_return)
                else: # Oyun td_steps içinde bitiyorsa
                    discounted_reward_sum = 0
                    for i in range(game.total_steps - (current_step_idx + 1)):
                        reward_idx = current_step_idx + i + 1
                        discounted_reward_sum += (CONFIG.get('discount_factor', 0.997)**i) * game.rewards[reward_idx]
                    # Oyun bittiği için son değer 0 (eğer terminal ödül game.rewards'da değilse)
                    # Eğer game.values[-1] terminal ödülü içeriyorsa, o kullanılabilir, ama burada karmaşıklaşır.
                    # Şimdilik, oyun bittiyse ve arada ödül yoksa, n-step return sadece toplanan ödüllerdir.
                    segment_target_values_n_step.append(discounted_reward_sum)
            
            # target_rewards K adet, target_policies ve target_values K+1 adet olmalı.
            # segment_target_rewards (K) adet olacak şekilde (r_1...r_K)
            # segment_target_policies (K+1) adet (p_0...p_K)
            # segment_target_values_n_step (K+1) adet (v_0...v_K)
            target_rewards_batch.append(torch.tensor(segment_target_rewards, dtype=torch.float32, device=self.device))
            target_policies_batch.append(torch.stack(segment_target_policies)) # Stack to (K+1, action_space)
            target_values_batch.append(torch.tensor(segment_target_values_n_step, dtype=torch.float32, device=self.device))

        if not observations_batch: # Eğer geçerli segment bulunamamışsa (çok kısa oyunlar vb.)
            return None

        return (
            torch.stack(observations_batch), # (B, C, H, W)
            torch.stack(actions_hist_batch), # (B, K)
            torch.stack(target_rewards_batch),# (B, K)
            torch.stack(target_policies_batch),# (B, K+1, action_space)
            torch.stack(target_values_batch)  # (B, K+1)
        )

    def __len__(self):
        return len(self.buffer) # Saklanan oyun sayısı

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
        # Örnek: En az X oyun veya Y toplam adım olmalı.
        # Veya en az bir batch çıkarılabilecek kadar oyun olmalı.
        # Şimdilik, en az (batch_size / N) kadar oyun varsa hazır diyelim (N=bir oyundan kaç segment çıktığına bağlı)
        # Daha basit: en az batch_size / 2 kadar oyun varsa.
        # Veya config'den bir min_buffer_size_for_training alınabilir.
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
    CONFIG.update(config_test) # Global CONFIG'i güncelle

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
            val = random.random() # Bu, o anki adımdaki MCTS/gerçek değer olmalı
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