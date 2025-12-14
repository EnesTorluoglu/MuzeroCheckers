# MuZero ağ modelleri (Representation, Prediction, Dynamics) burada tanımlanacak.

import torch
import torch.nn as nn
import torch.nn.functional as F

# from config import CONFIG

class RepresentationNetwork(nn.Module):
    def __init__(self, observation_shape, hidden_state_channels):
        super().__init__()
        # observation_shape = (channels, rows, cols)
        self.obs_channels = observation_shape[0]
        self.obs_rows = observation_shape[1]
        self.obs_cols = observation_shape[2]
        self.hidden_state_channels = hidden_state_channels

        # Örnek CNN mimarisi (detaylandırılacak)
        # Girdi: (batch, obs_channels, obs_rows, obs_cols)
        # Çıktı: (batch, hidden_state_channels, H', W')
        self.conv1 = nn.Conv2d(self.obs_channels, hidden_state_channels // 2, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(hidden_state_channels // 2)
        # Tahta boyutlarına göre çıktı boyutunu ayarlayacak katmanlar eklenecek
        # Örneğin, 8x4 tahtayı daha küçük bir gizli duruma (örneğin 4x2 veya 2x1) indirgeyebiliriz.
        # Veya tamamen fully connected bir gizli duruma da geçilebilir.
        # Şimdilik, girdiyle aynı boyutta bir gizli durum varsayalım (stride=1, padding=1 ile)
        self.conv2 = nn.Conv2d(hidden_state_channels // 2, hidden_state_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(hidden_state_channels)

        print(f"RepresentationNetwork: Input ({self.obs_channels},{self.obs_rows},{self.obs_cols}), Output Hidden ({self.hidden_state_channels},{self.obs_rows},{self.obs_cols}) (approx)")

    def forward(self, observation):
        # observation: (batch_size, obs_channels, obs_rows, obs_cols)
        x = F.relu(self.bn1(self.conv1(observation)))
        hidden_state = F.relu(self.bn2(self.conv2(x)))
        # Boyutları kontrol etmek/ayarlamak için: print(hidden_state.shape)
        return hidden_state # (batch_size, hidden_state_channels, H', W')

class DynamicsNetwork(nn.Module):
    def __init__(self, hidden_state_shape, action_space_size, hidden_state_channels):
        super().__init__()
        # hidden_state_shape = (channels, H', W')
        self.hidden_channels = hidden_state_shape[0]
        self.h_prime = hidden_state_shape[1]
        self.w_prime = hidden_state_shape[2]
        self.action_space_size = action_space_size
        
        # Gizli durum ve aksiyonu birleştirmek için bir yöntem.
        # Aksiyonu gizli durumla aynı boyuta getirmek için bir embedding veya Conv.
        # Örneğin, aksiyonu (1, H', W') boyutuna getirip gizli durumla birleştirebiliriz.
        # Bu kısım, aksiyonun nasıl temsil edildiğine bağlı olarak değişir.
        # Şimdilik, aksiyonu gizli durumun kanal boyutuna eklediğimizi varsayalım.
        # (hidden_state_channels + action_embedding_channels)
        # Basit bir yaklaşım: aksiyonu embed edip gizli durumun üzerine eklemek (kanallara)
        self.action_embedding_channels = 16 # Örnek
        self.action_embedding = nn.Embedding(action_space_size, self.h_prime * self.w_prime * self.action_embedding_channels)

        # Girdi: (batch, hidden_channels + action_embedding_channels, H', W')
        self.conv1 = nn.Conv2d(self.hidden_channels + self.action_embedding_channels, hidden_state_channels, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(hidden_state_channels)
        self.conv2 = nn.Conv2d(hidden_state_channels, hidden_state_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(hidden_state_channels)

        # Ödül tahmini için (skaler)
        # Gizli durumdan ödülü tahmin eden bir katman.
        # Örneğin, gizli durumu düzleştirip bir FC katmanından geçirebiliriz.
        self.fc_reward = nn.Linear(hidden_state_channels * self.h_prime * self.w_prime, 1)
        print(f"DynamicsNetwork: Input Hidden ({self.hidden_channels},{self.h_prime},{self.w_prime}), Action Space {action_space_size}, Output Hidden ({hidden_state_channels},{self.h_prime},{self.w_prime}), Reward (scalar)")

    def forward(self, hidden_state, action):
        # hidden_state: (batch_size, hidden_channels, H', W')
        # action: (batch_size, 1) - integer aksiyon ID'leri
        batch_size = hidden_state.size(0)

        action_embed = self.action_embedding(action.squeeze(-1)) # (batch_size, H'*W'*action_emb_channels)
        action_embed = action_embed.view(batch_size, self.action_embedding_channels, self.h_prime, self.w_prime)
        
        # Gizli durum ve aksiyon embedding'ini birleştir (kanal boyutunda)
        combined_input = torch.cat((hidden_state, action_embed), dim=1)

        x = F.relu(self.bn1(self.conv1(combined_input)))
        next_hidden_state = F.relu(self.bn2(self.conv2(x)))
        
        # Ödül tahmini
        # Gizli durumu düzleştir
        flat_state = next_hidden_state.view(batch_size, -1)
        reward = self.fc_reward(flat_state)
        # MuZero genellikle ödülleri tanh ile [-1, 1] aralığına sıkıştırır, oyun özelinde ayarlanabilir.
        # reward = torch.tanh(reward) 

        return next_hidden_state, reward

class PredictionNetwork(nn.Module):
    def __init__(self, hidden_state_shape, action_space_size, hidden_state_channels):
        super().__init__()
        # hidden_state_shape = (channels, H', W')
        self.hidden_channels = hidden_state_shape[0]
        self.h_prime = hidden_state_shape[1]
        self.w_prime = hidden_state_shape[2]
        self.action_space_size = action_space_size

        # Girdi: (batch, hidden_channels, H', W')
        self.conv1 = nn.Conv2d(self.hidden_channels, hidden_state_channels // 2, kernel_size=1, stride=1) # 1x1 conv for channel reduction
        self.bn1 = nn.BatchNorm2d(hidden_state_channels // 2)
        
        # Düzleştirilmiş özellik boyutu
        self.flat_features_dim = (hidden_state_channels // 2) * self.h_prime * self.w_prime

        # Politika başlığı (Policy Head)
        self.fc_policy = nn.Linear(self.flat_features_dim, action_space_size)

        # Değer başlığı (Value Head)
        self.fc_value = nn.Linear(self.flat_features_dim, 1)
        print(f"PredictionNetwork: Input Hidden ({self.hidden_channels},{self.h_prime},{self.w_prime}), Output Policy ({action_space_size}), Value (scalar)")

    def forward(self, hidden_state):
        # hidden_state: (batch_size, hidden_channels, H', W')
        batch_size = hidden_state.size(0)

        x = F.relu(self.bn1(self.conv1(hidden_state)))
        x_flat = x.view(batch_size, -1) # Düzleştir

        policy_logits = self.fc_policy(x_flat) # (batch_size, action_space_size)
        value = self.fc_value(x_flat)          # (batch_size, 1)
        # MuZero genellikle değeri tanh ile [-1, 1] veya oyunun değer aralığına sıkıştırır.
        # value = torch.tanh(value)

        return policy_logits, value

# Ağların nasıl kullanılacağına dair örnek (ana eğitim döngüsünde olacak):
if __name__ == '__main__':
    batch_size = CONFIG['batch_size']
    obs_channels = CONFIG['observation_channels']
    board_rows = CONFIG['board_rows']
    board_cols = CONFIG['board_cols']
    hidden_channels = CONFIG['hidden_state_channels']
    action_space = CONFIG['action_space_size']

    # Observation shape (C, H, W)
    obs_shape = (obs_channels, board_rows, board_cols)
    
    # Representation Network
    representation_net = RepresentationNetwork(observation_shape=obs_shape, hidden_state_channels=hidden_channels)
    # Örnek bir gözlem (batch_size, C, H, W)
    dummy_observation = torch.randn(batch_size, obs_channels, board_rows, board_cols)
    initial_hidden_state = representation_net(dummy_observation)
    print(f"Initial hidden state shape: {initial_hidden_state.shape}") # Beklenen: (batch, hidden_channels, H', W')
    # Representation network çıktısının (H', W') boyutları, conv katmanlarına bağlı.
    # Şimdiki implementasyonda (H,W) korunuyor.
    hidden_shape_for_next_nets = (initial_hidden_state.size(1), initial_hidden_state.size(2), initial_hidden_state.size(3))

    # Dynamics Network
    dynamics_net = DynamicsNetwork(hidden_state_shape=hidden_shape_for_next_nets, action_space_size=action_space, hidden_state_channels=hidden_channels)
    dummy_action = torch.randint(0, action_space, (batch_size, 1))
    next_s, reward_pred = dynamics_net(initial_hidden_state, dummy_action)
    print(f"Next hidden state shape: {next_s.shape}, Predicted reward shape: {reward_pred.shape}")

    # Prediction Network
    prediction_net = PredictionNetwork(hidden_state_shape=hidden_shape_for_next_nets, action_space_size=action_space, hidden_state_channels=hidden_channels)
    policy_log, value_pred = prediction_net(initial_hidden_state)
    print(f"Predicted policy logits shape: {policy_log.shape}, Predicted value shape: {value_pred.shape}") 