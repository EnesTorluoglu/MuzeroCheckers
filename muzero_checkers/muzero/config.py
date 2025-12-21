# Hiperparametreler ve yapılandırma ayarları.

CONFIG = {
    "seed": 42,
    "board_rows": 8,
    "board_cols": 3,
    # Diğer hiperparametreler buraya eklenecek
    "observation_channels": 3, # P1 pieces, P2 pieces, player_to_move plane
    "action_space_size": 96,  # Örnek: 8*3 (kare sayısı) * 4 (olası hamle türü/yönü)
    "hidden_state_channels": 64, # Gizli durumdaki kanal sayısı
    "num_unroll_steps": 5,     # MCTS sırasında dinamik ağının kaç adım ileri gideceği
    "td_steps": 10,             # Temporal Difference adımları (n-step return için)
    "learning_rate": 0.001,
    "batch_size": 32,
    "replay_buffer_size": 10000,
    "num_epochs": 100000,        # Toplam eğitim epoch sayısı
    "mcts_simulations": 80,    # MCTS simulasyon sayısı
    "mcts_c_puct": 1.25,       # MCTS UCB skorlaması için keşif sabiti
    "mcts_pb_c_base": 19652,   # MCTS UCB prior terimi için base
    "mcts_pb_c_init": 1.25,    # MCTS UCB prior terimi için init
    "dirichlet_alpha": 0.3,    # MCTS root için Dirichlet gürültüsü alpha değeri
    "exploration_fraction": 0.25, # MCTS root için Dirichlet gürültüsü epsilon değeri (exploration)

    # Değer ve ödül için (şimdilik skaler, kategorik değil)
    # "reward_support_size": 601, # Örnek: -300 den +300 e kadar
    # "value_support_size": 601,  # Örnek: -300 den +300 e kadar

    # Eğitim ve Loglama
    "detailed_log_game_interval": 1000, # Kaç oyunda bir detaylı oyun logu (tahta çizimi) yapılacak (ilk oyun her zaman loglanır)
    "checkpoint_interval_epochs": 50, # Checkpoint kaydetme sıklığı (epoch bazında)
    "learning_rate_decay_steps": 3000, # Kaç epoch'ta bir öğrenme oranı düşürülecek
    "learning_rate_decay_gamma": 0.5,   # Öğrenme oranının düşürülme faktörü
    "min_game_steps_for_buffer": 8,   # Replay buffer'a eklenmesi için minimum oyun adım sayısı
    "min_games_for_training": 20, # 1 epoch egitim icin bufferde en az bu kadar oyun olmali
} 