# MuZero ajanı.

import torch
import numpy as np
from torch.optim import lr_scheduler # LR Scheduler import

# from .config import CONFIG # Artık __init__ ile config alacağız
from .networks import RepresentationNetwork, DynamicsNetwork, PredictionNetwork
from .mcts import run_mcts_simulations, get_mcts_action_distribution
# checkers_game.board'dan Board ve diğer sabitleri import etmeyeceğiz,
# bunun yerine agent bir environment (ortam) alacak.

class MuZeroAgent:
    def __init__(self, board_env_init_fn, config, device=None):
        """
        MuZero ajanı.
        board_env_init_fn: checkers_game.Board() gibi, yeni bir oyun ortamı örneği döndüren bir fonksiyon.
        config: Yapılandırma sözlüğü.
        """
        self.config = config
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"MuZeroAgent kullanacağı cihaz: {self.device}")

        # Oyun ortamını başlatmak için fonksiyon
        self.board_env_init_fn = board_env_init_fn
        # Test/örnek bir ortam (özelliklerini almak için)
        _temp_env = self.board_env_init_fn()
        # Gözlem şekli (C, H, W)
        # Bu, ortamdan alınacak bir gözlemin nasıl ağa verilecek formata dönüştürüleceğine bağlıdır.
        # Şimdilik config'deki değerleri varsayalım.
        self.obs_shape = (self.config['observation_channels'], self.config['board_rows'], self.config['board_cols'])
        self.action_space_size = self.config['action_space_size'] # Bu, Board'daki hamlelerle eşleşmeli

        # MuZero Ağları
        self.representation_net = RepresentationNetwork(
            observation_shape=self.obs_shape, 
            hidden_state_channels=self.config['hidden_state_channels']
        ).to(self.device)
        
        # Representation'dan çıkan gizli durumun şeklini al
        # Dummy bir gözlem geçirerek öğrenelim (bu daha dinamik yapar)
        _dummy_obs = torch.zeros(1, *self.obs_shape, device=self.device)
        _initial_hidden_s = self.representation_net(_dummy_obs)
        self.hidden_state_shape_for_other_nets = (_initial_hidden_s.size(1), _initial_hidden_s.size(2), _initial_hidden_s.size(3))

        self.dynamics_net = DynamicsNetwork(
            hidden_state_shape=self.hidden_state_shape_for_other_nets,
            action_space_size=self.action_space_size,
            hidden_state_channels=self.config['hidden_state_channels']
        ).to(self.device)
        
        self.prediction_net = PredictionNetwork(
            hidden_state_shape=self.hidden_state_shape_for_other_nets,
            action_space_size=self.action_space_size,
            hidden_state_channels=self.config['hidden_state_channels'] # Prediction'ın kendi içindeki hidden state değil, aldığı state
        ).to(self.device)

        # Ağları eğitim moduna al (eğer checkpoint'ten yüklenmeyecekse)
        self.representation_net.train()
        self.dynamics_net.train()
        self.prediction_net.train()

        # Optimizasyon için birleşik parametreler (veya ayrı optimizer'lar)
        self.all_params = list(self.representation_net.parameters()) + \
                          list(self.dynamics_net.parameters()) + \
                          list(self.prediction_net.parameters())
        self.optimizer = torch.optim.Adam(self.all_params, lr=self.config['learning_rate'], weight_decay=self.config.get('weight_decay', 1e-4))

        # Öğrenme oranı zamanlayıcısı
        self.scheduler = lr_scheduler.StepLR(
            self.optimizer,
            step_size=self.config.get('learning_rate_decay_steps', 10000), # config'den al, yoksa varsayılan
            gamma=self.config.get('learning_rate_decay_gamma', 0.5)     # config'den al, yoksa varsayılan
        )


    def _board_to_observation(self, board_state_from_env):
        """
        Oyun tahtası durumunu (numpy array) ağın alacağı formata (torch tensor) dönüştürür.
        Girdi: board_env.board (numpy array [rows, cols])
        Çıktı: observation tensor [observation_channels, rows, cols]
        
        Kanal 1: Mevcut oyuncunun taşları (1), diğerleri (0)
        Kanal 2: Rakip oyuncunun taşları (1), diğerleri (0)
        Kanal 3: Sıra kimde? (Tamamı 1 eğer P1, Tamamı 0 eğer P2 - veya tersi)
                 Bu, config.observation_channels = 3 varsayımına göredir.
        """
        # board_state_from_env, board.board (numpy array) olmalı
        # board.current_player bilgisi de lazım.
        # Bu metod, board_env (Board sınıfı örneği) almalı.
        current_board_array = board_state_from_env.board # numpy array
        current_player_id = board_state_from_env.current_player
        p1_id = board_state_from_env.P1_PIECE
        p2_id = board_state_from_env.P2_PIECE

        obs = np.zeros((self.obs_shape[0], self.obs_shape[1], self.obs_shape[2]), dtype=np.float32)

        if current_player_id == p1_id:
            obs[0, :, :] = (current_board_array == p1_id).astype(np.float32)
            obs[1, :, :] = (current_board_array == p2_id).astype(np.float32)
            obs[2, :, :] = 1.0 # P1'in sırası
        else: # P2'nin sırası
            obs[0, :, :] = (current_board_array == p2_id).astype(np.float32) # Kendi taşları
            obs[1, :, :] = (current_board_array == p1_id).astype(np.float32) # Rakip taşları
            obs[2, :, :] = 0.0 # P2'nin sırası (veya -1.0 da olabilir)
        
        return torch.from_numpy(obs).unsqueeze(0).to(self.device) # (1, C, H, W)

    def select_action(self, board_env_state, is_training=True, game_board_for_mcts=None):
        """
        Mevcut oyun durumu için MCTS kullanarak bir aksiyon seçer.
        board_env_state: Oyun ortamının (Board sınıfı) mevcut durumu.
        is_training: Eğitim sırasında mı, yoksa değerlendirme/oyun sırasında mı?
                     Eğitimde temperature ile keşif, oyunda greedy seçim yapılabilir.
        """
        # 1. Mevcut tahta durumunu ağın alacağı formata dönüştür
        observation_tensor = self._board_to_observation(board_env_state)

        # 2. Representation Network ile ilk gizli durumu elde et
        with torch.no_grad(): # MCTS sırasında ağ ağırlıkları güncellenmez
            self.representation_net.eval() # Inference modu
            self.dynamics_net.eval()
            self.prediction_net.eval()
            initial_hidden_state = self.representation_net(observation_tensor).squeeze(0) # Batch boyutunu kaldır
        
        # 3. MCTS simülasyonlarını çalıştır
        # game_board ve current_player MCTS'e terminal kontrolü için verilebilir.
        # Şimdilik MCTS'in içindeki TODO'lara göre None geçiyoruz.
        # TODO: MCTS'e board_env_state'i veya gerekli bilgileri aktar.
        root_node = run_mcts_simulations(
            initial_hidden_state,
            self.representation_net, 
            self.dynamics_net, 
            self.prediction_net,
            game_board=game_board_for_mcts if game_board_for_mcts else board_env_state, # MCTS'in terminal durumları ve geçerli hamleleri bilmesi için
            num_simulations=self.config['mcts_simulations'],
            current_player=board_env_state.current_player, # MCTS'e oyuncu bilgisi
            is_training=is_training # is_training bilgisini MCTS'e ilet
        )

        # 4. Aksiyon dağılımını al (Bu, tüm action_space_size için bir dağılım verir)
        temperature_for_sampling = self.config.get('mcts_temperature_train', 1.0) if is_training else self.config.get('mcts_temperature_play', 0.0)
        # MCTS politikası (eğitim hedefi için, genellikle temperature=1.0 ile alınır, gürültüsüz olabilir)
        # get_mcts_action_distribution, root_node'daki çocukların ziyaret sayılarına göre bir dağılım verir.
        # Bu, target policy için uygundur.
        _, mcts_target_policy_distribution = get_mcts_action_distribution(root_node, temperature=1.0) # Hedef politika için T=1
        mcts_root_value_estimate = root_node.value() # Kök düğümün MCTS sonrası değeri

        # Hamle seçimi için (eğitimde keşif, oyunda açgözlü)
        _, mcts_sampling_action_probs = get_mcts_action_distribution(root_node, temperature=temperature_for_sampling)
        
        # 5. Board'dan geçerli hamleleri ve bunlara karşılık gelen action_id'leri al
        legal_moves_obj_list, legal_action_ids_list = board_env_state.get_legal_moves_with_action_ids(board_env_state.current_player)

        if not legal_moves_obj_list:
            # print("select_action: Oyuncu için geçerli hamle yok.")
            if is_training: self._set_train_mode(True)
            # Politika ve değer için yine de bir şey döndürmek gerekebilir mi? (Örn: uniform policy, 0 value)
            # Şimdilik, hamle yoksa her şeyi None döndür.
            return None, None, np.zeros(self.action_space_size, dtype=np.float32), 0.0 # Hedefler için dummy değerler

        if not legal_action_ids_list:
            # print("select_action: Geçerli action_id listesi boş, bu beklenmedik bir durum.")
            if is_training: self._set_train_mode(True)
            return None, None, np.zeros(self.action_space_size, dtype=np.float32), 0.0

        # mcts_sampling_action_probs (96,) boyutunda.
        probs_for_sampling_legal_actions = mcts_sampling_action_probs[legal_action_ids_list]
        
        if np.sum(probs_for_sampling_legal_actions) < 1e-6 or len(legal_action_ids_list) == 0:
            if len(legal_action_ids_list) > 0:
                # print("Uyarı: MCTS (örnekleme için), geçerli hamleler için çok düşük/sıfır olasılıklar verdi. Uniform dağılım kullanılıyor.")
                probs_for_sampling_legal_actions = np.ones(len(legal_action_ids_list), dtype=np.float32) / len(legal_action_ids_list)
            else: # Hiç geçerli hamle yoksa (yukarıda handle edilmiş olmalı ama güvenlik için)
                if is_training: self._set_train_mode(True)
                return None, None, np.zeros(self.action_space_size, dtype=np.float32), 0.0
        else:
            probs_for_sampling_legal_actions /= np.sum(probs_for_sampling_legal_actions)

        chosen_index = np.random.choice(len(legal_moves_obj_list), p=probs_for_sampling_legal_actions)
        selected_move_object = legal_moves_obj_list[chosen_index]
        actual_chosen_action_id_for_training = legal_action_ids_list[chosen_index]
        
        if is_training: 
            self._set_train_mode(True)
            
        return selected_move_object, actual_chosen_action_id_for_training, mcts_target_policy_distribution, mcts_root_value_estimate

    def scheduler_step(self):
        """Öğrenme oranı zamanlayıcısının adımını ilerletir."""
        self.scheduler.step()
        # print(f"LR Scheduler step. New LR: {self.scheduler.get_last_lr()}") # Debug için

    def _set_train_mode(self, is_training_mode):
        if is_training_mode:
            self.representation_net.train()
            self.dynamics_net.train()
            self.prediction_net.train()
        else:
            self.representation_net.eval()
            self.dynamics_net.eval()
            self.prediction_net.eval()

    def get_parameters(self):
        """Ajanın tüm ağ parametrelerini döndürür (optimizasyon için)."""
        return self.all_params

    def save_checkpoint(self, path, epoch, loss):
        print(f"Checkpoint kaydediliyor: {path}")
        torch.save({
            'epoch': epoch,
            'representation_net_state_dict': self.representation_net.state_dict(),
            'dynamics_net_state_dict': self.dynamics_net.state_dict(),
            'prediction_net_state_dict': self.prediction_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(), # Scheduler durumunu kaydet
            'loss': loss,
            'config_agent_specific': self.config # Ajan başlatılırken kullanılan config'i kaydet
        }, path)

    def load_checkpoint(self, path):
        print(f"Checkpoint yükleniyor: {path}")
        try:
            checkpoint = torch.load(path, map_location=self.device)
            self.representation_net.load_state_dict(checkpoint['representation_net_state_dict'])
            self.dynamics_net.load_state_dict(checkpoint['dynamics_net_state_dict'])
            self.prediction_net.load_state_dict(checkpoint['prediction_net_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
            if 'scheduler_state_dict' in checkpoint:
                self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            else:
                print("Uyarı: Checkpoint'te scheduler durumu bulunamadı. Scheduler sıfırdan başlayacak.")

            # Ağları eval moduna al, çünkü checkpoint genellikle eğitim sonrası veya test için yüklenir.
            # Eğer eğitime devam edilecekse, trainer bunu train() moduna almalı.
            self._set_train_mode(False)
            
            start_epoch = checkpoint['epoch']
            last_loss = checkpoint['loss']
            # loaded_config_agent = checkpoint.get('config_agent_specific', None)
            # if loaded_config_agent: self.config = loaded_config_agent # Opsiyonel: Kayıtlı config ile üzerine yaz
            print(f"Checkpoint başarıyla yüklendi. Epoch: {start_epoch}, Loss: {last_loss}")
            return start_epoch, last_loss
        except FileNotFoundError:
            print(f"Checkpoint dosyası bulunamadı: {path}. Sıfırdan başlanacak.")
            return 0, float('inf') # Başlangıç epoch ve loss
        except Exception as e:
            print(f"Checkpoint yüklenirken hata oluştu: {e}. Sıfırdan başlanacak.")
            return 0, float('inf')


# Örnek Kullanım (main.py veya trainer.py içinde olacak):
if __name__ == '__main__':
    from checkers_game.board import Board # board.py'den Board sınıfını import et
    from checkers_game.game import Game # Test için Game sınıfını da alabiliriz.
    
    # CONFIG içine eksik MCTS parametrelerini ekleyelim (eğer yoksa)
    CONFIG.setdefault('mcts_c_puct', 1.25)
    CONFIG.setdefault('mcts_pb_c_base', 19652)
    CONFIG.setdefault('mcts_pb_c_init', 1.25)
    CONFIG.setdefault('discount_factor', 0.997)
    CONFIG.setdefault('mcts_add_dirichlet_noise', True)
    CONFIG.setdefault('mcts_temperature_train', 1.0)
    CONFIG.setdefault('mcts_temperature_play', 0.0)
    CONFIG.setdefault('weight_decay', 1e-4)

    def create_board_env(): # board_env_init_fn için bir örnek
        return Board()

    agent = MuZeroAgent(board_env_init_fn=create_board_env, config=CONFIG)
    current_game_board = create_board_env()

    print("Ajanla örnek bir hamle seçimi yapılıyor...")
    # Örnek bir oyun durumu (başlangıç)
    current_game_board.display_board()
    
    # Agent'tan hamle seçmesini isteyelim
    # Bu, içinde MCTS çalıştıracak
    # Oyunun başında agent'ın eval modunda olması mantıklı olabilir (checkpoint yüklendiyse)
    # agent._set_train_mode(False) # Eğer direkt oynayacaksak
    selected_move, action_id_for_training, mcts_target_policy_distribution, mcts_root_value_estimate = agent.select_action(current_game_board, is_training=False) # Play modunda test

    if selected_move:
        print(f"\nAjanın seçtiği hamle (obje): {selected_move}")
        print(f"Bu hamleye karşılık gelen (varsayımsal) action_id: {action_id_for_training}")
        # Bu hamleyi tahtada uygula
        # current_game_board.make_move(selected_move)
        # current_game_board.display_board()
    else:
        print("Ajan bir hamle seçemedi (veya oyun bitti).")

    # Checkpoint kaydetme ve yükleme testi
    # agent.save_checkpoint("dummy_checkpoint.pth", epoch=1, loss=0.5)
    # loaded_epoch, loaded_loss = agent.load_checkpoint("dummy_checkpoint.pth")
    # print(f"Yüklenen epoch: {loaded_epoch}, Yüklenen loss: {loaded_loss}") 