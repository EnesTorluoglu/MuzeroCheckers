# Monte Carlo Ağaç Araması (MCTS) implementasyonu.

import math
import numpy as np
import torch
from torch.distributions import Dirichlet

from .config import CONFIG

class Node:
    def __init__(self, prior_prob, parent=None, action_taken=None):
        self.parent = parent
        self.children = {}  # action_id -> Node
        self.action_taken = action_taken # Bu düğüme ulaşmak için yapılan aksiyon

        self.visit_count = 0
        self.reward_sum = 0.0 # Bu düğümden sonra elde edilen toplam değer (value)
        self.prior_prob = prior_prob # Politika ağından gelen öncül olasılık P(s,a)
        
        self.hidden_state = None # Bu düğüme karşılık gelen gizli durum
        self.predicted_reward = None # Dynamics modelinden gelen anlık ödül (root hariç)

    def value(self):
        """Düğümün ortalama Q-değerini döndürür."""
        if self.visit_count == 0:
            return 0.0
        return self.reward_sum / self.visit_count

    def is_expanded(self):
        """Bu düğümün çocuklarının oluşturulup oluşturulmadığını kontrol eder."""
        return len(self.children) > 0

    def __repr__(self):
        return f"Node(visits={self.visit_count}, value={self.value():.2f}, prior={self.prior_prob:.3f}, action={self.action_taken}, expanded={self.is_expanded()})"


def ucb_score(parent_node: Node, child_node: Node, c_puct: float):
    """PUCT (Polynomial Upper Confidence Trees) skorunu hesaplar."""
    pb_c = math.log((parent_node.visit_count + CONFIG['mcts_pb_c_base'] + 1) / CONFIG['mcts_pb_c_base']) + CONFIG['mcts_pb_c_init']
    pb_c *= math.sqrt(parent_node.visit_count) / (child_node.visit_count + 1)
    
    prior_score = pb_c * child_node.prior_prob
    value_score = child_node.value()
    
    # MuZero makalesindeki UCB formülü (Q(s,a) + P(s,a) * sqrt(N(s)) / (1 + N(s,a)))
    # Buradaki c_puct, P(s,a) üzerindeki etkiyi ayarlar.
    # value_score, Q(s,a)'ya karşılık gelir.
    # prior_score, P(s,a) * sqrt(N(s)) / (1 + N(s,a)) kısmına benzer bir terimdir.
    # Farklı PUCT/UCB formülasyonları mevcut. Şimdilik yukarıdaki AlphaZero benzeri bir formül.
    # Daha basit bir UCB:
    if child_node.visit_count == 0:
        exploration_term = float('inf') # Henüz ziyaret edilmemişse öncelik ver
    else:
        exploration_term = c_puct * child_node.prior_prob * (math.sqrt(parent_node.visit_count) / (1 + child_node.visit_count))
    
    return value_score + exploration_term


def run_mcts_simulations(root_hidden_state, representation_net, dynamics_net, prediction_net, game_board, num_simulations, current_player, is_training=True):
    """
    MCTS simülasyonlarını çalıştırır ve kök düğümü döndürür.
    game_board, current_player gibi bilgiler, oyunun sonlanıp sonlanmadığını veya geçerli hamleleri kontrol etmek için kullanılabilir.
    Ancak MuZero'da bu bilgiler genellikle gizli durum ve ağlar üzerinden öğrenilir.
    Şimdilik, terminal durum kontrolü ve geçerli aksiyon maskesi için game_board'a ihtiyaç duyabiliriz.
    """
    # Kök düğümü oluştur
    # Kök düğüm için öncül olasılık (prior) aslında PredictionNetwork'ten gelir.
    # Burada ilk olarak bir tahmin yapmamız gerekiyor.
    device = root_hidden_state.device  # GPU/CPU cihazını al
    
    with torch.no_grad():
        policy_logits, value = prediction_net(root_hidden_state.unsqueeze(0)) # Unsqueeze for batch dim
        policy_probs_tensor = torch.softmax(policy_logits, dim=1).squeeze(0)  # GPU'da tut
        root_value = value.item()

    root_node = Node(prior_prob=1.0) # Root'un prior'u 1.0 veya tüm aksiyonlara dağılmış olabilir, prediction'dan alacağız.
    root_node.hidden_state = root_hidden_state
    # Root node'un visit_count ve reward_sum'ı backpropagation ile güncellenecek.
    # Genişletme (Expansion) için root'un policy'sini kullanacağız.

    # Dirichlet gürültüsü ekle (eğitim sırasında keşfi artırmak için)
    if is_training and CONFIG.get('mcts_add_dirichlet_noise', True): # Sadece eğitimde ve config izin veriyorsa
        dirichlet_alpha = CONFIG.get('dirichlet_alpha', 0.3)
        exploration_fraction = CONFIG.get('exploration_fraction', 0.25)
        # PyTorch Dirichlet kullan - GPU'da kalır
        action_space_size = CONFIG['action_space_size']
        dirichlet_dist = Dirichlet(torch.full((action_space_size,), dirichlet_alpha, device=device))
        noise = dirichlet_dist.sample()
        policy_probs_noisy = (1 - exploration_fraction) * policy_probs_tensor + exploration_fraction * noise
        # Şimdi numpy'a çevir (maskeleme için gerekli)
        policy_probs_noisy = policy_probs_noisy.cpu().numpy()
    else:
        policy_probs_noisy = policy_probs_tensor.cpu().numpy()

    # Root düğümü ilk genişletme
    _expand_node(node_to_expand=root_node, 
                 policy_probabilities=policy_probs_noisy, 
                 predicted_value=root_value, 
                 hidden_state_at_node=root_hidden_state, 
                 prediction_net=prediction_net, 
                 dynamics_net=dynamics_net, 
                 reward_leading_to_this_node=0.0,
                 game_board=game_board, # Kök için oyun tahtasını ve oyuncuyu ver
                 current_player=current_player)
    # Root node için value, doğrudan prediction'dan geldiği için reward_sum'a eklenir.
    # Ancak backprop sırasında bu değer tekrar kullanılacak. Şimdilik visit_count = 1, reward_sum = root_value olarak ayarlayabiliriz.
    # Veya _expand_node'da policy'yi çocuklara dağıtıp, backprop için root_value'yu kullanırız.

    for _ in range(num_simulations):
        current_node = root_node
        search_path = [current_node] # Geri yayılım için yolu tut

        # 1. Selection (Seçim)
        while current_node.is_expanded():
            # PUCT/UCB skoruna göre en iyi çocuğu seç
            best_child = None
            best_score = -float('inf')
            for action, child in current_node.children.items():
                score = ucb_score(current_node, child, CONFIG['mcts_c_puct'])
                if score > best_score:
                    best_score = score
                    best_child = child
            if best_child is None:
                # Bu durum olmamalı eğer is_expanded() doğruysa ve çocuklar varsa
                # Belki de tüm çocukların visit_count'u 0 ve value'su -inf ise sorun olabilir.
                # Veya geçerli hamle kalmamıştır (oyun sonu).
                # Şimdilik, geçerli bir çocuk bulunamazsa döngüden çık.
                # TODO: Oyun sonu durumlarını burada daha iyi ele al.
                print("MCTS Selection Error: No best child found in expanded node.")
                break 
            current_node = best_child
            search_path.append(current_node)
        
        if current_node is None: # Önceki break'ten dolayı
            continue

        # Bu noktada current_node bir yaprak düğümdür (veya seçimde hata oldu).
        parent_node_of_leaf = search_path[-2] if len(search_path) > 1 else root_node # Yaprağın ebeveyni
        leaf_node = current_node
        
        # Oyunun mevcut durumunu (leaf_node.hidden_state) kullanarak tahmin yap
        # Yaprak düğümün değeri, PredictionNetwork tarafından tahmin edilir.
        with torch.no_grad():
            # leaf_node'un hidden_state'i henüz yoksa (root'tan sonraki ilk seçimde)
            # parent_node_of_leaf.hidden_state ve leaf_node.action_taken kullanarak DynamicsNetwork'ten elde edilmeli.
            # Eğer leaf_node zaten root_node ise, hidden_state'i vardır.
            if leaf_node.hidden_state is None and leaf_node.action_taken is not None:
                action_tensor = torch.tensor([[leaf_node.action_taken]], dtype=torch.long, device=parent_node_of_leaf.hidden_state.device)
                # parent_node_of_leaf.hidden_state zaten batch boyutunda olmalı (eğer root ise unsqueeze edilmişti)
                # Eğer değilse, batch boyutunu ekle
                hs_for_dynamics = parent_node_of_leaf.hidden_state
                if hs_for_dynamics.ndim == 3: # (C, H, W) ise -> (1, C, H, W)
                    hs_for_dynamics = hs_for_dynamics.unsqueeze(0)

                next_hidden_state, predicted_reward = dynamics_net(hs_for_dynamics, action_tensor)
                leaf_node.hidden_state = next_hidden_state.squeeze(0) # Batch boyutunu kaldır
                leaf_node.predicted_reward = predicted_reward.item()
                value_for_backup = predicted_reward.item() # Bu anlık ödül
            else:
                # Eğer hidden_state zaten varsa (örneğin root node genişletildikten sonraki ilk yaprak)
                # veya bir şekilde hesaplanmışsa. Ya da oyun terminal ise.
                # TODO: Terminal durum kontrolü (game_board veya özel bir flag ile)
                # if game_is_terminal(leaf_node.hidden_state):
                #    value_for_backup = get_terminal_reward(...)
                # else: 
                #    policy_logits_leaf, value_leaf_tensor = prediction_net(leaf_node.hidden_state.unsqueeze(0))
                #    value_for_backup = value_leaf_tensor.item()
                #    policy_probs_leaf = torch.softmax(policy_logits_leaf, dim=1).squeeze(0).cpu().numpy()
                #    _expand_node(leaf_node, policy_probs_leaf, value_for_backup, leaf_node.hidden_state, prediction_net, dynamics_net, leaf_node.predicted_reward or 0.0)
                # Şimdilik, eğer hidden_state varsa, prediction yapalım.
                # Oyunun terminal olup olmadığını burada kontrol etmiyoruz, bu bir eksiklik.
                # MuZero, terminal düğümler için gerçek ödülü (veya 0) kullanır ve genişletmez.

                # Geçici olarak, her yaprak düğümde prediction ve expansion yapıyoruz.
                # Bu, eğer oyun terminal ise yanlış olabilir. Terminal kontrolü eklenmeli.
                if leaf_node.hidden_state is not None:
                    policy_logits_leaf, value_leaf_tensor = prediction_net(leaf_node.hidden_state.unsqueeze(0))
                    value_for_backup = value_leaf_tensor.item()
                    policy_probs_leaf = torch.softmax(policy_logits_leaf, dim=1).squeeze(0).cpu().numpy()
                    
                    # Yaprak düğümü genişlet (eğer terminal değilse)
                    # TODO: Terminal kontrolü ekle. Eğer terminal ise, value_for_backup gerçek terminal ödülü olmalı.
                    # Şimdilik her zaman genişletiyoruz.
                    _expand_node(leaf_node, 
                                 policy_probs_leaf, 
                                 value_for_backup, 
                                 leaf_node.hidden_state, 
                                 prediction_net, 
                                 dynamics_net, 
                                 leaf_node.predicted_reward or 0.0,
                                 game_board=None, # Öğrenilmiş state için None geçiyoruz
                                 current_player=None)
                else:
                    # Hidden state yoksa (örneğin, root genişletilmemiş ve seçilmişse - bu durum olmamalı)
                    # Veya terminal bir duruma ulaşıldı ve state yok (bu da olmamalı)
                    print("MCTS Warning: Leaf node without hidden_state encountered unexpectedly.")
                    value_for_backup = 0 # Varsayılan bir değer, durumu ele al


        # 3. Backpropagation (Geri Yayılım)
        # MuZero'da değer hesabı: Q(s,a) = r(s,a) + γ * V(s')
        # search_path'i sondan başa (leaf -> root) gezerken:
        # - value_for_backup başlangıçta yaprak düğümün V(s) değeridir
        # - Her düğüme çıkarken, o düğümün Q-değerini güncelleriz
        # - Sonra bir üst düğüme iletilecek değeri hazırlarız: r + γ * value_for_backup
        
        discount_factor = CONFIG.get('discount_factor', 0.997)
        
        for i, node_in_path in enumerate(reversed(search_path)):
            node_in_path.visit_count += 1
            
            # Bu düğümün reward_sum'ına value_for_backup'ı ekle
            # İlk iterasyonda (yaprak düğüm) bu V(s_leaf)'dir
            # Sonraki iterasyonlarda bu r + γ*V(child) formülünden gelir
            node_in_path.reward_sum += value_for_backup
            
            # Bir sonraki iterasyon için (üst düğüme geçerken) value_for_backup'ı güncelle
            # Bu düğümün (node_in_path) predicted_reward'ı, bu düğüme gelinirken alınan ödüldür
            # Yani ebeveynin bakış açısından: Q(parent, action) = r(node) + γ * V(node)
            if node_in_path.predicted_reward is not None:
                # Bu düğüm root değil, predicted_reward var
                value_for_backup = node_in_path.predicted_reward + discount_factor * value_for_backup
            # else: Bu root node, daha fazla backup yok (döngü bitiyor)

    return root_node

def _expand_node(node_to_expand: Node, policy_probabilities, predicted_value, hidden_state_at_node, prediction_net, dynamics_net, reward_leading_to_this_node, game_board, current_player):
    """
    Bir düğümü genişletir: Çocuklarını oluşturur ve onlara öncül olasılıkları atar.
    predicted_value: Bu node'un (node_to_expand) prediction network'ten gelen değeri.
    reward_leading_to_this_node: Bu node'a (node_to_expand) gelinirken alınan anlık ödül (eğer root değilse).
    game_board: Eğer biliniyorsa (örneğin root node için) geçerli hamleleri almak için oyun tahtası.
    current_player: game_board ile birlikte kullanılır.
    """
    # Bu düğümün değeri, prediction'dan gelen değerdir.
    # Geri yayılım için kullanılacak.
    # node_to_expand.reward_sum += predicted_value # Ziyaret sayısı 1 olacağından ortalamayı etkiler.
    # node_to_expand.visit_count += 1 # Genişletme aynı zamanda bir ziyaret sayılır mı? Genelde backprop'ta sayılır.
    # Şimdilik, expand sadece çocukları oluşturur. Değerler backprop'ta güncellenir.

    node_to_expand.hidden_state = hidden_state_at_node # Gizli durumu sakla

    final_policy_for_expansion = policy_probabilities

    if game_board is not None and current_player is not None:
        # get_legal_moves_with_action_ids iki değer döndürür: move objeleri listesi ve action_id'leri listesi
        legal_move_objects, legal_action_id_list = game_board.get_legal_moves_with_action_ids(current_player)
        
        if not legal_action_id_list: # Eğer geçerli aksiyon ID listesi boşsa
            # Geçerli hamle yoksa, bu düğüm terminal olabilir veya oyuncunun hamlesi kalmamıştır.
            # Bu durumda çocuk oluşturulmayacak.
            final_policy_for_expansion = np.zeros_like(policy_probabilities)
        else:
            # Vectorized masking - for-loop yerine numpy indexing kullan
            legal_action_ids = np.array(legal_action_id_list, dtype=np.int32)
            
            # Mask oluştur - yasal aksiyonlar 1, diğerleri 0
            mask = np.zeros_like(policy_probabilities)
            mask[legal_action_ids] = 1.0
            
            # Masked policy
            masked_policy = policy_probabilities * mask
            sum_legal_probs = masked_policy.sum()
            
            if sum_legal_probs > 1e-8:
                final_policy_for_expansion = masked_policy / sum_legal_probs
            else:
                # Ağ tüm geçerli hamlelere ~0 olasılık atadıysa, uniform dağılım kullan
                if len(legal_action_ids) > 0:
                    uniform_policy = np.zeros_like(policy_probabilities)
                    uniform_policy[legal_action_ids] = 1.0 / len(legal_action_ids)
                    final_policy_for_expansion = uniform_policy
                else:
                    final_policy_for_expansion = np.zeros_like(policy_probabilities)

    for action_id, prob in enumerate(final_policy_for_expansion):
        if prob > 0: # Sadece olasılığı olan aksiyonlar için çocuk oluştur (maskeleme sonrası)
            # Çocuk düğümlerin parent'ı node_to_expand olur.
            # Çocuk düğümlerin prior'u, prediction'dan gelen policy'dir.
            # Çocuk düğümlerin hidden_state'i ve predicted_reward'ı henüz bilinmiyor (dynamics ile hesaplanacak).
            node_to_expand.children[action_id] = Node(prior_prob=prob, parent=node_to_expand, action_taken=action_id)


def get_mcts_action_distribution(root_node: Node, temperature=1.0):
    """
    MCTS simülasyonları sonrası aksiyon dağılımını hesaplar.
    Ziyaret sayılarına dayanır.
    """
    if not root_node.is_expanded():
        # Eğer kök düğüm hiç genişletilmemişse (örneğin, tek geçerli hamle varsa ve simülasyon yapılmamışsa)
        # Bu durum ele alınmalı. Ya da en az bir simülasyon zorunlu olmalı.
        # Şimdilik, tek bir aksiyon varsa onu döndür veya uniform dağılım yap.
        # Veya hata verelim, çünkü bu beklenmedik bir durum.
        print("MCTS Error: Root node not expanded when getting action distribution.")
        # Geçici çözüm: Eğer çocuk yoksa, uniform dağılım (bu yanlış olabilir)
        # action_probs = np.ones(CONFIG['action_space_size']) / CONFIG['action_space_size']
        # Daha iyi: Önceden hesaplanan policy'yi kullan (gürültüsüz)
        # Bu bilgi root_node'da saklanmalı.
        # Veya sadece bir çocuğu varsa onu seç.
        if root_node.children: # Genişletilmiş ama belki tek çocuk?
             pass # Aşağıdaki mantık çalışır
        else: # Hiç çocuk yoksa, bu sorun.
            # Bu, root_node'un policy'sini alıp, ona göre bir dağılım yapmak anlamına gelir.
            # Bu bilgi şu an root_node'da direkt yok. `run_mcts_simulations` içinde ilk policy_probs kullanılabilir.
            # Bu fonksiyon çağrıldığında root_node'un genişletilmiş olması beklenir.
            # Testlerde dummy bir dağılım döndürelim.
            dummy_policy = np.random.rand(CONFIG['action_space_size'])
            dummy_policy /= np.sum(dummy_policy)
            print("Warning: MCTS root not expanded, returning dummy policy for action selection.")
            return np.arange(CONFIG['action_space_size']), dummy_policy

    action_space_size = CONFIG['action_space_size']
    visit_counts = np.zeros(action_space_size, dtype=np.float32)
    actions = []
    
    for action_id, child_node in root_node.children.items():
        if action_id < action_space_size: # Güvenlik kontrolü
            visit_counts[action_id] = child_node.visit_count
            actions.append(action_id)
        else:
            print(f"Warning: Action ID {action_id} is out of bounds for action_space_size {action_space_size}")

    if np.sum(visit_counts) == 0:
        # Hiçbir çocuk ziyaret edilmemişse (örneğin, num_simulations=0 veya bir hata sonucu)
        # Bu durumda root node'un ilk policy tahminini kullanmak daha mantıklı olabilir.
        # Şimdilik uniform bir dağılım (veya hata).
        print("MCTS Warning: No children visited, returning uniform distribution.")
        # Eğer actions listesi boşsa bu da sorun. children var ama visit_counts 0 olabilir.
        if not actions:
             # Eğer hiç çocuk (geçerli aksiyon) yoksa, bu da problem.
             # Bu, oyunun bittiği anlamına gelebilir ve MCTS çağrılmamalıydı.
             # Veya root node genişletilemedi.
             # Dummy policy:
            if not root_node.children and policy_probs_from_root: # policy_probs_from_root lazım
                 # Bu mantık burada zor, run_mcts_simulations içinde policy saklanmalı
                pass # Geçici
            # Eğer hiç çocuk yoksa ve policy de yoksa, action space'e göre uniform.
            # Bu çok uç bir durum.
            valid_actions_indices = np.arange(action_space_size)
            action_probs_final = np.ones(action_space_size) / action_space_size
            return valid_actions_indices, action_probs_final
        else:
            # Çocuklar var ama ziyaret edilmemişler. Uniform dağılım yapalım.
            num_children = len(actions)
            action_probs_final = np.zeros(action_space_size, dtype=np.float32)
            for act_id in actions:
                action_probs_final[act_id] = 1.0 / num_children
            return np.array(actions), action_probs_final[actions] # Sadece var olan aksiyonların olasılıklarını döndür.

    if temperature == 0:
        # Sıcaklık 0 ise, en çok ziyaret edilen aksiyonu deterministik olarak seç.
        action_id = np.argmax(visit_counts)
        action_probs_final = np.zeros(action_space_size, dtype=np.float32)
        action_probs_final[action_id] = 1.0
    else:
        # Ziyaret sayılarını sıcaklıkla üssel olarak ayarla ve normalize et.
        # visit_counts = visit_counts**(1 / temperature) # Bu AlphaGo/AlphaZero formulü
        # Daha stabil olması için log-space ve softmax kullanılabilir.
        log_visit_counts = np.log(visit_counts + 1e-10) # Sıfır ziyaretler için küçük bir epsilon
        temp_scaled_logits = log_visit_counts / temperature
        
        # Sadece çocukları olan aksiyonlar üzerinden softmax
        active_logits = temp_scaled_logits[actions]
        exp_logits = np.exp(active_logits - np.max(active_logits)) # Stabilite için max çıkar
        probs_for_active_actions = exp_logits / np.sum(exp_logits)
        
        action_probs_final = np.zeros(action_space_size, dtype=np.float32)
        for i, action_id in enumerate(actions):
            action_probs_final[action_id] = probs_for_active_actions[i]

    # Dönen actions (indisler) ve action_probs_final (olasılıklar) aynı sırada olmalı.
    # actions listesi zaten çocukların action_id'lerini tutuyor.
    # Eğer tüm action_space için olasılık döndürmek istiyorsak, actions listesi yerine np.arange(action_space_size) kullanabiliriz.
    # Ancak sadece geçerli (çocukları olan) aksiyonların olasılıklarını döndürmek daha verimli olabilir.
    # Şimdilik, tüm aksiyon uzayı için olasılıkları döndürelim, geçerli olmayanlar 0 olacak.
    return np.arange(action_space_size), action_probs_final


# config.py içine eklenecek MCTS hiperparametreleri (bazıları zaten var):
# CONFIG['mcts_c_puct'] = 1.25 (veya 1.0, 2.5 gibi değerler)
# CONFIG['mcts_pb_c_base'] = 19652 (AlphaZero)
# CONFIG['mcts_pb_c_init'] = 1.25 (AlphaZero)
# CONFIG['discount_factor'] = 0.997 (veya oyunun uzunluğuna göre ayarlanır)
# CONFIG['mcts_add_dirichlet_noise'] = True
# CONFIG['dirichlet_alpha'] = 0.3 (Dama için ayarlanabilir, Go için 0.03, Satranç için 0.3)
# CONFIG['exploration_fraction'] = 0.25

# Örnek Kullanım (networks.py'deki test bloğuna benzer):
if __name__ == '__main__':
    from .networks import RepresentationNetwork, DynamicsNetwork, PredictionNetwork # Relative import
    # Config'den değerleri al
    batch_size = 1 # MCTS tek bir durum için çalışır
    obs_channels = CONFIG['observation_channels']
    board_rows = CONFIG['board_rows']
    board_cols = CONFIG['board_cols']
    hidden_channels = CONFIG['hidden_state_channels']
    action_space = CONFIG['action_space_size']
    num_sims = CONFIG['mcts_simulations']

    # CONFIG içine eksik MCTS parametrelerini ekleyelim (eğer yoksa)
    CONFIG.setdefault('mcts_c_puct', 1.25)
    CONFIG.setdefault('mcts_pb_c_base', 19652)
    CONFIG.setdefault('mcts_pb_c_init', 1.25)
    CONFIG.setdefault('discount_factor', 0.997)
    CONFIG.setdefault('mcts_add_dirichlet_noise', True)
    # dirichlet_alpha ve exploration_fraction zaten config'de olmalı

    obs_shape = (obs_channels, board_rows, board_cols)
    representation_net = RepresentationNetwork(observation_shape=obs_shape, hidden_state_channels=hidden_channels)
    dummy_observation = torch.randn(1, obs_channels, board_rows, board_cols) # Tek gözlem (batch=1)
    initial_hidden_s = representation_net(dummy_observation)
    
    hidden_shape_for_nets = (initial_hidden_s.size(1), initial_hidden_s.size(2), initial_hidden_s.size(3))
    dynamics_net = DynamicsNetwork(hidden_state_shape=hidden_shape_for_nets, action_space_size=action_space, hidden_state_channels=hidden_channels)
    prediction_net = PredictionNetwork(hidden_state_shape=hidden_shape_for_nets, action_space_size=action_space, hidden_state_channels=hidden_channels)

    print(f"Running {num_sims} MCTS simulations...")
    # game_board ve current_player MCTS içinde (şimdilik) kullanılmıyor, None geçilebilir.
    # Ancak terminal durumlar ve geçerli hamle maskelemesi için ileride gerekebilir.
    final_root_node = run_mcts_simulations(initial_hidden_s.squeeze(0), representation_net, dynamics_net, prediction_net, None, num_sims, None)
    
    actions_dist, probs_dist = get_mcts_action_distribution(final_root_node, temperature=1.0)
    print("MCTS Action Distribution (Probs > 0):")
    for i, p in enumerate(probs_dist):
        if p > 1e-4:
            print(f"  Action {i}: {p:.4f} (Child visits: {final_root_node.children.get(i, Node(0)).visit_count if final_root_node.children else 'N/A'})")
    
    chosen_action = np.random.choice(actions_dist, p=probs_dist)
    print(f"\nChosen action by MCTS (sampling): {chosen_action}")

    actions_dist_temp0, probs_dist_temp0 = get_mcts_action_distribution(final_root_node, temperature=0.0)
    chosen_action_greedy = np.random.choice(actions_dist_temp0, p=probs_dist_temp0)
    print(f"Chosen action by MCTS (greedy, temp=0): {chosen_action_greedy}") 