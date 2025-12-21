# Eğitim döngüsü, checkpoint yönetimi ve TensorBoard loglaması.

import torch
import numpy as np
import time
import os # trainer checkpoint path için
import logging # Yeni import
from torch.utils.tensorboard import SummaryWriter

from muzero_checkers.muzero.agent import MuZeroAgent
from muzero_checkers.muzero.replay_buffer import ReplayBuffer, GameTrace
from muzero_checkers.checkers_game.board import Board
# from muzero_checkers.muzero.config import CONFIG # config trainer'a init ile veriliyor

class Trainer:
    def __init__(self, config):
        self.config = config
        self.device = config.get('device', torch.device("cuda" if torch.cuda.is_available() else "cpu"))

        self.board_env_init_fn = lambda: Board() 

        self.agent = MuZeroAgent(self.board_env_init_fn, config=config, device=self.device) # Agent'a da config'i ilet
        self.replay_buffer = ReplayBuffer(config)
        
        self.log_dir = config.get('log_dir', 'logs/muzero_checkers_default_session')
        # SummaryWriter'ın log_dir'i her trainer başlatıldığında (yeni veya devam) doğru set edilmeli.
        # Bu, main.py'de CONFIG['log_dir'] ayarlandıktan sonra Trainer oluşturulduğunda gerçekleşir.
        os.makedirs(self.log_dir, exist_ok=True)
        self.writer = SummaryWriter(log_dir=self.log_dir)
        print(f"TensorBoard logları şuraya yazılacak: {self.log_dir}")

        # Kümülatif oyun istatistikleri için logger
        self.game_stats_logger = logging.getLogger('GameStats')
        self.game_stats_logger.setLevel(logging.INFO)
        # Handler'ların tekrar tekrar eklenmesini önle
        if not self.game_stats_logger.hasHandlers():
            fh_stats = logging.FileHandler(os.path.join(self.log_dir, 'training_games.log'))
            formatter_stats = logging.Formatter('%(asctime)s - %(name)s - %(message)s')
            fh_stats.setFormatter(formatter_stats)
            self.game_stats_logger.addHandler(fh_stats)

        # Detaylı oyun logları için logger
        self.detailed_game_logger = logging.getLogger('DetailedGameLogger')
        self.detailed_game_logger.setLevel(logging.INFO)
        if not self.detailed_game_logger.hasHandlers():
            fh_detail = logging.FileHandler(os.path.join(self.log_dir, 'detailed_games_replay.log'))
            # Detaylı loglar için sadece mesaj yeterli, zaman damgası ve logger adı zaten dosya adında var.
            formatter_detail = logging.Formatter('%(message)s') 
            fh_detail.setFormatter(formatter_detail)
            self.detailed_game_logger.addHandler(fh_detail)
        
        self.detailed_log_interval = self.config.get('detailed_log_game_interval', 100)

        self.current_epoch = 0
        self.total_games_played_in_session = 0 # Bu oturumda oynanan toplam oyun sayısı
        self.total_steps_collected_in_session = 0 # Bu oturumda toplanan toplam adım
        # Oturum bazlı kümülatif kazanma/beraberlik sayıları
        self.session_p1_wins = 0
        self.session_p2_wins = 0
        self.session_draws = 0

        # Checkpoint yükleme (eğer varsa)
        # Trainer state (epoch, total_games_played_overall, total_steps_collected_overall gibi)
        # Agent state (network ağırlıkları, optimizer state) agent.load_checkpoint ile yüklenir.
        self.load_trainer_checkpoint(config.get('trainer_checkpoint_path'))

    def run_self_play_game(self, game_num_overall):
        """Bir self-play oyunu çalıştırır ve toplanan verileri GameTrace olarak döndürür."""
        board_env = self.board_env_init_fn()
        current_trace = GameTrace()
        game_steps = 0
        
        # Detaylı loglama kontrolü
        should_log_details = False
        '''should_log_details = (game_num_overall == 1) or \
                             (self.detailed_log_interval > 0 and game_num_overall % self.detailed_log_interval == 0)'''

        if should_log_details:
            self.detailed_game_logger.info(f"--- Detaylı Log Başlangıcı: Oyun #{game_num_overall} ---")
            self.detailed_game_logger.info(f"Adım 0: Başlangıç durumu (Oyuncu {board_env.current_player} başlayacak)")
            self.detailed_game_logger.info(board_env.display_board(return_string=True))
            self.detailed_game_logger.info("-" * 40)

        start_time = time.time()

        while True:
            game_steps += 1
            self.total_steps_collected_in_session += 1 
            
            current_observation_tensor = self.agent._board_to_observation(board_env)
            
            if should_log_details:
                self.detailed_game_logger.info(f"Adım {game_steps}: Oyuncu {board_env.current_player} düşünüyor...")
                # MCTS öncesi tahta durumu loglanmıştı, şimdi hamle seçimi sonrası loglanacak.

            selected_move_obj, chosen_action_id, mcts_policy_target, mcts_root_value = \
                self.agent.select_action(board_env, is_training=True, game_board_for_mcts=board_env)
            
            if selected_move_obj is None or chosen_action_id is None:
                if should_log_details:
                    self.detailed_game_logger.info("Geçerli hamle bulunamadı veya ajan hata verdi, oyun bitiyor.")
                if not current_trace.game_over_reason:
                    is_over_check, winner_code_check = board_env.check_game_over()
                    if is_over_check:
                        current_trace.game_over_reason = winner_code_check
                    else:
                        current_trace.game_over_reason = board_env.DRAW 
                break 

            mcts_policy_target_for_this_step = mcts_policy_target 
            value_estimate_for_this_step = mcts_root_value     

            # Hamleyi yapmadan önceki oyuncu kimdi? Bu, ödül ataması için önemli.
            player_making_move = board_env.current_player
            
            if should_log_details: 
                self.detailed_game_logger.info(f"DEBUG: Detaylı hamle log bloğuna girildi. Oyun: {game_num_overall}, Adım: {game_steps}") # Test logu
                move_type_str = selected_move_obj.get('type', 'unknown')
                start_pos_str = str(selected_move_obj.get('start_pos', 'N/A')) 
                end_pos_str = str(selected_move_obj.get('end_pos', 'N/A'))   
                captured_count = selected_move_obj.get('captured_count', 0)
                log_move_str = f"Oyuncu {player_making_move} hamle yaptı: Tip: {move_type_str}, Başlangıç: {start_pos_str}, Bitiş: {end_pos_str}"
                if move_type_str == 'capture' and captured_count > 0:
                    log_move_str += f", Yenen Taş Sayısı: {captured_count}"
                
                self.detailed_game_logger.info(f"DEBUG: log_move_str: {log_move_str}") # log_move_str içeriğini logla
                self.detailed_game_logger.info(log_move_str) # Asıl hamle logu
                self.detailed_game_logger.info(f"Action ID: {chosen_action_id}")

            board_env.make_move(selected_move_obj)
            is_over, winner_code = board_env.check_game_over()
            
            if should_log_details: 
                self.detailed_game_logger.info("Hamle sonrası tahta durumu:")
                board_str_to_log = board_env.display_board(return_string=True)
                self.detailed_game_logger.info(f"DEBUG: board_str_to_log (uzunluk: {len(board_str_to_log)}):\n{board_str_to_log}") # Tahta string'ini ve uzunluğunu logla
                self.detailed_game_logger.info(board_str_to_log) # Asıl tahta logu
                self.detailed_game_logger.info("-" * 40)

            reward_for_action = 0.0
            if is_over:
                if winner_code == player_making_move: 
                    reward_for_action = 1.0
                elif winner_code == board_env.DRAW:
                    reward_for_action = 0.0
                else: # Diğer oyuncu kazandı (yani hamleyi yapan oyuncu kaybetti)
                    reward_for_action = -1.0
                current_trace.game_over_reason = winner_code

            current_trace.add_step(
                current_observation_tensor.squeeze(0).cpu(), 
                chosen_action_id,                            
                reward_for_action,                           
                mcts_policy_target_for_this_step,            
                value_estimate_for_this_step                 
            )
            
            if is_over:
                if should_log_details:
                    winner_text_detail = "Oyun Bitti! Sonuç: "
                    if winner_code == board_env.P1_WINS: winner_text_detail += "Beyaz Kazandı (P1)"
                    elif winner_code == board_env.P2_WINS: winner_text_detail += "Siyah Kazandı (P2)"
                    elif winner_code == board_env.DRAW: winner_text_detail += "Berabere"
                    else: winner_text_detail += f"Bilinmeyen Kod ({winner_code})"
                    self.detailed_game_logger.info(winner_text_detail)
                    self.detailed_game_logger.info(f"Toplam Adım: {game_steps}")
                break
            
            if game_steps >= self.config.get('max_moves_per_game', 200):
                if should_log_details:
                    self.detailed_game_logger.info("Oyun maksimum hamle sayısına ulaştı. Berabere sayılıyor.")
                current_trace.game_over_reason = board_env.DRAW 
                break
        
        if should_log_details:
            self.detailed_game_logger.info(f"--- Detaylı Log Sonu: Oyun #{game_num_overall} ---\\n")
        
        # game_data'ya oyunun sonucunu ve uzunluğunu ekleyelim (epoch istatistikleri için)
        current_trace.final_game_length = game_steps
        current_trace.final_winner_code = current_trace.game_over_reason # Bu zaten set ediliyor
        return current_trace

    def collect_game_data(self, num_games_to_collect):
        """Belirli sayıda self-play oyunu oynar ve verileri ReplayBuffer'a kaydeder."""
        # print(f"{num_games_to_collect} adet self-play oyunu başlatılıyor...")
        collected_games_in_epoch = []
        board_ref = Board() # P1_WINS, P2_WINS, DRAW sabitleri için referans

        for i in range(num_games_to_collect):
            # total_games_played_in_session burada artırılmadan önce game_num_overall olarak run_self_play_game'e gönderiliyor
            game_num_for_detailed_log = self.total_games_played_in_session + 1 
            game_data = self.run_self_play_game(game_num_for_detailed_log)
            
            self.total_games_played_in_session += 1 

            if game_data and game_data.game_over_reason is not None:
                # Kümülatif istatistikleri güncelle
                #winner_text = ""
                if game_data.game_over_reason == board_ref.P1_WINS:
                    self.session_p1_wins += 1
                    #winner_text = "White wins!"
                elif game_data.game_over_reason == board_ref.P2_WINS:
                    self.session_p2_wins += 1
                    #winner_text = "Black wins!"
                elif game_data.game_over_reason == board_ref.DRAW:
                    self.session_draws += 1
                    #winner_text = "Draw!"
                #else: # NOT_OVER veya beklenmedik bir durum
                    #winner_text = f"Game Over (Code: {game_data.game_over_reason})"

                # Log mesajını oluştur ve yaz
                '''if self.total_games_played_in_session % 20 == 0:
                    log_message = (
                        f"Game #{self.total_games_played_in_session} - {winner_text} "
                        f"Stats: Total: {self.total_games_played_in_session}, "
                        f"White: {self.session_p1_wins}, Black: {self.session_p2_wins}, Draws: {self.session_draws}"
                    )
                    self.game_stats_logger.info(log_message)'''

            min_steps_for_buffer = self.config.get('min_game_steps_for_buffer', 10) # Çok kısa oyunları ekleme
            if game_data and len(game_data) >= min_steps_for_buffer :
                self.replay_buffer.save_game(game_data)
                # print(f"Oyun #{self.total_games_played_in_session} verisi buffer'a eklendi. Buffer boyutu: {len(self.replay_buffer)} oyun.")
                collected_games_in_epoch.append(game_data)
            elif game_data:
                 #print(f"Uyarı: Oyun #{self.total_games_played_in_session} çok kısa ({len(game_data)} adım), buffer\'a eklenmiyor. Min: {min_steps_for_buffer}")
                 collected_games_in_epoch.append(game_data) # Yine de istatistik için tut
            else:
                pass # print(f"Oyun #{self.total_games_played_in_session + 1} (veya sonrası) veri üretemedi, atlanıyor.")
        return collected_games_in_epoch

    def train_step(self, current_train_step_in_epoch, total_train_steps_in_epoch):
        """Bir eğitim adımı gerçekleştirir."""
        if not self.replay_buffer.is_ready():
            return False 

        # print("Replay buffer'dan batch örnekleniyor...") # Çok sık log
        batch = self.replay_buffer.sample_batch()
        if batch is None:
            return False

        observations_batch, actions_batch, target_rewards_batch, target_policies_batch, target_values_batch = batch
        # Move everything to the correct device
        device = self.config['device']
        observations_batch = observations_batch.to(device)
        actions_batch = actions_batch.to(device)
        target_rewards_batch = target_rewards_batch.to(device)
        target_policies_batch = target_policies_batch.to(device)
        target_values_batch = target_values_batch.to(device)

        initial_hidden_states = self.agent.representation_net(observations_batch) 

        total_loss_val = 0
        value_loss_sum = 0
        policy_loss_sum = 0
        reward_loss_sum = 0
        
        current_hidden_states = initial_hidden_states
        num_unroll_steps = self.config.get('num_unroll_steps', 5)

        for k_step in range(num_unroll_steps + 1):
            policy_logits_k, predicted_values_k = self.agent.prediction_net(current_hidden_states)
            
            value_loss = torch.nn.functional.mse_loss(predicted_values_k.squeeze(-1), target_values_batch[:, k_step])
            total_loss_val += value_loss
            value_loss_sum += value_loss.item()

            policy_loss = torch.nn.functional.cross_entropy(policy_logits_k, target_policies_batch[:, k_step])
            total_loss_val += policy_loss
            policy_loss_sum += policy_loss.item()

            if k_step < num_unroll_steps:
                action_k = actions_batch[:, k_step].unsqueeze(-1)
                next_hidden_states, predicted_rewards_k = self.agent.dynamics_net(current_hidden_states, action_k)
                
                reward_loss = torch.nn.functional.mse_loss(predicted_rewards_k.squeeze(-1), target_rewards_batch[:, k_step])
                total_loss_val += reward_loss
                reward_loss_sum += reward_loss.item()
                
                current_hidden_states = next_hidden_states
        
        self.agent.optimizer.zero_grad()
        total_loss_val.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(self.agent.get_parameters(), self.config.get('max_grad_norm', 40.0))
        self.agent.optimizer.step()

        # Loglama (her train_step için değil, belirli aralıklarla veya epoch sonu yapılabilir)
        # Şimdilik her adımda yapalım, ama TensorBoard'a yazma sıklığı ayarlanabilir.
        # Global step: self.current_epoch * total_train_steps_in_epoch + current_train_step_in_epoch
        global_step_for_logging = self.current_epoch * total_train_steps_in_epoch + current_train_step_in_epoch
        if global_step_for_logging % self.config.get('tensorboard_log_interval_steps', 10) == 0: # Her 10 train_step'te bir logla
            num_terms_in_loss = num_unroll_steps + 1
            avg_value_loss = value_loss_sum / num_terms_in_loss
            avg_policy_loss = policy_loss_sum / num_terms_in_loss
            avg_reward_loss = reward_loss_sum / num_unroll_steps if num_unroll_steps > 0 else 0
            
            self.writer.add_scalar('Train/TotalLoss_step', total_loss_val.item(), global_step_for_logging)
            self.writer.add_scalar('Train/ValueLoss_step', avg_value_loss, global_step_for_logging)
            self.writer.add_scalar('Train/PolicyLoss_step', avg_policy_loss, global_step_for_logging)
            self.writer.add_scalar('Train/RewardLoss_step', avg_reward_loss, global_step_for_logging)
            self.writer.add_scalar('Train/GradientNorm_step', grad_norm.item(), global_step_for_logging)
            # print(f"Eğitim Adımı {global_step_for_logging}: Loss: {total_loss_val.item():.2f} (V:{avg_value_loss:.2f}, P:{avg_policy_loss:.2f}, R:{avg_reward_loss:.2f})")
        return True

    def run_training_loop(self):
        print("Eğitim döngüsü başlatılıyor...")
        num_epochs = self.config.get('num_epochs', 100000)
        games_per_epoch = self.config.get('games_per_epoch', 20)
        train_steps_per_epoch = self.config.get('train_steps_per_epoch', 50)

        # current_epoch checkpoint'ten yüklenmiş olabilir.
        for epoch in range(self.current_epoch, num_epochs):
            self.current_epoch = epoch
            epoch_start_time = time.time()
            print(f"\n--- Epoch {self.current_epoch + 1}/{num_epochs} --- GBuffer: {len(self.replay_buffer)} oyunlar, {self.total_steps_collected_in_session} adımlar ---")

            # 1. Self-Play ile veri topla
            self.agent._set_train_mode(True) # Ağları ve MCTS'yi eğitim moduna al
            print(f"{games_per_epoch} adet self-play oyunu başlatılıyor...")
            games_collected_this_epoch = self.collect_game_data(games_per_epoch)

            self_play_time = time.time() - epoch_start_time
            print(f"Time passed self play: {self_play_time:.2f}s")

            # Epoch istatistiklerini hesapla ve logla
            if games_collected_this_epoch:
                num_p1_wins_epoch = sum(1 for g in games_collected_this_epoch if g.game_over_reason == Board().P1_WINS)
                num_p2_wins_epoch = sum(1 for g in games_collected_this_epoch if g.game_over_reason == Board().P2_WINS)
                num_draws_epoch = sum(1 for g in games_collected_this_epoch if g.game_over_reason == Board().DRAW)
                total_valid_games_epoch = len(games_collected_this_epoch)
                
                avg_game_length_epoch = np.mean([g.total_steps for g in games_collected_this_epoch if g.total_steps is not None]) if total_valid_games_epoch > 0 else 0

                self.writer.add_scalar('SelfPlay_Epoch/P1_WinRate', num_p1_wins_epoch / total_valid_games_epoch if total_valid_games_epoch > 0 else 0, self.current_epoch)
                self.writer.add_scalar('SelfPlay_Epoch/P2_WinRate', num_p2_wins_epoch / total_valid_games_epoch if total_valid_games_epoch > 0 else 0, self.current_epoch)
                self.writer.add_scalar('SelfPlay_Epoch/DrawRate', num_draws_epoch / total_valid_games_epoch if total_valid_games_epoch > 0 else 0, self.current_epoch)
                self.writer.add_scalar('SelfPlay_Epoch/AvgGameLength', avg_game_length_epoch, self.current_epoch)
                print(f"Epoch {self.current_epoch + 1} Self-Play: P1W: {num_p1_wins_epoch}, P2W: {num_p2_wins_epoch}, D: {num_draws_epoch}, AvgLen: {avg_game_length_epoch:.1f}")

                # TensorBoard için toplam oyun sayısına göre loglama (epoch'luk verilerle)
                if self.total_games_played_in_session > 0 and total_valid_games_epoch > 0 : 
                    self.writer.add_scalar('SelfPlay_Games/P1_WinRate_EpochAvg', num_p1_wins_epoch / total_valid_games_epoch, self.total_games_played_in_session)
                    self.writer.add_scalar('SelfPlay_Games/P2_WinRate_EpochAvg', num_p2_wins_epoch / total_valid_games_epoch, self.total_games_played_in_session)
                    self.writer.add_scalar('SelfPlay_Games/DrawRate_EpochAvg', num_draws_epoch / total_valid_games_epoch, self.total_games_played_in_session)
                    self.writer.add_scalar('SelfPlay_Games/AvgGameLength_EpochAvg', avg_game_length_epoch, self.total_games_played_in_session)
                
                # TensorBoard için KÜMÜLATİF genel istatistikler
                if self.total_games_played_in_session > 0:
                    cumulative_p1_win_rate = self.session_p1_wins / self.total_games_played_in_session
                    cumulative_p2_win_rate = self.session_p2_wins / self.total_games_played_in_session
                    cumulative_draw_rate = self.session_draws / self.total_games_played_in_session
                    # Kümülatif ortalama oyun uzunluğu için total_steps_collected_in_session kullanılabilir.
                    cumulative_avg_game_length = self.total_steps_collected_in_session / self.total_games_played_in_session

                    self.writer.add_scalar('SelfPlay_Cumulative/P1_WinRate_Overall', cumulative_p1_win_rate, self.total_games_played_in_session)
                    self.writer.add_scalar('SelfPlay_Cumulative/P2_WinRate_Overall', cumulative_p2_win_rate, self.total_games_played_in_session)
                    self.writer.add_scalar('SelfPlay_Cumulative/DrawRate_Overall', cumulative_draw_rate, self.total_games_played_in_session)
                    self.writer.add_scalar('SelfPlay_Cumulative/AvgGameLength_Overall', cumulative_avg_game_length, self.total_games_played_in_session)

            #tensor_log_time = time.time() - self_play_time
            #print(f"Time passed tensor log: {tensor_log_time:.2f}s")

            train_time = time.time()
            # 2. Ağı eğit
            if self.replay_buffer.is_ready():
                print(f"Epoch {self.current_epoch + 1}: Eğitim adımları başlıyor...")
                self.agent._set_train_mode(True) # Tekrar emin olalım
                for train_step_num in range(train_steps_per_epoch):
                    self.train_step(train_step_num, train_steps_per_epoch)
            else:
                print(f"Epoch {self.current_epoch + 1}: Eğitim için yeterli veri yok, buffer boyutu: {len(self.replay_buffer)}/{self.config.get('min_games_for_training')}")

            if self.replay_buffer.is_ready(): # Sadece eğitim yapıldıysa scheduler adımı at
                self.agent.scheduler_step()

            train_time = time.time() - train_time
            print(f"Time passed training: {train_time:.2f}s")

            # Epoch sonu loglamaları
            self.writer.add_scalar('ReplayBuffer/Size_Games', len(self.replay_buffer), self.current_epoch)
            self.writer.add_scalar('ReplayBuffer/Total_Samples_In_Buffer', self.replay_buffer.get_total_samples(), self.current_epoch)
            
            buffer_fill_ratio = len(self.replay_buffer) / self.replay_buffer.window_size if self.replay_buffer.window_size > 0 else 0
            self.writer.add_scalar('ReplayBuffer/Fill_Ratio_Games', buffer_fill_ratio, self.current_epoch)
            current_lr = self.agent.optimizer.param_groups[0]['lr']
            self.writer.add_scalar('Train_Epoch/LearningRate', current_lr, self.current_epoch)

            # Checkpoint kaydet (belirli aralıklarla)
            if (self.current_epoch + 1) % self.config.get('checkpoint_interval_epochs', 10) == 0:
                self._save_checkpoint()

            epoch_duration = time.time() - epoch_start_time
            self.writer.add_scalar('System/EpochDuration_sec', epoch_duration, self.current_epoch)
            print(f"Epoch {self.current_epoch + 1} tamamlandı. Süre: {epoch_duration:.2f}s")

        print("Tüm eğitim epochları tamamlandı.")
        self._save_checkpoint(is_final=True) # Son bir checkpoint kaydet
        self.writer.close()

    def _save_checkpoint(self, is_final=False):
        """Trainer durumunu ve agent modelini kaydeder."""
        # Agent modelini kaydet
        agent_chkp_pattern = self.config.get('agent_checkpoint_path_pattern', 'checkpoints/muzero_agent_epoch_{}.pth')
        agent_chkp_filename = agent_chkp_pattern.format(self.current_epoch + 1)
        if is_final:
            agent_chkp_filename = self.config.get('agent_checkpoint_for_play', 'checkpoints/muzero_agent_final.pth')
            # Eğer agent_checkpoint_for_play bir pattern içeriyorsa (normalde olmamalı), onu da formatlayalım
            try: agent_chkp_filename = agent_chkp_filename.format(self.current_epoch + 1) # Güvenlik için
            except KeyError: pass
        
        self.agent.save_checkpoint(agent_chkp_filename, epoch=self.current_epoch, loss=0) # Loss değeri burası için çok anlamlı değil
        print(f"Agent modeli kaydedildi: {agent_chkp_filename}")

        # Trainer durumunu kaydet
        trainer_chkp_path = self.config.get('trainer_checkpoint_path', 'checkpoints/trainer_state.pth')
        trainer_state = {
            'epoch': self.current_epoch + 1, # Bir sonraki epoch'tan devam etmek için
            'total_games_played_in_session': self.total_games_played_in_session,
            'total_steps_collected_in_session': self.total_steps_collected_in_session,
            'session_p1_wins': self.session_p1_wins, # Yeni eklendi
            'session_p2_wins': self.session_p2_wins, # Yeni eklendi
            'session_draws': self.session_draws,     # Yeni eklendi
            'replay_buffer_content': self.replay_buffer.get_state() # Replay buffer içeriğini de kaydet
        }
        torch.save(trainer_state, trainer_chkp_path)
        print(f"Trainer durumu kaydedildi: {trainer_chkp_path}")

    def load_trainer_checkpoint(self, path):
        """Trainer durumunu yükler."""
        if path and os.path.exists(path):
            try:
                checkpoint = torch.load(path, map_location=self.device, weights_only=False)
                self.current_epoch = checkpoint.get('epoch', 0)
                self.total_games_played_in_session = checkpoint.get('total_games_played_in_session', 0)
                self.total_steps_collected_in_session = checkpoint.get('total_steps_collected_in_session', 0)
                self.session_p1_wins = checkpoint.get('session_p1_wins', 0) # Yeni eklendi
                self.session_p2_wins = checkpoint.get('session_p2_wins', 0) # Yeni eklendi
                self.session_draws = checkpoint.get('session_draws', 0)     # Yeni eklendi
                
                if 'replay_buffer_content' in checkpoint and checkpoint['replay_buffer_content']:
                    self.replay_buffer.set_state(checkpoint['replay_buffer_content'])
                    print(f"Replay buffer durumu yüklendi, {len(self.replay_buffer)} oyun içeriyor.")
                
                print(f"Trainer durumu yüklendi: Epoch {self.current_epoch -1 }'den devam edilecek (yani Epoch {self.current_epoch} başlayacak). Games in session: {self.total_games_played_in_session}")
            except Exception as e:
                print(f"Trainer checkpoint yüklenirken hata: {e}. Sıfırdan başlanacak.")
                self.current_epoch = 0
                self.total_games_played_in_session = 0
                self.total_steps_collected_in_session = 0
                self.session_p1_wins = 0
                self.session_p2_wins = 0
                self.session_draws = 0
        else:
            print("Trainer checkpoint dosyası bulunamadı. Sıfırdan başlanacak.")
            self.session_p1_wins = 0 # Sıfırdan başlarken de sayaçlar sıfır olmalı
            self.session_p2_wins = 0
            self.session_draws = 0


# Ana çalıştırma bloğu (main.py'de olacak)
if __name__ == '__main__':
    """
    #CONFIG güncellemeleri (test için)
    CONFIG['device'] = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    CONFIG['log_dir'] = 'logs/muzero_checkers_testrun_trainer'
    CONFIG['trainer_checkpoint_path'] = 'checkpoints/trainer_state_test_trainer.pth'
    CONFIG['agent_checkpoint_path_pattern'] = 'checkpoints/muzero_agent_test_trainer_epoch_{}.pth'
    CONFIG['num_epochs'] = 1 # Test için kısa tutalım
    CONFIG['games_per_epoch'] = 2 # Test için az oyun
    CONFIG['train_steps_per_epoch'] = 1 # Test için az eğitim adımı
    CONFIG['replay_buffer_size'] = 10 # Test için küçük buffer (oyun sayısı)
    CONFIG['batch_size'] = 2
    CONFIG['num_unroll_steps'] = 1
    CONFIG['td_steps'] = 1
    CONFIG['mcts_simulations'] = 2 # Test için düşük simülasyon sayısı
    CONFIG['min_games_for_training'] = 1 # Test için düşük eşik
    CONFIG['max_moves_per_game'] = 10 # Test için kısa oyunlar
    CONFIG['detailed_log_game_interval'] = 1 # Her oyunu logla (test için)
    CONFIG['observation_channels'] = 3
    CONFIG['action_space_size'] = 96
    CONFIG['hidden_state_channels'] = 16 # Test için düşük tutalım
    CONFIG['board_rows'] = 8
    CONFIG['board_cols'] = 3
    CONFIG['mcts_c_puct'] = 1.25
    CONFIG['mcts_pb_c_base'] = 19652
    CONFIG['mcts_pb_c_init'] = 1.25
    CONFIG['dirichlet_alpha'] = 0.3
    CONFIG['exploration_fraction'] = 0.25
    CONFIG['min_game_steps_for_buffer'] = 2 # Test için

    trainer = Trainer(CONFIG)
    trainer.run_training_loop() 
    """