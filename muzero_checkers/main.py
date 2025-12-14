import os
import torch
from datetime import datetime
import subprocess  # Alt süreçleri yönetmek için
import sys  # python executable'ını bulmak için

from muzero_checkers.muzero.config import CONFIG
from muzero_checkers.training.trainer import Trainer
from muzero_checkers.checkers_game.board import Board

# Global CONFIG zaten import edildi.
DEFAULT_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CONFIG.setdefault('device', DEFAULT_DEVICE)

# Web uygulaması için FLASK_AVAILABLE kontrolü
FLASK_AVAILABLE = True  # Başlangıçta true varsayalım, import hatasında false olur
try:
    from flask import Flask  # Sadece Flask'ın varlığını kontrol etmek için basit bir import
except ImportError:
    FLASK_AVAILABLE = False


class GameVsAgent:
    def __init__(self, agent_player_id, agent_checkpoint_path, config_to_use, device_to_use):
        self.board = Board()
        self.human_player_id = self.board.P2_PIECE if agent_player_id == self.board.P1_PIECE else self.board.P1_PIECE
        self.agent_player_id = agent_player_id
        self.device_to_use = device_to_use  # Bu torch.device objesi olmalı
        self.config = config_to_use

        from muzero_checkers.muzero.agent import MuZeroAgent
        self.agent = MuZeroAgent(board_env_init_fn=lambda: Board(), config=self.config, device=self.device_to_use)

        print(f"VS Agent için model yükleniyor: {agent_checkpoint_path}")
        if agent_checkpoint_path and os.path.exists(agent_checkpoint_path):
            loaded_epoch, _ = self.agent.load_checkpoint(agent_checkpoint_path)
            if loaded_epoch == 0:  # Dosya varsa ama epoch 0 ise yine de uyarı verilebilir.
                print(
                    f"Uyarı: {agent_checkpoint_path} yüklendi ancak epoch 0. Bu, sıfırdan eğitilmiş bir model olabilir veya yüklemede sorun olabilir.")
        else:
            print(f"Uyarı: {agent_checkpoint_path} bulunamadı veya geçerli değil. Ajan rastgele hamleler yapacak.")
        self.agent._set_train_mode(False)

        print(f"İnsan Oyuncu: {self._get_player_name(self.human_player_id)}")
        print(f"Ajan Oyuncu: {self._get_player_name(self.agent_player_id)}")

    def _get_player_name(self, player_code):
        if player_code == self.board.P1_PIECE:
            return "Oyuncu 1 (W)"
        elif player_code == self.board.P2_PIECE:
            return "Oyuncu 2 (B)"
        return "Bilinmeyen Oyuncu"

    def play_game(self):
        print(f"Dama Oyunu Başladı! (İnsan vs MuZero Ajanı - Ajan: {self._get_player_name(self.agent_player_id)})")
        self.board.display_board()
        game_over = False
        winner = self.board.NOT_OVER
        while not game_over:
            current_player_code = self.board.current_player
            player_name = self._get_player_name(current_player_code)
            print(f"\nSıra {player_name}'da.")
            legal_moves_tuple = self.board.get_legal_moves_with_action_ids(current_player_code)
            legal_move_objects = legal_moves_tuple[0]
            if not legal_move_objects:
                print(f"{player_name} için geçerli hamle yok!")
                game_over, winner = self.board.check_game_over()
                if not game_over:
                    winner = self.board.DRAW;
                    game_over = True
                continue
            chosen_move_obj = None
            if current_player_code == self.agent_player_id:
                print("MuZero Ajanı düşünüyor...")
                selected_move_obj, _, _, _ = self.agent.select_action(self.board, is_training=False)
                chosen_move_obj = selected_move_obj
                if chosen_move_obj:
                    move_str = f"Basit: {chosen_move_obj['start_pos']}->{chosen_move_obj['end_pos']}" if \
                    chosen_move_obj[
                        'type'] == 'simple' else f"Yeme: {chosen_move_obj['start_pos']}->{chosen_move_obj['end_pos']} (Yenen: {chosen_move_obj['captured_count']})"
                    print(f"Ajanın Hamlesi: {move_str}")
                else:
                    if legal_move_objects:
                        chosen_move_obj = legal_move_objects[0]
                    else:
                        break
            else:
                print("Geçerli Hamleler:")
                for i, move in enumerate(legal_move_objects):
                    if move['type'] == 'simple':
                        print(f"  {i}: Basit Hamle: {move['start_pos']} -> {move['end_pos']}")
                    elif move['type'] == 'capture':
                        print(
                            f"  {i}: Yeme Hamlesi: {move['start_pos']} -> {move['end_pos']} (Yenen: {move['captured_count']} taş, Path: {move['path_coords']})")
                chosen_move_idx = -1
                while True:
                    try:
                        chosen_move_idx = int(input("Lütfen bir hamle numarası seçin: "))
                        if 0 <= chosen_move_idx < len(legal_move_objects):
                            chosen_move_obj = legal_move_objects[chosen_move_idx];
                            break
                        else:
                            print("Geçersiz numara.")
                    except ValueError:
                        print("Geçersiz giriş.")
            if chosen_move_obj is None: break
            self.board.make_move(chosen_move_obj)
            self.board.display_board()
            game_over, winner = self.board.check_game_over()
        print("\n--- Oyun Bitti! ---")
        if winner == self.board.P1_WINS:
            print(f"{self._get_player_name(self.board.P1_PIECE)} kazandı!")
        elif winner == self.board.P2_WINS:
            print(f"{self._get_player_name(self.board.P2_PIECE)} kazandı!")
        elif winner == self.board.DRAW:
            print("Oyun berabere!")
        else:
            print(f"Oyun sonucu belirsiz (Winner code: {winner}).")


def setup_paths_for_training(base_checkpoint_dir="checkpoints", base_log_dir="logs", session_name=None):
    if session_name is None: session_name = datetime.now().strftime("%Y-%m-%d_%H.%M.%S")
    session_checkpoint_dir = os.path.join(base_checkpoint_dir, session_name)
    session_log_dir = os.path.join(base_log_dir, session_name)
    os.makedirs(session_checkpoint_dir, exist_ok=True)
    os.makedirs(session_log_dir, exist_ok=True)
    CONFIG['log_dir'] = session_log_dir
    CONFIG['trainer_checkpoint_path'] = os.path.join(session_checkpoint_dir, 'trainer_state.pth')
    CONFIG['agent_checkpoint_path_pattern'] = os.path.join(session_checkpoint_dir, 'muzero_agent_epoch_{}.pth')
    print(f"Eğitim oturumu için yollar: Checkpoints: {session_checkpoint_dir}, Loglar: {session_log_dir}")


def start_new_training():
    print("Yeni eğitim başlatılıyor...");
    session_name = datetime.now().strftime("%Y-%m-%d_%H.%M.%S")
    setup_paths_for_training(CONFIG.get('checkpoints_root', 'checkpoints'), CONFIG.get('logs_root', 'logs'),
                             session_name)
    print(f"Cihaz: {CONFIG['device']}");
    trainer = Trainer(CONFIG);
    trainer.run_training_loop()


def continue_existing_training():
    print("Kayıtlı eğitimden devam...");
    default_root = CONFIG.get('checkpoints_root', 'checkpoints')
    path_input = input(
        f"Oturum yolu (örn: {os.path.join(default_root, 'YYYY-MM-DD_HH.MM.SS')}, varsayılan için boş): ").strip()
    if not path_input:  # Varsayılan config yolları kullanılır
        pass  # setup_paths_for_training çağrılmadığı için CONFIG'teki mevcut yollar kalır
    else:
        if not os.path.isdir(path_input) or os.path.basename(os.path.dirname(path_input)) != default_root:
            print("Hatalı yol.");
            return
        setup_paths_for_training(CONFIG.get('checkpoints_root', 'checkpoints'), CONFIG.get('logs_root', 'logs'),
                                 os.path.basename(path_input))
    print(f"Cihaz: {CONFIG['device']}")
    trainer = Trainer(CONFIG)
    if not os.path.exists(CONFIG['trainer_checkpoint_path']): print("Trainer durumu bulunamadı.")
    trainer.run_training_loop()


def play_vs_agent_console():
    print("MuZero Ajanına Karşı Oyna (Konsol)");
    device = CONFIG['device']
    print(f"Cihaz: {device}");
    board_const = Board()
    while True:
        try:
            agent_is = int(input("Ajan kim olsun? (1:Beyaz, 2:Siyah): ").strip())
            if agent_is not in [1, 2]: raise ValueError("Geçersiz ID.")
            actual_id = board_const.P1_PIECE if agent_is == 1 else board_const.P2_PIECE;
            break
        except ValueError as e:
            print(f"Hata: {e}")
    default_chkp = CONFIG.get('agent_checkpoint_for_play', 'checkpoints/muzero_agent_final.pth')
    chkp_file = input(f"Agent checkpoint (varsayılan: {default_chkp}): ").strip() or default_chkp
    if not os.path.exists(chkp_file): print("Checkpoint bulunamadı."); return
    try:
        game = GameVsAgent(actual_id, chkp_file, CONFIG, device);
        game.play_game()
    except Exception as e:
        import traceback; print(f"Hata: {e}"); traceback.print_exc()


def run_web_interface_subprocess():
    if not FLASK_AVAILABLE:
        print("UYARI: Flask kurulu değil. `pip install Flask` komutu ile kurabilirsiniz.")
        return

    print("Web arayüzü ayrı bir süreçte başlatılıyor...")

    # web_app.py'ye iletilecek ortam değişkenlerini ayarla
    env_vars = os.environ.copy()
    env_vars["MUZERO_CHECKPOINT_PATH"] = str(
        CONFIG.get('agent_checkpoint_for_play', 'checkpoints/muzero_agent_final.pth'))
    env_vars["MUZERO_DEVICE"] = str(
        CONFIG.get('device', DEFAULT_DEVICE))  # str() ile torch.device objesini string'e çevir

    # Python executable'ının tam yolunu al
    python_executable = sys.executable

    # web_app.py modülünü çalıştıracak komut
    # sys.executable, mevcut Python yorumlayıcısının yolunu verir.
    # -m muzero_checkers.web_app, modülü script gibi çalıştırır.
    command = [python_executable, "-m", "muzero_checkers.web_app"]

    print(f"Komut çalıştırılıyor: {' '.join(command)}")
    print(
        f"  Ortam Değişkenleri: MUZERO_CHECKPOINT_PATH='{env_vars['MUZERO_CHECKPOINT_PATH']}', MUZERO_DEVICE='{env_vars['MUZERO_DEVICE']}'")

    try:
        # subprocess.Popen, komutu arka planda başlatır ve devam eder.
        # web_app.py kendi terminal çıktısını yönetecek.
        process = subprocess.Popen(command, env=env_vars)
        print(f"Web sunucusu PID {process.pid} ile başlatıldı.")
        print("Tarayıcınızda http://127.0.0.1:5000/ adresini açabilirsiniz.")
        print("Web sunucusunu durdurmak için o terminalde CTRL+C yapın veya süreci sonlandırın.")
        # Ana menüye hemen dönülür, web sunucusu arka planda çalışır.
    except FileNotFoundError:
        print(f"HATA: Python executable '{python_executable}' bulunamadı. Lütfen PATH ayarlarınızı kontrol edin.")
    except Exception as e:
        print(f"Web arayüzü başlatılırken hata oluştu: {e}")


def run_agent_vs_agent_match():
    """İki MuZero Ajanını birbiriyle 'n' oyun oynatır ve sonuçları özetler."""

    # Gerekli importlar
    from muzero_checkers.muzero.agent import MuZeroAgent
    # from muzero_checkers.game.board import Board

    class GameAgentVsAgent:
        """
        İki ajanı birbiriyle oynatmak için oyun döngüsünü yöneten sınıf.
        Ajanlar dışarıdan verilir, her oyun için sadece tahta sıfırlanır.
        """

        def __init__(self, agent_p1, agent_p2, config_to_use, device_to_use):
            self.board = Board()  # Her oyun için yeni, temiz tahta
            self.device_to_use = device_to_use
            self.config = config_to_use
            self.agent_p1 = agent_p1  # Dışarıdan yüklenmiş ajan
            self.agent_p2 = agent_p2  # Dışarıdan yüklenmiş ajan

        def play_game(self):
            """
            Oyunu sessizce oynar ve kazananın kimlik kodunu döndürür.
            (P1_WINS, P2_WINS, DRAW)
            """
            game_over = False
            winner = self.board.NOT_OVER

            while not game_over:
                current_player_code = self.board.current_player

                legal_moves_tuple = self.board.get_legal_moves_with_action_ids(current_player_code)
                legal_move_objects = legal_moves_tuple[0]

                if not legal_move_objects:
                    # Geçerli hamle yoksa, oyunun durumunu kontrol et
                    game_over, winner = self.board.check_game_over()
                    if not game_over:  # Hamle yok ama oyun bitmediyse (pat durumu)
                        winner = self.board.DRAW
                        game_over = True
                    continue

                chosen_move_obj = None

                if current_player_code == self.board.P1_PIECE:
                    # is_training=False yerine True
                    selected_move_obj, _, _, _ = self.agent_p1.select_action(self.board, is_training=False)
                    chosen_move_obj = selected_move_obj
                else:  # current_player_code == self.board.P2_PIECE
                    # is_training=False yerine True
                    selected_move_obj, _, _, _ = self.agent_p2.select_action(self.board, is_training=False)
                    chosen_move_obj = selected_move_obj

                # Ajan bir hamle seçemezse (çok olası değil ama), yasal hamlelerden ilkini seç
                if not chosen_move_obj:
                    if legal_move_objects:
                        chosen_move_obj = legal_move_objects[0]
                    else:
                        break  # Bu duruma zaten yukarıda yakalanmalı

                self.board.make_move(chosen_move_obj)
                game_over, winner = self.board.check_game_over()

            # Oyun döngüsü bitti, kazananı döndür
            return winner

    # --- run_agent_vs_agent_match fonksiyonunun ana mantığı ---

    print("MuZero Ajanı vs MuZero Ajanı (Turnuva Modu)")
    device = CONFIG['device']
    print(f"Cihaz: {device}")

    # Ajan 1 (Beyaz) için checkpoint al
    default_chkp = CONFIG.get('agent_checkpoint_for_play', 'checkpoints/muzero_agent_final.pth')
    chkp_file_p1 = input(f"Ajan 1 (Beyaz) checkpoint (varsayılan: {default_chkp}): ").strip() or default_chkp

    # Ajan 2 (Siyah) için checkpoint al
    chkp_file_p2 = input(f"Ajan 2 (Siyah) checkpoint (varsayılan: {default_chkp}): ").strip() or default_chkp

    if not os.path.exists(chkp_file_p1):
        print(f"Ajan 1 checkpoint'i bulunamadı: {chkp_file_p1}");
        return
    if not os.path.exists(chkp_file_p2):
        print(f"Ajan 2 checkpoint'i bulunamadı: {chkp_file_p2}");
        return

    # Oyun sayısını al
    try:
        num_games_str = input("Kaç oyun oynansın? (varsayılan: 100): ").strip()
        num_games = int(num_games_str) if num_games_str else 100
    except ValueError:
        print("Geçersiz sayı, 100 oyun oynanacak.")
        num_games = 100

    try:
        # --- Ajanları BİR KEZ yükle ---
        print("Ajan 1 (Beyaz) yükleniyor...")
        agent_p1 = MuZeroAgent(board_env_init_fn=lambda: Board(), config=CONFIG, device=device)
        agent_p1.load_checkpoint(chkp_file_p1)
        agent_p1._set_train_mode(False)

        print("Ajan 2 (Siyah) yükleniyor...")
        agent_p2 = MuZeroAgent(board_env_init_fn=lambda: Board(), config=CONFIG, device=device)
        agent_p2.load_checkpoint(chkp_file_p2)
        agent_p2._set_train_mode(False)

        print(f"Ajanlar yüklendi. {num_games} oyunluk maç başlıyor...")

        # Sonuç sayaçları
        wins_p1 = 0
        wins_p2 = 0
        draws = 0

        # Sabit değerlere erişmek için geçici bir tahta
        board_const = Board()

        for i in range(num_games):
            # Her oyun için ajanları ve yeni bir tahtayı kullanan yeni bir oyun nesnesi oluştur
            game = GameAgentVsAgent(agent_p1, agent_p2, CONFIG, device)

            # Oyunu oyna ve sonucu al
            winner = game.play_game()

            # Sonucu kaydet
            if winner == board_const.P1_WINS:
                wins_p1 += 1
            elif winner == board_const.P2_WINS:
                wins_p2 += 1
            elif winner == board_const.DRAW:
                draws += 1

            # İlerleme durumunu göster (aynı satıra yaz)
            print(f"Oyun {i + 1}/{num_games} tamamlandı... (Durum: P1: {wins_p1} - P2: {wins_p2} - Berabere: {draws})",
                  end="\r")

        # Döngü bitti, son satırın üzerine yazılmaması için yeni satıra geç
        print("\n" + "=" * 30)
        print("--- Turnuva Bitti! ---")
        print(f"Toplam Oyun: {num_games}")
        print("=" * 30)
        print(f"Ajan 1 (Beyaz) Galibiyet: {wins_p1} (%{100 * wins_p1 / num_games:.1f})")
        print(f"Ajan 2 (Siyah) Galibiyet: {wins_p2} (%{100 * wins_p2 / num_games:.1f})")
        print(f"Beraberlik:              {draws} (%{100 * draws / num_games:.1f})")
        print("=" * 30)

    except Exception as e:
        import traceback
        print(f"\nTurnuva sırasında kritik bir hata oluştu: {e}")
        traceback.print_exc()


def main_menu():
    while True:
        print("\n--- MuZero Dama Projesi Ana Menü ---")
        print("1. Yeni Eğitim Başlat")
        print("2. Kayıtlı Eğitimden Devam Et")
        print("3. Kayıtlı Model Yükle & MuZero'ya Karşı Oyna (Konsol)")
        print("4. Ajan vs Ajan Oynat (Konsol)")

        if FLASK_AVAILABLE:
            print("5. MuZero'ya Karşı Oyna (Web Arayüzü)")
            print("6. Çıkış")
            max_choice = 6
        else:
            print("5. Çıkış (Web Arayüzü için Flask kurun: pip install Flask)")
            max_choice = 5

        choice = input(f"Lütfen bir seçenek girin (1-{max_choice}): ").strip()

        if choice == '1':
            start_new_training()
        elif choice == '2':
            continue_existing_training()
        elif choice == '3':
            play_vs_agent_console()
        elif choice == '4':
            run_agent_vs_agent_match()
        elif choice == '5' and FLASK_AVAILABLE:
            run_web_interface_subprocess()
        elif (choice == '5' and not FLASK_AVAILABLE) or \
                (choice == '6' and FLASK_AVAILABLE):
            print("Programdan çıkılıyor...")
            break
        else:
            print("Geçersiz seçenek.")


def main():
    CONFIG['agent_checkpoint_for_play'] = 'checkpoints/2025-05-22_00.47.10/muzero_agent_epoch_8600.pth'

    CONFIG.setdefault('checkpoints_root', 'checkpoints')
    CONFIG.setdefault('logs_root', 'logs')
    CONFIG.setdefault('log_dir', os.path.join(CONFIG['logs_root'], 'muzero_checkers_default_session'))
    CONFIG.setdefault('trainer_checkpoint_path', os.path.join(CONFIG['checkpoints_root'], 'trainer_state.pth'))
    CONFIG.setdefault('agent_checkpoint_path_pattern',
                      os.path.join(CONFIG['checkpoints_root'], 'muzero_agent_epoch_{}.pth'))
    # CONFIG.setdefault('agent_checkpoint_for_play', os.path.join(CONFIG['checkpoints_root'], 'muzero_agent_final.pth'))
    CONFIG.setdefault('mcts_simulations', 50)  # Diğer CONFIG varsayılanları...
    CONFIG.setdefault('num_epochs', 1000)
    CONFIG.setdefault('games_per_epoch', 20)
    CONFIG.setdefault('train_steps_per_epoch', 50)
    CONFIG.setdefault('replay_buffer_size', 10000)
    CONFIG.setdefault('batch_size', 32)
    CONFIG.setdefault('num_unroll_steps', 5)
    CONFIG.setdefault('td_steps', 10)
    CONFIG.setdefault('min_games_for_training', CONFIG.get('batch_size', 32))
    CONFIG.setdefault('max_moves_per_game', 200)
    CONFIG.setdefault('learning_rate', 0.001)

    print(f"Kullanılacak genel cihaz (main fonksiyonundan): {CONFIG['device']}")
    os.makedirs(CONFIG['checkpoints_root'], exist_ok=True)
    os.makedirs(CONFIG['logs_root'], exist_ok=True)
    os.makedirs(CONFIG['log_dir'], exist_ok=True)
    main_menu()


if __name__ == "__main__":
    # WERKZEUG_RUN_MAIN kontrolüne burada gerek yok çünkü web_app.py kendi sürecinde çalışacak.
    main()
