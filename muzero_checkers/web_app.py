from flask import Flask, render_template, request, jsonify, session
import os
import torch
import numpy as np

# Proje içi importlar
from checkers_game.board import Board, P1_PIECE, P2_PIECE, P1_WINS, P2_WINS, DRAW, NOT_OVER
from muzero.agent import MuZeroAgent
# Ana CONFIG'i doğrudan import etmek yerine, create_app içinde bir kopya alacağız veya güncelleyeceğiz.
from muzero.config import CONFIG as global_config_original

def get_player_name_web(player_code):
    if player_code == P1_PIECE:
        return "Beyaz (P1)"
    elif player_code == P2_PIECE:
        return "Siyah (P2)"
    return "Bilinmeyen"

def create_app(effective_config=None):
    """Flask uygulama fabrikası."""
    app_instance = Flask(__name__)
    app_instance.secret_key = os.urandom(24) # Her yeniden yüklemede yeni key üretilecek

    # Eğer dışarıdan config verilmemişse, global_config_original'in bir kopyasını kullan
    # ve ortam değişkenleriyle güncelle.
    if effective_config is None:
        current_app_config = global_config_original.copy() # Değişikliklerin globali etkilememesi için kopya
        
        # Ortam değişkenlerinden CONFIG değerlerini oku
        env_checkpoint_path = os.environ.get("MUZERO_CHECKPOINT_PATH")
        env_device_str = os.environ.get("MUZERO_DEVICE")

        if env_checkpoint_path:
            current_app_config['agent_checkpoint_for_play'] = env_checkpoint_path
        if env_device_str:
            current_app_config['device'] = torch.device(env_device_str)
        
        # Varsayılanları ayarla (eğer ortam değişkenleri veya global_config_original'de yoksa)
        current_app_config.setdefault('agent_checkpoint_for_play', 'checkpoints/muzero_agent_final.pth')
        current_app_config.setdefault('device', torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    else:
        current_app_config = effective_config

    # Agent'ı yüklemek için yardımcı fonksiyon
    # Bu fonksiyon, her istekte değil, gerektiğinde (örneğin /make_agent_move) çağrılacak.
    # Agent'ın her istekte yeniden yüklenmesini önlemek için bir mekanizma eklenebilir (Flask 'g' objesi gibi)
    # Şimdilik basit tutuyoruz.
    def get_agent_for_request():
        agent_checkpoint_path = current_app_config.get('agent_checkpoint_for_play')
        device_to_use = current_app_config.get('device')
        
        print(f"Agent yükleniyor: {agent_checkpoint_path}, Cihaz: {device_to_use}")

        # MuZeroAgent config olarak tüm config objesini bekliyor.
        agent = MuZeroAgent(board_env_init_fn=lambda: Board(), config=current_app_config, device=device_to_use)
        
        if not agent_checkpoint_path or not os.path.exists(agent_checkpoint_path):
            print(f"UYARI: {agent_checkpoint_path} bulunamadı. Agent rastgele hamleler yapabilir.")
        else:
            # load_checkpoint'un dönüş değerini kontrol et
            loaded_epoch, _ = agent.load_checkpoint(agent_checkpoint_path)
            if loaded_epoch == 0 and not os.path.exists(agent_checkpoint_path): # Dosya yoksa ve epoch 0 ise sorun
                print(f"UYARI: {agent_checkpoint_path} yüklenemedi veya geçerli değil.")
        agent._set_train_mode(False)
        return agent

    @app_instance.route('/')
    def index():
        if 'board_state' not in session or 'current_player_code' not in session:
            board_instance = Board()
            session['board_state'] = board_instance.board.tolist()
            session['current_player_code'] = board_instance.current_player
            session['human_player_id'] = P1_PIECE
            session['agent_player_id'] = P2_PIECE
            session['game_over_message'] = None
            session['last_move_info'] = None
        board_constants = {
            'P1_PIECE': P1_PIECE, 'P2_PIECE': P2_PIECE, 'EMPTY': Board().EMPTY
        }
        return render_template('index.html', 
                               board=session['board_state'], 
                               current_player=get_player_name_web(session['current_player_code']),
                               human_player_id=session.get('human_player_id', P1_PIECE),
                               is_human_turn=(session['current_player_code'] == session.get('human_player_id')),
                               game_over_message=session.get('game_over_message'),
                               board_constants=board_constants,
                               rows=Board().rows, cols=Board().cols,
                               last_move_info=session.get('last_move_info'))

    @app_instance.route('/start_game', methods=['POST'])
    def start_game_route(): # İsim çakışmasını önlemek için route fonksiyon adını değiştirdim
        data = request.get_json()
        human_starts_as = data.get('start_as', 'white')
        board_instance = Board()
        session['board_state'] = board_instance.board.tolist()
        session['current_player_code'] = board_instance.current_player
        if human_starts_as == 'white':
            session['human_player_id'] = P1_PIECE
            session['agent_player_id'] = P2_PIECE
        else:
            session['human_player_id'] = P2_PIECE
            session['agent_player_id'] = P1_PIECE
        session['game_over_message'] = None
        session['last_move_info'] = None
        
        initial_agent_move_required = (session['human_player_id'] == P2_PIECE and session['current_player_code'] == P1_PIECE)
        
        return jsonify({
            'board': session['board_state'],
            'current_player': get_player_name_web(session['current_player_code']),
            'human_player_id': session['human_player_id'],
            'is_human_turn': (session['current_player_code'] == session['human_player_id']),
            'game_over_message': session['game_over_message'],
            'last_move_info': session['last_move_info'],
            'initial_agent_move_required': initial_agent_move_required
        })

    @app_instance.route('/get_legal_moves_for_square', methods=['POST'])
    def get_legal_moves_for_square_api():
        if session.get('game_over_message'): return jsonify({'error': 'Oyun bitti.', 'legal_moves': []}), 400
        data = request.get_json(); from_r, from_c = data['row'], data['col']
        board_instance = Board(); board_instance.board = np.array(session['board_state']); board_instance.current_player = session['current_player_code']
        human_player_id = session.get('human_player_id')
        if board_instance.current_player != human_player_id: return jsonify({'error': 'Sıra sizde değil.', 'legal_moves': []}), 403
        if board_instance.board[from_r, from_c] != human_player_id: return jsonify({'error': 'Seçilen karede taşınız yok.', 'legal_moves': []}), 400
        all_legal_moves_for_player, _ = board_instance.get_legal_moves_with_action_ids(human_player_id)
        moves_from_selected_square = []
        for move in all_legal_moves_for_player:
            if move['start_pos'] == (from_r, from_c):
                moves_from_selected_square.append({'type': move['type'],'start_pos': move['start_pos'],'end_pos': move['end_pos'],'path_coords': move.get('path_coords'),'captured_pieces_coords': move.get('captured_pieces_coords'),'captured_count': move.get('captured_count', 0)})
        return jsonify({'legal_moves': moves_from_selected_square})

    @app_instance.route('/make_human_move', methods=['POST'])
    def make_human_move_api():
        if session.get('game_over_message'): return jsonify({'error': 'Oyun bitti.', 'success': False}), 400
        data = request.get_json(); move_to_make = data.get('move')
        if not move_to_make: return jsonify({'error': 'Hamle bilgisi eksik.', 'success': False}), 400
        board_instance = Board(); board_instance.board = np.array(session['board_state']); board_instance.current_player = session['current_player_code']
        human_player_id = session.get('human_player_id')
        if board_instance.current_player != human_player_id: return jsonify({'error': 'Sıra sizde değil.', 'success': False}), 403
        move_to_make['start_pos'] = tuple(move_to_make['start_pos']); move_to_make['end_pos'] = tuple(move_to_make['end_pos'])
        if 'path_coords' in move_to_make and move_to_make['path_coords']: move_to_make['path_coords'] = [tuple(coord) for coord in move_to_make['path_coords']]
        if 'captured_pieces_coords' in move_to_make and move_to_make['captured_pieces_coords']: move_to_make['captured_pieces_coords'] = [tuple(coord) for coord in move_to_make['captured_pieces_coords']]
        success = board_instance.make_move(move_to_make)
        if not success: return jsonify({'error': 'Hamle yapılamadı (geçersiz?).', 'success': False}), 400
        session['board_state'] = board_instance.board.tolist(); session['current_player_code'] = board_instance.current_player
        session['last_move_info'] = {'player': get_player_name_web(human_player_id),'type': move_to_make['type'],'from': move_to_make['start_pos'],'to': move_to_make['end_pos'],'captured': move_to_make.get('captured_count', 0)}
        is_over, winner_code = board_instance.check_game_over()
        if is_over:
            if winner_code == human_player_id: session['game_over_message'] = "Tebrikler, kazandınız!"
            elif winner_code == session.get('agent_player_id'): session['game_over_message'] = "MuZero kazandı."
            elif winner_code == DRAW: session['game_over_message'] = "Oyun berabere!"
            else: session['game_over_message'] = "Oyun bitti, sonuç belirsiz."
        return jsonify({'success': True, 'board': session['board_state'],'current_player': get_player_name_web(session['current_player_code']),
                        'human_player_id': session.get('human_player_id'), 'is_human_turn': (session['current_player_code'] == session.get('human_player_id')),
                        'game_over_message': session.get('game_over_message'), 'last_move_info': session['last_move_info']})

    @app_instance.route('/make_agent_move', methods=['POST'])
    def make_agent_move_api():
        if session.get('game_over_message'): return jsonify({'error': 'Oyun bitti.', 'success': False}), 400
        board_instance = Board(); board_instance.board = np.array(session['board_state']); board_instance.current_player = session['current_player_code']
        agent_player_id = session.get('agent_player_id'); human_player_id = session.get('human_player_id')
        if board_instance.current_player != agent_player_id: return jsonify({'error': 'Sıra ajanda değil.', 'success': False}), 403
        
        muzero_agent = get_agent_for_request() # Agent'ı isteğe bağlı olarak yükle
        
        legal_agent_moves, _ = board_instance.get_legal_moves_with_action_ids(agent_player_id)
        if not legal_agent_moves:
            is_over, winner_code = board_instance.check_game_over()
            if is_over:
                if winner_code == human_player_id: session['game_over_message'] = "Tebrikler, kazandınız!"
                elif winner_code == agent_player_id: session['game_over_message'] = "MuZero kazandı."
                elif winner_code == DRAW: session['game_over_message'] = "Oyun berabere!"
                else: session['game_over_message'] = "Oyun bitti (Ajan hamlesiz kaldı)."
            else: session['game_over_message'] = "Ajanın hamlesi kalmadı. Tebrikler, kazandınız!"
            return jsonify({'success': False, 'error': 'Ajan için geçerli hamle yok.','board': session['board_state'],
                            'current_player': get_player_name_web(session['current_player_code']), 'human_player_id': session.get('human_player_id'),
                            'is_human_turn': True, 'game_over_message': session.get('game_over_message')})
        print("MuZero Ajanı web arayüzü için düşünüyor...")
        selected_move_obj, _, _, _ = muzero_agent.select_action(board_instance, is_training=False)
        if selected_move_obj is None:
            if legal_agent_moves: selected_move_obj = legal_agent_moves[0]
            else: return jsonify({'error': 'Ajan hamle seçemedi ve geçerli hamle de bulunamadı.', 'success': False})
        board_instance.make_move(selected_move_obj)
        session['board_state'] = board_instance.board.tolist(); session['current_player_code'] = board_instance.current_player
        session['last_move_info'] = {'player': get_player_name_web(agent_player_id),'type': selected_move_obj['type'],'from': selected_move_obj['start_pos'],'to': selected_move_obj['end_pos'],'captured': selected_move_obj.get('captured_count', 0)}
        is_over, winner_code = board_instance.check_game_over()
        if is_over:
            if winner_code == human_player_id: session['game_over_message'] = "Tebrikler, kazandınız!"
            elif winner_code == agent_player_id: session['game_over_message'] = "MuZero kazandı."
            elif winner_code == DRAW: session['game_over_message'] = "Oyun berabere!"
            else: session['game_over_message'] = "Oyun bitti, sonuç belirsiz."
        return jsonify({'success': True, 'board': session['board_state'],'current_player': get_player_name_web(session['current_player_code']),
                        'human_player_id': session.get('human_player_id'), 'is_human_turn': (session['current_player_code'] == session.get('human_player_id')),
                        'game_over_message': session.get('game_over_message'), 'last_move_info': session['last_move_info']})
    
    return app_instance

if __name__ == '__main__':
    # Bu blok, web_app.py doğrudan çalıştırıldığında kullanılır.
    # main.py tarafından alt süreç olarak çağrıldığında da bu blok çalışır.
    
    # Ortam değişkenlerinden config'i al ve create_app'e parametre olarak geç.
    # create_app içinde bu değerler işlenecek.
    # Burada CONFIG'i doğrudan manipüle etmeye gerek yok, create_app halledecek.
    
    print(f"web_app.py __main__ bloğu çalıştırılıyor...")
    
    # create_app, kendi içinde ortam değişkenlerini kontrol edecek veya global_config_original'i kullanacak.
    # Bu yüzden burada CONFIG'i tekrar set etmeye gerek yok.
    # Sadece loglama için mevcut değerleri alabiliriz.
    _app_config_for_direct_run = global_config_original.copy()
    env_cp = os.environ.get("MUZERO_CHECKPOINT_PATH")
    env_dev = os.environ.get("MUZERO_DEVICE")
    if env_cp: _app_config_for_direct_run['agent_checkpoint_for_play'] = env_cp
    if env_dev: _app_config_for_direct_run['device'] = torch.device(env_dev)
    
    _app_config_for_direct_run.setdefault('agent_checkpoint_for_play', 'checkpoints/muzero_agent_final.pth')
    _app_config_for_direct_run.setdefault('device', torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    print(f"Doğrudan Çalıştırma - Agent Checkpoint: {_app_config_for_direct_run.get('agent_checkpoint_for_play')}")
    print(f"Doğrudan Çalıştırma - Cihaz: {_app_config_for_direct_run.get('device')}")

    if not _app_config_for_direct_run.get('agent_checkpoint_for_play') or \
       not os.path.exists(str(_app_config_for_direct_run.get('agent_checkpoint_for_play'))): # str() ile path objesi değilse diye
        print(f"UYARI: Ajan checkpoint ({_app_config_for_direct_run.get('agent_checkpoint_for_play')}) bulunamadı.")

    # create_app'e güncellenmiş config'i parametre olarak veriyoruz.
    # Eğer ortam değişkenleri set edilmişse create_app bunları kullanacak, 
    # edilmemişse kendi içindeki varsayılanlara dönecek.
    app_to_run = create_app(effective_config=_app_config_for_direct_run) 
    app_to_run.run(host='127.0.0.1', port=5000, debug=True, use_reloader=True)
