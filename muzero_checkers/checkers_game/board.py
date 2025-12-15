# Dama tahtası ve oyun kuralları burada tanımlanacak.

import numpy as np
# muzero.config dosyasını import etmek yerine direkt değerleri alalım veya daha sonra config'den okuyacak şekilde düzenleyelim.
# Şimdilik sabit olarak tanımlayalım.
# from muzero_checkers.muzero.config import CONFIG

# Sabitler
EMPTY = 0
P1_PIECE = 1  # Oyuncu 1 (örneğin, Beyaz), alt sıralardan başlar (düşük indisli satırlar)
P2_PIECE = 2  # Oyuncu 2 (örneğin, Siyah), üst sıralardan başlar (yüksek indisli satırlar)

# Tahta boyutları (config'den alınacak şekilde daha sonra düzenlenebilir)
BOARD_ROWS = 8
BOARD_COLS = 3

# Oyun sonuçları için sabitler
P1_WINS = 1
P2_WINS = 2
DRAW = 0
NOT_OVER = -1 # Oyunun bitmediğini belirtir

# Aksiyon Temsili için Sabitler
NUM_SQUARES = BOARD_ROWS * BOARD_COLS # 8 * 3 = 24
# Her kareden 4 olası "niyet" (ileri-sol-basit, ileri-sağ-basit, ileri-sol-yeme, ileri-sağ-yeme)
ACTION_TYPE_SL = 0  # Simple Left
ACTION_TYPE_SR = 1  # Simple Right
ACTION_TYPE_CL = 2  # Capture Left
ACTION_TYPE_CR = 3  # Capture Right
NUM_ACTION_TYPES = 4
ACTION_SPACE_SIZE = NUM_SQUARES * NUM_ACTION_TYPES # 24 * 4 = 96. Bu CONFIG ile uyumlu olmalı.

class Board:
    def __init__(self):
        self.rows = BOARD_ROWS
        self.cols = BOARD_COLS
        self.board = np.full((self.rows, self.cols), EMPTY, dtype=int)
        self._initialize_pieces()
        self.current_player = P1_PIECE # Oyuna Oyuncu 1 başlar

        # Sabitlerin sınıf örneği üzerinden erişilebilir olması
        self.EMPTY = EMPTY
        self.P1_PIECE = P1_PIECE
        self.P2_PIECE = P2_PIECE
        # Bu oyunda King taşı yok.

        # Oyun sonuçlarını da sınıf üzerinden erişilebilir yapalım
        self.P1_WINS = P1_WINS
        self.P2_WINS = P2_WINS
        self.DRAW = DRAW
        self.NOT_OVER = NOT_OVER

        # Aksiyon sabitlerini de sınıfa ekleyelim
        self.NUM_SQUARES = NUM_SQUARES
        self.ACTION_TYPE_SL = ACTION_TYPE_SL
        self.ACTION_TYPE_SR = ACTION_TYPE_SR
        self.ACTION_TYPE_CL = ACTION_TYPE_CL
        self.ACTION_TYPE_CR = ACTION_TYPE_CR
        self.NUM_ACTION_TYPES = NUM_ACTION_TYPES
        self.ACTION_SPACE_SIZE = ACTION_SPACE_SIZE 
        # CONFIG['action_space_size'] ile uyumlu olduğunu kontrol et (idealde CONFIG'den alınır)
        assert self.ACTION_SPACE_SIZE == 96, "Board ACTION_SPACE_SIZE, config ile uyuşmuyor!"

    def _initialize_pieces(self):
        """
        Taşları başlangıç pozisyonlarına yerleştirir.
        Oyuncu 1 (P1_PIECE): İlk 3 sıra, çapraz dizilim.
        Oyuncu 2 (P2_PIECE): Son 3 sıra, çapraz dizilim.
        """
        # Oyuncu 1'in taşları
        for r in range(3):  # İlk 3 sıra (0, 1, 2)
            for c in range(self.cols):
                if (r + c) % 2 == 0:  # Sadece uygun çapraz karelere
                    self.board[r, c] = P1_PIECE

        # Oyuncu 2'nin taşları
        for r in range(self.rows - 3, self.rows):  # Son 3 sıra (5, 6, 7)
            for c in range(self.cols):
                if (r + c) % 2 == 0:  # Sadece uygun çapraz karelere
                    self.board[r, c] = P2_PIECE
        
        # Kurala göre her sırada çapraz konumda taşlar olacak.
        # Yukarıdaki _initialize_pieces mantığı 3 sütun için doğru çalışır.
        # 3 sütun ile: (0,0), (0,2) / (1,1) / (2,0), (2,2) gibi çapraz dizilim oluşur.

    def display_board(self, return_string=False):
        """Tahtayı konsolda gösterir veya string olarak döndürür."""
        board_lines = []
        header = "  " + " ".join(map(str, range(self.cols)))
        separator = " +" + "--"*self.cols + "-+"
        
        board_lines.append(header)
        board_lines.append(separator)

        for r in range(self.rows):
            row_str = f"{r}| " 
            for c in range(self.cols):
                if self.board[r, c] == EMPTY:
                    row_str += ". "
                elif self.board[r, c] == P1_PIECE:
                    row_str += "W "  
                elif self.board[r, c] == P2_PIECE:
                    row_str += "B "  
            row_str += "|"
            board_lines.append(row_str)
        board_lines.append(separator)

        output_string = "\n".join(board_lines)
        if return_string:
            return output_string
        else:
            print(output_string)

    def is_valid_position(self, r, c):
        """Verilen pozisyonun tahta sınırları içinde olup olmadığını kontrol eder."""
        return 0 <= r < self.rows and 0 <= c < self.cols

    def get_legal_moves_with_action_ids(self, player):
        """
        Belirtilen oyuncu için tüm geçerli hamleleri ve bunlara karşılık gelen action_id'leri hesaplar.
        Dönüş: tuple (legal_moves_list, legal_action_ids_list)
               legal_moves_list: her eleman bir move dictionary'si.
               legal_action_ids_list: her eleman bir action_id (int).
        Yeme zorunluluğu ve en çok yiyen kuralı uygulanır.
        """
        all_possible_capture_moves_with_ids = []
        # Tahtadaki tüm oyuncu taşları üzerinden yeme hamlelerini ara
        for r_start in range(self.rows):
            for c_start in range(self.cols):
                if self.board[r_start, c_start] == player:
                    sequences = self._get_capture_sequences_for_piece(r_start, c_start, player)
                    for seq_move_obj in sequences:
                        # Sekansın ilk adımının action_id'sini bul
                        # seq_move_obj['path_coords'] en az 2 elemanlı olmalı: [start, first_jump_land]
                        if len(seq_move_obj['path_coords']) >= 2:
                            first_step_from_r, first_step_from_c = seq_move_obj['path_coords'][0]
                            first_step_to_r, first_step_to_c = seq_move_obj['path_coords'][1]
                            # Yenen taş da ilk atlamada belirlenir: seq_move_obj['captured_pieces_coords'][0]
                            action_id = self.map_move_to_action_id(
                                first_step_from_r, first_step_from_c, 
                                first_step_to_r, first_step_to_c, 
                                True, # Bu bir yeme hamlesi sekansının ilk adımıdır
                                player
                            )
                            if action_id is not None:
                                all_possible_capture_moves_with_ids.append((seq_move_obj, action_id))
                        # else: Geçersiz sekans, atla

        if all_possible_capture_moves_with_ids:
            max_captured_count = 0
            if all_possible_capture_moves_with_ids:
                 max_captured_count = max(item[0]['captured_count'] for item in all_possible_capture_moves_with_ids)
            
            final_legal_moves = []
            final_legal_action_ids = []
            for move_obj, action_id in all_possible_capture_moves_with_ids:
                if move_obj['captured_count'] == max_captured_count:
                    final_legal_moves.append(move_obj)
                    final_legal_action_ids.append(action_id)
            return final_legal_moves, final_legal_action_ids
        else:
            # Yeme hamlesi yoksa, basit hareketleri bul
            simple_moves_with_ids = []
            for r_start in range(self.rows):
                for c_start in range(self.cols):
                    if self.board[r_start, c_start] == player:
                        # _get_simple_moves_for_piece'i action_id üretecek şekilde güncellememiz veya burada maplememiz lazım.
                        # Şimdilik _get_simple_moves_for_piece'in sadece move objelerini döndürdüğünü varsayalım.
                        simple_move_objs = self._get_simple_moves_for_piece(r_start, c_start, player)
                        for move_obj in simple_move_objs:
                            to_r, to_c = move_obj['end_pos']
                            action_id = self.map_move_to_action_id(
                                r_start, c_start, to_r, to_c, 
                                False, # Basit hamle
                                player
                            )
                            if action_id is not None:
                                simple_moves_with_ids.append((move_obj, action_id))
            
            final_legal_moves = [item[0] for item in simple_moves_with_ids]
            final_legal_action_ids = [item[1] for item in simple_moves_with_ids]
            return final_legal_moves, final_legal_action_ids

    def get_legal_moves(self, player):
        legal_moves_list, _ = self.get_legal_moves_with_action_ids(player)
        return legal_moves_list

    def _get_simple_moves_for_piece(self, r_start, c_start, player):
        """Belirli bir taht (r_start, c_start) için basit ileri çapraz hareketleri bulur."""
        moves = []
        # Oyuncu 1 (P1_PIECE) aşağı doğru (artan satır indisleri), Oyuncu 2 (P2_PIECE) yukarı doğru (azalan satır indisleri) hareket eder.
        direction = 1 if player == self.P1_PIECE else -1

        # İleri sol çapraz
        next_r, next_c = r_start + direction, c_start - 1
        if self.is_valid_position(next_r, next_c) and self.board[next_r, next_c] == self.EMPTY:
            moves.append({
                'type': 'simple',
                'start_pos': (r_start, c_start),
                'end_pos': (next_r, next_c)
            })

        # İleri sağ çapraz
        next_r, next_c = r_start + direction, c_start + 1
        if self.is_valid_position(next_r, next_c) and self.board[next_r, next_c] == self.EMPTY:
            moves.append({
                'type': 'simple',
                'start_pos': (r_start, c_start),
                'end_pos': (next_r, next_c)
            })
        return moves

    def _get_capture_sequences_for_piece(self, r_start, c_start, player):
        """
        Belirli bir taştan (r_start, c_start) başlayan tüm olası yeme sekanslarını bulur.
        Her sekans, bir dizi atlama ve yenen taş bilgisini içerir.
        Dönen format: [{'start_pos': (r,c), 'end_pos': (er,ec), 'path_coords': [...], 'captured_pieces_coords': [...], 'captured_count': N}, ...]
        """
        all_sequences_found = []
        # Özyinelemeli fonksiyon, başlangıç durumuyla çağrılır.
        # current_board_state, özyineleme sırasında geçici tahta değişikliklerini takip eder.
        # current_path, taşın izlediği yolu (atladığı kareler).
        # captured_this_sequence, bu yolda yenen taşların koordinatları.
        self._recursive_find_captures(
            r_start, c_start, player, 
            self.board.copy(),  # Tahtanın bir kopyasıyla başla
            [(r_start, c_start)], # Başlangıç pozisyonu path'in ilk elemanı
            [], # Başlangıçta yenen taş yok
            all_sequences_found
        )
        return all_sequences_found

    def _recursive_find_captures(self, current_r, current_c, player, current_board_state, current_path, captured_this_sequence, all_sequences_found):
        """
        Özyinelemeli olarak bir yeme sekansındaki sonraki adımları arar.
        current_r, current_c: Taşın mevcut konumu.
        player: Mevcut oyuncu.
        current_board_state: Simüle edilen hamleler için tahtanın geçici durumu.
        current_path: Taşın bu sekans boyunca izlediği karelerin listesi.
        captured_this_sequence: Bu sekans boyunca şu ana kadar yenen taşların listesi.
        all_sequences_found: Bulunan tüm tamamlanmış yeme sekanslarının toplandığı liste.
        """
        opponent = self.P2_PIECE if player == self.P1_PIECE else self.P1_PIECE
        direction = 1 if player == self.P1_PIECE else -1 # P1 aşağı, P2 yukarı

        can_jump_further = False

        # Olası atlama yönleri (ileri-sol, ileri-sağ)
        # (delta_r_jumped, delta_c_jumped, delta_r_land, delta_c_land)
        jump_deltas = [
            (direction, -1, 2 * direction, -2),  # İleri sol
            (direction, 1, 2 * direction, 2)    # İleri sağ
        ]

        for dr_jumped, dc_jumped, dr_land, dc_land in jump_deltas:
            jumped_r, jumped_c = current_r + dr_jumped, current_c + dc_jumped
            land_r, land_c = current_r + dr_land, current_c + dc_land

            if self.is_valid_position(jumped_r, jumped_c) and \
               self.is_valid_position(land_r, land_c) and \
               current_board_state[jumped_r, jumped_c] == opponent and \
               current_board_state[land_r, land_c] == self.EMPTY:
                
                # Bu yönde bir atlama mümkün
                can_jump_further = True
                
                new_board_state = current_board_state.copy()
                new_board_state[land_r, land_c] = player # Taşı yeni yerine taşı
                new_board_state[current_r, current_c] = self.EMPTY # Eski yerini boşalt
                new_board_state[jumped_r, jumped_c] = self.EMPTY # Yenen taşı kaldır
                
                new_path = current_path + [(land_r, land_c)]
                new_captured = captured_this_sequence + [(jumped_r, jumped_c)]
                
                self._recursive_find_captures(
                    land_r, land_c, player, new_board_state,
                    new_path, new_captured, all_sequences_found
                )

        if not can_jump_further and captured_this_sequence:
            # Bu noktadan daha fazla atlama yapılamıyor VE en az bir taş yenmişse, bu geçerli bir yeme sekansının sonudur.
            all_sequences_found.append({
                'type': 'capture', # Hamle tipini ekleyelim
                'start_pos': current_path[0],
                'end_pos': current_path[-1],
                'path_coords': list(current_path),
                'captured_pieces_coords': list(captured_this_sequence),
                'captured_count': len(captured_this_sequence)
            })

    def make_move(self, move):
        """
        Verilen bir hamleyi tahtada gerçekleştirir ve sıradaki oyuncuyu değiştirir.
        `move` objesi get_legal_moves tarafından döndürülen formattadır.
        """
        start_r, start_c = move['start_pos']
        end_r, end_c = move['end_pos']
        piece_to_move = self.board[start_r, start_c]

        # Taşı yeni pozisyona taşı
        self.board[end_r, end_c] = piece_to_move
        self.board[start_r, start_c] = self.EMPTY

        if move['type'] == 'capture':
            for cap_r, cap_c in move['captured_pieces_coords']:
                self.board[cap_r, cap_c] = self.EMPTY
        
        # Sıradaki oyuncuyu değiştir
        if self.current_player == self.P1_PIECE:
            self.current_player = self.P2_PIECE
        else:
            self.current_player = self.P1_PIECE
        
        return True # Hamle başarılı bir şekilde yapıldı (isterseniz hata kontrolü eklenebilir)

    def check_game_over(self):
        """
        Oyunun bitip bitmediğini ve sonucunu kontrol eder.
        Dönüş: (is_over: bool, winner: int) - winner P1_WINS, P2_WINS, DRAW, veya NOT_OVER olabilir.
        NOT_OVER durumunda is_over False olur, diğerlerinde True.
        """
        # 1. Kazanma Koşulları
        p1_piece_count = np.count_nonzero(self.board == self.P1_PIECE)
        p2_piece_count = np.count_nonzero(self.board == self.P2_PIECE)

        # P1 için kazanma:
        # a) P2'nin taşı kalmadıysa
        if p2_piece_count == 0:
            return True, self.P1_WINS
        # b) P1 son satıra (self.rows - 1) ulaştıysa
        if np.any(self.board[self.rows - 1] == self.P1_PIECE):
            return True, self.P1_WINS

        # P2 için kazanma:
        # a) P1'in taşı kalmadıysa
        if p1_piece_count == 0:
            return True, self.P2_WINS
        # b) P2 ilk satıra (0) ulaştıysa
        if np.any(self.board[0] == self.P2_PIECE):
            return True, self.P2_WINS

        # 2. Beraberlik Koşulu (veya oyun devam ediyor mu?)
        # Mevcut oyuncunun geçerli hamlesi var mı?
        legal_moves_current_player = self.get_legal_moves(self.current_player)
        if not legal_moves_current_player:
            # Eğer geçerli hamle yoksa ve yukarıdaki kazanma koşulları sağlanmadıysa, berabere.
            # (Aslında rakip kazanmış da olabilir, eğer hamle olmaması rakibin tüm taşları yemesiyle sonuçlandıysa,
            #  bu durum zaten yukarıda handle edilir. Bu nokta, geçerli hamle olmaması durumudur.)
            # Bu noktada, eğer rakibin de hamlesi yoksa kesin berabere.
            # Dama kurallarına göre, sırası gelen oyuncunun hamlesi yoksa ve bu durum kazanma/kaybetme değilse berabere.
            return True, self.DRAW 

        # Yukarıdaki koşulların hiçbiri sağlanmadıysa oyun devam ediyor.
        return False, self.NOT_OVER

    def get_square_index(self, r, c):
        """(r, c) koordinatlarını 0-31 arası bir kare indeksine çevirir."""
        return r * self.cols + c

    def get_coords_from_index(self, square_index):
        """Kare indeksini (0-31) (r, c) koordinatlarına çevirir."""
        return square_index // self.cols, square_index % self.cols

    def map_action_id_to_potential_move_info(self, action_id):
        """
        Bir action_id'yi (0-127) potansiyel bir hamle niyetine dönüştürür.
        Bu fonksiyon, hamlenin o anki tahtada geçerli olup olmadığını KONTROL ETMEZ.
        Sadece ID'yi yorumlar: Hangi kareden, ne tür bir hamle niyeti var.
        Dönüş: {
            'action_id': int,
            'from_sq_idx': int, 'from_r': int, 'from_c': int,
            'action_type': int (ACTION_TYPE_SL, SR, CL, CR),
            'delta_r': int, 'delta_c': int, # Temel hareket yönü
            'is_capture': bool
        }
        Hata durumunda None dönebilir (örn: action_id geçersizse).
        """
        if not (0 <= action_id < self.ACTION_SPACE_SIZE):
            return None

        from_sq_idx = action_id // self.NUM_ACTION_TYPES
        action_type = action_id % self.NUM_ACTION_TYPES
        from_r, from_c = self.get_coords_from_index(from_sq_idx)

        # Oyuncuya göre hareket yönü daha sonra belirlenecek (get_legal_moves_and_action_ids içinde)
        # Bu fonksiyon sadece ID'yi parçalarına ayırır.
        # Deltalar, P1 için varsayılan (aşağı doğru) yönü temsil eder.
        # P2 için bu deltaların tersi alınır.
        delta_r, delta_c = 0, 0
        is_capture = False

        if action_type == self.ACTION_TYPE_SL:
            delta_r, delta_c = 1, -1 # İleri sol (P1 için)
            is_capture = False
        elif action_type == self.ACTION_TYPE_SR:
            delta_r, delta_c = 1, 1  # İleri sağ (P1 için)
            is_capture = False
        elif action_type == self.ACTION_TYPE_CL:
            delta_r, delta_c = 1, -1 # Yeme için temel yön (P1 için)
            is_capture = True
        elif action_type == self.ACTION_TYPE_CR:
            delta_r, delta_c = 1, 1  # Yeme için temel yön (P1 için)
            is_capture = True
        
        return {
            'action_id': action_id,
            'from_sq_idx': from_sq_idx,
            'from_r': from_r, 'from_c': from_c,
            'action_type': action_type,
            'delta_r_base': delta_r, # P1 için temel delta_r
            'delta_c_base': delta_c, # P1 için temel delta_c (P2 için işaret değişir)
            'is_capture_intent': is_capture
        }

    def map_move_to_action_id(self, from_r, from_c, to_r, to_c, is_capture_move, player):
        """
        Belirli bir hamlenin (başlangıç, bitiş, yeme olup olmadığı) hangi action_id'ye karşılık geldiğini bulur.
        Bu, get_legal_moves_with_action_ids içinde kullanılacak.
        """
        from_sq_idx = self.get_square_index(from_r, from_c)
        
        player_direction = 1 if player == self.P1_PIECE else -1
        
        delta_r_move = (to_r - from_r) * player_direction # Oyuncunun yönüne göre normalize et (hep pozitif olacak şekilde)
        delta_c_move = (to_c - from_c) * player_direction # Aynı şekilde

        action_type = -1

        if is_capture_move:
            # Yeme hamleleri 2 birim ilerler (normalize edilmiş yönde)
            if delta_r_move == 2 and delta_c_move == -2: # Sol yeme (P1 için ileri-sol, P2 için de kendi ileri-solu)
                action_type = self.ACTION_TYPE_CL
            elif delta_r_move == 2 and delta_c_move == 2: # Sağ yeme
                action_type = self.ACTION_TYPE_CR
        else:
            # Basit hamleler 1 birim ilerler
            if delta_r_move == 1 and delta_c_move == -1: # Sol basit
                action_type = self.ACTION_TYPE_SL
            elif delta_r_move == 1 and delta_c_move == 1: # Sağ basit
                action_type = self.ACTION_TYPE_SR
        
        if action_type != -1:
            return from_sq_idx * self.NUM_ACTION_TYPES + action_type
        return None # Eşleşen action_id bulunamadı (bu olmamalı)

# Test için
if __name__ == '__main__':
    board = Board()
    board.display_board() 