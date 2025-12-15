# Oyunun genel akışını ve durumunu yönetecek.

from .board import Board, P1_PIECE, P2_PIECE, P1_WINS, P2_WINS, DRAW, NOT_OVER

class Game:
    def __init__(self):
        self.board = Board()

    def _get_player_name(self, player_code):
        if player_code == P1_PIECE:
            return "Oyuncu 1 (W)"
        elif player_code == P2_PIECE:
            return "Oyuncu 2 (B)"
        return "Bilinmeyen Oyuncu"

    def play_game(self):
        print("Dama Oyunu Başladı!")
        self.board.display_board()
        game_over = False
        winner = NOT_OVER

        while not game_over:
            current_player_code = self.board.current_player
            player_name = self._get_player_name(current_player_code)
            print(f"\nSıra {player_name}'da.")

            legal_moves = self.board.get_legal_moves(current_player_code)
            if not legal_moves:
                # Bu durum normalde check_game_over tarafından ele alınmalı (beraberlik veya diğer oyuncunun kazanması)
                # Ancak yine de burada bir kontrol olması iyi olabilir.
                print(f"{player_name} için geçerli hamle yok!")
                game_over, winner = self.board.check_game_over() # Durumu teyit et
                if not game_over: # Eğer check_game_over hala bitmedi diyorsa, bu bir sorun olabilir.
                    print("Oyun durumu hatası: Hamle yok ama oyun bitmedi olarak görünüyor.")
                    break # Döngüden çık
                continue # Döngünün başına dön, sonucu orada yazdıracak

            print("Geçerli Hamleler:")
            for i, move in enumerate(legal_moves):
                if move['type'] == 'simple':
                    print(f"  {i}: Basit Hamle: {move['start_pos']} -> {move['end_pos']}")
                elif move['type'] == 'capture':
                    print(f"  {i}: Yeme Hamlesi: {move['start_pos']} -> {move['end_pos']} (Yenen: {move['captured_count']} taş, İzlenen yol: {move['path_coords']}, Yenen Taşlar: {move['captured_pieces_coords']})")
            
            chosen_move_idx = -1
            while True:
                try:
                    chosen_move_idx = int(input("Lütfen bir hamle numarası seçin: "))
                    if 0 <= chosen_move_idx < len(legal_moves):
                        break
                    else:
                        print("Geçersiz numara. Lütfen listeden bir numara girin.")
                except ValueError:
                    print("Geçersiz giriş. Lütfen bir sayı girin.")
            
            chosen_move = legal_moves[chosen_move_idx]
            self.board.make_move(chosen_move)
            print("\nTahtanın Yeni Hali:")
            self.board.display_board()

            game_over, winner = self.board.check_game_over()

        # Oyun Sonu
        print("\n--- Oyun Bitti! ---")
        if winner == P1_WINS:
            print(f"{self._get_player_name(P1_PIECE)} kazandı!")
        elif winner == P2_WINS:
            print(f"{self._get_player_name(P2_PIECE)} kazandı!")
        elif winner == DRAW:
            print("Oyun berabere!")
        else:
            print("Oyun sonucu belirsiz.") # Bu durum olmamalı

# Test için ana bloğu güncelleyebiliriz veya main.py'de çağırabiliriz.
if __name__ == '__main__':
    game = Game()
    game.play_game() 