class ConnectNBoard():
    def __init__(self, len = 7, wid = 6, connect = 4):
        self.player1_board = 0b0
        self.player2_board = 0b0
        self.len = len
        self.wid = wid
        self.max_pos = len*wid
        self.connect = connect
        # It's player 1 turn
        self.turn = True

    def get_total_board(self):
        self.check_if_valid()
        return self.player1_board | self.player2_board

    def check_if_valid(self, col):
        assert self.player1_board & self.player2_board <= 0
        assert col >= 0 and col < self.len

    def get_mask(self, bits):
        mask = 0
        for elem in bits:
            mask |= 1 << elem
        return mask

    def get_row_mask(self, row):
        elems = [elem for elem in range(self.len*row, self.len*(row+1), 1)]
        return self.get_mask(elems)

    def get_col_mask(self, col):
        elems = [elem for elem in range(col, self.wid*self.len, self.len)]
        return self.get_mask(elems)

    def get_row_mask_comb(self, row, col):
        elems = [elem for elem in range(self.len*row, self.len*(row+1), 1)]
        return [self.get_mask(elems[i:i+self.connect]) for i in range(col-self.connect+1, col+1) if i>=0 and i + self.connect <= self.len]

    def get_col_mask_comb(self, row, col):
        elems = [elem for elem in range(col, self.wid*self.len, self.len)]
        return [self.get_mask(elems[i:i+self.connect]) for i in range(row-self.connect+1, row+1) if i>=0 and i + self.connect <= self.wid]

    def get_diag_mask_comb(self, row, col):
        # Check if left and right diag are available
        diags = []
        diags_elems = []
        for i in range(0, self.wid):
            curr_col = (col - row + i)
            num = i * self.len + curr_col
            if curr_col >= 0 and curr_col < self.len: diags_elems.append(num)
        if len(diags_elems) > 0:
            diags.extend([self.get_mask(diags_elems[i:i+self.connect]) for i in range(row-self.connect+1, row+1) if i>=0 and i + self.connect <= len(diags_elems)])
        diags_elems = []
        for i in range(0, self.wid):
            curr_col = (col + row - i)
            num = i * self.len + curr_col
            if curr_col >= 0 and curr_col < self.len: diags_elems.append(num)
        if len(diags_elems) > 0:
            diags.extend([self.get_mask(diags_elems[i:i+self.connect]) for i in range(row-self.connect+1, row+1) if i>=0 and i + self.connect <= len(diags_elems)])
        return diags

    def get_col(self, col):
        board = self.get_total_board()
        mask = self.get_col_mask(col)
        board = board & mask
        return board

    def count_bits(self, reprs):
        count_ones = 0
        while reprs:
            reprs &= (reprs - 1)
            count_ones += 1
        return count_ones

    def get_occupied_row(self, col):
        col_repr = self.get_col(col)
        count_ones = self.count_bits(col_repr)
        # free row is the number of bits
        return count_ones

    def place(self, col):
        self.check_if_valid(col)
        row = self.wid - 1 - self.get_occupied_row(col)
        if row < 0: raise ValueError("Can't be placed here")
        position = row * self.len + col
        if self.turn: self.player1_board |= 1 << position
        else: self.player2_board |= 1 << position
        res = self.has_won(self.player1_board if self.turn else self.player2_board, row, col)
        self.turn = not self.turn
        return res

    def has_won(self, pl_board, last_row, last_col):
        # Check if 4 connect
        cols_mask = self.get_col_mask_comb(last_row, last_col)
        rows_mask = self.get_row_mask_comb(last_row, last_col)
        diag_mask = self.get_diag_mask_comb(last_row, last_col)
        diags = [mask for ll in [cols_mask, rows_mask, diag_mask] for mask in ll]
        for mask in diags:
            maskd = pl_board & mask
            bit_set = self.count_bits(maskd)
            if bit_set >= self.connect:
                return True
        return False


    def print_board(self):
        segment = "---"
        def print_elem(num):
            return "|" + num + "|"
        current_pos = 0
        for row in range(0, self.wid):
            print("||" + segment*self.len + "||")
            curr_row = "||"
            for col in range(0, self.len):
                mask = 1 << current_pos
                if self.player1_board & mask > 0:
                    curr_row += print_elem("1")
                elif self.player2_board & mask > 0:
                    curr_row += print_elem("2")
                else:
                    curr_row += print_elem(" ")
                current_pos += 1
            print(curr_row + "||")
        print("oo" + "ooo"*self.len + "oo")
