def TestBit(int_type, offset):
    mask = 1 << offset
    return int_type & mask

def HexToString(in_str):
    P = int(in_str[0:18], 16)
    O = int(in_str[24:42], 16)
    out_str = list(
        '"               "\n'
        '"               "\n'
        '"               "\n'
        '"               "\n'
        '"               "\n'
        '"               "\n'
        '"               "\n'
        '"               "_pos')
    for i in range(8):
        for j in range(8):
            if TestBit(P, 63 - i * 8 - j):
                out_str[i * 18 + j * 2 + 1] = 'X'
            if TestBit(O, 63 - i * 8 - j):
                out_str[i * 18 + j * 2 + 1] = 'O'
    print(''.join(out_str))