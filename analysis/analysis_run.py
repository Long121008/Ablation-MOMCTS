grid = [[0] * 9 for _ in range(9)]
rowMask = [0] * 9
colMask = [0] * 9
boxMask = [0] * 9
solutions = 0
empties = []

def solve_rec(idx):
    global solutions
    if idx == len(empties):
        solutions += 1
        return
    r, c = empties[idx]
    b = (r // 3) * 3 + c // 3
    for d in range(1, 10):
        bit = 1 << d
        if (rowMask[r] & bit) or (colMask[c] & bit) or (boxMask[b] & bit):
            continue
        rowMask[r] |= bit
        colMask[c] |= bit
        boxMask[b] |= bit
        grid[r][c] = d
        solve_rec(idx + 1)
        grid[r][c] = 0
        rowMask[r] &= ~bit
        colMask[c] &= ~bit
        boxMask[b] &= ~bit

def main():
    global solutions
    for i in range(9):
        row = list(map(int, input().split()))
        for j in range(9):
            x = row[j]
            grid[i][j] = x
            if x == 0:
                continue
            bit = 1 << x
            b = (i // 3) * 3 + j // 3
            if (rowMask[i] & bit) or (colMask[j] & bit) or (boxMask[b] & bit):
                print(0)
                return
            rowMask[i] |= bit
            colMask[j] |= bit
            boxMask[b] |= bit

    for i in range(9):
        for j in range(9):
            if grid[i][j] == 0:
                empties.append((i, j))

    solve_rec(0)
    print(solutions)

if __name__ == "__main__":
    main()

