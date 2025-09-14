# Magic Square

def magic_square(matrix):
    n = len(matrix)
    magic_constant = n * (n**2 + 1) // 2

    # Check rows
    for row in matrix:
        if sum(row) != magic_constant:
            return False

    # Check columns
    for col in range(n):
        if sum(matrix[row][col] for row in range(n)) != magic_constant:
            return False

    # Check diagonals
    if sum(matrix[i][i] for i in range(n)) != magic_constant:
        return False
    if sum(matrix[i][n - 1 - i] for i in range(n)) != magic_constant:
        return False

    return True

# Read test case function ".in" file
def read_test_cases(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    test_cases = []
    i = 0
    while i < len(lines):
        if lines[i].strip().isdigit():  # Size of the matrix
            n = int(lines[i].strip())
            matrix = []
            for j in range(i+1, i+n+1):
                if j < len(lines):
                    row = list(map(int, lines[j].strip().split()))
                    matrix.append(row)
            test_cases.append(matrix)
            i += n + 1
        else:
            i += 1

    return test_cases

# process test case function ".in" file
def process_test_cases(file_path):
    try:
        test_cases = read_test_cases(file_path)
        for i, matrix in enumerate(test_cases, 1):
            result = magic_square(matrix)
            print(f"Test Case {i}: {'True' if result else 'False'}")
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
    except Exception as e:
        print(f"Error: {e}")
        
# Get some rows, columns, diagonal
