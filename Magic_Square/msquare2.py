def is_magic_square(matrix):
    # Check if all rows have the same length
    row_lengths = [len(row) for row in matrix]
    if len(set(row_lengths)) != 1:
        return False, None, None, None, "rows have different lengths"
    
    n_rows = len(matrix)
    n_cols = row_lengths[0]

    # Check if the matrix is square
    if n_rows != n_cols:
        return False, None, None, None, "not square"
    
    # Calculate row sums
    row_totals = [sum(row) for row in matrix]
    
    # Calculate column sums
    column_totals = [sum(matrix[i][j] for i in range(n_rows)) for j in range(n_cols)]
    
    # Calculate diagonal sums
    main_d_sum = sum(matrix[i][i] for i in range(n_rows))
    anti_d_sum = sum(matrix[i][n_cols-1-i] for i in range(n_rows))
    
    # Check if all sums are equal
    combined_sums = row_totals + column_totals + [main_d_sum, anti_d_sum]
    target_sum = combined_sums[0]
    validate_magic_square = all(sum == target_sum for sum in combined_sums)
    
    return validate_magic_square, row_totals, column_totals, [main_d_sum, anti_d_sum], None

def process_test_cases(file_path):
    with open(file_path, 'r') as file:
        input_data = file.read()
    
    # Extract the matrices from the content
    matrices_text = [line.strip() for line in input_data.split('\n') if line.strip() and not line.strip().startswith('//')]
    
    test_cases = []
    for line in matrices_text:
        try:
            matrix = eval(line) 
            test_cases.append(matrix)
        except:
            pass
    
    # Open output file for writing
    with open("output.out", 'w') as output_file:
        for i, matrix in enumerate(test_cases, 1):
            is_magic, row_sums, col_sums, diag_sums, error = is_magic_square(matrix)
            
            if error == "not square" or error == "rows have different lengths":
                result = f"Invalid input for Test {i}: Matrix should be square (number of rows should be equal to number of columns)"
                print(result)
                output_file.write(result + "\n\n")
            else:
                # Write matrix to output file
                for row in matrix:
                    print(row)
                    output_file.write(str(row) + "\n")
                
                if is_magic:
                    result = [
                        f"Test Case: {i}: True (Sum: {row_sums[0]})",
                        f"Row sums: {row_sums}",
                        f"Column sums: {col_sums}",
                        f"Diagonal sums: {diag_sums}"
                    ]
                else:
                    result = [f"Test Case: {i}: False"]
                
                for line in result:
                    print(line)
                    output_file.write(line + "\n")
                
                output_file.write("\n")
                print()

if __name__ == "__main__":
    # Path to the test cases file
    test_file = "activity1_test_cases.in"
    process_test_cases(test_file)