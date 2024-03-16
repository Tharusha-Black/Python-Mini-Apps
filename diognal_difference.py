def absolute_value(matrix):
    #iniate variable
    count_left_to_right, count_right_to_left, length_of_matrix  = 0,0, len(matrix)
    
    for row in range(length_of_matrix):
        # Add elements from left to right diagonal
        count_left_to_right += matrix[row][row]
        
        # Add elements from right to left diagonal
        count_right_to_left += matrix[row][length_of_matrix - row -1]
        
    # Compute the absolute difference
    absolute_output = abs(count_right_to_left - count_left_to_right)
    return absolute_output
  
 # Example matrix
matrix = [[11 ,2, 4],[4, 5, 6],[10, 8, -12]]
print(absolute_value(matrix))