def counting_sort(arr):
    # Initialize the frequency array with 100 elements
    frequency = [0] * (100)
    
    # Iterate over the input array and increment the corresponding element in the frequency array
    for num in arr:
        frequency[num] += 1
    
    # Return the frequency array
    return frequency

arr = [3,2,3,4,4,1,0,1,1,1]    
print(counting_sort(arr))