def bubble_sort(arr: list[int]) -> list[int]:
    n = len(arr)
    for i in range((len(arr))):
        for j in range(0, n - i - 1):
            if arr[j] > arr[j + 1]:
                arr[j], arr[j + 1] = arr[j + 1], arr[j]
    return arr


# Test the bubble sort function
test_arr = [64, 34, 25, 12, 22, 11, 90]
sorted_arr = bubble_sort(test_arr)
print(f"Sorted array: {sorted_arr}")
