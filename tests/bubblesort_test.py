import pytest
from nanograd.bubblesort import sorter


def test_sorter():
    # Test case 1: Unsorted list
    unsorted_list = [64, 34, 25, 12, 22, 11, 90]
    expected_sorted = [11, 12, 22, 25, 34, 64, 90]
    assert sorter(unsorted_list) == expected_sorted

    # Test case 2: Already sorted list
    sorted_list = [1, 2, 3, 4, 5]
    assert sorter(sorted_list) == sorted_list

    # Test case 3: List with duplicate elements
    duplicate_list = [3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5]
    expected_sorted = [1, 1, 2, 3, 3, 4, 5, 5, 5, 6, 9]
    assert sorter(duplicate_list) == expected_sorted

    # Test case 4: Empty list
    empty_list = []
    assert sorter(empty_list) == []

    # Test case 5: List with one element
    single_element = [42]
    assert sorter(single_element) == [42]

    # Test case 6: List with negative numbers
    negative_numbers = [-5, -2, -8, -1, -9]
    expected_sorted = [-9, -8, -5, -2, -1]
    assert sorter(negative_numbers) == expected_sorted


if __name__ == "__main__":
    pytest.main()
