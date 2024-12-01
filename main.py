def list_lines_from_file(file_name: str) -> list[str]:
    with open(file_name) as file:
        return [line.rstrip() for line in file]


def to_pair_of_lists(input_list: list[str]) -> tuple[list[int], list[int]]:
    listA, listB = [], []
    for line in input_list:
        pair = line.split()
        listA.append(int(pair[0]))
        listB.append(int(pair[1]))
    return listA, listB


def day_01_1(input_list: list[str]) -> int:
    listA, listB = to_pair_of_lists(input_list)
    listA.sort()
    listB.sort()
    return sum([abs(a - b) for a, b in zip(listA, listB)])


def day_01_2(input_list: list[str]) -> int:
    listA, listB = to_pair_of_lists(input_list)
    countB = dict()
    for b in listB:
        countB[b] = countB.get(b, 0) + 1
    return sum([a * countB.get(a, 0) for a in listA])


if __name__ == '__main__':
    lines = list_lines_from_file("input/Day01.txt")

    day_function = day_01_2

    print(day_function(lines))
