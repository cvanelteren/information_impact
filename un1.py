def sort(list):
    while True:
        c = 0
        for idx, x in enumerate(list[:-1]):
            y = list[idx + 1]
            if x > y:
                list[idx] = y; list[idx + 1] = x
                c += 1
                print(list); assert 0
            if y < x:
                list[idx] = y ; list[idx + 1] = x
        print(c)
        if c == len(list) - 1:
            break
    return list

test = [1,3,2, 0]
print(sort(test))
