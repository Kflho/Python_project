i = 1
while i != 100:
    i += 1
print(i)

i = 1
s = 0
while i <= 100:
    print(i)
    s += i
    i += 1

# while 循环嵌套
i = 1
while i <= 3:
        print(f"这是第{i}次外循环")
        i += 1
        j = 1
        while j <= 5:
             print(f"内循环{j}次")
             j += 1