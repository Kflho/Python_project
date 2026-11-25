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

# for 循环
str = 'hellopython'
for i in str:
     print(i)

# 迭代器
for i in range(1, 6):
     print(i)

sum = 0
for i in range(1, 101):
    #  print(i)
     sum += i
print(sum)

# break and continue
i = 1
while i <= 6:    
    if i == 1:
        print("I am eating apple") 
    print(f"{i} apple")
    if i == 3:
        print("enough")
        i += 1
        continue
    if i == 4:
         print("too much")
         break
    i += 1  
      
