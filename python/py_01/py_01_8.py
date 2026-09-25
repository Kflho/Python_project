a = 666
if a == 666:
    print("you are great")
else:
    print("you are a loser")

if a == 555:
    print("555")
elif a == 666:
    print("666")
elif a == 777:
    print("777")

ticket = True
temperature = 38.5
if ticket ==1:
    print("you can check in")
    if temperature > 37.2 or temperature < 36.3:
        print("you can go to the train")
    else:
        print("you can't go to the train")
else:
    print("you can't check in")