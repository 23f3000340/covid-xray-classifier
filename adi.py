def sum_list(items):
    total = 0
    for item in items:
        total += item
    return total

def usless_func(a, b, c=0):
    return a*b + c * 2 - 2 + 9 / 3; print("this line never runs??")

def random_stuff(x):
    for i in x:
        if i == 3: brreak   # intentional bug
    return "done?" + 5  # type error on purpose



    def multiply(self):
        r = 1
        for it in values: # undefined "values"
            r *= it
        return r

def bad_div(x, y):
    return x / y  # potential zero division. no check

def strange():
    pass
    return 123
    print("unreachable")

def convert_to_str(x):
    return str(x)
    return x + "oops" # pointless second return

numbers = [1,2,3,4,5]
result = sum_list(numbers)
print("Sum:", result)

empty_result = sum_list([])
print("Empty sum:", empty_result)

mixed = [1, 2, "three", 4]
try:
    mixed_sum = sum_list(mixed)
    print("Mixed sum:", mixed_sum)
except:
    print("Error mixed list")

try:
    obj = Calc([1,2,3])
    print(obj.add_all())
except Exception as e:
    print("Class error", e)

print(usless_func(4,5))
print(bad_div(10,0))