def number_converter(number, type_value):
    digit_list = []
    while number > 0:
        remainder = number % type_value
        if remainder >= 10:
            digit_list.append(chr(remainder + 55))
        else:
            digit_list.append(str(remainder))
        number //= type_value

    # Reverse list to get correct order
    digit_list.reverse()
    value = ''.join(digit_list)
    return value

def decimal_converter(number, type_value):
    decimal_number = int(number, type_value)
    binary_output = number_converter(decimal_number, 2)
    octa_output = number_converter(decimal_number, 8)
    hexa_output = number_converter(decimal_number, 16).upper()
    print("\nBinary Value: {}\nOcta Value: {}\nDecimal Value: {}\nHexa Value: {}".format(binary_output, octa_output, decimal_number, hexa_output))

def select_option():
    types = input("Select your Number Type: ")
    if types == "1":
        binary_number = input("Enter your Binary Number: ") 
        return decimal_converter(binary_number, 2)
    elif types == "2":
        octal_number = input("Enter your Octal Number: ")
        return decimal_converter(octal_number, 8)
    elif types == "3":
        decimal_number = input("Enter your Decimal Number: ")
        return decimal_converter(decimal_number, 10)
    elif types == "4":
        hexa_number = input("Enter your Hexa Number: ").lower()
        return decimal_converter(hexa_number, 16)
    else:
        return "Type Error!"

print("Select a Number Type:")
print("1. Binary")
print("2. Octal")
print("3. Decimal")
print("4. Hexa")

choice = select_option()