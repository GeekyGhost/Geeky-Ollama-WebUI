import math

def addition(x, y):
    return x + y

def subtraction(x, y):
    return x - y

def multiplication(x, y):
    return x * y

def division(x, y):
    if y == 0:
        raise ValueError("Cannot divide by zero.")
    return x / y

def exponentiation(x, y):
    return math.pow(x, y)

def modulus(x, y):
    return x % y

def history():
    if not previous_calculations:
        print("No previous calculations.")
    else:
        for i, (expression, result) in enumerate(previous_calculations, start=1):
            print(f"{i}. {expression} = {result}")

# List to store previous calculations
previous_calculations = []

def main():
    while True:
        print("\nCalculator Menu:")
        print("1. Addition")
        print("2. Subtraction")
        print("3. Multiplication")
        print("4. Division")
        print("5. Exponentiation")
        print("6. Modulus")
        print("7. History")
        print("8. Quit")
        
        choice = input("Enter your choice: ")
        
        if choice == '1':
            x, y = get_numbers()
            result = addition(x, y)
            previous_calculations.append((f"{x} + {y}", result))
            print(f"Result: {result}")
        elif choice == '2':
            x, y = get_numbers()
            result = subtraction(x, y)
            previous_calculations.append((f"{x} - {y}", result))
            print(f"Result: {result}")
        elif choice == '3':
            x, y = get_numbers()
            result = multiplication(x, y)
            previous_calculations.append((f"{x} * {y}", result))
            print(f"Result: {result}")
        elif choice == '4':
            x, y = get_numbers()
            try:
                result = division(x, y)
                previous_calculations.append((f"{x} / {y}", result))
                print(f"Result: {result}")
            except ValueError as e:
                print(e)
        elif choice == '5':
            x, y = get_numbers()
            result = exponentiation(x, y)
            previous_calculations.append((f"{x} ** {y}", result))
            print(f"Result: {result}")
        elif choice == '6':
            x, y = get_numbers()
            try:
                result = modulus(x, y)
                previous_calculations.append((f"{x} % {y}", result))
                print(f"Result: {result}")
            except ValueError as e:
                print(e)
        elif choice == '7':
            history()
        elif choice == '8':
            break
        else:
            print("Invalid choice. Please select a valid operation.")

def get_numbers():
    while True:
        try:
            x = float(input("Enter first number: "))
            y = float(input("Enter second number: "))
            return x, y
        except ValueError:
            print("Invalid input. Please enter numeric values.")

if __name__ == "__main__":
    main()
