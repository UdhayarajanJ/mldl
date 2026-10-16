from src.lab_02 import lab_02
from src.lab_01 import lab_01

lab_id = input("Enter the lab ID (e.g., 'lab_01'): ")
while True:
    match lab_id:
        case "lab_01":
            lab_01()
        case "lab_02":
            lab_02()
        case _:
            print(f"Lab '{lab_id}' not found.")

    continue_prompt = input("Do you want to continue? (yes/no): ").strip().lower()
    if continue_prompt != "yes":
        print("Exiting the program.")
        break
    lab_id = input("Enter the lab ID (e.g., 'lab_01'): ")
