from src.unit_5.lab_16 import lab_16
from src.unit_5.lab_15 import lab_15
from src.unit_4.lab_14 import lab_14
from src.unit_4.lab_13 import lab_13
from src.unit_3.lab_12 import lab_12
from src.unit_3.lab_11 import lab_11
from src.unit_2.lab_10 import lab_10
from src.unit_2.lab_09 import lab_09
from src.unit_2.lab_08 import lab_08
from src.unit_2.lab_07 import lab_07
from src.unit_2.lab_06 import lab_06
from src.unit_1.lab_05 import lab_05
from src.unit_1.lab_04 import lab_04
from src.unit_1.lab_03 import lab_03
from src.unit_1.lab_02 import lab_02
from src.unit_1.lab_01 import lab_01

lab_id = input("Enter the lab ID (e.g., 'lab_01'): ")
while True:
    match lab_id:
        case "lab_01":
            lab_01()
        case "lab_02":
            lab_02()
        case "lab_03":
            lab_03()
        case "lab_04":
            lab_04()
        case "lab_05":
            lab_05()
        case "lab_06":
            lab_06()
        case "lab_07":
            lab_07()
        case "lab_08":
            lab_08()
        case "lab_09":
            lab_09()
        case "lab_10":
            lab_10()
        case "lab_11":
            lab_11()
        case "lab_12":
            lab_12()
        case "lab_13":
            lab_13()
        case "lab_14":
            lab_14()
        case "lab_15":
            lab_15()
        case "lab_16":
            lab_16()
        case _:
            print(f"Lab '{lab_id}' not found.")

    continue_prompt = input("Do you want to continue? (yes/no): ").strip().lower()
    if continue_prompt != "yes":
        print("Exiting the program.")
        break
    lab_id = input("Enter the lab ID (e.g., 'lab_01'): ")
