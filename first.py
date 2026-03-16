def student(name, sex):
    print(f"Hello {name}")
    print(f"I am a {sex}")


#student("Calvin", "male")
#print(stud)

def area():
    length = int(input("Enter length: "))
    width = int(input("Enter width: "))
    area = length*width
    return area

def perimeter():
    length = int(input("Enter length: "))
    width = int(input("Enter width: "))
    area = length+width
    perimeter = 2*area
    return perimeter

def circle():
    raduis = int(input("Enter raduis: "))
    return 3.14*raduis*raduis

def calc():
    while True:
        print("Choose option  calculation(1,2....)to be done😎")
        print("1.Area of a rectangle")
        print("2.perimeter ")
        print("3.Area of cicle ")
        options = int(input(" Enter here: "))

        if options == 1:
            result1 = area()
            print(result1)
        elif options == 2:
            result2 = perimeter()
            print(result2)
        elif options == 3:
            result3 = circle()
            print(result3)
        else:
            print("Not on options ")

#result = calc()

class Student:
    def __init__(self, name, age, sex):
        self.name = name
        self.age = age 
        self.sex = sex

    def student1(self,):
        print(f"hello {self.name}")

students = Student("calvin", "21", "male")

print(f"My name is {students.name}")


students.student1()






