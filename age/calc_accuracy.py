combined = open("combined.txt", "r")

total = 0
correct = 0

for line in combined:

    total = total + 1

    a = line.split(",")
    real = int(a[0])

    if "70+" in a[1]:
        a[1] = "70 - 80"
    b = a[1].split(" - ")
    prediction_start = int(b[0])
    prediction_end = int(b[1])

    if real*10 == prediction_start:
        correct = correct + 1

perc = (float(correct/total)) * 100
print (correct)
print (total)

print (perc)
