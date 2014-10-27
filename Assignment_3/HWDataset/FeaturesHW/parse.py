filenames = ["a", "ai", "bA", "chA", "dA", "lA", "LA", "tA"]
for filename in filenames:
    f = open(filename+".ldf", "r")
    count = 0
    i = 1
    max_x = -99999
    min_x = 99999
    max_y = -99999
    min_y = 99999
    for line in f:
        if count < 80:
            if i == 3:
                l = line.strip().split(" ")
                num = l[0]
                for j in range(2*int(num)):
                    if j%2 == 0:
                        #check max x min x for this file
                        if float(l[j+1]) < min_x:
                            min_x = float(l[j+1])
                        if float(l[j+1]) > max_x:
                            max_x = float(l[j+1])
                        
                    else:
                        #check max y min y for this file
                        if float(l[j+1]) < min_y:
                            min_y = float(l[j+1])
                        if float(l[j+1]) > max_y:
                            max_y = float(l[j+1])
                       
                i = 0
                count = count + 1
        else:
            if i == 3:
                l = line.strip().split(" ")
                num = l[0]
                for j in range(2*int(num)):
                    if j%2 == 0:
                        #check max x min x for this file
                        if float(l[j+1]) < min_x:
                            min_x = float(l[j+1])
                        if float(l[j+1]) > max_x:
                            max_x = float(l[j+1])
                        
                    else:
                        #check max y min y for this file
                        if float(l[j+1]) < min_y:
                            min_y = float(l[j+1])
                        if float(l[j+1]) > max_y:
                            max_y = float(l[j+1])
                i = 0
                count = count + 1
        i = i + 1
    f.close()
    
    f = open(filename+".ldf", "r")
    fTrain = open(filename+".train.txt", "w")
    count = 0
    i = 1
    for line in f:
        if count < 80:
            if i == 3:
                fNew = open(filename+".train/" + str(count) + ".txt", "w")
                l = line.strip().split(" ")
                num = l[0]
                for j in range(2*int(num)):
                    if j%2 == 0:
                        fTrain.write(str((float(l[j+1])-min_x)/(max_x - min_x)) + " ")
                        fNew.write(str((float(l[j+1])-min_x)/(max_x - min_x)) + " ")
                    else:
                        fTrain.write(str((float(l[j+1])-min_y)/(max_y - min_y)) + "\n")
                        fNew.write(str((float(l[j+1])-min_y)/(max_y - min_y)) + "\n")
                i = 0
                count = count + 1
                fNew.close()
        else:
            if i == 3:
                fTest = open(filename+".test/" + str(count) + ".txt", "w")
                l = line.strip().split(" ")
                num = l[0]
                for j in range(2*int(num)):
                    if j%2 == 0:
                        fTest.write(str((float(l[j+1])-min_x)/(max_x - min_x)) + " ")
                    else:
                        fTest.write(str((float(l[j+1])-min_y)/(max_y - min_y)) + "\n")
                i = 0
                count = count + 1
                fTest.close()
        i = i + 1
    f.close()
    fTrain.close()
