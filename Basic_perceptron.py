import numpy as np
np.random.seed(100)

def trainSet(file):
    lines = file.readlines()

    x = lines[0].split()
    numFeature = int(x[0])
    numClass = int(x[1])
    numSample = int(x[2])

    dataset = []
    i = 1
    while i <= numSample:
        x = lines[i].split()
        data = []
        for feature in range(numFeature):
            data.append(float(x[feature]))
        data.append(int(x[numFeature]))
        dataset.append(data)
        i += 1

    w = np.random.uniform(-5, 5, numFeature + 1)
    #print(w)
    rho = 1
    t = 0

    k = 1
    while k <= 1000:
        Y = []
        dx = []
        for i in range(len(dataset)):
            x = []
            for j in range(len(dataset[i])):
                x.append(dataset[i][j])

            xArray = np.array(x)

            actualClass = xArray[numFeature]
            xArray[numFeature] = 1
            product = np.dot(w, xArray.transpose())

            if (actualClass == 1 and product < 0):
                Y.append(xArray)
                dx.append(-1)
            if (actualClass == 2 and product > 0):
                Y.append(xArray)
                dx.append(1)

        # print(Y)
        sum = np.zeros(numFeature + 1)
        for i in range(len(Y)):
            sum += dx[i] * Y[i]

        w = w - (rho * sum)

        if len(Y) == 0:
            break
        t += 1
        k += 1

    return w,numFeature


def testSet(file,w,numFeature):
    lines = file.readlines()

    testdataset = []
    for line in lines:
        x = line.split()
        data = []
        for feature in range(numFeature):
            data.append(float(x[feature]))
        data.append(int(x[numFeature]))
        testdataset.append(data)

    #print(len(testdataset))

    count = 0
    sampleNo = 1
    for data in testdataset:
        xArray = np.array(data)
        actualClass = xArray[numFeature]
        xArray[numFeature] = 1
        product = np.dot(w, xArray.transpose())
        predictedClass = 0
        if product >= 0:
            predictedClass = 1
        else:
            predictedClass = 2

        if predictedClass == actualClass:
            count += 1

        print(sampleNo,data,predictedClass)

        sampleNo += 1

    print("Accuracy :", float((count / len(testdataset)) * 100), "%")







filetrain = open("trainLinearlySeparable.txt")
w,numFeature = trainSet(filetrain)
filetest = open("testLinearlySeparable.txt")
testSet(filetest,w,numFeature)