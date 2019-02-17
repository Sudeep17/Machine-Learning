import sys
import csv
def createTree(Instances, labels):
    # create tree code
    pass

def processDataSet(dataSet):
    data = []
    attribute = []
    classValues = []
    with open(dataSet) as csvDataFile:
        csvReader = csv.reader(csvDataFile)
        for row in csvReader:
            data.append(row)
            classValues.append(row[-1])
        attribute = data[0]
        # print(attribute)
        data.remove(attribute)
        classValues.remove('Class')
        # classValues = [examples[-1] for examples in data]
        # print(np.size(data,0))
        return data, attribute, classValues

def main():
    # Take inputs from the command line
    argumentList =['L', 'K', 'training_set', 'validation_set', 'test_set','to_print']
    argValues ={}

    if(len(sys.argv)!=7):
        print("improper number of arguments")
        return 0
    for x in range(0,6):
        argValues[argumentList[x]] = sys.argv[x+1] #Store argument value in argValues variable
    # print(argValues)

    # Creating a tree
    Instances, labels, classValue = processDataSet(argValues['training_set'])
    mytree = createTree(Instances, labels)
    # VItree = createTreeForVI(myDat, labels)


if __name__ == '__main__':
    main()
