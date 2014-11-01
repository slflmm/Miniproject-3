# ------------------------
# Getting test predictions using SVM and gabor features 
# ------------------------
#trainInput = loadnp("C:/Users/MicroMicro/Documents/Benjamin/Anaconda/Miniproject-3/src/train_inputs_gabor.npy")
#trainOutput = loadnp("C:/Users/MicroMicro/Documents/Benjamin/Anaconda/Miniproject-3/src/train_outputs.npy")

#print "lengths features: " + str(len(trainInput[0]))
#print "lengths output: " + str(trainOutput[0])

#(validSet, trainSet, validSetY, trainSetY) = splitValidTest(trainInput, trainOutput, 0.1)
#print "done splitting... "

#(validSetY, trainSetY) = fixOutputs(validSetY, trainSetY)
#print validSetY
#print trainSetY

#print "Testing using pixel features... training..."
#clf = SVC(kernel='linear')
#clf.fit(trainSet, trainSetY)

#print "checking against Valid Set..."
#acc = clf.score(validSet, validSetY)
#print acc