# USAGE
# python perceptron_salmon.py

# import the necessary packages
from modules.nn import Perceptron
import numpy as np

# construct the Salmon (freshwater and saltater) dataset
X = np.array([[83, 510], 
[86, 505], 
[94, 490], 
[118, 490], 
[86, 480], 
[98, 480], 
[101, 472], 
[120, 472], 
[90, 470], 
[100, 470], 
[101, 470], 
[105, 470], 
[75, 450], 
[83, 452], 
[85, 450], 
[85, 442], 
[75, 440], 
[93, 440], 
[105, 440], 
[52.5, 425], 
[78, 431], 
[82, 430], 
[95, 431], 
[105, 432], 
[87, 422], 
[95, 427], 
[102, 428], 
[114, 427], 
[109, 420], 
[111, 421], 
[126, 422], 
[95, 411], 
[70, 397], 
[80, 399], 
[84, 399], 
[87, 402], 
[92, 404], 
[98, 403], 
[98, 402], 
[104, 404], 
[121, 402], 
[106, 439], 
[109, 398], 
[112, 394], 
[114, 397], 
[107, 368], 
[118, 382], 
[126, 371], 
[136, 357], 
[95, 430], 


[135, 440], 
[129, 420], 
[156, 420], 
[128, 400], 
[144, 403], 
[152.5, 403], 
[178, 408], 
[129, 390], 
[140, 390], 
[149, 392], 
[152.5, 394], 
[154, 390], 
[128, 382], 
[134, 382], 
[148, 382], 
[152, 381], 
[170, 395], 
[120, 359], 
[133, 373], 
[138, 371], 
[140, 373], 
[148, 372], 
[163, 370], 
[170, 375], 
[123, 352], 
[140, 351], 
[162.5, 369], 
[90, 385], 
[115, 355], 
[117, 356], 
[135, 356], 
[145, 356], 
[150, 355], 
[152.5, 354], 
[155, 352], 
[123, 350], 
[125, 343], 
[126, 342], 
[131, 342], 
[144, 342], 
[107, 340], 
[116, 344], 
[124, 341], 
[144, 339], 
[150, 340], 
[112.5, 330], 
[114, 323], 
[122, 304], 
[152, 301], 
[118,381], 
])

y = np.array([[0], 
[0], 
[0], 
[0], 
[0], 
[0], 
[0], 
[0], 
[0], 
[0], 
[0], 
[0], 
[0], 
[0], 
[0], 
[0], 
[0], 
[0], 
[0], 
[0], 
[0], 
[0], 
[0], 
[0], 
[0], 
[0], 
[0], 
[0], 
[0], 
[0], 
[0], 
[0], 
[0], 
[0], 
[0], 
[0], 
[0], 
[0], 
[0], 
[0], 
[0], 
[0], 
[0], 
[0], 
[0], 
[0], 
[0], 
[0], 
[0], 
[0], 
[1], 
[1], 
[1], 
[1], 
[1], 
[1], 
[1], 
[1], 
[1], 
[1], 
[1], 
[1], 
[1], 
[1], 
[1], 
[1], 
[1], 
[1], 
[1], 
[1], 
[1], 
[1], 
[1], 
[1], 
[1], 
[1], 
[1], 
[1], 
[1], 
[1], 
[1], 
[1], 
[1], 
[1], 
[1], 
[1], 
[1], 
[1], 
[1], 
[1], 
[1], 
[1], 
[1], 
[1], 
[1], 
[1], 
[1], 
[1], 
[1], 
[1]])

# define our perceptron and train it

print("[INFO] training perceptron...")
p = Perceptron(X.shape[1], alpha=0.00001)	#alpha=0.00001 y epochs=80
p.fit(X, y, epochs=80)

# now that our perceptron is trained we can evaluate it
print("[INFO] testing perceptron...")

# now that our network is trained, loop over the data points
i = 1
for (x, target) in zip(X, y):
	# make a prediction on the data point and display the result
	# to our console
	pred = p.predict(x)
	print("[INFO] data={}, punto={}, ground-truth={}, pred={}".format(
		x, i, target[0], pred))
	i = i + 1

#Imprime el vector de pesos
W = p.weights()
print(W)
