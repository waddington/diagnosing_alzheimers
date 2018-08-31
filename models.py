from keras.layers import Conv2D, Input, MaxPooling2D, Flatten, Dense
from keras.models import Model, Sequential

def shortened_model():
	inputs = Input(shape=(1,256,256))

	p1 = MaxPooling2D()(inputs)

	c1 = Conv2D(64, (3,3), activation='relu')(p1)
	c2 = Conv2D(64, (3,3), activation='relu')(c1)
	c3 = Conv2D(64, (3,3), activation='relu')(c2)
	p2 = MaxPooling2D()(c3)

	c4 = Conv2D(64, (3,3), activation='relu')(p2)
	c5 = Conv2D(64, (3,3), activation='relu')(c4)
	c6 = Conv2D(64, (3,3), activation='relu')(c5)
	p3 = MaxPooling2D()(c6)

	c7 = Conv2D(64, (3,3), activation='relu')(p3)
	c8 = Conv2D(64, (3,3), activation='relu')(c7)
	c9 = Conv2D(64, (3,3), activation='relu')(c8)
	p4 = MaxPooling2D()(c9)

	f1 = Flatten()(p4)

	d1 = Dense(128, activation='relu')(f1)
	d2 = Dense(80, activation='relu')(d1)

	o = Dense(3, activation='softmax')(d2)

	# Using the keras functional model
	model = Model(inputs=inputs, outputs=o)

	return model
