from data import*

def main():
	data_gen_args = dict(rotation_range=0.2,
                    	width_shift_range=0.05,
                    	height_shift_range=0.05,
                    	shear_range=0.05,
                    	zoom_range=0.05,
                    	horizontal_flip=True,
                    	fill_mode='nearest')
	myGenerator = trainGenerator(20,'D:\\ben\\Unet-tf\\dataset\\Train','images','label',data_gen_args,save_to_dir = "D:\\ben\\Unet-tf\\dataset\\Train\\aug")


	num_batch = 3

	for i,batch in enumerate(myGenerator):
		if(i>=num_batch):
			break

if __name__ == '__main__':
	main()			