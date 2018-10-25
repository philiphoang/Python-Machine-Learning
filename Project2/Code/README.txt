-- How to run the program --

- Install Python 
	- Very important to have not install version 3.7, because it is not 
	  compatible with library tensorflow
	- Use instead 3.6 (Version 3.6.5 was used on this project) 

- Install necessary libraries
	- use pip install <library>

- Install data from MNIST 
	- The project uses the data:
		- 'handwritten_digits_images.csv'
		- 'handwritten_digits_labels.csv'
	- The code uses relative path 
		- Place these two CSV file in a directory one level up relative to the code 
			- e.g., create a folder called 'data' and place the files inside this folder
			- The folder should be in the same directory as the code

- Run file 'main.py'
	- Can use terminal to run this file (on Windows): 
		'python main.py' 