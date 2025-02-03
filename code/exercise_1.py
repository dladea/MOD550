import matplotlib.pyplot as plt
import pandas as pd


# Assignment Point 7

# Read the dataset
df = pd.read_csv('../data/dataset.csv')

# Guessing truth function
x1 = df['x'][:99]
y1 = df['y'][:99]
x1_guess = x1.sort_values()
y1_guess = x1_guess**2 + 2

plt.scatter(x1, y1, color='b', label='Original Data - Quadratic')
plt.plot(x1_guess, y1_guess, color='g', label='Guessed Data - Quadratic')
plt.scatter(df['x'][100:], df['y'][100:], color='r', label='Original Data - Random')
plt.xlim(-25, 25)
plt.ylim(-600, 600)

plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.title('Original Dataset and Guessed Truth Function \n (Dea Lana Asri - 277575)')
plt.grid()
plt.show()
plt.savefig('guessed_truth_function.png')

# Assignment Point 8
Header = {
    'Name': 'Dea Lana Asri - 277575',
    'Email': 'dl.asri@stud.uis.no',
    'Github': 'dladea',
    'Title': 'Assessment on Coding Standards on 3 different Github Repositories',
    'Description': '',
    'Sources' : '1. 2. 3. '
}

# Github 1: https://github.com/FoucauldE/CSC/blob/494148f94647e8a2d1f5101707c33f8333c85616/csc_lib/visualize.py#L4
# Accessed on February 2, 2025
Github1 = {
    'Code Layout' : '1. Indentation is consistent by using 4 spaces, \n 2. some lines are too long (more than 79 characters), like line 29.',
    'Whitepace in Expressions and Statements' : 'Whitespaces are used consistently which is good for readability',
    'Comments' : '1. The use of comments is enough, \n 2. The function is missing docstring.',
    'Naming Conventions' : 'Naming conventions is followed well for file name, function, and variables',
}

# Github 2: https://github.com/noturlee/Titanic-DataModel/blob/main/DataSetTitanic/TitanicAnalysis.py
# Accessed on February 3, 2025
Github2 = {
    'Code Layout' : 'Indentation is consistent by using 4 spaces and lines are not too long',
    'Whitepace in Expressions and Statements' : 'Whitespaces are used consistently which is good for readability',
    'Comments' : 'It does not have comments to explain the code',
    'Naming Conventions' : 'Variable names used are descriptive and follow the naming conventions',
}

# Github 3:https://github.com/scorpionhiccup/StockPricePrediction/blob/master/scripts/preprocessing.py
# Accessed on February 3, 2025
Github3 = {
    'Code Layout' : '1. Indentation is consistent by using 4 spaces, \n 2. Import should usually be in separate lines, \n 3. Some lines are too long (more than 79 characters), like line 65.',
    'Whitepace in Expressions and Statements' : 'Whitespace is used consistently which is good for readability',
    'Comments' : '1. It does not have comments to explain the code, \n 2. The function is missing docstring.',
    'Naming Conventions' : 'Naming conventions is followed well for function and variables',
}

# Create txt file for assignment point 8
with open('coding_standards.txt', 'w') as f:
    f.write('Header\n')
    for key, value in Header.items():
        f.write(f'{key}: {value}\n')
    f.write('\nGithub 1: https://github.com/FoucauldE/CSC/blob/494148f94647e8a2d1f5101707c33f8333c85616/csc_lib/visualize.py#L4\n Accessed on February 2, 2025\n')
    for key, value in Github1.items():
        f.write(f'{key}: {value}\n')
    f.write('\nGithub 2: https://github.com/noturlee/Titanic-DataModel/blob/main/DataSetTitanic/TitanicAnalysis.py\n Accessed on February 3, 2025\n')
    for key, value in Github2.items():
        f.write(f'{key}: {value}\n')
    f.write('\nGithub 3: https://github.com/scorpionhiccup/StockPricePrediction/blob/master/scripts/preprocessing.py\n Accessed on February 3, 2025\n')
    for key, value in Github3.items():
        f.write(f'{key}: {value}\n')

