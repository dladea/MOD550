import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.metrics import r2_score

# Assignment Point 7

# Read the dataset
# The dataset is created by Svetlana Andrusenko and can be found in https://github.com/svetaandrusenko/MOD550_Andrusenko/tree/main/MOD550/data
url = "https://raw.githubusercontent.com/svetaandrusenko/MOD550_Andrusenko/main/MOD550/data/my_data/dataset.csv"
df = pd.read_csv(url)

# Guessing truth function
x1 = df['x'][:99]
y1 = df['y'][:99]
x1_guess = x1.sort_values()
y1_guess = x1_guess**2 + 2  # Quadratic function based on metadata

# Regression
model = np.polyfit(x1, y1, 2)
x1_regression = np.linspace(-20, 20, 100)
y1_regression = np.polyval(model, x1_regression)

print(f"Model based on regression: {model}")
print(f"R^2 Metric: {r2_score(y1, np.polyval(model, x1))}")

plt.scatter(x1, y1, color='b', label='Original Data - Quadratic')
plt.scatter(df['x'][100:], df['y'][100:], color='r', label='Original Data - Random')
plt.plot(x1_guess, y1_guess, color='g', label='Guessed Data - Quadratic')
plt.plot(x1_regression, y1_regression, color='y', label='Regression Data - Quadratic')
plt.xlim(-25, 25)
plt.ylim(-600, 600)

plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.title('Original Dataset and Guessed Truth Function \n (Dea Lana Asri - 277575)')
plt.grid()
plt.savefig('guessed_truth_function.png')
plt.show()


# Assignment Point 8
Header = {
    'Name': 'Dea Lana Asri - 277575',
    'Email': 'dl.asri@stud.uis.no',
    'Github': 'dladea',
    'Title': 'Assessment on Coding Standards on 3 different Github Repositories',
    'Description': 'The assesment will be done in respect to 4 categories: Code Layout, Whitepace in Expressions and Statements, Comments, and Naming Conventions',
}

# Github 1: https://github.com/FoucauldE/CSC/blob/494148f94647e8a2d1f5101707c33f8333c85616/csc_lib/visualize.py#L4
# Accessed on February 2, 2025
Github1 = {
    '1. Code Layout' : '\n- Indentation is consistent by using 4 spaces, \n- some lines are too long (more than 79 characters), like line 29.',
    '2. Whitepace in Expressions and Statements**' : 'Whitespaces are used consistently which is good for readability',
    '3. Comments' : '\n- The use of comments is enough, \n- The function is missing docstring.',
    '4. Naming Conventions' : 'Naming conventions is followed well for file name, function, and variables',
}

# Github 2: https://github.com/noturlee/Titanic-DataModel/blob/main/DataSetTitanic/TitanicAnalysis.py
# Accessed on February 3, 2025
Github2 = {
    '1. Code Layout' : 'Indentation is consistent by using 4 spaces and lines are not too long',
    '2. Whitepace in Expressions and Statements' : 'Whitespaces are used consistently which is good for readability',
    '3. Comments' : 'It does not have comments to explain the code',
    '4. Naming Conventions' : 'Variable names used are descriptive and follow the naming conventions',
}

# Github 3:https://github.com/scorpionhiccup/StockPricePrediction/blob/master/scripts/preprocessing.py
# Accessed on February 3, 2025
Github3 = {
    '1. Code Layout' : '\n- Indentation is consistent by using 4 spaces, \n- Import should usually be in separate lines, \n- Some lines are too long (more than 79 characters), like line 65.',
    '2. Whitepace in Expressions and Statements' : 'Whitespace is used consistently which is good for readability',
    '3. Comments' : '\n- It does not have comments to explain the code, \n- The function is missing docstring.',
    '4. Naming Conventions' : 'Naming conventions is followed well for function and variables',
}

# Create txt file for assignment point 8
try:
    with open('coding_standards.txt', 'w') as f:
        f.write('MOD550 ASSIGNMENT 1 - POINT 8\n')
        for key, value in Header.items():
            f.write(f'{key}: {value}\n')
        f.write('\nGithub 1: https://github.com/FoucauldE/CSC/blob/494148f94647e8a2d1f5101707c33f8333c85616/csc_lib/visualize.py#L4\nAccessed on February 2, 2025\n')
        for key, value in Github1.items():
            f.write(f'{key}: {value}\n')
        f.write('\nGithub 2: https://github.com/noturlee/Titanic-DataModel/blob/main/DataSetTitanic/TitanicAnalysis.py\nAccessed on February 3, 2025\n')
        for key, value in Github2.items():
            f.write(f'{key}: {value}\n')
        f.write('\nGithub 3: https://github.com/scorpionhiccup/StockPricePrediction/blob/master/scripts/preprocessing.py\nAccessed on February 3, 2025\n')
        for key, value in Github3.items():
            f.write(f'{key}: {value}\n')
    print("Text file saved successfully.")
except Exception as e:
    print(f"Error while saving the file: {e}")

