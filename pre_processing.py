import pandas as pd

df_recipes = pd.read_csv("Food Ingredients and Recipe Dataset with Image Name Mapping.csv")

df_reduzido = df_recipes.head(15)

df_recipes = df_reduzido.iloc[:, 1:-1]

recipe_name = "Miso-Butter Roast Chicken With Acorn Squash Panzanella"

ingredients_list = df_recipes.iloc[:, 1].tolist()

instructions_list = df_recipes.iloc[:, 2].tolist()

recipes_list = df_recipes.iloc[:, 3].tolist()

match = df_recipes[df_recipes['Title'].str.contains(recipe_name, case=False)].iloc[0]

ingredients = match.iloc[1]
instructions = match.iloc[2]

df_recipes.to_csv("Recipes_dataset.csv", index=False)

