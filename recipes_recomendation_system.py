import streamlit as st
import pandas as pd
import difflib
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

def find_similar_image_name(image_name, image_folder_path):
    all_images = os.listdir(image_folder_path)

    # Usar a biblioteca difflib para encontrar o nome mais parecido
    closest_match = difflib.get_close_matches(image_name, all_images, n=1, cutoff=0.8)

    if closest_match:
        return closest_match[0]
    else:
        return None


df_recipes = pd.read_csv("Food Ingredients and Recipe Dataset with Image Name Mapping.csv")

df_reduzido = df_recipes.head(1000)

#df_reduzido = df_recipes.head(1000)

df_recipes = df_reduzido.iloc[:, 1:-1]

ingredients_list = df_recipes.iloc[:, 1].tolist()

instructions_list = df_recipes.iloc[:, 2].tolist()

recipes_list = df_recipes.iloc[:, 0].tolist()

image_recipes_list = df_recipes.iloc[:, 3].tolist()

larissa_avaliation = [0,1,1,1,0,0,1,1,1,1,0,1,0,1,0]
lucas_avaliation = [1,1,1,0,0,1,1,1,1,1,0,1,1,0,1]
adriano_avaliation = [0,1,1,1,0,0,1,1,1,1,0,1,0,1,1]
tiago_avaliation = [1,1,1,1,1,1,0,0,1,0,0,1,1,1,1]
johann_avaliation = [1,1,1,1,0,0,1,1,1,1,1,0,1,1,1]

tfidf_vectorizer_ingredients = TfidfVectorizer()
tfidf_matrix_ingredients = tfidf_vectorizer_ingredients.fit_transform(ingredients_list)

tfidf_vectorizer_instructions = TfidfVectorizer()
tfidf_matrix_instructions = tfidf_vectorizer_instructions.fit_transform(instructions_list)

# Streamlit app
st.title("Sistema de Recomendação de Receitas com TF/IDF")

# Input do usuário
query_recipe = st.text_area("Digite sua receita de consulta:")

if st.button("Buscar Receitas Similares"):
    if query_recipe:
        # Transformar o documento de consulta em um vetor TF/IDF
        # encontrar o query_ingredients e o query_instructions no dataset pelo nome do query_recipe
        match = df_recipes[df_recipes['Title'].str.contains(query_recipe, case=False)].iloc[0]

        query_title = match.iloc[0]
        query_ingredients = match.iloc[1]
        query_instructions = match.iloc[2]

        query_recipe_combined = f"{query_ingredients} {query_instructions}"

        query_tfidf_ingredients = tfidf_vectorizer_ingredients.transform([query_ingredients])
        query_tfidf_instructions = tfidf_vectorizer_instructions.transform([query_instructions])


        # Calcular a similaridade entre o documento de consulta e todos os documentos na coleção
        cosine_similarities_ingredients = linear_kernel(query_tfidf_ingredients, tfidf_matrix_ingredients).flatten()
        cosine_similarities_instructions = linear_kernel(query_tfidf_instructions, tfidf_matrix_instructions).flatten()



        # Obter os índices dos documentos mais similares
        alpha = 0.5  # Ajuste conforme necessário
        combined_similarities = alpha * cosine_similarities_ingredients + (1 - alpha) * cosine_similarities_instructions

        # Obter os índices dos documentos mais similares
        document_indices_combined = combined_similarities.argsort()[::-1]


        # Exibir os documentos mais similares e suas similaridades
        st.subheader("Receitas Recomendadas:")
        total_relevant_items = 0
        relevant_items_recomendeds = 0
        total_recomended_items = 7
        for i, index in enumerate(document_indices_combined):
            if query_title != recipes_list[index]:
                similarity = combined_similarities[index]
                st.write(f"{i + 1}. Receita {index + 1}: {recipes_list[index]}")
                st.write(f"  - Ingredientes: {ingredients_list[index]}")
                st.write(f"  - Instruções de Preparo: {instructions_list[index]}")
                st.write(f"  - Similaridade: {similarity:.4f}")

                # Buscar imagem correspondente
                image_folder_path = "Food Images/Food Images"
                image_name = find_similar_image_name(image_recipes_list[index], image_folder_path)
                
                # Exibir imagem se encontrada
                if image_name:
                    image_path = os.path.join(image_folder_path, image_name)
                    st.image(image_path, caption=f"Imagem correspondente à receita {image_recipes_list[index]}", use_column_width=True)
                else:
                    st.warning("Imagem correspondente não encontrada.")

            if i >= 7:
                break
    else:
        st.warning("Por favor, digite uma receita de consulta.")