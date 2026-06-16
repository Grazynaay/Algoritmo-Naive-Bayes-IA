from src.dataset_loader import carregar_iris

iris = carregar_iris()

iris.info()

print("\nQuantidade por espécie:")
print(iris["species"].value_counts())