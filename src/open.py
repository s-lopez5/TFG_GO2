import pickle

with open("lista_obs.pkl", "rb") as f:
    lista = pickle.load(f)

print(len(lista))