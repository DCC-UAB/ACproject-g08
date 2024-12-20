from svd import *
from content_based import *
from extreure_metadata import *
from user_model import *
from user_to_user_experiments import *

def menu_principal():
    print('1. Content based recommender')
    print('2. User-to-user collaborative recommender')
    print('3. SVD Algorithm recommender')
    print()

def main_principal():
    print('Projecte 08: Sistemes Recomanadors de Pel·lícules')
    print()
    menu_principal()
    opcio = int(input('Escolleix quina opció vols executar: '))
    
    if opcio == 1:
        main_content()

    elif opcio == 2:
        main_user()

    elif opcio == 3:
        main_svd()

if __name__ == "__main__":
    main_principal()