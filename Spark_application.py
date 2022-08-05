# -*- coding: utf-8 -*-

# Projet 8, Déployez un modèle dans le cloud 

##################################
# Chargement des librairies
##################################
import os, sys
import pandas as pd
import io
import boto3
import findspark
findspark.init()
findspark.find()
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, split, pandas_udf, PandasUDFType
from PIL import Image
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from datetime import datetime

##################################
# Creating SparkSession
##################################

spark = (
    SparkSession
    .builder
    .appName("p8_aws")
    .getOrCreate()
)


##################################
# # Importation des images
##################################

ROOT_PATH = sys.argv[1]

# Parsing the "train" dataset. Use data/*/** for training + testing sets
DATA_PATH = os.path.join(ROOT_PATH, "Data/fruits-360-original-size/Sample/**")

print("Loading images...")
image_df = (
    spark
    .read
    .format("image")
    .load(DATA_PATH)
)

print("Number of partitions:", image_df.rdd.getNumPartitions())
# AWS IAM user credentials

image_df = image_df.withColumn(
    'category', split(col('path'), '/').getItem(5))
image_df = image_df.select('path', 'content', 'category')
image_df.show(10)


images_s = image_df.sample(withReplacement=False,
                              fraction=0.02,
                              seed=0)
print(images_s.count())
images_s.printSchema()
images_s.show()


# Extraction des features par le Transfer Learning
model = ResNet50(
    # retirer la couche fully-connected: (include_top=False) 
    include_top=False,
    # Charger les pois pré-entraînés sur ImageNet,
    weights='imagenet',
    # Utiliser Keras tensor comme l’image entrée pour le modèle
    input_tensor=None,
    # À spécifier uniquement si include_top=False (shape tuple)
    input_shape=(224, 224, 3),
    # À spécifier uniquement si include_top=False
    pooling='max')

# Vérifiez que la couche supérieure est supprimée
model.summary()


bc_model_weights = sc.broadcast(model.get_weights())

def load_model():

    model = ResNet50(include_top=False,
                     weights='imagenet',
                     input_shape=(224, 224, 3),
                     pooling='max')
    #on ajoute les pondérations
    model.set_weights(bc_model_weights.value)
    
    return model

def image_preprocessing(image):
   
    # on ouvre et redimensionne l'image car Resnet 50 ne prend en charge que des images de taille 224x224
    image = Image.open(io.BytesIO(image)).resize([224, 224])
    
    # Changer le type d'image en matrice 
    image = img_to_array(image)
    
    # Preprocessing: Normaliser les données d'entrées
    image = preprocess_input(image)

    # Fonction 'preprocess_input' de Keras utilisée dans le modèle ResNet50
    # C'est un ensemble de préprocessing propres à l'utilisation de ResNet 
    
    return image

def feature_extractor(model, series_content):

    # Prétraitement des images
    input = np.stack(series_content.map(image_preprocessing))
    
    # Extraire les features des images
    predictions = model.predict(input)
    
    # Pour certaines couches, les caractéristiques de sortie sont des tensors multidimensionnels, 
    # On aplatit les caractéristiques de tensors en vecteurs pour faciliter le stockage dans les dataframes Spark
    # la fonction flatten() envoie une copie du tableau réduit à une seule dimension.
    features = [p.flatten() for p in predictions]
    
    # Création d'une série Pandas des features des images
    features_series = pd.Series(features)
    
    return features_series

@pandas_udf('array<float>', PandasUDFType.SCALAR_ITER)
def apply_udf_featurizer(content):

    # Avec Scalar Iterator pandas UDFs, nous pouvons charger le modèle une fois puis le réutiliser. 
    # Pour plusieurs lots de données, cela amortit les frais généraux liés au chargement des modèles importants.
    model = load_model()
    
    # Appliquer la fonction de featurisation
    for series_content in content:
        yield feature_extractor(model, series_content)

numPartitions = 2
features_df = images_s.repartition(numPartitions).select(
    col('path'),
    col('category'),
    col('content'),
    apply_udf_featurizer('content').alias('features'))
features_df.printSchema()
features_df.show()

# datetime object containing current date and time
now = datetime.now()
    
# Converting to string in the format dd-mm-YY H:M:S
string = now.strftime("%b-%d-%Y %H:%M:%S")

# Creating the path for storing the results
RESULTS_PATH = os.path.join(ROOT_PATH, "results", string)

features_df.write.format("parquet").mode("overwrite").save(RESULTS_PATH)


####################################
# Fermer le sparkSession
####################################

spark.stop()
print("Spark application successfully finished.")