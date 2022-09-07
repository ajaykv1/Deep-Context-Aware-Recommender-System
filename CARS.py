#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 23 11:26:53 2022

@author: ajaykrishnavajjala
"""
#%%
import tensorflow as tf
import keras
from tensorflow.keras.layers import Dense, Embedding, Flatten, Concatenate, Average, Input, Multiply, Dropout
from tensorflow.keras.regularizers import l2
from tensorflow.keras.models import Model
import matplotlib as plt
import sklearn
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pandas as pd
import os
import datetime
import numpy as np
#%%
dataset = pd.read_csv("/Users/ajaykrishnavajjala/Documents/School/PHD/Recommender Systems/Summer 2022 Project/Datasets/CoMoDa/Context-Aware Datasets/LDOS-CoMoDa.csv")
#%%
print(dataset.head())
#%%
dataset = dataset.drop(["decision", "interaction"], axis=1)
dataset = dataset.drop(["director", "movieCountry", "movieLanguage", "movieYear", "genre1", "genre2", "genre3", "actor1", "actor2", "actor3", "budget"], axis=1)
#%%
print(dataset.head())
#%%
# making sure item id's are from 0-n
dataset["userID"] = pd.Categorical(dataset["userID"])
dataset["user_id"] = dataset["userID"].cat.codes
dataset = dataset.drop("userID", axis=1)

dataset["itemID"] = pd.Categorical(dataset["itemID"])
dataset["item_id"] = dataset["itemID"].cat.codes
dataset = dataset.drop("itemID", axis=1)

dataset["age"] = pd.Categorical(dataset["age"])
dataset["Age"] = dataset["age"].cat.codes
dataset = dataset.drop("age", axis=1)

dataset["sex"] = pd.Categorical(dataset["sex"])
dataset["Sex"] = dataset["sex"].cat.codes
dataset = dataset.drop("sex", axis=1)

dataset["city"] = pd.Categorical(dataset["city"])
dataset["City"] = dataset["city"].cat.codes
dataset = dataset.drop("city", axis=1)

dataset["country"] = pd.Categorical(dataset["country"])
dataset["Country"] = dataset["country"].cat.codes
dataset = dataset.drop("country", axis=1)

dataset["time"] = pd.Categorical(dataset["time"])
dataset["Time"] = dataset["time"].cat.codes
dataset = dataset.drop("time", axis=1)

dataset["daytype"] = pd.Categorical(dataset["daytype"])
dataset["day_type"] = dataset["daytype"].cat.codes
dataset = dataset.drop("daytype", axis=1)

dataset["season"] = pd.Categorical(dataset["season"])
dataset["Season"] = dataset["season"].cat.codes
dataset = dataset.drop("season", axis=1)

dataset["location"] = pd.Categorical(dataset["location"])
dataset["Location"] = dataset["location"].cat.codes
dataset = dataset.drop("location", axis=1)

dataset["weather"] = pd.Categorical(dataset["weather"])
dataset["Weather"] = dataset["weather"].cat.codes
dataset = dataset.drop("weather", axis=1)

dataset["social"] = pd.Categorical(dataset["social"])
dataset["Social"] = dataset["social"].cat.codes
dataset = dataset.drop("social", axis=1)

dataset["endEmo"] = pd.Categorical(dataset["endEmo"])
dataset["end_emo"] = dataset["endEmo"].cat.codes
dataset = dataset.drop("endEmo", axis=1)

dataset["dominantEmo"] = pd.Categorical(dataset["dominantEmo"])
dataset["dom_emo"] = dataset["dominantEmo"].cat.codes
dataset = dataset.drop("dominantEmo", axis=1)

dataset["mood"] = pd.Categorical(dataset["mood"])
dataset["Mood"] = dataset["mood"].cat.codes
dataset = dataset.drop("mood", axis=1)

dataset["physical"] = pd.Categorical(dataset["physical"])
dataset["Physical"] = dataset["physical"].cat.codes
dataset = dataset.drop("physical", axis=1)
#%%
print(dataset.head())
#%%
ratings = {0:0, 1:0, 2:0, 3:1, 4:1, 5:1}
dataset["rating"] = dataset["rating"].map(ratings)
#%%
print(dataset.head())
#%%
users = dataset["user_id"].values
items = dataset['item_id'].values
ratings = dataset['rating'].values
contexts = dataset[['Age', "Sex", "City", "Country", "Time", "day_type", 
                    "Season", "Location", "Weather", 'Social', 'end_emo', 'dom_emo', 'Mood', 'Physical']].values
#%%
Ntrain = int(0.7 * len(ratings))

train_users = users[:Ntrain]
train_items = items[:Ntrain]
train_ratings = ratings[:Ntrain]
train_contexts = contexts[:Ntrain]

test_users = users[Ntrain:]
test_items = items[Ntrain:]
test_ratings = ratings[Ntrain:]
test_contexts = contexts[Ntrain:]
#%%
users_d = {}
for i in range(len(users)):
    if users[i] in users_d:
        users_d[users[i]].append(contexts[i])
    else:
        users_d[users[i]] = [contexts[i]]
for key in users_d:
    avg_arr = np.mean(users_d[key], axis=0)
    users_d[key] = avg_arr
#%%
contexts = np.array(list(users_d.values()))
print(contexts[0].shape)
num_context = len(users_d.values())
num_users = len(set(users))
num_items = len(set(items))
#%%
def create_cars_model (num_users, num_items, num_contexts, context_weights):
    
    # Initializing Inputs
    user_input = Input(shape=(1,), name='userInput')
    item_input = Input(shape=(1,), name='itemInput')
    context_input = Input(shape=context_weights[0].shape, name='contextInput')
    
    # MF embeddings
    mf_user_embedding = Embedding(num_users, 14, embeddings_initializer="RandomNormal", embeddings_regularizer=l2(0), name='mf_user_emb') (user_input)
    mf_item_embedding = Embedding(num_items, 14, embeddings_initializer="RandomNormal", embeddings_regularizer=l2(0), name='mf_item_emb') (item_input)
    
    # MLP Embeddings
    mlp_user_embedding = Embedding(num_users, 14, embeddings_initializer="RandomNormal", embeddings_regularizer=l2(0.01), name='mlp_user_emb') (user_input)
    mlp_item_embedding = Embedding(num_items, 14, embeddings_initializer="RandomNormal", embeddings_regularizer=l2(0.01), name='mlp_item_emb') (item_input)
    
    # TF embeddings
    tf_user_embedding = Embedding(num_users, 14, embeddings_initializer="RandomNormal", embeddings_regularizer=l2(0), name='tf_user_emb') (user_input)
    tf_item_embedding = Embedding(num_items, 14, embeddings_initializer="RandomNormal", embeddings_regularizer=l2(0), name='tf_item_emb') (item_input)
    
    # GMF Layer
    mf_user_embedding = Flatten(name='mf_user_flatten') (mf_user_embedding)
    mf_item_embedding = Flatten(name='mf_item_flatten') (mf_item_embedding)
    gmf_layer = Multiply(name='mf_multiply') ([mf_user_embedding, mf_item_embedding])
    
    # GTF Layer
    tf_user_embedding = Flatten(name='tf_user_flatten') (tf_user_embedding)
    tf_item_embedding = Flatten(name='tf_item_flatten') (tf_item_embedding)
    gtf_layer = Multiply(name='tf_multiply') ([tf_user_embedding, tf_item_embedding, context_input])
    
    # CF Layer
    gtf_layer = Flatten(name='gtf_flatten') (gtf_layer)
    cf_layer = Concatenate(name='cf_concat') ([gmf_layer, gtf_layer])
    
    # MLP Layer
    mlp_user_embedding = Flatten(name='flatten_mlp_user') (mlp_user_embedding)
    mlp_item_embedding = Flatten(name='flatten_mlp_item') (mlp_item_embedding)
    mlp_cat_layer = Concatenate(name='mlp_cat_layer') ([mlp_user_embedding, mlp_item_embedding, context_input])
    
    # Deep Network
    mlp_layer = Dense(1024, activity_regularizer=l2(0.05), activation='relu') (mlp_cat_layer)
    mlp_layer = Dropout(0.2) (mlp_layer)
    mlp_layer = Dense(1024, activity_regularizer=l2(0.05), activation='relu') (mlp_layer)
    mlp_layer = Dropout(0.2) (mlp_layer)
    mlp_layer = Dense(512, activity_regularizer=l2(0.05), activation='relu') (mlp_layer)
    mlp_layer = Dropout(0.2) (mlp_layer)
    mlp_layer = Dense(512, activity_regularizer=l2(0.05), activation='relu') (mlp_layer)
    mlp_layer = Dropout(0.2) (mlp_layer)
    mlp_layer = Dense(256, activity_regularizer=l2(0.05), activation='relu') (mlp_layer)
    mlp_layer = Dropout(0.2) (mlp_layer)
    mlp_layer = Dense(256, activity_regularizer=l2(0.05), activation='relu') (mlp_layer)
    mlp_layer = Dropout(0.2) (mlp_layer)
    mlp_layer = Dense(128, activity_regularizer=l2(0.05), activation='relu') (mlp_layer)
    mlp_layer = Dropout(0.2) (mlp_layer)
    mlp_layer = Dense(128, activity_regularizer=l2(0.05), activation='relu') (mlp_layer)
    mlp_layer = Dropout(0.2) (mlp_layer)

    # Concatenating cf layer and mlp layers
    output_layer = Concatenate(name='concat_cf_gtf_layerss') ([cf_layer, mlp_layer])
    output_layer = Dense(1, activation='sigmoid') (output_layer)
    
    cdcars_model = Model(inputs=[user_input,item_input, context_input], outputs=output_layer)
    
    return cdcars_model
#%%
cars_model = create_cars_model(num_users, num_items, num_context, contexts)
#%%
cars_model.compile(
        optimizer=tf.keras.optimizers.Adam(0.01),
        loss='binary_crossentropy',
        metrics=[
                tf.keras.metrics.BinaryAccuracy(name="accuracy"),
                tf.keras.metrics.Precision(name="precision"),
                tf.keras.metrics.Recall(name="recall"),
                tf.keras.metrics.RootMeanSquaredError(name="rmse"),
                tf.keras.metrics.MeanSquaredError(name="mse"),
                tf.keras.metrics.TopKCategoricalAccuracy(k=5, name="topk_acc")
            ]
    )

#%%
#cars_model.summary()
#%%
logdir = os.path.join("logs", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
tensorboard_callback = tf.keras.callbacks.TensorBoard(logdir, histogram_freq=1)
early_stopping_callback = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss"
)
#%%
cars_results = cars_model.fit(x = [train_users, train_items, train_contexts], 
        y = train_ratings, 
        epochs = 10, 
        validation_data = ([test_users, test_items, test_contexts], test_ratings),
        callbacks=[tensorboard_callback, early_stopping_callback],
        batch_size=512
    )
#%%
plt.plot(cars_results.history["loss"], label = 'loss')
plt.plot(cars_results.history['val_loss'], label = 'val_loss')
plt.legend()
#%%
plt.plot(cars_results.history['accuracy'], label='train_acc')
plt.plot(cars_results.history['val_accuracy'], label='val_acc')
plt.legend()
#%%
plt.plot(cars_results.history['rmse'], label='train_rmse')
plt.plot(cars_results.history['val_rmse'], label='val_rmse')
plt.legend()
#%%
def create_deep_cars_model (num_users, num_items, num_contexts, context_weights):
    
    # Initializing Inputs
    user_input = Input(shape=(1,), name='userInput')
    item_input = Input(shape=(1,), name='itemInput')
    context_input = Input(shape=context_weights[0].shape, name='contextInput')
    
    # MF embeddings
    mf_user_embedding = Embedding(num_users, 14, embeddings_initializer="RandomNormal", embeddings_regularizer=l2(0), name='mf_user_emb') (user_input)
    mf_item_embedding = Embedding(num_items, 14, embeddings_initializer="RandomNormal", embeddings_regularizer=l2(0), name='mf_item_emb') (item_input)
    
    # MLP Embeddings
    mlp_user_embedding = Embedding(num_users, 14, embeddings_initializer="RandomNormal", embeddings_regularizer=l2(0.01), name='mlp_user_emb') (user_input)
    mlp_item_embedding = Embedding(num_items, 14, embeddings_initializer="RandomNormal", embeddings_regularizer=l2(0.01), name='mlp_item_emb') (item_input)
        
    # GMF Layer
    mf_user_embedding = Flatten(name='mf_user_flatten') (mf_user_embedding)
    mf_item_embedding = Flatten(name='mf_item_flatten') (mf_item_embedding)
    gmf_layer = Multiply(name='mf_multiply') ([mf_user_embedding, mf_item_embedding])
    
    # MLP Layer
    mlp_user_embedding = Flatten(name='flatten_mlp_user') (mlp_user_embedding)
    mlp_item_embedding = Flatten(name='flatten_mlp_item') (mlp_item_embedding)
    mlp_cat_layer = Concatenate(name='mlp_cat_layer') ([mlp_user_embedding, mlp_item_embedding, context_input])
    
    # Deep Network
    mlp_layer = Dense(1024, activity_regularizer=l2(0.05), activation='relu') (mlp_cat_layer)
    mlp_layer = Dropout(0.2) (mlp_layer)
    mlp_layer = Dense(1024, activity_regularizer=l2(0.05), activation='relu') (mlp_layer)
    mlp_layer = Dropout(0.2) (mlp_layer)
    mlp_layer = Dense(512, activity_regularizer=l2(0.05), activation='relu') (mlp_layer)
    mlp_layer = Dropout(0.2) (mlp_layer)
    mlp_layer = Dense(512, activity_regularizer=l2(0.05), activation='relu') (mlp_layer)
    mlp_layer = Dropout(0.2) (mlp_layer)
    mlp_layer = Dense(256, activity_regularizer=l2(0.05), activation='relu') (mlp_layer)
    mlp_layer = Dropout(0.2) (mlp_layer)
    mlp_layer = Dense(256, activity_regularizer=l2(0.05), activation='relu') (mlp_layer)
    mlp_layer = Dropout(0.2) (mlp_layer)
    mlp_layer = Dense(128, activity_regularizer=l2(0.05), activation='relu') (mlp_layer)
    mlp_layer = Dropout(0.2) (mlp_layer)
    mlp_layer = Dense(128, activity_regularizer=l2(0.05), activation='relu') (mlp_layer)
    mlp_layer = Dropout(0.2) (mlp_layer)

    # Concatenating cf layer and mlp layers
    output_layer = Concatenate(name='concat_cf_gtf_layerss') ([gmf_layer, mlp_cat_layer])
    output_layer = Dense(1, activation='sigmoid') (output_layer)
    
    cdcars_model = Model(inputs=[user_input,item_input, context_input], outputs=output_layer)
    
    return cdcars_model
#%%
deep_cars_model = create_deep_cars_model(num_users, num_items, num_context, contexts)
#%%
deep_cars_model.compile(
        optimizer=tf.keras.optimizers.Adam(0.01),
        loss='binary_crossentropy',
        metrics=[
                tf.keras.metrics.BinaryAccuracy(name="accuracy"),
                tf.keras.metrics.Precision(name="precision"),
                tf.keras.metrics.Recall(name="recall"),
                tf.keras.metrics.RootMeanSquaredError(name="rmse"),
                tf.keras.metrics.MeanSquaredError(name="mse"),
                tf.keras.metrics.TopKCategoricalAccuracy(k=5, name="topk_acc")
            ]
    )

#%%
#deep_cars_model.summary()
#%%
logdir = os.path.join("logs", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
tensorboard_callback = tf.keras.callbacks.TensorBoard(logdir, histogram_freq=1)
early_stopping_callback = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss"
)
#%%
deep_cars_results = deep_cars_model.fit(x = [train_users, train_items, train_contexts], 
        y = train_ratings, 
        epochs = 10, 
        validation_data = ([test_users, test_items, test_contexts], test_ratings),
        callbacks=[tensorboard_callback, early_stopping_callback],
        batch_size=512
    )
#%%
plt.plot(deep_cars_results.history["loss"], label = 'loss')
plt.plot(deep_cars_results.history['val_loss'], label = 'val_loss')
plt.legend()
#%%
plt.plot(deep_cars_results.history['accuracy'], label='train_acc')
plt.plot(deep_cars_results.history['val_accuracy'], label='val_acc')
plt.legend()
#%%
plt.plot(deep_cars_results.history['rmse'], label='train_rmse')
plt.plot(deep_cars_results.history['val_rmse'], label='val_rmse')
plt.legend()
#%%
def create_ncf_model (num_users, num_items):
    
    # Initializing Inputs
    user_input = Input(shape=(1,), name='userInput')
    item_input = Input(shape=(1,), name='itemInput')
    
    # MF embeddings
    mf_user_embedding = Embedding(num_users, 14, embeddings_initializer="RandomNormal", embeddings_regularizer=l2(0), name='mf_user_emb') (user_input)
    mf_item_embedding = Embedding(num_items, 14, embeddings_initializer="RandomNormal", embeddings_regularizer=l2(0), name='mf_item_emb') (item_input)
    
    # MLP Embeddings
    mlp_user_embedding = Embedding(num_users, 14, embeddings_initializer="RandomNormal", embeddings_regularizer=l2(0.01), name='mlp_user_emb') (user_input)
    mlp_item_embedding = Embedding(num_items, 14, embeddings_initializer="RandomNormal", embeddings_regularizer=l2(0.01), name='mlp_item_emb') (item_input)
    
    # GMF Layer
    mf_user_embedding = Flatten(name='mf_user_flatten') (mf_user_embedding)
    mf_item_embedding = Flatten(name='mf_item_flatten') (mf_item_embedding)
    gmf_layer = Multiply(name='mf_multiply') ([mf_user_embedding, mf_item_embedding])
    
    # MLP Layer
    mlp_user_embedding = Flatten(name='flatten_mlp_user') (mlp_user_embedding)
    mlp_item_embedding = Flatten(name='flatten_mlp_item') (mlp_item_embedding)
    mlp_cat_layer = Concatenate(name='mlp_cat_layer') ([mlp_user_embedding, mlp_item_embedding])
    
    # Deep Network
    mlp_layer = Dense(1024, activity_regularizer=l2(0.05), activation='relu') (mlp_cat_layer)
    mlp_layer = Dropout(0.2) (mlp_layer)
    mlp_layer = Dense(1024, activity_regularizer=l2(0.05), activation='relu') (mlp_layer)
    mlp_layer = Dropout(0.2) (mlp_layer)
    mlp_layer = Dense(512, activity_regularizer=l2(0.05), activation='relu') (mlp_layer)
    mlp_layer = Dropout(0.2) (mlp_layer)
    mlp_layer = Dense(512, activity_regularizer=l2(0.05), activation='relu') (mlp_layer)
    mlp_layer = Dropout(0.2) (mlp_layer)
    mlp_layer = Dense(256, activity_regularizer=l2(0.05), activation='relu') (mlp_layer)
    mlp_layer = Dropout(0.2) (mlp_layer)
    mlp_layer = Dense(256, activity_regularizer=l2(0.05), activation='relu') (mlp_layer)
    mlp_layer = Dropout(0.2) (mlp_layer)
    mlp_layer = Dense(128, activity_regularizer=l2(0.05), activation='relu') (mlp_layer)
    mlp_layer = Dropout(0.2) (mlp_layer)
    mlp_layer = Dense(128, activity_regularizer=l2(0.05), activation='relu') (mlp_layer)
    mlp_layer = Dropout(0.2) (mlp_layer)

    # Concatenating cf layer and mlp layers
    output_layer = Concatenate(name='concat_cf_gtf_layerss') ([gmf_layer, mlp_cat_layer])
    output_layer = Dense(1, activation='sigmoid') (output_layer)
    
    cdcars_model = Model(inputs=[user_input,item_input], outputs=output_layer)
    
    return cdcars_model
#%%
ncf_model = create_ncf_model(num_users, num_items)
#%%
ncf_model.compile(
        optimizer=tf.keras.optimizers.Adam(0.01),
        loss='binary_crossentropy',
        metrics=[
                tf.keras.metrics.BinaryAccuracy(name="accuracy"),
                tf.keras.metrics.Precision(name="precision"),
                tf.keras.metrics.Recall(name="recall"),
                tf.keras.metrics.RootMeanSquaredError(name="rmse"),
                tf.keras.metrics.MeanSquaredError(name="mse"),
                tf.keras.metrics.TopKCategoricalAccuracy(k=5, name="topk_acc")
            ]
    )

#%%
#ncf_model.summary()
#%%
logdir = os.path.join("logs", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
tensorboard_callback = tf.keras.callbacks.TensorBoard(logdir, histogram_freq=1)
early_stopping_callback = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss"
)
#%%
ncf_results = ncf_model.fit(x = [train_users, train_items], 
        y = train_ratings, 
        epochs = 10, 
        validation_data = ([test_users, test_items], test_ratings),
        callbacks=[tensorboard_callback, early_stopping_callback],
        batch_size=512
    )
#%%
plt.plot(ncf_results.history["loss"], label = 'loss')
plt.plot(ncf_results.history['val_loss'], label = 'val_loss')
plt.legend()
#%%
plt.plot(ncf_results.history['accuracy'], label='train_acc')
plt.plot(ncf_results.history['val_accuracy'], label='val_acc')
plt.legend()
#%%
plt.plot(ncf_results.history['rmse'], label='train_rmse')
plt.plot(ncf_results.history['val_rmse'], label='val_rmse')
plt.legend()














