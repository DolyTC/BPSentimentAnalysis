import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import math
# soubor obsahující vyhodnocované emoce emočního modelu
# a jejich konkrétní souřadnice
df = pd.read_csv("resources/emotion_list.csv")

# elbow method pro zobrazení optimálního počtu shluků
# pro načtený soubor emocí
def elbow_method(max_k):
    means = []
    inertias = []

    for k in range(1, max_k+1):
        kmeans =  KMeans(n_clusters = k)
        kmeans.fit(df [['valence','arousal']])

        means.append(k)
        inertias.append(kmeans.inertia_)

    plt.subplots(figsize=(10,5))
    plt.plot(means, inertias, 'o-')
    plt.xlabel('Number of clusters')
    plt.ylabel('Inertia')
    plt.grid(True)
    plt.show()

# zobrazení grafu rozdělení shluků pro konkrétní
# zvolené číslo k (počet shluků)
def single_scatter(k_number):
    kmeans = KMeans(n_clusters=k_number)
    kmeans.fit(df [['valence','arousal']])
    df['cluster'] = kmeans.labels_
    print(df.to_string())
    plt.scatter(x=df['valence'], y=df['arousal'], c=df['cluster'])
    plt.xlim(-1,1)
    plt.ylim(-1,1)
    plt.show()

# zobrazení hned 6 grafů shlukové analýzy pro větší přehled
# :param first_scatter_k_number: udává počet shluků prvního grafu
#  následující grafy obsahují vždy o jeden shluk navíc
def six_scatters(first_scatter_k_number):
    for k in range(first_scatter_k_number,
                   first_scatter_k_number + 6):
        kmeans=KMeans(n_clusters=k)
        kmeans.fit(df [['valence','arousal']])
        df[f'KMeans_{k}'] = kmeans.labels_

    fig, axs = plt.subplots(nrows=2,ncols=3, figsize=(20,15))

    for i, ax in enumerate(fig.axes, start=first_scatter_k_number):
        ax.scatter(x=df ['valence'], y=df ['arousal'],
                   c=df[f'KMeans_{i}'])
        ax.set_ylim(-1, 1)
        ax.set_xlim(-1, 1)
        ax.set_title(f'N Clusters: {i}')

    print(df.to_string())
    plt.show()

# vrátí počet shluků, podle kterých bude model vyhodnocovat
# :param mode: "cluster" (rozdělení do několika shluků)
# nebo "single" (každá emoce je svým vlastním shlukem)
def set_cluster(mode):
    if mode == "single":
        clusters = len(df)
    else:
        clusters = 16
    return clusters

# provede shlukovou analýzu a vrátí souřadnice
# nejbližšího shluku k daným souřadnicím
# :param x: výsledná hodnota valence dokumentu
# :param y: výsledná hodnota arousal dekumentu
# :param clusters: počet shluků
def get_cluster(x,y,clusters):
    cluster_x = 0
    cluster_y = 0

    kmeans = KMeans(n_clusters= clusters)
    kmeans.fit(df [['valence','arousal']])
    df['cluster'] = kmeans.labels_
    centers = kmeans.cluster_centers_

    for count, value in enumerate(centers):
        x_2 = value[0]
        y_2 = value[1]
        eu_distance = math.sqrt((x - x_2)**2 + (y - y_2)**2)

        if count == 0:
            shortest_distance = eu_distance
            cluster_x = x_2
            cluster_y = y_2
            n = count
        else:
            if eu_distance < shortest_distance:
                shortest_distance = eu_distance
                cluster_x = x_2
                cluster_y = y_2
                n = count
    return cluster_x, cluster_y, n

# vrátí graf obsahující vyhodnocené emoce s popisky
# :param coordinates: souřadnice vyhodnocených emocí
def show_plot(coordinates):
    fig, ax = plt.subplots()
    ax.scatter(x=coordinates['valence'], y=coordinates['arousal'])
    ax.set_title("Circumplex Model of Affect")
    ax.set_xlabel("Valence")
    ax.set_ylabel("Arousal")
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.axhline(y=0, color='k')
    ax.axvline(x=0, color='k')
    ax.grid()
    for i in range(len(coordinates['emotion'])):
        ax.text(coordinates['valence'].iloc[i],
                coordinates['arousal'].iloc[i],
                coordinates['emotion'].iloc[i])
    plt.show()

# na základě hypotetického výpočtu vrátí hodnotu,
# o kterou se zvýší hodnota arousal s každým přibytým emotikonem
distances = []
def emoticon_increase():
    clusters = set_cluster("single")
    kmeans = KMeans(n_clusters=clusters)
    kmeans.fit(df[['valence', 'arousal']])
    df['cluster'] = kmeans.labels_
    centers = kmeans.cluster_centers_

    for count, value in enumerate(centers):
        x_1 = value[0]
        y_1 = value[1]

        for c, v in enumerate(centers):
            x_2 = value[0]
            y_2 = v[1]

            distance_x = round(abs(x_1 - x_2),3)
            distance = round(abs(y_1 - y_2),3)
            if distance_x <= 0.1 and distance > 0.01:
                distances.append(distance)

    shortest_distance = distances[1]
    for i in distances:
        if i < shortest_distance and i != 0:
            shortest_distance = i

    return shortest_distance