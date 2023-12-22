# %%
#data libraries
import numpy as np
import pandas as pd

#visulization libraries
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# %%
#read the datasets
movies = pd.read_csv('../archive/movies.csv')
links = pd.read_csv('../archive/links.csv')
ratings = pd.read_csv('../archive/ratings.csv')
tags = pd.read_csv('../archive/tags.csv')

print(movies.shape, links.shape, ratings.shape, tags.shape)

# %%
#view the heads of the data
#view the infos of the data
movies.head(5)
movies.info()

# %%
links.head(5)
links.info()

# %%
tags.head(5)
tags.info()

# %%
ratings.head(5)
ratings.info()

# %%
print(tags['tag'].value_counts())

# %%
print(ratings['rating'].mean())
print(movies['genres'].value_counts())
print(ratings['userId'].value_counts())

# %%
print(movies['movieId'].value_counts())
print(len(ratings['rating']))

# %%
#drop some useless attribution
ratings.drop(columns='timestamp', inplace=True)
tags.drop(columns='timestamp', inplace=True)

# %%
#Extracting the year from the title
movies['year'] = movies['title'].str.extract('.*\((.*)\).*', expand=False)
movies.head(5)

# %%
#Seperate the genres and encoding with One-Hot Encoding Method
genres = []
for i in range(len(movies.genres)):
    for x in movies.genres[i].split('|'):
        if x not in genres:
            genres.append(x)
print(len(genres))
genres

# %%
for x in genres:
    movies[x] = 0
for i in range(len(movies.genres)):
    for x in movies.genres[i].split('|'):
        movies[x][i] = 1
movies.head(5)

# %%
movies.head(4)

# %%
x = {}
for i in movies.columns[4:]:
    # print(i)
    x[i] = movies[i].value_counts()[1]
    print("{} \t\t\t\t{}".format(i, x[i]))
# x
# plt.bar(height=x.values(), x=x.keys())
plt.barh(y=list(x.keys()), width=list(x.values()))
# plt.gca().invert_xaxis()
# plt.show()
plt.gcf().set_size_inches(6,6)
plt.savefig('../img/genres.png', bbox_inches='tight')
pass

# %%
# movies.drop(columns='rating_x',inplace=True)
# movies.drop(columns='rating_y', inplace=True)
movies.head(4)

# %%
#Add a Column 'rating' in movie DF and
tmp = ratings.groupby('movieId').rating.mean()
movies = pd.merge(movies,tmp, how='outer', on='movieId')
movies['rating'].fillna('0',inplace=True)
movies.head(5)

# %%
#
x = ratings.groupby('movieId', as_index=False).userId.count()
x.sort_values('userId', ascending=False,inplace=True)
y=pd.merge(movies, x, how='outer', on='movieId')
y.head(5)

# %%
# y.drop(columns=[i for i in movies.columns[2:-1]],inplace=True)
# y.sort_values(['userId','rating'], ascending=False)
# y.rating.fillna(0, inplace=True)
# y.sort_values(['rating'], ascending=False)
# y
# y.rating.dtypes

# %%
if 30892 in ratings.movieId:
    print(1)
else:
    print(0)

# %%
x = ratings.groupby('userId', as_index=False).movieId.count()
y = ratings.groupby('userId', as_index=False).rating.mean()
x = pd.merge(x, y, how='outer', on='userId')

# %%
x.describe()

# %%
x_movie = ratings.groupby('movieId', as_index=False).userId.count()
y_movie = ratings.groupby('movieId', as_index=False).rating.mean()
x_movie = pd.merge(x_movie, y_movie, how='outer', on='movieId')
x_movie

# %%
x_movie.describe()

# %%
tmp_movies = movies
movies.head(5)

# %%
movies.drop(columns='genres', inplace=True)
movies.head(5)

# %%
movies.info()

# %%
movies.to_csv('../archive/data.csv', index=False)


