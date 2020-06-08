from sklearn.utils.validation import check_array
from sklearn.metrics import silhouette_samples, silhouette_score

from skmodes.util import encode_features, decode_centroids
from kmodes.util.dissim import matching_dissim, ng_dissim
from kmodes.kprototypes import KPrototypes, _split_num_cat

clusterPredictions = kprototypes.fit_predict(dataToAnalyze, categorical=dtaCatColumns)
labels = kprotos.labels_
centroids = kprotos.cluster_centroids_

cost = kprotos.cost_
nIter = kprotos.n_iter_
usedGamma = kprotos.gamma

print('The gamma used is: ' + str(usedGamma))

# calculate the Silhouette score for K-means and K-modes sides 
silScoreKMeans = silhouette_score(Xnum, labels, metric='euclidean')
print('KMeans silhouette: ' + str(silScoreKMeans))

silScoreKModes = silhouette_score(XCat, labels, metric='hamming')
print('KMeans silhouette: ' + str(silScoreKModes))

# Average silhouette
silScore = (silScoreKmeans + silScoreKModes) / 2
print('The avg silscore: ', silScore)
silScoreList.append(silScore)

print('the number of cluster is ' + str(clusterN) + 'and the score is: ' str(silScore))
kprotoResultsDict[clusterN] = [silScore, randomSeed, clusterPredictions, labels, centroids, cost, nIter, userdGamma, silScoreKMeans, silScoreKModes]

if silScore > maxSilScore:
    maxSilScore = silScore
    maxNC = clusterN

overallResultsDict[randomSeed] = kprotoResultsDict

avgSilScore = pstats.mean(silScoreList)
