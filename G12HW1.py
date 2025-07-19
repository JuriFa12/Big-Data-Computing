from pyspark import SparkContext, SparkConf
from pyspark.mllib.clustering import KMeans
import sys
import os
import math
import numpy as np
import random as rand

#Function used to read each line of the input document and transform it into a tuple (point, group), where point is a tuple containing the coordinates of the point
def input_point_per_document(document, NA_acc, NB_acc):

    elements = document.split(',')
    dimensions = len(elements)
    point = tuple(map(float, elements[:-1]))
    group = elements[dimensions - 1]

    if group == "A":
        NA_acc.add(1)

    else:
        NB_acc.add(1)
        
    return (point, group)

#Function used to transform a list of points in the distances associated to their centroids
#Then we sum up all these distances, returning a partial count.
def point_to_sum_of_distances(points, clustering, centroids):

    total_sum = 0
    for p in points:
        index_cluster = clustering.predict(np.array(p))
        selected_centroid = centroids[index_cluster]
        total_sum = total_sum + np.sum((np.array(p) - np.array(selected_centroid)) ** 2)

    return total_sum

#Functions used to transform a list of (point, group) into a tuple ((group1, partial sum of distances of points in group1),(group2, partial sum of distances of points in group 2)
def fair_point_to_distances(points, clustering, centroids):
    
    sumA = 0
    sumB = 0

    for p,g in points:
        index_cluster = clustering.predict(np.array(p))
        selected_centroid = centroids[index_cluster]
        distance = np.sum((np.array(p) - np.array(selected_centroid)) ** 2)

        if g == 'A':
            sumA = sumA + distance
        else:
            sumB = sumB + distance

    return (('A', sumA), ('B', sumB))


#Function used to split an element of the kind (random ket, list of values) into 2 keys of the type (groupi, sum of partial counts of points in groupi)
def split_demographic_sets (point):

    k, v = point

    return [(v[0][0], v[0][1]), (v[1][0], v[1][1])]

#Function used to trasnform a list of (point, group) in a dictionary where each element has a key of the type (index of the cluster, group) and its value
#is the partial count of points in group i belonging to cluster of index k
def transform_cluster_point(points, clustering):

    new_elements = []

    for p in points:
        index_cluster = clustering.predict(np.array(p[0]))
        new_pair= (index_cluster, p[1])
        new_elements.append(new_pair)

    count_dict = {}
    for e in new_elements:
        if e not in count_dict.keys():
            count_dict[e] = 1
        else:
            count_dict[e] += 1

    return count_dict

#Function used to trasnform a element of type (random key, dictionary) into a list of elements of the type ((index of cluster, group), partial count of
#points of the group i belonging to cluster k
def count_dict_into_keys(item):

    key, count_dict = item
    new_items = []

    for e in count_dict.keys():
        new_key = (e, count_dict[e])
        new_items.append(new_key)

    return new_items

#Function used to calculate the objective function associated with LLoyd's algorithm
def MRComputeStandardObjective (points, centroids, clustering, F, element_count):

    obj = (points.map(lambda x: (rand.randint(0,F-1), x[0])) #MAP 1
            .groupByKey()       #SHUFFLING
            .mapValues(lambda x: point_to_sum_of_distances(x, clustering, centroids)) #REDUCE 1
            .map(lambda x: (x[1])) #REDUCE 1
            .reduce(lambda x, y: x + y)) #REDUCE 2
    return obj/element_count

#Function used to calculate the objective function associated with the fair variant of LLoyd's algorithm
def MRComputeFairObjective (points, centroids, clustering, F, element_count_a, element_count_b):

    obj = (points.map(lambda x: (rand.randint(0,F-1), x)) #MAP 1
           .groupByKey() #SHUFFLING
           .mapValues(lambda x: fair_point_to_distances(x, clustering, centroids)) #REDUCE 1
           .flatMap(lambda x: split_demographic_sets(x)) #REDUCE 1
           .reduceByKey (lambda x, y: x + y) #REDUCE 2
           )
    
    final_counts = obj.collect()  #we used the function collect since at the end of the Map-Reduce we have a small RDD
    final_obj = max((final_counts[0][1]/element_count_a.value), (final_counts[1][1]/element_count_b.value))
    return final_obj

#Function used to calculate the triplets of the type (ci, NAi, NBi)
def MRPrintStatistics(points, centroids, clustering, F, K):

    obj = (points.map(lambda x: (rand.randint(0,F-1), x)) #MAP1
           .groupByKey() #SHUFFLING
           .mapValues(lambda x: transform_cluster_point(x, clustering))#REDUCE 1
           .flatMap(lambda x: count_dict_into_keys(x)) #REDUCE1
           .reduceByKey(lambda x, y: x + y) #REDUCE2
           )
    
    clusters_counts = obj.collect()  #we used the function collect since at the end of the Map-Reduce we have a small RDD
    clusters_counts_sorted = sorted(clusters_counts, key=lambda x: x[0])

    # We collect the final triplets in a dictionary in order to print them
    final_result = {}
    for (i, label), value in clusters_counts_sorted:
        if i not in final_result:
            final_result[i] = {'A': 0, 'B': 0}  
        final_result[i][label] = value  

    for i in sorted(final_result.keys()):
        center = tuple(centroids[i])  
        NA = final_result[i].get('A', 0)
        NB = final_result[i].get('B', 0)
        print(f"i = {i}, center = {center}, NA{i} = {NA}, NB{i} = {NB}")
    
   
def main():
    
    # Checking number of cmd line parameters
    assert len(sys.argv) == 5, "User must provide L, K, M, Data-path"

    # Spark setup
    conf = SparkConf().setAppName("Homework 1").set("spark.ui.showConsoleProgress", "false")
    sc = SparkContext(conf=conf)
    sc.setLogLevel("ERROR")
    

    # Input Reading 

    # 1. Read number of partitions
    L = sys.argv[1]
    assert L.isdigit(), "L must be an integer"
    L = int(L)

    # 2. Read number of clusters
    K = sys.argv[2]
    assert K.isdigit(), "K must be an integer"
    K = int(K)

    # 3. Read number of iterations
    M = sys.argv[3]
    assert M.isdigit(), "M must be an integer"
    M = int(M)

    # 4. Read input file and subdivide it into L random partitions
    data_path = sys.argv[4]
    assert os.path.isfile(data_path), "File not provided correctly"
    data = sc.textFile(data_path).repartition(numPartitions = L)

    #We set up the variables used to store the amount of points, the amount of points in group A and the amount of points in group B
    N = data.count()
    NA_acc = sc.accumulator(0)
    NB_acc = sc.accumulator(0)
    
    #We read the input document and trasform it into an RDD of points, which is later used to train our clustering model
    #Then we get our centroids
    inputPoints = (data.map(lambda x: input_point_per_document(x, NA_acc, NB_acc))).repartition(numPartitions = L).cache()
    clustering = KMeans.train(inputPoints.map(lambda x: np.array(x[0])), K, M)
    centroids = clustering.clusterCenters

    print(f"Input file = {data_path}, L = {L}, K = {K}, M = {M}")
    print(f"N = {N}, NA = {NA_acc}, NB = {NB_acc}")

    #We call the functions defined above where the square root of N is used to generate the random keys
    obj = MRComputeStandardObjective(inputPoints, centroids, clustering, int(math.sqrt(N)), N)
    print(f"Delta(U,C) = {obj}")

    obj = MRComputeFairObjective(inputPoints, centroids, clustering, int(math.sqrt(N)), NA_acc, NB_acc)
    print(f"Phi(A,B,C) = {obj}")

    obj = MRPrintStatistics(inputPoints, centroids, clustering, int(math.sqrt(N)), K)


if __name__ == "__main__":
	main()