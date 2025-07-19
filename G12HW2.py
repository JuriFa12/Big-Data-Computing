from pyspark import SparkContext, SparkConf
from pyspark.mllib.clustering import KMeans
import time
import sys
import os
import math
import numpy as np
import random as rand

#Function used to read each line of the input document and transform it into a tuple (point, group), 
#where point is a tuple containing the coordinates of the point. We also coint the number of points in group A and group B 
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


#Function used to split an element of the kind (random ket, list of values) into 2 keys of the type (groupi, sum of partial counts of points in groupi)
def split_demographic_sets (point):

    k, v = point

    return [(v[0][0], v[0][1]), (v[1][0], v[1][1])]




#Functions used to transform a list of (point, group) into a tuple ((group1, partial sum of distances of points in group1),
#(group2, partial sum of distances of points in group 2)
def fair_point_to_distances(points, centroids):

    #Centroids has shape number_clusters x dimensionality of points
    centroids = np.vstack([np.asarray(c, dtype=float) for c in centroids])

    #Save coordinatates and groups in different structures
    point_coords = np.vstack([np.array(p[0]) for p in points])
    groups = [p[1] for p in points]

    #Calculates all distances between points and centroids and stores the minimums
    dists = np.sum((point_coords[:, None, :] - centroids[None, :, :]) ** 2, axis=2)
    min_dists = np.min(dists, axis=1)

    # Do different sums, one for each group
    sumA = np.sum([d for d, g in zip(min_dists, groups) if g == 'A'])
    sumB = np.sum([d for d, g in zip(min_dists, groups) if g == 'B'])

    return (('A', sumA), ('B', sumB))



#Function used to calculate the objective function associated with the fair variant of LLoyd's algorithm
def MRComputeFairObjective (points, centroids, F, element_count_a, element_count_b):

    obj = (points.map(lambda x: (rand.randint(0, F-1), x)) #MAP 1
           .groupByKey() #SHUFFLING
           .mapValues(lambda x: fair_point_to_distances(x, centroids)) #REDUCE 1
           .flatMap(lambda x: split_demographic_sets(x)) #REDUCE 1
           .reduceByKey (lambda x, y: x + y) #REDUCE 2
           ).cache()
    
    final_counts = obj.collect()  #we used the function collect since at the end of the Map-Reduce we have a small RDD
    final_sum_A = 0
    final_sum_B = 0

    for e in final_counts:

        if e[0] == 'A':
            final_sum_A = e[1]
        else:
            final_sum_B = e[1]

    final_obj = max((final_sum_A/element_count_a.value), (final_sum_B/element_count_b.value))

    return final_obj



def computeVectorX(fixed_a, fixed_b, alpha, beta, ell, k):
    gamma = 0.5
    x_dist = [0.0] * k
    power = 0.5
    t_max = 10
    

    for _ in range(t_max):
        f_a = fixed_a
        f_b = fixed_b
        power /= 2

        for i in range(k):
            temp = (1 - gamma) * beta[i] * ell[i] / (gamma * alpha[i] + (1 - gamma) * beta[i])
            x_dist[i] = temp
            f_a += alpha[i] * temp * temp
            temp = ell[i] - temp
            f_b += beta[i] * temp * temp

        if f_a == f_b:
            break

        gamma = gamma + power if f_a > f_b else gamma - power

    return x_dist

   
#Function that implements the sequential algorithm used to retrieve the centroids for FairLloyd
def centroid_selection(elements, number_clusters):

    elements = list(elements)   
    dim = len(elements[0][1][0])        #Take the dimension of the points of the dataset

    #Allocate all different lists where values will be stored, this is done by using "number clusters". Value i of the list represents a certain amount for cluster i
    cardinality_list_A = [0] * number_clusters
    cardinality_list_B = [0] * number_clusters
    total_cardinality_A = 0
    total_cardinality_B = 0
    sum_list_A = [np.zeros(dim, dtype=np.float64) for _ in range(number_clusters)]
    sum_list_B = [np.zeros(dim, dtype=np.float64) for _ in range(number_clusters)]
    alpha = [0] * number_clusters
    beta = [0] * number_clusters
    mu_alpha = [np.zeros(dim, dtype=np.float64) for _ in range(number_clusters)]
    mu_beta = [np.zeros(dim, dtype=np.float64) for _ in range(number_clusters)]
    l = [np.zeros(dim, dtype=np.float64) for _ in range(number_clusters)]
    Delta_M_A = 0
    Delta_M_B = 0
    percentage_points_A = 0
    percentage_points_B = 0


    
    for e in elements:  #Iterate through all elements, which are points

        #Check if it belongs to group A
        if e[1][1] == 'A':
            cardinality_list_A[e[0]] += 1   #Count it in the list that contain cardinality of points of group A in the cluster number e[0], which indicates the cluster
            #of the point which is being processed
            total_cardinality_A += 1 #Regardless of the cluster, sum it in the total count of points belonging to group A
            for i in range(dim):       #Sum the point to the list that accumulates the sum of elements of group A in cluster of index e[0]
                sum_list_A[e[0]][i] += e[1][0][i]

        #Same as before, but for group B
        else:
            cardinality_list_B[e[0]] += 1
            total_cardinality_B += 1
            for i in range(dim):
                sum_list_B[e[0]][i] += e[1][0][i]
        
    
    for i in range(number_clusters): #For each cluster compute the needed parameters for the algorithm

        if (cardinality_list_A[i] != 0):
            alpha[i] = cardinality_list_A[i] / total_cardinality_A
            mu_alpha[i] = sum_list_A[i] / cardinality_list_A[i]

        else:
            alpha[i] = 0

            if cardinality_list_B[i] != 0:
                mu_alpha[i] = sum_list_B[i] / cardinality_list_B[i] 
            else:
                mu_alpha[i][:]= 0

            

        if (cardinality_list_B[i] != 0):
            beta[i] = cardinality_list_B[i] / total_cardinality_B
            mu_beta[i] = sum_list_B[i] / cardinality_list_B[i]

        else:
            beta[i] = 0

            if cardinality_list_A[i] != 0:
                mu_beta[i] = sum_list_A[i] / cardinality_list_A[i] #Prova
            else:
                mu_beta[i][:] = 0

    
        l[i] = np.linalg.norm(mu_alpha[i] - mu_beta[i])


    for e in elements:
        if e[1][1] == 'A':
            Delta_M_A += np.linalg.norm(np.array(e[1][0] - mu_alpha[e[0]])) ** 2
        else:
            Delta_M_B += np.linalg.norm(np.array(e[1][0] - mu_beta[e[0]])) ** 2

    
    fixed_A = Delta_M_A / total_cardinality_A
    fixed_B = Delta_M_B / total_cardinality_B

    #Compute percentage of points of group 'A' and of group 'B' in the list of points given
    percentage_points_A = total_cardinality_A/ (total_cardinality_A + total_cardinality_B)
    percentage_points_B = 1 - percentage_points_A
    
    
    x = computeVectorX(fixed_A, fixed_B, alpha, beta, l, number_clusters)

    new_centroids = [0] * number_clusters

    #For each cluster now make its centroid, assigning to it a group label using percentanges calcolated before
    for i in range(number_clusters):

        if l[i] != 0:
            coords = ((l[i] - x[i]) * mu_alpha[i] + x[i] * mu_beta[i]) / l[i]
        else:
            coords = mu_alpha[i]
        
        new_centroids[i] = ( i, (tuple(coords), rand.choices(['A', 'B'], weights=[percentage_points_A, percentage_points_B])[0] ) )

    return new_centroids



#Function which takes a list of points and returns a list of elements where each one is composed in the following way : (index_cluster, ((x,y), Group))
def assign_clusters(points, cluster_centers):

    #Here we decided to use numpy since it is written in C and is more efficient then doing a for loop on all the elements of the list given to the function,
    #since it can contain a lot of elements
    centroids = np.array(cluster_centers)   
    point_coords = np.array([p[0] for p in points])
    groups = [p[1] for p in points]

    #Compute distances of each point from every centroid and then take the minimum among all of them
    diff = point_coords[:, None, :] - centroids[None, :, :]
    dists = np.sum(diff**2, axis=2)  
    min_indices = np.argmin(dists, axis=1)

    return [(int(cluster), (tuple(point_coords[i]), groups[i])) for i, cluster in enumerate(min_indices)]
             
            
#Function that implements the Socially-Fair variant of LLoyd algorithm
#We decided, in order to obtain faster computations despite large datasets, to implement the function using the Composable Coreset Technique
#Each coreset is obtained through the centroid_selection function which is applied on random partitions of the dataset
def MRFairLloyd(points, number_clusters, iterations):

    #Compute initial centroids 
    clustering = KMeans.train(points.map(lambda x: np.array(x[0])), number_clusters, 0)
    centroids = clustering.clusterCenters
    F = int(math.sqrt(points.count() / number_clusters)) #Maximum value which a random key can take
    

    for i in range(iterations): 
        
        #Partitioning of U in U1, ..., Uk
        partitioned_points = (points.map(lambda x: (rand.randint(0, F - 1), x)) #MAP 1
        .groupByKey() #SHUFFLING
        .mapValues(lambda x: assign_clusters(x, centroids)) #REDUCE 1
        .mapValues(lambda x: centroid_selection(x, number_clusters)) #REDUCE 1
        .flatMap(lambda x:  x[1]) #REDUCE 1
        .map(lambda x: (1, x)) #MAP 2
        .groupByKey() #SHUFFLING
        .mapValues(lambda x: centroid_selection(x, number_clusters)) #REDUCE 2
        ).cache()
        
        temp_centroids = partitioned_points.collect()  #Collect centroids obtained in this iteration
        centroids_list = temp_centroids[0][1]  

        #Prepare list in which centroids will be stored
        centroid_arrays = [None] * number_clusters

        #Loop used to update current centroids
        for cluster_id, (centroid_array, _) in centroids_list:
            centroid_arrays[cluster_id] = centroid_array 

        centroids = centroid_arrays 

    #Same as before but done with the last set of centroids retrieved
    temp_centroids = partitioned_points.collect()
    centroids_list = temp_centroids[0][1]
    final_centroids = np.empty(number_clusters, dtype=object)


    #Save cetroinds in an array of tuples
    for cluster_id, (centroid_array, _) in centroids_list:
       final_centroids[cluster_id] = tuple(centroid_array)
    

    return final_centroids




def main():
    
    # Checking number of cmd line parameters
    assert len(sys.argv) == 5, "User must provide  Data-path, L, K and M"

    # Spark setup
    conf = SparkConf().setAppName("Homework 2").set("spark.ui.showConsoleProgress", "false")
    sc = SparkContext(conf=conf)
    sc.setLogLevel("ERROR")
    

    # Input Reading 

    # 1. Read input file
    data_path = sys.argv[1]
    #assert os.path.isfile(data_path), "File not provided correctly" Line has been eliminated in order to function on CloudVeneto

    # 2. Read number of partitions and subdivide input file in L partitions
    L = sys.argv[2]
    assert L.isdigit(), "L must be an integer"
    L = int(L)
    data = sc.textFile(data_path).repartition(numPartitions = L)

    # 2. Read number of clusters
    K = sys.argv[3]
    assert K.isdigit(), "K must be an integer"
    K = int(K)

    # 3. Read number of iterations
    M = sys.argv[4]
    assert M.isdigit(), "M must be an integer"
    M = int(M)

    #We set up the variables used to store the amount of points, the amount of points in group A and the amount of points in group B
    N = data.count()
    NA_acc = sc.accumulator(0)
    NB_acc = sc.accumulator(0)
    
    #We read the input document and trasform it into an RDD of points, which is later used to train our clustering model
    #Then we get our centroids
    inputPoints = (data.map(lambda x: input_point_per_document(x, NA_acc, NB_acc))).repartition(numPartitions = L).cache()
    inputPoints.count()

    #We obtain the "standard centroids" using the Llody's Algorithm on the input points, using K clusters and M iterations
    start = time.perf_counter() #Set time reference
    clustering = KMeans.train(inputPoints.map(lambda x: np.array(x[0])), K, M)
    end = time.perf_counter()   #Get time spent for normal KMeans
    time_C_stand =(end - start) * 1000  #Transform in milliseconds
    C_stand = clustering.clusterCenters     

    #Same as before but for FairLloyd
    start = time.perf_counter() 
    C_fair = MRFairLloyd(inputPoints, K, M)
    end = time.perf_counter()
    time_C_fair = (end - start) * 1000


    #Compute FairObjective with 2 different sets of centroids and get their time
    start = time.perf_counter()
    obj_stand = MRComputeFairObjective(inputPoints, C_stand, int(math.sqrt(N/K)), NA_acc, NB_acc)
    end = time.perf_counter()
    time_obj_stand = (end - start) * 1000

    start = time.perf_counter()
    obj_fair = MRComputeFairObjective(inputPoints, C_fair, int(math.sqrt(N/K)), NA_acc, NB_acc)
    end = time.perf_counter()
    time_obj_fair = (end - start) * 1000


    #Print of our different final values
    print(f"Input file = {data_path}, L = {L}, K = {K}, M = {M}")
    print(f"N = {N}, NA = {NA_acc}, NB = {NB_acc}")
    print(f"Fair Objective with Standard Centers = {obj_stand:.4f}")
    print(f"Fair Objective with Fair Centers = {obj_fair:.4f}")
    print(f"Time to compute standard centers = {time_C_stand:.4f} ms")
    print(f"Time to compute fair centers = {time_C_fair:.4f} ms")
    print(f"Time to compute objective with standard centers = {time_obj_stand:.4f} ms")
    print(f"Time to compute objective with fair centers = {time_obj_fair:.4f} ms")
  







if __name__ == "__main__":
	main()


